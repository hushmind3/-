"""One registered, vertically composed TradingMoE with frozen native experts."""
import hashlib
import io
import inspect
import json
from pathlib import Path
import tempfile
import time
import zipfile
import numpy as np
import torch
from torch import nn
from .expert_system import adapter_features, build_fusion_head, decode_trading_output, registry_owner
from .moe_native import NativeExpert, native_call
from . import expert_backends
from .gpu_scheduler import FairGpuScheduler
from .moe_inputs import MacroHFTInputAdapter,macro_adapter_metadata
from .paths import EXPERT_ASSETS_DIR, GPU_OWNER_LOCK


def parameter_digest(module):
    digest=hashlib.sha256()
    for name,p in module.named_parameters():
        digest.update(name.encode());digest.update(str(p.dtype).encode());digest.update(str(tuple(p.shape)).encode())
        digest.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class EvidenceAdapter(nn.Module):
    def __init__(self,size):
        super().__init__()
        self.scale=nn.Parameter(torch.ones(size))
        self.bias=nn.Parameter(torch.zeros(size))
    def forward(self,packet,symbols):
        row=adapter_features([packet],symbols)[0]
        x=np.asarray(row["features"],np.float32)
        scale=np.maximum(np.sqrt(np.mean(x.astype(np.float64)**2,axis=-1,keepdims=True)),1e-6)
        meta=np.column_stack([np.log1p(scale[:,0]),np.full(len(x),np.log1p(row["horizon"])),
                             np.full(len(x),np.log1p(row["sampling_seconds"] or 0))])
        values=np.concatenate([x/scale,meta],axis=-1).astype(np.float32)
        mask=np.asarray(row["coverage_mask"],bool);values[~mask]=0
        return torch.from_numpy(values[None]).to(self.scale.device)*self.scale+self.bias,torch.from_numpy(mask[None]).to(self.scale.device)


class VerticalController(nn.Module):
    def __init__(self,sizes,stock_policy_ids=()):
        super().__init__()
        self.stock_policy_ids=list(stock_policy_ids)
        self.macro_policy_ids=sorted(k for k in sizes if k.startswith("macrophft_"))
        self.market_ids=sorted(k for k in sizes if k not in self.stock_policy_ids and k not in self.macro_policy_ids)
        self.policy_ids=self.macro_policy_ids+self.stock_policy_ids
        self.sizes=sizes
        self.router=nn.ModuleDict({k:nn.Linear(sizes[k],1) for k in self.market_ids})
        self.router_context=nn.Linear(16,len(self.market_ids))
        self.market_fusion=build_fusion_head({k:sizes[k] for k in self.market_ids})
        self.policy_adapters=nn.ModuleDict({k:nn.Linear(sizes[k],64) for k in self.policy_ids})
        self.policy_attention=nn.MultiheadAttention(64,4,batch_first=True)
        if self.stock_policy_ids:
            self.policy_router=nn.ModuleDict({k:nn.Linear(sizes[k],1) for k in self.policy_ids})
            # Equal initial preferences; training may learn symbol/context-specific preferences.
            for router in self.policy_router.values():
                nn.init.zeros_(router.weight);nn.init.zeros_(router.bias)
        self.account_context=nn.Linear(16,64)
        self.controller_norm=nn.LayerNorm(64)
        # The residual starts at zero; the learned ETH policy decides first.
        for head in (self.market_fusion.policy,self.market_fusion.allocation,self.market_fusion.cash):
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)

    def forward(self,evidence,validity,account,policy_q=None):
        logits=torch.cat([self.router[k](evidence[k]) for k in self.market_ids],-1)+self.router_context(account)
        market_mask=torch.stack([validity[k] for k in self.market_ids],-1)
        routing=getattr(self,"assembly_routing",{})
        logits=logits/max(.05,float(routing.get("market",{}).get("temperature",1)))
        top_k=int(routing.get("market",{}).get("top_k",0))
        if 0<top_k<len(self.market_ids):
            selected=torch.zeros_like(market_mask).scatter_(-1,logits.masked_fill(~market_mask,-1e9).topk(top_k,-1).indices,True)
            market_mask=market_mask&selected
        gates=logits.masked_fill(~market_mask,-1e9).softmax(-1)*market_mask
        gates=gates/gates.sum(-1,keepdim=True).clamp_min(1e-9)
        # Every available historical expert participates; no initial fixed top-2.
        gates=.95*gates+.05*market_mask/market_mask.sum(-1,keepdim=True).clamp_min(1)
        fused=self.market_fusion({k:evidence[k] for k in self.market_ids},account,
            key_padding_mask=~market_mask,allow_untrained=True,expert_gates=gates*len(self.market_ids))
        latent=fused["shared_latent"]
        policy_mask=torch.stack([validity[k] for k in self.policy_ids],-1)
        tokens=torch.stack([self.policy_adapters[k](evidence[k]) for k in self.policy_ids],-2)
        policy_gates=None
        if self.stock_policy_ids:
            policy_logits=torch.cat([self.policy_router[k](evidence[k]) for k in self.policy_ids],-1)
            policy_logits=policy_logits/max(.05,float(routing.get("policy",{}).get("temperature",1)))
            policy_k=int(routing.get("policy",{}).get("top_k",0))
            if 0<policy_k<len(self.policy_ids):
                selected=torch.zeros_like(policy_mask).scatter_(-1,policy_logits.masked_fill(~policy_mask,-1e9).topk(policy_k,-1).indices,True)
                policy_mask=policy_mask&selected
            policy_gates=policy_logits.masked_fill(~policy_mask,-1e9).softmax(-1)*policy_mask
            policy_gates=policy_gates/policy_gates.sum(-1,keepdim=True).clamp_min(1e-9)
            tokens=tokens*policy_gates[...,None]*policy_mask.sum(-1)[...,None,None]
        b,n,e,w=tokens.shape
        unavailable=~policy_mask.any(-1)
        safe_mask=~policy_mask.clone();safe_mask[unavailable,0]=False
        tokens=tokens.masked_fill(unavailable[...,None,None],0)
        policy,_=self.policy_attention(latent.reshape(b*n,1,w),tokens.reshape(b*n,e,w),tokens.reshape(b*n,e,w),
            key_padding_mask=safe_mask.reshape(b*n,e))
        policy=policy.reshape(b,n,w).masked_fill(unavailable[...,None],0)
        final=self.controller_norm(latent+policy+self.account_context(account))
        head=self.market_fusion
        prior=torch.zeros_like(head.policy(final));allocation_prior=torch.zeros_like(head.allocation(final).squeeze(-1))
        if policy_q is not None:
            q=torch.stack([policy_q[k] for k in self.macro_policy_ids],-2)
            macro_mask=policy_mask[...,:len(self.macro_policy_ids)]
            centered=q-q.mean(-1,keepdim=True)
            direction=centered/(centered.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-8))
            votes=direction.softmax(-1)*macro_mask[...,None]
            votes=votes.sum(-2)/macro_mask.sum(-1,keepdim=True).clamp_min(1)
            flat,long=votes[...,0].clamp_min(1e-6).log(),votes[...,1].clamp_min(1e-6).log()
            held=account[...,1]>1e-6
            prior[...,0]=torch.where(held,flat,torch.full_like(flat,-4))
            prior[...,1]=torch.where(held,long,flat)
            prior[...,2]=torch.where(held,torch.full_like(long,-4),long)
            macro_unavailable=~macro_mask.any(-1)
            prior=prior.masked_fill(macro_unavailable[...,None],0)
            allocation_prior=(long-flat).masked_fill(macro_unavailable,0)
            if self.stock_policy_ids:
                stock_mask=policy_mask[...,len(self.macro_policy_ids):]
                stock_votes=torch.stack([policy_q[k][...,:3] for k in self.stock_policy_ids],-2)
                stock_gates=policy_gates[...,len(self.macro_policy_ids):]*stock_mask
                stock_gates=stock_gates/stock_gates.sum(-1,keepdim=True).clamp_min(1e-9)
                probabilities=(stock_votes*stock_gates[...,None]).sum(-2)
                stock_prior=probabilities.clamp_min(1e-6).log()
                stock_targets=torch.stack([policy_q[k][...,3] for k in self.stock_policy_ids],-1)
                target=(stock_targets*stock_gates).sum(-1).clamp(1e-6,1-1e-6)
                active=stock_mask.any(-1)
                prior=torch.where(active[...,None],stock_prior,prior)
                allocation_prior=torch.where(active,torch.logit(target),allocation_prior)
        return {"policy_logits":head.policy(final)+prior,"value":head.value(final).squeeze(-1),
            "allocation_scores":head.allocation(final).squeeze(-1)+allocation_prior,"cash_scores":head.cash(final.mean(1)),
            "shared_latent":final,"router_probabilities":gates,"policy_validity":policy_mask,
            "coverage":market_mask.any(-1)|policy_mask.any(-1),
            **({"policy_router_probabilities":policy_gates} if policy_gates is not None else {})}


class TradingMoE(nn.Module):
    def __init__(self,experts,config,metadata,root):
        super().__init__()
        self.experts=nn.ModuleDict(experts)
        self.adapters=nn.ModuleDict({k:EvidenceAdapter(config["feature_sizes"][k]) for k in experts})
        self.controller=VerticalController(config["feature_sizes"],config.get("stock_policy_ids",()))
        self.config,self.metadata,self.root=config,metadata,Path(root)
        self.macro_input_adapter=MacroHFTInputAdapter(**(metadata.get("macro_input_adapter") or macro_adapter_metadata(root)))
        self.optimizer_updates=0
        self.scheduler=FairGpuScheduler()
        self.gpu_lock=GPU_OWNER_LOCK
        self.experts.requires_grad_(False).eval()

    def parameter_groups(self,train_experts=()):
        groups=[]
        for key,expert in self.experts.items():
            expert.requires_grad_(key in train_experts)
            if key in train_experts:groups.append({"name":"expert:"+key,"params":list(expert.parameters())})
        for name,module in (("adapter",self.adapters),("controller_router_fusion",self.controller)):
            groups.append({"name":name,"params":list(module.parameters())})
        return groups

    def set_learning_device(self,device):
        """Move only the small learned modules, never all frozen native experts."""
        self.adapters.to(device)
        self.controller.to(device)
        return self

    def _apply(self,fn,recurse=True):
        # Native execution transfers ONE expert. Upper model moves must never
        # materialize all 14 on CUDA as a side effect of model.cuda()/to().
        raise RuntimeError("Move one native expert explicitly; TradingMoE stays on CPU")

    def train(self,mode=True):
        super().train(mode);self.experts.eval();return self

    def prepare(self,packets,symbols):
        evidence,validity={},{}
        packets={p["expert"]:p for p in packets}
        for key,size in self.config["feature_sizes"].items():
            packet=packets.get(key)
            if hasattr(self,"assembly_enabled") and key not in self.assembly_enabled:packet=None
            if packet is None:
                device=self.adapters[key].scale.device
                evidence[key]=torch.zeros(1,len(symbols),size,device=device);validity[key]=torch.zeros(1,len(symbols),dtype=torch.bool,device=device)
            else:
                evidence[key],validity[key]=self.adapters[key](packet,symbols)
                if evidence[key].shape[-1]!=size:raise ValueError(f"native shape changed for {key}")
                # Native adapters already identify which symbols have real outputs.
                # The controller uses evidence coverage, not a second universe gate.
        return evidence,validity

    def forward(self,snapshot,account_state,*,packets=None,device="cpu",explore=False):
        started=time.perf_counter();profiles=[];unavailable_policies={}
        if packets is None:
            packets=[]
            with registry_owner(self.gpu_lock,wait=True,on_wait=getattr(self,"gpu_wait_callback",None)):
                for key,expert in self.experts.items():
                    if hasattr(self,"assembly_enabled") and key not in self.assembly_enabled:continue
                    data=snapshot["expert_inputs"].get(key)
                    if key in self.config.get("stock_policy_ids",()) and data is None:
                        data=expert.prepare_input(snapshot)
                        if data is None:unavailable_policies[key]="symbol outside universe or native history/account unavailable"
                    if key in self.config.get("stock_policy_ids",()) and data is not None:
                        data={**data,"requested_symbols":snapshot["symbols"]}
                    if data is None:continue
                    if key.startswith("macrophft_") and (not data.get("native_features_verified") or
                        data.get("feature_schema")!="MacroHFT_36+9" or data.get("symbols")!=["ETHUSDT"]):continue
                    if key=="marketgpt" and (not data.get("native_features_verified") or
                        data.get("token_schema")!="MarketGPT_ITCH_Vocab_v3"):continue
                    with self.scheduler.work("champion_live"):
                        packet=expert(self.root,data,device)
                    packet["expert"]=key
                    packet["native_features_verified"]=bool(data.get("native_features_verified"))
                    packets.append(packet)
                    profiles.append({"expert":key,"timings":{k:packet.get(k) for k in
                        ("cold_load_seconds","gpu_transfer_seconds","forward_seconds","worker_seconds")}})
                    if device.startswith("cuda"):torch.cuda.empty_cache()
        for packet in packets:
            if np.datetime64(packet["as_of"])>np.datetime64(snapshot["as_of"]):raise ValueError("expert evidence contains future information")
            if packet["expert"].startswith("macrophft_") and not packet.get("native_features_verified"):
                raise ValueError("unverified MacroHFT evidence cannot enter the controller")
        evidence,validity=self.prepare(packets,snapshot["symbols"])
        policy_q=self.policy_q(packets,snapshot["symbols"])
        outputs=self.controller(evidence,validity,account_state.to(next(self.controller.parameters()).device),policy_q)
        trading=decode_trading_output(outputs,snapshot,outputs["coverage"][0].tolist())
        trading["policy_status"]="trained_vertical_controller" if self.optimizer_updates else "native_policy_prior"
        trading["reason"]="Applicable frozen stock/crypto policy prior plus market/account-conditioned controller"
        probabilities=outputs["policy_logits"][0].softmax(-1)
        for n,s in enumerate(snapshot["symbols"]):
            if outputs["policy_validity"][0,n].any():
                chosen=(torch.distributions.Categorical(probabilities[n]).sample() if explore else probabilities[n].argmax())
                trading["actions"][s]=["SELL","HOLD","BUY"][int(chosen)]
        trading["exploration_enabled"]=bool(explore)
        trading["paper_executable"]=bool(outputs["coverage"].any())
        result={"as_of":snapshot["as_of"],"currencies":snapshot["currencies"],"trading_output":trading,
            "tradable_symbols":snapshot.get("tradable_symbols",snapshot["symbols"]),
            "adapter_status":"connected","pipeline_timings":{"total_seconds":time.perf_counter()-started},
            "fusion_output":{"trained":self.optimizer_updates>0,"status":"vertical_controller",
                "shapes":{k:list(v.shape) for k,v in outputs.items()},"native_head_output":{k:v.detach().tolist() for k,v in outputs.items()}},
            "raw_outputs":packets,"used_experts":[p["expert"] for p in packets],"selected_experts":[p["expert"] for p in packets],
            "evidence_as_of":{p["expert"]:p["as_of"] for p in packets},
            "policy_validity":outputs["policy_validity"].tolist(),"profiles":profiles,
            "unavailable_stock_policies":unavailable_policies,
            "decision_seconds":time.perf_counter()-started,"training_performed":False}
        return result,outputs

    def apply_assembly_recipe(self,recipe):
        unknown=set(recipe["enabled_experts"])-set(self.config["feature_sizes"])
        if unknown:raise ValueError("공용 PT에 아직 없는 expert: "+", ".join(sorted(unknown)))
        if recipe.get("controller_variant")!="vertical_native_prior_v1":raise ValueError("unsupported controller variant")
        self.assembly_enabled=set(recipe["enabled_experts"])
        self.controller.assembly_routing={"market":recipe["market_routing"],"policy":recipe["policy_routing"]}

    def save_assembly_state(self,path,optimizer=None):
        """Only learned modules; frozen expert weights never enter this file."""
        destination=Path(path);destination.parent.mkdir(parents=True,exist_ok=True)
        temporary=destination.with_suffix(".partial")
        torch.save({"format":"trading_moe_assembly_v1","feature_sizes":self.config["feature_sizes"],
            "controller":self.controller.state_dict(),"adapters":self.adapters.state_dict(),
            "optimizer":optimizer.state_dict() if optimizer else None,"optimizer_updates":self.optimizer_updates,
            "learning_state":{key:self.config[key] for key in ("replay_account_episode","applied_replay_rows") if key in self.config}},temporary)
        temporary.replace(destination)

    def load_assembly_state(self,path):
        state=torch.load(path,map_location=next(self.controller.parameters()).device,weights_only=True)
        if state["format"]!="trading_moe_assembly_v1" or state["feature_sizes"]!=self.config["feature_sizes"]:
            raise ValueError("assembly state does not match shared base")
        self.controller.load_state_dict(state["controller"]);self.adapters.load_state_dict(state["adapters"])
        self.optimizer_updates=state["optimizer_updates"]
        self.config.update(state.get("learning_state",{}))
        return state.get("optimizer")

    def policy_q(self,packets,symbols):
        device=next(self.controller.parameters()).device
        values={k:torch.zeros(1,len(symbols),2 if k in self.controller.macro_policy_ids else 4,device=device) for k in self.controller.policy_ids}
        for packet in packets:
            if packet["expert"] not in values:continue
            key=packet["expert"]
            if key in self.controller.stock_policy_ids:
                raw=torch.tensor([[r["sell_score"],r["hold_score"],r["buy_score"],r["target_weight"]] for r in packet["common_output"]],device=device)
            else:raw=torch.tensor(packet["native_output"],dtype=torch.float32,device=device).reshape(-1,2)
            for j,s in enumerate(packet["symbols"]):
                if s in symbols:values[key][0,symbols.index(s)]=raw[j]
        return values

    def save_checkpoint(self,path,optimizer=None):
        path=Path(path);temporary=path.with_suffix(".partial")
        path.parent.mkdir(parents=True,exist_ok=True)
        torch.save({"format":"registered_vertical_trading_moe_v1","config":self.config,"metadata":self.metadata,
            "state_dict":self.state_dict(),"expert_mapping":{k:e.entry for k,e in self.experts.items()},
            "optimizer_state":optimizer.state_dict() if optimizer else None,"optimizer_updates":self.optimizer_updates},temporary)
        temporary.replace(path)

    @classmethod
    def load_checkpoint(cls,path,*,cached_market=False):
        saved=torch.load(path,map_location="cpu",weights_only=True,mmap=True)
        if saved["format"]!="registered_vertical_trading_moe_v1":raise ValueError("unknown MoE format")
        temp=tempfile.TemporaryDirectory(prefix="stockrl-moe-native-")
        root=Path(temp.name)
        with zipfile.ZipFile(io.BytesIO(saved["metadata"]["architecture_sources"])) as archive:
            for name in archive.namelist():
                if not (root/name).resolve().is_relative_to(root.resolve()):raise ValueError("invalid source archive path")
            archive.extractall(root)
        experts={}
        for key,entry in saved["expert_mapping"].items():
            if cached_market and not key.startswith("macrophft_"):
                # Replay trials reuse actual native market/stock outputs; no
                # native body materialization. Six crypto Q modules remain real.
                experts[key]=nn.Identity()
                continue
            prefix=f"experts.{key}.models."
            count=saved["config"]["native_module_counts"][key]
            states=[{k.removeprefix(prefix+str(i)+"."):v for k,v in saved["state_dict"].items() if k.startswith(prefix+str(i)+".")} for i in range(count)]
            if entry.get("stock_policy"):
                from .moe_stock_policies import StockPolicyExpert
                experts[key]=StockPolicyExpert.restore(entry,states[0])
                continue
            models=native_call(entry["backend"],root,saved["metadata"]["construction_inputs"][key],states=states,
                load_only=True,runner_source=saved["metadata"]["native_runner_source"])
            experts[key]=NativeExpert(models,entry)
        # Older local packing predates embedding the native feature list.
        if "macro_input_adapter" not in saved["metadata"]:
            import os
            artifact_root=EXPERT_ASSETS_DIR
            saved["metadata"]["macro_input_adapter"]=macro_adapter_metadata(artifact_root)
        model=cls(experts,saved["config"],saved["metadata"],root)
        # Native states are already attached with assign; copy only the small head.
        model.controller.load_state_dict({k.removeprefix("controller."):v for k,v in saved["state_dict"].items() if k.startswith("controller.")},strict=True)
        adapters={k.removeprefix("adapters."):v for k,v in saved["state_dict"].items() if k.startswith("adapters.")}
        adapters={k.replace("calibration.weight","scale").replace("calibration.bias","bias"):
                  (v.diagonal() if k.endswith("calibration.weight") else v) for k,v in adapters.items()}
        if adapters:model.adapters.load_state_dict(adapters,strict=True)
        if not saved["config"].get("policy_prior_version"):
            for head in (model.controller.market_fusion.policy,model.controller.market_fusion.allocation,model.controller.market_fusion.cash):
                nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
            saved["config"]["policy_prior_version"]=1;saved["optimizer_state"]=None;saved["optimizer_updates"]=0
        model.optimizer_updates=saved["optimizer_updates"];model._native_sources=temp
        model.gpu_lock=GPU_OWNER_LOCK
        optimizer_state=saved["optimizer_state"]
        previous=saved["metadata"].get("pre_expansion_optimizer_state")
        if optimizer_state is None and previous and model.controller.stock_policy_ids:
            optimizer_state=model._expand_optimizer_state(previous)
        return model,optimizer_state

    def _expand_optimizer_state(self,previous):
        """Keep existing Adam moments by parameter NAME when new modules are added."""
        stock=set(self.controller.stock_policy_ids)
        original_ids=[k for k in self.experts if k not in stock]
        old_names={"adapter":[f"adapters.{k}.{name}" for k in original_ids for name,_ in self.adapters[k].named_parameters()]}
        with torch.random.fork_rng(devices=[]):
            legacy=VerticalController({k:v for k,v in self.config["feature_sizes"].items() if k not in stock})
        old_names["controller_router_fusion"]=["controller."+name for name,_ in legacy.named_parameters()]
        moments={}
        for group in previous["param_groups"]:
            names=old_names[group["name"]]
            if len(names)!=len(group["params"]):raise ValueError("legacy optimizer parameter layout changed")
            moments.update({name:previous["state"][index] for name,index in zip(names,group["params"]) if index in previous["state"]})
        by_identity={id(p):name for name,p in self.named_parameters()}
        migrated={"state":{},"param_groups":[]};index=0
        for group in self.parameter_groups():
            previous_group=next(g for g in previous["param_groups"] if g["name"]==group["name"])
            record={**previous_group,"params":[]}
            for parameter in group["params"]:
                record["params"].append(index)
                name=by_identity[id(parameter)]
                if name in moments:migrated["state"][index]=moments[name]
                index+=1
            migrated["param_groups"].append(record)
        return migrated


def package_verified_experts(root,baseline_result,construction_snapshot):
    root=Path(root);catalog=json.loads((root/"expert_catalog.json").read_text(encoding="utf-8"))
    experts={};inputs={}
    for entry in catalog["experts"]:
        key=entry["id"];data=dict(construction_snapshot["expert_inputs"][key])
        if entry.get("variant"):data["variant"]=entry["variant"]
        inputs[key]=data
        models=native_call(entry["backend"],root,data,load_only=True)
        if sum(p.numel() for m in models for p in m.parameters())!=entry["parameters"]:
            raise ValueError(f"registered native parameter mismatch: {key}")
        experts[key]=NativeExpert(models,entry)
    sources=io.BytesIO()
    with zipfile.ZipFile(sources,"w",zipfile.ZIP_DEFLATED) as archive:
        for base in (root/"sources",root/"checkpoints"):
            for path in base.rglob("*"):
                if path.is_file() and path.suffix in (".py",".json") and ".git" not in path.parts and path.stat().st_size<2_000_000:
                    archive.write(path,path.relative_to(root).as_posix())
        archive.write(root/"marketgpt_config.json","marketgpt_config.json")
    sizes={r["expert"]:r["shape"][1]+3 for r in baseline_result["shared_representation"]}
    config={"feature_sizes":sizes,"max_gpu_experts":1,"policy_prior_version":1,"native_module_counts":{k:len(e.models) for k,e in experts.items()}}
    metadata={"architecture_sources":sources.getvalue(),"construction_inputs":inputs,
              "native_runner_source":inspect.getsource(expert_backends.run_native),"baseline":"1c497ef",
              "macro_input_adapter":macro_adapter_metadata(root)}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026);model=TradingMoE(experts,config,metadata,root)
    fusion=baseline_result.get("fusion_checkpoint")
    # Reuse all matching baseline projections/attention/head weights.
    registry=json.loads((root/"TradingMoE.manifest.json").read_text(encoding="utf-8"))
    checkpoint=registry.get("fusion_checkpoint")
    if checkpoint:
        old=torch.load(root/checkpoint["path"],map_location="cpu",weights_only=True)
        current=model.controller.market_fusion.state_dict()
        model.controller.market_fusion.load_state_dict({k:(v if k.startswith(("policy.","allocation.","cash.")) else old.get(k,v)) for k,v in current.items()},strict=True)
        for k,projection in model.controller.policy_adapters.items():
            projection.load_state_dict({"weight":old[f"projections.{k}.weight"],"bias":old[f"projections.{k}.bias"]})
    return model
