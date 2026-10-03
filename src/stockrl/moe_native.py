"""Register the existing native architectures without copying their forward code.

The existing backend function is specialized in memory: checkpoint reads become
state reads when packaging, constructors become registered modules on reuse.
No original source/checkpoint is changed and no expert probe is repeated.
"""
import ast
import inspect
import json
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from . import expert_backends


class NativeExpert(nn.Module):
    def __init__(self, models, entry):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.entry = entry
        self.requires_grad_(False).eval()

    def forward_tensors(self,*args,module_index=0,**kwargs):
        """Differentiable native entrypoint for a selected expert parameter group.

        Diagnostic JSON evidence detaches outputs. Native fine tuning instead
        calls this registered module with that architecture's native tensors.
        """
        return self.models[module_index](*args,**kwargs)

    def forward(self, root, data, device="cpu"):
        # Frozen CPU tensors are views into the single mmap PT. Retain those
        # views rather than copying every expert GPU -> newly allocated RAM.
        frozen=all(not p.requires_grad for p in self.parameters())
        cpu_parameters=[(p,p.detach()) for p in self.parameters()] if frozen else []
        cpu_buffers=[(module,name,value) for module in self.modules() for name,value in module._buffers.items() if value is not None] if frozen else []
        try:
            return native_call(self.entry["backend"], root, data, device,
                               modules=list(self.models))
        finally:
            if frozen:
                for parameter,value in cpu_parameters:parameter.data=value
                for module,name,value in cpu_buffers:module._buffers[name]=value
            else:self.cpu()


@lru_cache(maxsize=8)
def _compiled_runner(runner):
    """Cache code only; model instances and per-call bindings stay outside."""
    source = runner if isinstance(runner, str) else inspect.getsource(runner)
    tree = ast.parse(source)
    constructors = {"EXAONEFinance", "PatchedTimeSeriesDecoder", "PatchedTimeSeriesDecoder_MOE",
                    "TimeMoeForPrediction", "Toto2Model", "Transformer", "subagent"}

    class Reuse(ast.NodeTransformer):
        def visit_Call(self,node):
            node=self.generic_visit(node)
            name=node.func.id if isinstance(node.func,ast.Name) else node.func.attr if isinstance(node.func,ast.Attribute) else ""
            if name=="load_native_pretrained":
                return ast.Call(ast.Name("_pretrained",ast.Load()),node.args,node.keywords)
            if name in constructors:
                return ast.Call(ast.Name("_construct",ast.Load()),[ast.Lambda(ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),node)],[])
            if name=="from_pretrained" and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id in {"ChronosPipeline","Kronos","KronosTokenizer"}:
                return ast.Call(ast.Name("_pretrained",ast.Load()),[node.func.value,*node.args],node.keywords)
            if name=="load_state_dict":
                return ast.Call(ast.Name("_restore",ast.Load()),[node.func.value,*node.args],node.keywords)
            if name=="load_file" or name=="load" and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=="torch":
                return ast.Call(ast.Name("_weights",ast.Load()),[ast.Constant(name),*node.args],node.keywords)
            return node

    tree=Reuse().visit(tree)
    function=tree.body[0]
    for index,node in enumerate(function.body):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="loaded_seconds" for t in node.targets):
            function.body.insert(index,ast.If(ast.Name("_load_only",ast.Load()),[ast.Return(ast.Name("models",ast.Load()))],[]))
            break
    ast.fix_missing_locations(tree)
    return compile(tree, "<registered baseline native backend>", "exec")


def native_call(backend, root, data, device="cpu", *, modules=None, states=None,
                load_only=False, runner_source=None):
    """Use exactly the baseline native data preparation and native forward."""
    code = _compiled_runner(runner_source or expert_backends.run_native)
    namespace=dict(vars(expert_backends))
    cursor=0
    restored=0

    def construct(factory):
        nonlocal cursor
        model=modules[cursor] if modules is not None else factory()
        cursor+=1
        return model

    def pretrained(cls,directory,**kwargs):
        nonlocal cursor
        if modules is None and states is None:
            result=expert_backends.load_native_pretrained(cls,directory,**kwargs)
            cursor+=1
            return result
        cfg=json.loads((Path(directory)/"config.json").read_text(encoding="utf-8"))
        if cls.__name__=="ChronosPipeline":
            from chronos import ChronosConfig, ChronosModel
            from transformers import AutoConfig, AutoModelForSeq2SeqLM
            if modules is None:
                model=AutoModelForSeq2SeqLM.from_config(AutoConfig.from_pretrained(directory,local_files_only=True))
                restore(model,states[0],strict=True)
            else: model=modules[cursor]
            cc=ChronosConfig(**cfg["chronos_config"])
            result=cls(cc.create_tokenizer(),ChronosModel(cc,model))
        else:
            result=modules[cursor] if modules is not None else cls(**cfg)
            if modules is None: restore(result,states[cursor],strict=True)
        cursor+=1
        return result

    def weights(kind,path,**kwargs):
        if modules is not None or states is not None:
            if backend=="marketgpt":
                return {"model_args":json.loads((root/"marketgpt_config.json").read_text()),"model":states[0] if states else {}}
            return states[0] if states else {}
        if kind=="load": return torch.load(path,**kwargs)
        from safetensors.torch import load_file
        return load_file(path,**kwargs)

    def restore(model,state,**kwargs):
        nonlocal restored
        if modules is not None: return SimpleNamespace(missing_keys=[],unexpected_keys=[])
        if states is not None:
            state=states[restored]
            kwargs["assign"]=True
        restored+=1
        result=model.load_state_dict(state,**kwargs)
        if hasattr(model,"tie_weights"): model.tie_weights()
        if backend=="marketgpt": model.output.weight=model.tok_embeddings.weight
        return result

    namespace.update(_construct=construct,_pretrained=pretrained,_weights=weights,_restore=restore,_load_only=load_only)
    exec(code,namespace)
    # Baseline backend reseeds isolated workers; do not overwrite policy RNG here.
    with torch.random.fork_rng(devices=[]):
        return namespace["run_native"](backend,Path(root),data,device)
