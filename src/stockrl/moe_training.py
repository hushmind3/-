"""Small controller update from the existing executed paper-account replay."""
import torch
from torch.nn import functional as F
from torch.distributions import Dirichlet
from .trading_moe import parameter_digest


def update_controller(model,optimizer,experience,snapshot,packets,target_weights,cash_weight,actions=None):
    if experience.origin_model!="trading_moe" or experience.source!="paper_account_portfolio":
        raise ValueError("controller learns only the MoE's actual paper outcomes")
    if experience.portfolio_reward is None:raise ValueError("paper NAV reward is missing")
    pstate=torch.as_tensor(experience.portfolio_state,dtype=torch.float32)
    astate=torch.as_tensor(experience.account_state,dtype=torch.float32).expand(len(pstate),-1)
    device=next(model.controller.parameters()).device
    account=torch.cat([pstate,astate],-1)[None].to(device)
    evidence,validity=model.prepare(packets,snapshot["symbols"])
    before=parameter_digest(model.controller)
    versions=[p._version for p in model.experts.parameters()]
    optimizer.zero_grad(set_to_none=True)
    output=model.controller(evidence,validity,account,model.policy_q(packets,snapshot["symbols"]))
    reward=torch.tensor(100*float(experience.portfolio_reward),dtype=torch.float32,device=device)
    predicted=output["value"].mean()
    advantage=(reward-predicted).detach()
    # Action credit plus the actual submitted portfolio allocation, not a
    # made-up profitable label or a price prediction substituted for reward.
    log_probabilities=output["policy_logits"].log_softmax(-1)[0]
    if actions is None:
        log_action=log_probabilities[experience.symbol_index,experience.action]
    else:
        actionable=snapshot.get("tradable_symbols",snapshot["symbols"])
        indices=[j for j,s in enumerate(snapshot["symbols"]) if s in actionable]
        if not indices:raise ValueError("no actionable paper decision to credit")
        log_action=torch.stack([log_probabilities[j,{"SELL":0,"HOLD":1,"BUY":2}[actions[snapshot["symbols"][j]]]] for j in indices]).mean()
    allocation=torch.cat([output["allocation_scores"][0],output["cash_scores"][0]])
    distribution=Dirichlet(F.softplus(allocation)+1)
    weights=torch.tensor([target_weights[s] for s in snapshot["symbols"]]+[cash_weight],device=device).clamp_min(1e-8)
    weights=weights/weights.sum()
    allocation_logp=distribution.log_prob(weights)
    router=output["router_probabilities"]
    entropy=-(router*router.clamp_min(1e-8).log()).sum(-1).mean()
    value_loss=F.smooth_l1_loss(predicted,reward)
    loss=-advantage*(log_action+allocation_logp)+.5*value_loss-.005*entropy
    loss.backward()
    grads={name:float(p.grad.norm()) for name,p in model.controller.named_parameters() if p.grad is not None}
    if not grads or not all(torch.isfinite(torch.tensor(v)) for v in grads.values()):raise ValueError("invalid controller gradients")
    torch.nn.utils.clip_grad_norm_(model.controller.parameters(),1)
    optimizer.step();model.optimizer_updates+=1
    after=parameter_digest(model.controller)
    if after==before:raise ValueError("controller update did not change weights")
    if versions!=[p._version for p in model.experts.parameters()] or any(p.grad is not None or p.requires_grad for p in model.experts.parameters()):
        raise ValueError("frozen native expert was changed")
    return {"loss":float(loss.detach()),"value_loss":float(value_loss.detach()),"reward_points":float(reward),
        "adapter_gradient_norm":sum(float(p.grad.norm()) for p in model.adapters.parameters() if p.grad is not None),
        "router_gradient_norm":sum(v for k,v in grads.items() if k.startswith("router")),
        "fusion_gradient_norm":sum(v for k,v in grads.items() if k.startswith("market_fusion.projections") or k.startswith("market_fusion.cross_attention")),
        "controller_gradient_norm":sum(v for k,v in grads.items() if k.startswith("controller_norm") or k.startswith("market_fusion.policy")),
        "controller_before":before,"controller_after":after,"expert_versions_unchanged":True,"optimizer_updates":model.optimizer_updates}
