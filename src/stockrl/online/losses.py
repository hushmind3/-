"""Vectorized losses for experiences sharing one model forward.

Keep the scalar objective intact, but transfer validity once per group rather
than synchronizing CUDA for every individual experience.
"""
import torch
from torch import nn
from torch.distributions import Categorical
from .data import PAPER_EXPLORATION_EPSILON


def shared_experience_losses(logits, values, allocations, experiences, successors):
    device = logits.device
    indices = torch.tensor([e.symbol_index for e in experiences], device=device)
    chosen = logits[0, indices].float()
    predicted = values[0, indices].float()
    finite = (torch.isfinite(chosen).all(-1) & torch.isfinite(predicted)).detach().cpu().tolist()
    accepted = [i for i, ok in enumerate(finite) if ok]
    rejected = len(experiences) - len(accepted)
    if not accepted:
        return None, [], rejected
    chosen = chosen[accepted]
    predicted = predicted[accepted]
    rows = [experiences[i] for i in accepted]
    metadata = torch.tensor([
        [e.action, e.reward * 100.0 + e.goal_reward_points,
         e.behavior_log_prob if e.behavior_log_prob is not None else 0.0,
         float(not e.portfolio_transition),
         float(e.behavior_log_prob is not None and e.trade_executed),
         float(e.portfolio_transition and e.portfolio_value_transition),
         100.0 * float(e.portfolio_reward or 0.0) + e.portfolio_goal_reward_points,
         float(e.portfolio_transition and e.trade_executed),
         float(e.source.startswith("teacher"))] for e in rows
    ], device=device, dtype=torch.float32)
    actions = metadata[:, 0].long()
    target = metadata[:, 1]
    account_target = metadata[:, 6]
    if any(e.bootstrap_discount for e in rows):
        zero = values.new_zeros((), dtype=torch.float32)
        future, account_future = [], []
        for i, e in zip(accepted, rows):
            if e.bootstrap_discount:
                successor = successors[i]
                if successor is None:
                    raise ValueError("future-credit successor value is missing; replay must be retained")
                future.append(float(e.bootstrap_discount) * successor[0, int(e.bootstrap_symbol_index)].float().detach())
                account_future.append(float(e.bootstrap_discount) * successor[0].float().mean().detach())
            else:
                future.append(zero); account_future.append(zero)
        target = target + torch.stack(future)
        account_target = account_target + torch.stack(account_future)
    needs_target=metadata[:,3].bool() | (metadata[:,7].bool() if allocations is not None else False)
    eligible=(~needs_target | torch.isfinite(target))
    eligible &= ~(metadata[:,3].bool() & metadata[:,4].bool() & torch.isnan(metadata[:,2]))
    account_mean=values[0].float().mean()
    eligible &= ~metadata[:,5].bool() | (torch.isfinite(account_target) & torch.isfinite(account_mean))
    target=torch.nan_to_num(target)
    account_target=torch.nan_to_num(account_target)
    account_mean=torch.nan_to_num(account_mean)
    metadata[:,2]=torch.nan_to_num(metadata[:,2],nan=0.0,posinf=float("inf"),neginf=-float("inf"))
    epsilon = min(PAPER_EXPLORATION_EPSILON, 1.0 / max(1, int(rows[0].features.shape[1])))
    probabilities = (1.0 - epsilon) * torch.softmax(chosen, -1) + epsilon / 3.0
    dist = Categorical(probs=probabilities, validate_args=False)
    losses = chosen.sum(-1) * 0
    ordinary = [i for i, e in enumerate(rows) if not e.portfolio_transition]
    if ordinary:
        term = .5 * nn.functional.smooth_l1_loss(predicted[ordinary], target[ordinary], reduction="none")
        term = term - .0005 * dist.entropy()[ordinary]
        losses = losses.index_add(0, torch.tensor(ordinary, device=device), term)
    actor = [i for i, e in enumerate(rows) if not e.portfolio_transition and e.behavior_log_prob is not None and e.trade_executed]
    if actor:
        advantage = target[actor] - predicted[actor].detach()
        ratio = torch.exp((dist.log_prob(actions)[actor] - metadata[actor, 2]).clamp(-20, 20))
        term = -torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage)
        losses = losses.index_add(0, torch.tensor(actor, device=device), term)
    portfolio = [i for i, e in enumerate(rows) if e.portfolio_transition and e.portfolio_value_transition]
    if portfolio:
        term = .25 * nn.functional.smooth_l1_loss(account_mean.expand(len(portfolio)), account_target[portfolio], reduction="none")
        losses = losses.index_add(0, torch.tensor(portfolio, device=device), term)
    allocation = [i for i, e in enumerate(rows) if e.portfolio_transition and e.trade_executed]
    if allocations is not None and allocation:
        selected = allocations[0, indices[accepted][allocation]].clamp_min(1e-7)
        eligible[allocation] &= torch.isfinite(selected)
        selected=torch.nan_to_num(selected,nan=1e-7,posinf=1e-7,neginf=1e-7)
        term = -.10 * target[allocation].detach() * torch.log(selected)
        losses = losses.index_add(0, torch.tensor(allocation, device=device), term)
    teacher = [i for i, e in enumerate(rows) if e.source.startswith("teacher")]
    if teacher:
        term = nn.functional.cross_entropy(chosen[teacher], actions[teacher], reduction="none")
        losses = losses.index_add(0, torch.tensor(teacher, device=device), term)
    finite_loss = (torch.isfinite(losses) & eligible).detach().cpu().tolist()
    valid = [i for i, ok in enumerate(finite_loss) if ok]
    rejected += len(rows) - len(valid)
    return losses[valid], [rows[i] for i in valid], rejected
