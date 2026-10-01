"""Keep Adam moments across checkpoint batches without retaining GPU models."""
import torch


def restore_optimizer(owner, learner, optimizer, model_version):
    saved = getattr(owner, "_optimizer_states", {}).get(learner)
    shapes = tuple(tuple(p.shape) for group in optimizer.param_groups for p in group["params"])
    if saved is None or saved["version"] != model_version or saved["shapes"] != shapes:
        return False
    optimizer.load_state_dict(saved["state"])
    return True


def remember_optimizer(owner, learner, optimizer, model_version):
    def copy_cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().to("cpu", copy=True)
        if isinstance(value, dict):
            return {key: copy_cpu(item) for key, item in value.items()}
        if isinstance(value, list):
            return [copy_cpu(item) for item in value]
        if isinstance(value, tuple):
            return tuple(copy_cpu(item) for item in value)
        return value

    if not hasattr(owner, "_optimizer_states"):
        owner._optimizer_states = {}
    owner._optimizer_states[learner] = {
        "version": model_version,
        "shapes": tuple(tuple(p.shape) for group in optimizer.param_groups for p in group["params"]),
        "state": copy_cpu(optimizer.state_dict()),
    }
