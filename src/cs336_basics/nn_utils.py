import torch
from .model import softmax


def cross_entropy(inputs,targets):
    flat = inputs.float().reshape(-1, inputs.shape[-1])
    labels = targets.reshape(-1)
    return (torch.logsumexp(flat, -1)-flat.gather(-1,labels[:,None]).squeeze(-1)).mean()


@torch.no_grad()
def gradient_clipping(parameters,max_l2_norm):
    if max_l2_norm < 0:
        raise ValueError('max_l2_norm must be nonnegative')
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return 0.
    norm = torch.stack([p.grad.float().square().sum() for p in params]).sum().sqrt()
    norm_value=norm.item()
    if not torch.isfinite(norm): raise FloatingPointError('Nonfinite global gradient norm; optimizer update aborted')
    scale = min(1.,max_l2_norm/(norm_value+1e-6))
    for p in params:
        p.grad.mul_(scale)
    return norm_value
