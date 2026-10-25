import math
import torch


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0 or eps <= 0 or weight_decay < 0 or any(not 0 <= b < 1 for b in betas):
            raise ValueError('Invalid AdamW hyperparameters')
        super().__init__(params,dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay))
    @torch.no_grad()
    def step(self,closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1,b2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW requires dense gradients')
                g = p.grad
                state = self.state[p]
                if not state:
                    state.update(t=0,m=torch.zeros_like(p),v=torch.zeros_like(p))
                state['t'] += 1
                t,m,v = state['t'],state['m'],state['v']
                m.mul_(b1).add_(g,alpha=1-b1)
                v.mul_(b2).addcmul_(g,g,value=1-b2)
                alpha = group['lr'] * math.sqrt(1-b2**t)/(1-b1**t)
                # Course formula: epsilon is added to the uncorrected second moment.
                p.addcdiv_(m,v.sqrt().add_(group['eps']),value=-alpha)
                p.mul_(1-group['lr']*group['weight_decay'])
        return loss


def get_lr_cosine_schedule(it,max_learning_rate,min_learning_rate,warmup_iters,cosine_cycle_iters):
    if warmup_iters and it < warmup_iters:
        return it/warmup_iters*max_learning_rate
    if it >= cosine_cycle_iters:
        return min_learning_rate
    progress = (it-warmup_iters)/(cosine_cycle_iters-warmup_iters)
    return min_learning_rate+0.5*(max_learning_rate-min_learning_rate)*(1+math.cos(math.pi*progress))
