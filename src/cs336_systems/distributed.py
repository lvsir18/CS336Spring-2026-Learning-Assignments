"""Educational DDP, optimizer sharding, and per-layer FSDP.

FSDP supports the A1 Linear/Embedding modules, retains replicated norms,
gathers on demand, and releases full weights between forward/backward.
"""
from types import MethodType
import math
import torch
from torch import nn
import torch.distributed as dist
from cs336_basics.model import Linear,Embedding


def world():
    return (dist.get_rank(),dist.get_world_size()) if dist.is_initialized() else (0,1)


class DDP(nn.Module):
    def __init__(self,module):
        super().__init__()
        self.module = module
        self.rank,self.world_size = world()
        self.pending = []
        if self.world_size > 1:
            for tensor in list(module.parameters())+list(module.buffers()):
                dist.broadcast(tensor.data,src=0)
        self.handles = [p.register_post_accumulate_grad_hook(self._ready) for p in module.parameters() if p.requires_grad]
    def _ready(self,param):
        if self.world_size > 1:
            self.pending.append((dist.all_reduce(param.grad,async_op=True),param.grad))
    def forward(self,*args,**kwargs):
        return self.module(*args,**kwargs)
    def finish_gradient_synchronization(self):
        for work,gradient in self.pending:
            work.wait(); gradient.div_(self.world_size)
        self.pending.clear()


class NaiveDDP(DDP):
    def __init__(self,module):
        super().__init__(module)
        for handle in self.handles:handle.remove()
        self.handles=[]
    def finish_gradient_synchronization(self):
        if self.world_size>1:
            for param in self.module.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad);param.grad.div_(self.world_size)


class FlatDDP(NaiveDDP):
    def finish_gradient_synchronization(self):
        groups={}
        for param in self.module.parameters():
            if param.grad is not None:groups.setdefault((param.grad.device,param.grad.dtype),[]).append(param.grad)
        if self.world_size>1:
            for gradients in groups.values():
                flat=torch.cat([g.reshape(-1) for g in gradients]);dist.all_reduce(flat);flat.div_(self.world_size)
                offset=0
                for gradient in gradients:
                    gradient.copy_(flat[offset:offset+gradient.numel()].view_as(gradient));offset+=gradient.numel()


class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self,params,optimizer_cls,**kwargs):
        params = list(params)
        super().__init__(params,kwargs)
        self.rank,self.world_size = world()
        self.owners, local_groups = {},[]
        seen = set()
        i = 0
        for group in self.param_groups:
            selected = []
            for p in group['params']:
                if id(p) in seen:
                    continue
                seen.add(id(p)); self.owners[p] = i%self.world_size; i += 1
                if self.owners[p] == self.rank:
                    selected.append(p)
            local_groups.append({**group,'params':selected})
        self.local_optimizer = optimizer_cls(local_groups,**kwargs) if self.owners else None
        if self.local_optimizer is not None:
            self.state = self.local_optimizer.state
    def step(self,closure=None):
        if self.local_optimizer is not None:
            for local,global_group in zip(self.local_optimizer.param_groups,self.param_groups):
                local.update({k:v for k,v in global_group.items() if k!='params'})
            loss = self.local_optimizer.step(closure)
        else:
            loss = None
        if self.world_size > 1:
            handles = [dist.broadcast(p.data,owner,async_op=True) for p,owner in self.owners.items()]
            for work in handles:
                work.wait()
        return loss


    def state_dict(self):
        return {'rank':self.rank,'world_size':self.world_size,
                'local_optimizer':self.local_optimizer.state_dict() if self.local_optimizer else None,
                'group_options':[{k:v for k,v in g.items() if k!='params'} for g in self.param_groups]}
    def load_state_dict(self,state):
        if state['rank']!=self.rank or state['world_size']!=self.world_size:
            raise ValueError('Optimizer shard resume requires the same rank and world size')
        if len(state['group_options'])!=len(self.param_groups):raise ValueError('Parameter groups changed')
        for group,options in zip(self.param_groups,state['group_options']):group.update(options)
        if self.local_optimizer is not None:
            self.local_optimizer.load_state_dict(state['local_optimizer']);self.state=self.local_optimizer.state


def _gather(shard,shape,compute_dtype):
    _,size = world()
    chunk = shard.to(compute_dtype or shard.dtype)
    if size == 1:
        return chunk[:math.prod(shape)].view(shape)
    pieces = [torch.empty_like(chunk) for _ in range(size)]
    dist.all_gather(pieces,chunk)
    return torch.cat(pieces)[:math.prod(shape)].view(shape)


class _ShardedLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,shard,module):
        weight = _gather(shard,module._fsdp_shape,module._fsdp_compute_dtype)
        ctx.module = module
        ctx.save_for_backward(x,shard)
        if isinstance(module,Embedding):
            return weight[x]
        return x.to(weight.dtype) @ weight.T
    @staticmethod
    def backward(ctx,go):
        x,shard = ctx.saved_tensors
        module = ctx.module
        owner = module._fsdp_owner
        # Complete the previous layer's scatter before retaining another full buffer.
        # Its communication has overlapped with this layer's backward computation.
        owner._finish_shards()
        shape = module._fsdp_shape
        if isinstance(module,Embedding):
            full_grad = torch.zeros(shape,device=go.device,dtype=go.dtype)
            full_grad.index_add_(0,x.reshape(-1),go.reshape(-1,shape[-1]))
            dx = None
        else:
            weight = _gather(shard,shape,module._fsdp_compute_dtype)
            dx = (go @ weight).to(x.dtype)
            full_grad = go.reshape(-1,shape[0]).T @ x.to(go.dtype).reshape(-1,shape[1])
        # Reduce-scatter in compute dtype; master gradients are converted to FP32 on completion.
        padded = torch.zeros(shard.numel()*owner.world_size,device=go.device,dtype=full_grad.dtype)
        padded[:full_grad.numel()] = full_grad.reshape(-1)
        reduced = torch.empty(shard.numel(),device=go.device,dtype=full_grad.dtype)
        if owner.world_size > 1:
            work = dist.reduce_scatter_tensor(reduced,padded,op=dist.ReduceOp.SUM,async_op=True)
        else:
            reduced.copy_(padded); work = None
        owner.pending.append((work,shard,reduced,padded))
        return dx,None,None


def _sharded_forward(module,x):
    return _ShardedLayer.apply(x,module.weight,module)


class FSDP(nn.Module):
    def __init__(self,module,compute_dtype=None):
        super().__init__()
        self.module = module
        self.rank,self.world_size = world()
        self.pending,self.replicated = [],[]
        if self.world_size > 1:
            for tensor in list(module.parameters())+list(module.buffers()):
                dist.broadcast(tensor.data,0)
        sharded_ids = set()
        for submodule in module.modules():
            if not isinstance(submodule,(Linear,Embedding)):
                continue
            weight = submodule.weight
            if id(weight) in sharded_ids:
                raise ValueError('FSDP educational implementation does not support shared weight objects')
            sharded_ids.add(id(weight))
            shape = tuple(weight.shape)
            chunk_size = math.ceil(weight.numel()/self.world_size)
            full = torch.zeros(chunk_size*self.world_size,device=weight.device,dtype=torch.float32)
            full[:weight.numel()] = weight.detach().reshape(-1).float()
            shard = full[self.rank*chunk_size:(self.rank+1)*chunk_size].clone()
            submodule.weight = nn.Parameter(shard,requires_grad=weight.requires_grad)
            submodule._fsdp_shape = shape
            submodule._fsdp_compute_dtype = compute_dtype
            object.__setattr__(submodule,'_fsdp_owner',self)
            submodule.forward = MethodType(_sharded_forward,submodule)
        sharded_params = {id(m.weight) for m in module.modules() if isinstance(m,(Linear,Embedding))}
        self.handles = [p.register_post_accumulate_grad_hook(self._ready_replicated) for p in module.parameters() if p.requires_grad and id(p) not in sharded_params]
    def _ready_replicated(self,param):
        if self.world_size > 1:
            self.replicated.append((dist.all_reduce(param.grad,async_op=True),param.grad))
    def forward(self,*args,**kwargs):
        return self.module(*args,**kwargs)
    def finish_gradient_synchronization(self):
        self._finish_shards()
        for work,gradient in self.replicated:
            work.wait(); gradient.div_(self.world_size)
        self.replicated.clear()
    def _finish_shards(self):
        for work,param,reduced,_keepalive in self.pending:
            if work is not None:
                work.wait()
            gradient = reduced.to(param.dtype)/self.world_size
            if param.grad is None:
                param.grad = gradient
            else:
                param.grad.add_(gradient)
        self.pending.clear()
    @torch.no_grad()
    def gather_full_params(self):
        result = {}
        modules = dict(self.module.named_modules())
        for name,param in self.module.named_parameters():
            path = name.rsplit('.',1)[0] if '.' in name else ''
            m = modules[path]
            result[name] = _gather(param,m._fsdp_shape,None) if isinstance(m,(Linear,Embedding)) else param.detach().clone()
        return result
