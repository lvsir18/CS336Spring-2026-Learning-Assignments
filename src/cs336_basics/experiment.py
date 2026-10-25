"""Deterministic epoch sampling and immutable, resumable validation selection."""
import json, math, os, random
from pathlib import Path
import torch
from .training import atomic_json


def atomic_torch(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as f:
        torch.save(value,f);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


class EpochSampler:
    def __init__(self,size,seed,state=None):
        if size<1:raise ValueError('Empty sampler')
        self.size,self.seed=size,seed;self.seen=(state or {}).get('seen',0)
        if state and (state['size']!=size or state['seed']!=seed):raise ValueError('Sampler identity changed')
        self._epoch=None

    def take(self,count):
        result=[]
        while len(result)<count:
            epoch,offset=divmod(self.seen,self.size)
            if epoch!=self._epoch:
                self.order=list(range(self.size));random.Random(self.seed+epoch).shuffle(self.order);self._epoch=epoch
            n=min(count-len(result),self.size-offset)
            result.extend(self.order[offset:offset+n]);self.seen+=n
        return result

    def state_dict(self):return {'size':self.size,'seed':self.seed,'seen':self.seen}


class BestModel:
    def __init__(self,run,mode='min',state=None):
        self.run=Path(run);self.mode=mode;self.state=state or {'step':None,'value':None,'bad_evaluations':0}
        # A crash may leave a newer pointer than the restored optimizer boundary.
        if self.state['step'] is not None:
            if not (self.run/self.state['file']).is_file():raise FileNotFoundError('Selected weights are missing')
            atomic_json(self.run/'best.json',self.state)

    def update(self,value,step,model,min_delta=0.):
        if not math.isfinite(value):raise FloatingPointError('Nonfinite validation metric')
        old=self.state['value']
        improved=old is None or (value<old-min_delta if self.mode=='min' else value>old+min_delta)
        if improved:
            filename=f'best-step-{step}.pt'
            atomic_torch(self.run/filename,{'model':model.state_dict(),'iteration':step,'metric':value})
            self.state={'step':step,'value':value,'file':filename,'bad_evaluations':0}
            atomic_json(self.run/'best.json',self.state)
        else:self.state={**self.state,'bad_evaluations':self.state['bad_evaluations']+1}
        return improved

    def load(self,model):
        if self.state['step'] is None:raise RuntimeError('No validated model to export')
        saved=torch.load(self.run/self.state['file'],map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model'])
