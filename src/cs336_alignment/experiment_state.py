"""Full model/optimizer/RNG state at optimizer boundaries for long HF runs."""
from contextlib import contextmanager
import json,os,random,shutil,signal,time
from pathlib import Path
import numpy as np
import torch
from cs336_basics.training import atomic_json,restore,reconcile_metrics


def rng_state():
    return {'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng(state):
    random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'])
    if torch.cuda.is_available() and state.get('cuda_rng') is not None:torch.cuda.set_rng_state_all(state['cuda_rng'])


class ExperimentState:
    def __init__(self,cfg,model,optimizer,resume=None):
        self.cfg,self.model,self.optimizer=cfg,model,optimizer
        self.run=Path(cfg['output_dir']);self.run.mkdir(parents=True,exist_ok=True)
        self.step=0;self.stopping=False;self.last_save=time.monotonic()
        self.extra={}
        if resume:
            saved=restore(resume,self.run,cfg)
            model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);self.step=saved['iteration'];self.extra=saved.get('extra',{});set_rng(saved)
            del saved;reconcile_metrics(self.run/'metrics.jsonl',self.step)
        elif any((self.run/n).exists() for n in ['checkpoint.pt','metrics.jsonl']):
            raise FileExistsError('Existing alignment run; use --resume auto')
        atomic_json(self.run/'config.json',cfg)

    @contextmanager
    def signals(self):
        def request(number,frame):
            self.stopping=True;print('STOP_REQUESTED',number,'save after optimizer boundary',flush=True)
        previous={s:signal.signal(s,request) for s in [signal.SIGTERM,signal.SIGINT]}
        try:yield
        finally:
            for s,handler in previous.items():signal.signal(s,handler)

    def record(self,row):
        self.step=row['step']
        with (self.run/'metrics.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n');f.flush()
        print(json.dumps(row),flush=True)

    def save(self,reason):
        target=self.run/'checkpoint.pt';temporary=self.run/'checkpoint.tmp'
        with temporary.open('wb') as f:
            torch.save({'format_version':2,'model':self.model.state_dict(),'optimizer':self.optimizer.state_dict(),
                        'iteration':self.step,'config':self.cfg,'extra':self.extra,**rng_state()},f)
            f.flush();os.fsync(f.fileno())
        if target.exists():
            prev=self.run/'checkpoint.previous.tmp'
            if prev.exists():prev.unlink()
            try:os.link(target,prev)
            except OSError:shutil.copy2(target,prev)
            os.replace(prev,self.run/'checkpoint.previous.pt')
        os.replace(temporary,target)
        if os.name!='nt':
            fd=os.open(self.run,os.O_RDONLY)
            try:os.fsync(fd)
            finally:os.close(fd)
        self.last_save=time.monotonic()
        atomic_json(self.run/'status.json',{'step':self.step,'target_steps':self.cfg['steps'],'reason':reason,'pid':os.getpid(),'time_unix':time.time()})
        print('CHECKPOINT',self.step,reason,flush=True)

    def boundary(self):
        due=self.step%self.cfg.get('checkpoint_every',25)==0 or time.monotonic()-self.last_save>=60*self.cfg.get('checkpoint_minutes',30)
        if due or self.stopping or self.step==self.cfg['steps']:
            self.save('signal' if self.stopping else ('complete' if self.step==self.cfg['steps'] else 'periodic'))
        return self.stopping
