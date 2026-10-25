"""Resumable single-device pretraining, independent of the CLI entry point."""
import argparse
from contextlib import contextmanager, nullcontext
import json, math, os, random, shutil, signal, subprocess, time
from pathlib import Path
import numpy as np
import torch
from .model import TransformerLM
from .nn_utils import cross_entropy, gradient_clipping
from .optimizer import AdamW, get_lr_cosine_schedule
from .data import get_batch


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)


@contextmanager
def run_lock(run):
    """The OS releases the lock even after SIGKILL; never unlink the lock file."""
    with (run/'.train.lock').open('a+b') as f:
        if os.name == 'nt':
            import msvcrt
            if f.seek(0, 2) == 0: f.write(b'0'); f.flush()
            f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try: yield
        finally:
            if os.name == 'nt':
                f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else: fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def signature(cfg):
    ignored = {'output_dir','eval_every','eval_batches','checkpoint_every','checkpoint_minutes','log_every','cpu_threads'}
    return {k:v for k,v in cfg.items() if k not in ignored}


def restore(path, run, cfg):
    candidates = [run/'checkpoint.pt',run/'checkpoint.previous.pt'] if path == 'auto' else [Path(path)]
    errors=[]
    for candidate in candidates:
        if not candidate.exists(): continue
        try:
            state=torch.load(candidate,map_location='cpu',weights_only=False)
            required={'model','optimizer','iteration','config','torch_rng','numpy_rng','python_rng'}
            if not isinstance(state,dict) or not required.issubset(state): raise ValueError('Incomplete checkpoint schema')
        except Exception as exc:
            errors.append(f'{candidate}: {exc}')
            # Do not later rotate a corrupt latest file over the good fallback.
            if path=='auto':
                archived=candidate.with_name(candidate.stem+f'.corrupt-{time.time_ns()}'+candidate.suffix)
                os.replace(candidate,archived)
                print('CORRUPT_CHECKPOINT_ARCHIVED',str(archived),flush=True)
            continue
        if signature(state['config']) != signature(cfg):
            raise ValueError('Resume configuration changed training semantics; use a new run')
        print('RESUME',str(candidate),'completed_steps',state['iteration'],flush=True)
        return state
    raise RuntimeError('No readable checkpoint: '+repr(errors))


def reconcile_metrics(path, step):
    if not path.exists(): return
    kept=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        try: row=json.loads(line)
        except json.JSONDecodeError: continue
        if row['step'] <= step: kept.append(json.dumps(row)+'\n')
    shutil.copy2(path,path.with_name(f'metrics.before-resume-{time.time_ns()}.jsonl'))
    temporary=path.with_suffix('.tmp'); temporary.write_text(''.join(kept),encoding='utf-8'); os.replace(temporary,path)


def train(args,cfg,run):
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    seed=cfg.get('seed',42); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type=='cuda': torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(cfg.get('cpu_threads',4))
    if not args.resume and any((run/p).exists() for p in ['checkpoint.pt','metrics.jsonl']):
        raise FileExistsError('Existing run: use --resume auto or a fresh output_dir')
    if cfg.get('architecture')=='a4_reference':
        from cs336_a4_reference.model import BasicsTransformerLM
        model=BasicsTransformerLM(**cfg['model']).to(device)
        decay=[p for p in model.parameters() if p.dim()>=2];norm=[p for p in model.parameters() if p.dim()<2]
        optimizer=torch.optim.AdamW([{'params':decay,'weight_decay':.1},{'params':norm,'weight_decay':0.}],
                                    lr=cfg['lr'],betas=(.9,.98),eps=1e-9,fused=device.type=='cuda')
    else:
        model=TransformerLM(**cfg['model']).to(device)
        optimizer=AdamW(model.parameters(),lr=cfg['lr'],betas=(.9,.95),weight_decay=cfg.get('weight_decay',.1))
    start=0;selection_state=None
    if args.resume:
        state=restore(args.resume,run,cfg)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer']); start=state['iteration']
        torch.set_rng_state(state['torch_rng']); np.random.set_state(state['numpy_rng']); random.setstate(state['python_rng'])
        if device.type=='cuda' and state.get('cuda_rng') is not None: torch.cuda.set_rng_state_all(state['cuda_rng'])
        selection_state=state.get('selection');del state; reconcile_metrics(run/'metrics.jsonl',start)
    from .experiment import BestModel
    selection=BestModel(run,'min',selection_state) if cfg.get('track_best') else None
    early_stopped=bool(selection and cfg.get('early_stopping_patience',0) and selection.state['bad_evaluations']>=cfg['early_stopping_patience'])
    def tokens(path):
        meta=json.loads(Path(str(path)+'.json').read_text(encoding='utf-8'))
        return np.memmap(path,mode='r',dtype=meta['dtype'])
    corpus,val=tokens(cfg['train_tokens']),tokens(cfg['val_tokens'])
    sequence,batch=cfg['model']['context_length'],cfg['micro_batch_size']; accum=cfg.get('gradient_accumulation_steps',1)
    steps=args.steps if args.steps is not None else cfg['steps']
    if accum<1 or not 0<steps<=cfg['steps'] or steps<start: raise ValueError('Invalid accumulation or step limit')
    if min(len(corpus),len(val))<=sequence: raise ValueError('Corpus shorter than context')
    amp=cfg.get('bf16',False) and device.type=='cuda'
    def autocast(): return torch.autocast('cuda',dtype=torch.bfloat16) if amp else nullcontext()
    stop={'signal':None}
    def request_stop(number,frame):
        stop['signal']=number; print('STOP_REQUESTED',number,'checkpoint after current optimizer step',flush=True)
    handlers={sig:signal.signal(sig,request_stop) for sig in [signal.SIGTERM,signal.SIGINT]}
    try: commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.SubprocessError): commit=None
    atomic_json(run/'config.json',cfg)
    atomic_json(run/'launch.json',{'pid':os.getpid(),'git_commit':commit,'resumed_from_step':start,
                                 'parameters':sum(p.numel() for p in model.parameters()),'time_unix':time.time()})
    last_checkpoint=time.monotonic()
    def checkpoint(step,reason):
        nonlocal last_checkpoint
        target,temporary=run/'checkpoint.pt',run/'checkpoint.tmp'
        with temporary.open('wb') as f:
            torch.save({'format_version':2,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'iteration':step,
                        'config':cfg,'git_commit':commit,'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),
                        'python_rng':random.getstate(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
                        'selection':selection.state if selection else None},f)
            f.flush(); os.fsync(f.fileno())
        if target.exists():
            previous=run/'checkpoint.previous.tmp'
            if previous.exists(): previous.unlink()
            try: os.link(target,previous)
            except OSError: shutil.copy2(target,previous)
            os.replace(previous,run/'checkpoint.previous.pt')
        os.replace(temporary,target)
        if os.name!='nt':
            fd=os.open(run,os.O_RDONLY)
            try: os.fsync(fd)
            finally: os.close(fd)
        last_checkpoint=time.monotonic()
        atomic_json(run/'status.json',{'step':step,'target_steps':cfg['steps'],'reason':reason,'checkpoint':str(target),
                                      'time_unix':time.time(),'pid':os.getpid()})
        print('CHECKPOINT',step,reason,flush=True)
    try:
        with (run/'metrics.jsonl').open('a',encoding='utf-8') as log:
            completed=start
            for step in range(start,start if early_stopped else steps):
                model.train(); optimizer.zero_grad(set_to_none=True)
                lr=get_lr_cosine_schedule(step,cfg['lr'],cfg.get('min_lr',cfg['lr']*.1),cfg.get('warmup_steps',100),cfg['steps'])
                for group in optimizer.param_groups: group['lr']=lr
                if device.type=='cuda': torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                t0=time.perf_counter(); total=0.
                for _ in range(accum):
                    x,y=get_batch(corpus,batch,sequence,device)
                    with autocast(): loss=cross_entropy(model(x),y)/accum
                    loss.backward(); total+=loss.detach().item()
                if not math.isfinite(total): raise FloatingPointError(f'Nonfinite loss at step {step}')
                grad_norm=gradient_clipping(model.parameters(),cfg.get('max_grad_norm',1.)); optimizer.step()
                if device.type=='cuda': torch.cuda.synchronize()
                duration=time.perf_counter()-t0; completed=step+1
                row={'step':completed,'train_loss':total,'gradient_norm':grad_norm,'lr':lr,'train_step_seconds':duration,
                     'tokens_per_second':batch*sequence*accum/duration,'tokens_seen':completed*batch*sequence*accum}
                if device.type=='cuda': row['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
                evaluate=completed%cfg.get('eval_every',100)==0 or completed==cfg['steps']
                if evaluate and stop['signal'] is None:
                    model.eval(); previous=np.random.get_state(); np.random.seed(seed+100000); losses=[]
                    try:
                        with torch.no_grad():
                            weights=[]
                            if cfg.get('validation_mode')=='sequential':
                                windows=(len(val)-1)//sequence;eval_batch=cfg.get('eval_batch_size',batch)
                                for first in range(0,windows,eval_batch):
                                    count=min(eval_batch,windows-first);offsets=np.arange(first,first+count)[:,None]*sequence+np.arange(sequence)[None,:]
                                    x=torch.tensor(np.asarray(val[offsets],dtype=np.int64),device=device)
                                    y=torch.tensor(np.asarray(val[offsets+1],dtype=np.int64),device=device)
                                    with autocast():losses.append(cross_entropy(model(x),y).item())
                                    weights.append(count*sequence)
                            else:
                                for _ in range(cfg.get('eval_batches',20)):
                                    x,y=get_batch(val,cfg.get('eval_batch_size',batch),sequence,device)
                                    with autocast(): losses.append(cross_entropy(model(x),y).item())
                                    weights.append(x.numel())
                    finally: np.random.set_state(previous)
                    value=sum(v*w for v,w in zip(losses,weights))/sum(weights)
                    row.update(val_loss=value,val_perplexity=math.exp(min(80.,value)),eval_tokens=sum(weights))
                    if selection:
                        selection.update(value,completed,model,cfg.get('selection_min_delta',0.))
                        patience=cfg.get('early_stopping_patience',0)
                        early_stopped=bool(patience and selection.state['bad_evaluations']>=patience)
                log.write(json.dumps(row)+'\n'); log.flush()
                if completed%cfg.get('log_every',10)==0 or evaluate or step==start: print(json.dumps(row),flush=True)
                due=completed%cfg.get('checkpoint_every',cfg.get('eval_every',100))==0
                due=due or time.monotonic()-last_checkpoint>=60*cfg.get('checkpoint_minutes',15)
                if stop['signal'] is not None or due or evaluate or completed==steps:
                    checkpoint(completed,'signal' if stop['signal'] else ('early_stop' if early_stopped else ('complete' if completed==cfg['steps'] else 'periodic')))
                if stop['signal'] is not None or early_stopped: break
    finally:
        for sig,handler in handlers.items(): signal.signal(sig,handler)
    if stop['signal'] is None and (completed==cfg['steps'] or early_stopped):
        atomic_json(run/'complete.json',{'step':completed,'early_stopped':early_stopped,'selection':selection.state if selection else None})
    return 128+stop['signal'] if stop['signal'] is not None else 0


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--device',default='cuda')
    p.add_argument('--resume',help='checkpoint path or auto (latest, then previous if unreadable)')
    p.add_argument('--steps',type=int,help='Stop at global step; preserve configured learning-rate horizon')
    args=p.parse_args(); cfg=json.loads(Path(args.config).read_text(encoding='utf-8'))
    run=Path(cfg['output_dir']); run.mkdir(parents=True,exist_ok=True)
    with run_lock(run): return train(args,cfg,run)
