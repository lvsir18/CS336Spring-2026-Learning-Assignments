"""Whole-model training-step benchmark, optional profiler and checkpointing."""
import argparse,json,time,statistics
from pathlib import Path
import torch
from torch.utils.checkpoint import checkpoint
from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy
p=argparse.ArgumentParser();p.add_argument('--config',default='configs/a1-formal-tinystories.json');p.add_argument('--device',default='cuda');p.add_argument('--warmup',type=int,default=5);p.add_argument('--repeats',type=int,default=20);p.add_argument('--checkpointing',action='store_true');p.add_argument('--checkpoint-every',type=int,default=1);p.add_argument('--memory-snapshot',action='store_true');p.add_argument('--compile',action='store_true');p.add_argument('--profile',action='store_true');p.add_argument('--output',default='runs/a2/model.json');a=p.parse_args()
cfg=json.loads(Path(a.config).read_text(encoding='utf-8'));torch.manual_seed(42);torch.set_num_threads(4)
model=TransformerLM(**cfg['model']).to(a.device)
if a.checkpointing:
    if a.checkpoint_every<1:raise ValueError('checkpoint-every must be positive')
    from types import MethodType
    def checkpointed_forward(self,ids):
        x=self.token_embeddings(ids)
        for start in range(0,len(self.layers),a.checkpoint_every):
            group=list(self.layers[start:start+a.checkpoint_every])
            def segment(x,_group=group):
                for layer in _group:x=layer(x)
                return x
            x=checkpoint(segment,x,use_reentrant=False)
        return self.lm_head(self.ln_final(x))
    model.forward=MethodType(checkpointed_forward,model)
if a.compile:model=torch.compile(model)
optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'])
x=torch.randint(cfg['model']['vocab_size'],(cfg['micro_batch_size'],cfg['model']['context_length']),device=a.device);y=torch.roll(x,-1,-1)
def operation():
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=a.device.startswith('cuda') and cfg.get('bf16',True)):
        loss=cross_entropy(model(x),y)
    loss.backward();optimizer.step()
for _ in range(a.warmup):operation()
cuda=x.is_cuda
if cuda:torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
if a.memory_snapshot and cuda:torch.cuda.memory._record_memory_history(max_entries=10000)
times=[]
for _ in range(a.repeats):
    if cuda:torch.cuda.synchronize()
    start=time.perf_counter();operation()
    if cuda:torch.cuda.synchronize()
    times.append(time.perf_counter()-start)
output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
if a.memory_snapshot and cuda:
    torch.cuda.memory._dump_snapshot(str(output.with_suffix('.memory.pickle')))
    torch.cuda.memory._record_memory_history(enabled=None)
if a.profile:
    activities=[torch.profiler.ProfilerActivity.CPU]+([torch.profiler.ProfilerActivity.CUDA] if cuda else [])
    with torch.profiler.profile(activities=activities,record_shapes=True,profile_memory=True) as prof:operation()
    prof.export_chrome_trace(str(output.with_suffix('.trace.json')))
row={'settings':vars(a),'torch':torch.__version__,'median_seconds':statistics.median(times),'all_seconds':times,'tokens_per_second':x.numel()/statistics.median(times),'peak_allocated_bytes':torch.cuda.max_memory_allocated() if cuda else None}
output.write_text(json.dumps(row,indent=2),encoding='utf-8');print(json.dumps(row,indent=2))
