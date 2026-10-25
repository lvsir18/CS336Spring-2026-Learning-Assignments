"""Synthetic full training-step timing for real multi-rank deployments.

Linux NCCL requires one GPU per rank; CPU Gloo is for smoke testing only.
"""
import argparse,json,os,time,statistics
from contextlib import nullcontext
from pathlib import Path
import torch,torch.distributed as dist
from cs336_basics.model import TransformerLM
from cs336_systems.distributed import DDP,NaiveDDP,FlatDDP,FSDP,ShardedOptimizer

def worker(rank,args):
 os.environ.update(MASTER_ADDR='127.0.0.1',MASTER_PORT=str(args.port),RANK=str(rank),WORLD_SIZE=str(args.cpu_workers),LOCAL_RANK=str(rank),USE_LIBUV='0');run(args)

def run(a):
 dist.init_process_group(a.backend)
 try:
  rank,size=dist.get_rank(),dist.get_world_size();device=f'cuda:{os.environ["LOCAL_RANK"]}' if a.backend=='nccl' else 'cpu'
  if a.backend=='nccl':torch.cuda.set_device(device)
  torch.manual_seed(42);torch.set_num_threads(2)
  cfg=json.loads(Path(a.config).read_text(encoding='utf-8'));model=TransformerLM(**cfg['model']).to(device)
  wrapper={'naive':NaiveDDP,'flat':FlatDDP,'ddp':DDP,'zero1':DDP,'fsdp':FSDP}[a.method]
  wrapped=wrapper(model,compute_dtype=torch.bfloat16 if a.bf16 else None) if a.method=='fsdp' else wrapper(model)
  optimizer=ShardedOptimizer(wrapped.parameters(),torch.optim.AdamW,lr=.001) if a.method=='zero1' else torch.optim.AdamW(wrapped.parameters(),lr=.001)
  torch.manual_seed(1000+rank);x=torch.randint(cfg['model']['vocab_size'],(cfg['micro_batch_size'],cfg['model']['context_length']),device=device);y=torch.roll(x,-1,-1)
  times=[]
  for i in range(a.warmup+a.repeats):
   optimizer.zero_grad(set_to_none=True)
   if i==a.warmup and a.backend=='nccl':torch.cuda.reset_peak_memory_stats()
   dist.barrier()
   if a.backend=='nccl':torch.cuda.synchronize()
   t0=time.perf_counter()
   context=torch.autocast('cuda',dtype=torch.bfloat16) if a.bf16 and a.backend=='nccl' and a.method!='fsdp' else nullcontext()
   with context:loss=torch.nn.functional.cross_entropy(wrapped(x).float().flatten(0,1),y.flatten())
   loss.backward();wrapped.finish_gradient_synchronization();optimizer.step()
   if a.backend=='nccl':torch.cuda.synchronize()
   if i>=a.warmup:times.append(time.perf_counter()-t0)
  duration=torch.tensor(times,device=device);dist.all_reduce(duration,op=dist.ReduceOp.MAX)
  peak=torch.tensor(torch.cuda.max_memory_allocated() if a.backend=='nccl' else 0,device=device,dtype=torch.int64);dist.all_reduce(peak,op=dist.ReduceOp.MAX)
  if rank==0:
   values=duration.cpu().tolist();row={'settings':vars(a),'world_size':size,'torch':torch.__version__,'all_seconds_max_rank':values,'median_seconds':statistics.median(values),'global_tokens_per_second':size*x.numel()/statistics.median(values),'max_rank_peak_allocated_bytes':int(peak),'dataset':'synthetic tensors; no data IO; reference equivalence checked separately'}
   out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(row,indent=2),encoding='utf-8');print(row)
 finally:dist.destroy_process_group()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/a1-formal-tinystories.json');p.add_argument('--method',choices=['naive','flat','ddp','zero1','fsdp'],default='ddp');p.add_argument('--backend',choices=['gloo','nccl'],default='gloo');p.add_argument('--cpu-workers',type=int,default=0);p.add_argument('--port',type=int,default=29547);p.add_argument('--bf16',action='store_true');p.add_argument('--warmup',type=int,default=5);p.add_argument('--repeats',type=int,default=20);p.add_argument('--output',default='runs/a2/distributed-benchmark.json');a=p.parse_args()
 if a.repeats<1 or a.warmup<0:raise ValueError('Invalid repeat/warmup counts')
 if a.cpu_workers:
  if a.backend!='gloo':raise ValueError('CPU launcher requires gloo')
  torch.multiprocessing.spawn(worker,args=(a,),nprocs=a.cpu_workers,join=True)
 else:run(a)
