import argparse
import json
from pathlib import Path
import statistics
import time
from contextlib import nullcontext
import torch
from cs336_systems.flash import FlashAttentionPyTorch

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--impl',choices=['naive','pytorch','triton','sdpa'],default='triton');parser.add_argument('--device',default='cuda');parser.add_argument('--lengths',type=int,nargs='+',default=[256,1024,4096]);parser.add_argument('--head-dim',type=int,default=64);parser.add_argument('--batch',type=int,default=4);parser.add_argument('--repeats',type=int,default=20);parser.add_argument('--warmup',type=int,default=5);parser.add_argument('--causal',action='store_true');parser.add_argument('--backward',action='store_true');parser.add_argument('--compile',action='store_true');parser.add_argument('--bf16',action='store_true');parser.add_argument('--output',default='runs/a2/attention.json')
    parser.add_argument('--sdpa-backend',choices=['auto','flash','math'],default='auto')
    args=parser.parse_args()
    if args.impl=='triton':
        from cs336_systems.flash_triton import FlashAttentionTriton
        function=FlashAttentionTriton.apply
    elif args.impl=='pytorch':function=FlashAttentionPyTorch.apply
    elif args.impl=='sdpa':
        def function(q,k,v,c):
            from torch.nn.attention import SDPBackend,sdpa_kernel
            context=nullcontext() if args.sdpa_backend=='auto' else sdpa_kernel(SDPBackend.FLASH_ATTENTION if args.sdpa_backend=='flash' else SDPBackend.MATH)
            with context:
                # SDPA's fused kernels require an explicit head dimension.
                return torch.nn.functional.scaled_dot_product_attention(q.unsqueeze(-3),k.unsqueeze(-3),v.unsqueeze(-3),is_causal=c).squeeze(-3)
    else:
        def function(q,k,v,c):
            scores=(q@k.transpose(-1,-2))/q.shape[-1]**.5
            if c:scores=scores.masked_fill(~torch.ones(q.shape[-2],k.shape[-2],device=q.device,dtype=torch.bool).tril(),float('-inf'))
            return scores.softmax(-1)@v
    if args.compile:function=torch.compile(function)
    rows=[];torch.manual_seed(42)
    for length in args.lengths:
        tensors=[torch.randn(args.batch,length,args.head_dim,device=args.device,dtype=torch.bfloat16 if args.bf16 else torch.float32,requires_grad=args.backward) for _ in range(3)]
        def operation():
            for tensor in tensors:tensor.grad=None
            result=function(*tensors,args.causal)
            if args.backward:result.square().mean().backward()
            return result
        for _ in range(args.warmup):operation()
        cuda=tensors[0].is_cuda
        if cuda:torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        times=[]
        for _ in range(args.repeats):
            if cuda:
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record();operation();end.record();end.synchronize();times.append(start.elapsed_time(end))
            else:
                start=time.perf_counter();operation();times.append((time.perf_counter()-start)*1000)
        row={'sequence_length':length,'median_ms':statistics.median(times),'mean_ms':statistics.mean(times),'stdev_ms':statistics.stdev(times) if len(times)>1 else 0}
        if cuda:row['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        rows.append(row);print(row,flush=True)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'settings':vars(args),'torch':torch.__version__,'device_name':torch.cuda.get_device_name() if cuda else 'cpu','results':rows},indent=2),encoding='utf-8')

if __name__=='__main__':main()
