"""GPU forward/gradient checks on ragged lengths and FP32/BF16 inputs."""
import argparse
import json
from pathlib import Path
import torch
from cs336_systems.flash_triton import FlashAttentionTriton


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='runs/round1/flash-numerics.json');args=parser.parse_args()
    torch.manual_seed(42);rows=[]
    for dtype in [torch.float32,torch.bfloat16]:
        for length,dim in [(39,16),(129,64),(513,128)]:
            for causal in [False,True]:
                tensors=[torch.randn(2,length,dim,device='cuda',dtype=dtype,requires_grad=True) for _ in range(3)]
                refs=[x.detach().float().requires_grad_() for x in tensors]
                gradient=torch.randn_like(tensors[0])
                out=FlashAttentionTriton.apply(*tensors,causal)
                scores=refs[0]@refs[1].transpose(-1,-2)/dim**.5
                if causal:scores=scores.masked_fill(~torch.ones(length,length,device='cuda',dtype=torch.bool).tril(),float('-inf'))
                ref=scores.softmax(-1)@refs[2]
                out.backward(gradient);ref.backward(gradient.float())
                atol,rtol=(2e-2,2e-2) if dtype==torch.bfloat16 else (2e-4,2e-4)
                actual=[out.float(),*[x.grad.float() for x in tensors]]
                expected=[ref,*[x.grad for x in refs]]
                for a,b in zip(actual,expected):torch.testing.assert_close(a,b,atol=atol,rtol=rtol)
                row={'dtype':str(dtype),'length':length,'head_dim':dim,'causal':causal,'atol':atol,'rtol':rtol,
                    'max_abs_errors':dict(zip(['output','dq','dk','dv'],[float((a-b).abs().max()) for a,b in zip(actual,expected)]))}
                rows.append(row);print(json.dumps(row),flush=True)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'passed_cases':len(rows),'reference':'explicit FP32 softmax attention','cases':rows},indent=2)+'\n')


if __name__=='__main__':main()
