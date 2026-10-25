"""A1 matrix-only FLOPs and parameter accounting, no GPU allocation."""
import argparse,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',default='reports/a1-accounting.json');a=p.parse_args()
rows=[]
for name,l,d,h in [('small',12,768,12),('medium',24,1024,16),('large',36,1280,20),('xl',48,1600,25)]:
 for t in ([1024,16384] if name=='xl' else [1024]):
    v=50257;f=round((8*d/3)/64)*64
    params=2*v*d+l*(4*d*d+3*d*f+2*d)+d
    flops={'qkv_and_output':l*8*t*d*d,'qk_and_pv':l*4*t*t*d,'swiglu':l*6*t*d*f,'lm_head':2*t*d*v}
    total=sum(flops.values())
    # A transparent simplified accounting model: retained intermediates of each operation,
    # excluding aliases, RoPE tensors, masks, autograd implementation details and workspaces.
    activation_elements=l*(8*t*d+4*t*f+2*h*t*t)+t*d+2*t*v
    row={'model':name,'layers':l,'d_model':d,'heads':h,'d_ff':f,'context':t,'parameters':params,'fp32_parameter_bytes':4*params,
         'forward_matmul_flops':total,'components_flops':flops,'component_fraction':{k:x/total for k,x in flops.items()},
         'fp32_static_adam_bytes':16*params,'simplified_activation_bytes_per_sequence':4*activation_elements}
    if name=='xl' and t==1024:row.update(simplified_max_batch_at_80GB=math.floor((80e9-16*params)/(4*activation_elements)),estimated_h100_hours_400k_steps_batch1024_at_50pct_TF32_MFU=400000*1024*3*total/(.5*495e12)/3600)
    rows.append(row)
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'assumptions':'untied embedding/head; SwiGLU width aligned to nearest64; dense full attention; matrix FLOPs only; activation model is explicitly approximate','rows':rows},indent=2),encoding='utf-8');print(json.dumps(rows,indent=2))
