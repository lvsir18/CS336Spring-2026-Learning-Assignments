import json,os
from pathlib import Path
from cs336_basics.training import atomic_json
from cs336_scaling_lab.fit import fit_isoflops,fit_loss_surface
root=Path('runs/formal/remaining');records=[]
for run in json.loads((root/'a3-accounting.json').read_text()):
    rows=[json.loads(x) for x in (Path(run['output_dir'])/'metrics.jsonl').read_text().splitlines()]
    if rows[-1]['step']!=run['steps'] or 'val_loss' not in rows[-1]:raise RuntimeError('Incomplete scaling run: '+run['id'])
    records.append(run|{'final_loss':rows[-1]['val_loss'],'train_seconds':sum(r['train_step_seconds'] for r in rows)})
fit=fit_isoflops(records);surface=fit_loss_surface(records)
sizes=sorted({r['parameters'] for r in records})
diagnostic={'boundary_minima':[m for m in fit['minima'] if m['parameters'] in [sizes[0],sizes[-1]]],
            'scope':'local estimated IsoFLOPs with integer steps; not official B200 API; boundary minima require expanding model grid'}
out=root/'a3';out.mkdir(exist_ok=True);atomic_json(out/'measurements.json',records);atomic_json(out/'fit.json',{'fit':fit,'loss_surface':surface,**diagnostic})
os.environ.setdefault('MPLCONFIGDIR',str(out/'.matplotlib'))
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(8,5))
for c in sorted({r['compute_budget'] for r in records}):
    group=sorted([r for r in records if r['compute_budget']==c],key=lambda r:r['parameters'])
    ax.plot([r['parameters'] for r in group],[r['final_loss'] for r in group],marker='o',label=f'{c:.0e} nominal FLOPs')
ax.set(xscale='log',xlabel='Parameters',ylabel='Validation loss',title='Measured local IsoFLOPs curves');ax.legend();ax.grid(alpha=.3);fig.tight_layout();fig.savefig(out/'isoflops.png',dpi=160)
print(json.dumps({'fit':fit,**diagnostic}),flush=True)
