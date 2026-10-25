"""Rebuild figures and paired statistics from the published numeric evidence."""
import json,os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load(rel):return json.loads((ROOT/rel).read_text(encoding='utf-8'))
def rows(rel):
    with (ROOT/rel).open(encoding='utf-8') as f:return [json.loads(s) for s in f if s.strip()]
def curve(ax,rel,label,color=None):
    r=[v for v in rows(rel) if 'val_loss' in v]
    ax.plot([v['step'] for v in r],[v['val_loss'] for v in r],label=label,color=color,lw=2)
def paired(a,b):
    assert len(a)==len(b) and [r['test_index'] for r in a]==[r['test_index'] for r in b]
    a=np.array([r['reward'] for r in a],int);b=np.array([r['reward'] for r in b],int);n=len(a)
    counts=np.bincount(a*2+b,minlength=4)
    boot=np.random.default_rng(42).multinomial(n,counts/n,size=10000)
    return {'gained':int(counts[1]),'lost':int(counts[2]),'delta_pp':100*(counts[1]-counts[2])/n,
            'paired_bootstrap_95_ci_pp':np.percentile(100*(boot[:,1]-boot[:,2])/n,[2.5,97.5]).tolist()}

def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.titleweight':'bold','axes.titlepad':12,'figure.facecolor':'#f4f7fb',
                         'axes.facecolor':'white','axes.labelcolor':'#334155','text.color':'#0f172a'})
    out=ROOT/'assets-figures';out.mkdir(exist_ok=True)
    colors=['#2563eb','#f59e0b','#10b981']
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    ax=axes[0,0];curve(ax,'results/a1/metrics.jsonl','Validation',colors[0]);ax.set(title='A1 | 22.7M LM from scratch',xlabel='Optimizer updates',ylabel='Token NLL');ax.legend()
    ax=axes[0,1]
    for impl,c in zip(['naive','triton','sdpa'],colors):
        r=load(f'results/a2/a2-{impl}-d64-fwd-bwd.json')['results']
        ax.plot([v['sequence_length'] for v in r],[v['median_ms'] for v in r],marker='o',label=impl,color=c,lw=2)
    ax.set(title='A2 | Attention forward + backward',xlabel='Sequence length (B=4, dh=64)',ylabel='Median milliseconds',yscale='log');ax.legend()
    ax=axes[0,2];measurements=load('results/a3/measurements.json')
    for budget,c in zip(sorted({r['compute_budget'] for r in measurements}),colors):
        r=sorted([v for v in measurements if v['compute_budget']==budget],key=lambda v:v['parameters'])
        ax.plot([v['parameters']/1e6 for v in r],[v['final_loss'] for v in r],marker='o',label=f'{budget:.0e} FLOPs',color=c,lw=2)
    ax.set(title='A3 | Nine local IsoFLOPs runs',xlabel='Model parameters (millions)',ylabel='Validation NLL');ax.legend(fontsize=9)
    ax=axes[1,0]
    for arm,c in zip(['raw','rules'],[colors[0],colors[2]]):curve(ax,f'results/a4/a4-expanded-{arm}/metrics.jsonl',arm,c)
    ax.set(title='A4 | Expanded web corpus',xlabel='Optimizer updates',ylabel='Full-stream validation NLL');ax.legend()
    scores={f:load(f'results/a5/optimized-{f}-scores.json') for f in ['sft','dpo','rl']}
    ax=axes[1,1];values=[100*np.mean([v['reward'] for v in scores[f]]) for f in scores]
    bars=ax.bar(['SFT','DPO','GRPO + KL'],values,color=colors,width=.6)
    for bar,v in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,v+.8,f'{v:.2f}%',ha='center',weight='bold')
    ax.set(title='A5 | GSM8K strict test accuracy',ylabel='Accuracy (%)',ylim=(0,45));ax.text(.02,.95,'n=1319 | one sample/question',transform=ax.transAxes,va='top',fontsize=9)
    ax=axes[1,2];labels=['A1','A4','SFT','DPO','GRPO+KL']
    memory=load('results/memory.json')
    a1_peak=max(r['peak_allocated_bytes'] for r in rows('results/a1/metrics.jsonl'))/2**30
    def peak(prefix):return max(r['peak_allocated_gib'] for r in memory if r.get('task','').startswith(prefix))
    v=[a1_peak,peak('a4-expanded'),peak('sft-'),peak('dpo-'),max(peak('rl-p4g8-kl01'),peak('rl-p8g4-kl001'),peak('rl-p8g4-kl01'))]
    ax.bar(labels,v,color=['#94a3b8','#94a3b8',*colors],width=.6);ax.axhline(42,color='#dc2626',ls='--',label='42 GiB allocator cap');ax.set(title='Shared RTX PRO 6000 | Memory',ylabel='Peak allocated GiB (approx.)',ylim=(0,48));ax.legend(fontsize=9)
    for ax in axes.flat:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('CS336 | Implementation, measurements, and honest limits',fontsize=19,weight='bold',y=.99)
    fig.text(.5,.012,'Single physical GPU. A2 has no multi-GPU scaling result. A3 uses estimated FLOPs. A5 starts from public OLMo.',ha='center',fontsize=10,color='#475569')
    fig.tight_layout(rect=(0,.035,1,.955));fig.savefig(out/'results-overview.png',dpi=150);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(13,4.8));ax=axes[0]
    names=['rl-p8g4-kl0','rl-p8g4-kl001','rl-p8g4-kl01','rl-p4g8-kl01']
    for name,c in zip(names,['#94a3b8','#f59e0b','#2563eb','#10b981']):
        samples=[]
        for p in (ROOT/'results/a5'/name).glob('eval-*.summary.json'):
            samples.append((int(p.name.split('-')[1].split('.')[0]),json.loads(p.read_text())['reward']*100))
        samples.sort();ax.plot(*zip(*samples),marker='o',label=name.replace('rl-',''),color=c,lw=2)
    ax.set(title='Validation chooses the checkpoint',xlabel='GRPO updates',ylabel='Strict accuracy (%)');ax.legend(fontsize=9)
    ax=axes[1];families=['sft','dpo','rl'];x=np.arange(3);width=.34
    for offset,prefix,label,c in [(-.5,'initial','Initial recipe','#94a3b8'),(.5,'optimized','Optimized recipe','#2563eb')]:
        vals=[100*np.mean([v['reward'] for v in load(f'results/a5/{prefix}-{f}-scores.json')]) for f in families]
        bars=ax.bar(x+offset*width,vals,width,label=label,color=c)
        for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+.4,f'{v:.1f}',ha='center',fontsize=9)
    ax.set(xticks=x,xticklabels=['SFT','DPO','RL / GRPO'],ylabel='Strict test accuracy (%)',ylim=(0,43),title='Held-out test | different recipes');ax.legend()
    for ax in axes:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.text(.5,.012,'Validation n=256; test n=1319. Recipe changes and feedback budgets differ; these are not universal algorithm rankings.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(out/'alignment-details.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.8));ax=axes[0]
    names=['raw','rules','ranked','random42','random43','random44']
    vals=[load(f'results/a4/a4-small-{name}/best.json')['value'] for name in names]
    bars=ax.bar(names,vals,color=['#2563eb','#10b981','#dc2626','#94a3b8','#94a3b8','#94a3b8'])
    for b,value in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,value+.015,f'{value:.3f}',ha='center',fontsize=9)
    ax.set(title='Matched-budget controls | 600 updates',ylabel='Best full-stream validation NLL',ylim=(5.2,6.15))
    ax=axes[1]
    for arm,c in zip(['raw','rules'],[colors[0],colors[2]]):curve(ax,f'results/a4/a4-expanded-{arm}/metrics.jsonl',arm,c)
    ax.set(title='Expanded data | 65.536M training tokens',xlabel='Optimizer updates',ylabel='Full-stream validation NLL');ax.legend()
    for ax in axes:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.text(.5,.012,'Ranked/random subsets have equal token counts. Expanded rules improves NLL by only 0.005; no robust filtering gain claimed.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(out/'data-ablation.png',dpi=160);plt.close(fig)
    result={'test':{f:{'correct':int(sum(r['reward'] for r in v)),'total':len(v),'accuracy':float(np.mean([r['reward'] for r in v]))} for f,v in scores.items()},
            'comparisons':{a+'_to_'+b:paired(scores[a],scores[b]) for a,b in [('sft','dpo'),('sft','rl'),('dpo','rl')]},
            'bootstrap_scope':'Paired question resampling conditional on fixed trained models and generated outputs; excludes training and sampling seed uncertainty.'}
    (ROOT/'results/summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
