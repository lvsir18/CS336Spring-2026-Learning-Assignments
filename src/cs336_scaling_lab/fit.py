"""IsoFLOPs fitting with discrete minima, as specified in the 2026 handout."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares


def fit_isoflops(records):
    groups = defaultdict(list)
    for row in records:
        n,c,l = float(row['parameters']),float(row['compute_budget']),float(row['final_loss'])
        if min(n,c,l) <= 0 or not np.isfinite([n,c,l]).all():
            raise ValueError('Only finite, positive, completed-run measurements are supported')
        groups[c].append(row)
    if len(groups) < 2:
        raise ValueError('At least two different compute budgets are required')
    minima = [min(rows,key=lambda row:row['final_loss']) for _,rows in sorted(groups.items())]
    c = np.array([float(row['compute_budget']) for row in minima])
    n = np.array([float(row['parameters']) for row in minima])
    d = c/(6*n)
    a,log_kn = np.polyfit(np.log(c),np.log(n),1)
    b,log_kd = np.polyfit(np.log(c),np.log(d),1)
    return {'a':float(a),'b':float(b),'k_n':float(np.exp(log_kn)),'k_d':float(np.exp(log_kd)),
            'minima':[{'compute_budget':float(ci),'parameters':float(ni),'tokens':float(di),'loss':float(row['final_loss'])} for ci,ni,di,row in zip(c,n,d,minima)]}


def predict(fit,compute):
    return {'compute_budget':float(compute),'parameters':float(fit['k_n']*compute**fit['a']),
            'tokens':float(fit['k_d']*compute**fit['b'])}


def fit_loss_surface(records):
    """Fit L=E+A*N^-alpha+B*D^-beta; parameters are positive via exponentiation.

    Returns residuals for diagnosis. Extrapolation is not a guarantee.
    """
    n = np.array([r['parameters'] for r in records],dtype=float)
    d = np.array([r.get('tokens',r['compute_budget']/(6*r['parameters'])) for r in records],dtype=float)
    y = np.array([r['final_loss'] for r in records],dtype=float)
    def unpack(x):
        return np.exp(x)
    def residual(x):
        e,a,b,alpha,beta = unpack(x)
        pred = e+a*n**(-alpha)+b*d**(-beta)
        return np.log(pred)-np.log(y)
    candidates = [least_squares(residual,np.log([max(0.1,y.min()/2),100,100,alpha,beta]),loss='soft_l1',max_nfev=2000)
                  for alpha,beta in [(0.1,0.1),(0.3,0.3),(0.5,0.5)]]
    best = min(candidates,key=lambda r:np.square(residual(r.x)).sum())
    return dict(zip(['E','A','B','alpha','beta'],map(float,unpack(best.x)))) | {'log_residual_rmse':float(np.sqrt(np.mean(residual(best.x)**2))), 'converged':bool(best.success)}


def save_results(input_path,output_dir):
    records = json.loads(Path(input_path).read_text(encoding='utf-8'))
    fit = fit_isoflops(records)
    predictions = [predict(fit,c) for c in [1e23,1e24]]
    output = Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    (output/'fit.json').write_text(json.dumps({'data_kind':'official synthetic IsoFLOPs fixture', 'fit':fit,'predictions':predictions},indent=2),encoding='utf-8')
    import os
    os.environ.setdefault('MPLCONFIGDIR',str(output/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = np.logspace(np.log10(min(r['compute_budget'] for r in fit['minima'])),24,200)
    fig,axes = plt.subplots(1,2,figsize=(11,4))
    for ax,key,k,exponent,label in zip(axes,['parameters','tokens'],[fit['k_n'],fit['k_d']],[fit['a'],fit['b']],['Parameters','Training tokens']):
        ax.loglog(x,k*x**exponent,label=f'power law exponent={exponent:.3f}')
        ax.scatter([r['compute_budget'] for r in fit['minima']],[r[key] for r in fit['minima']],label='observed discrete minima')
        ax.set_xlabel('Training compute (FLOPs)'); ax.set_ylabel(label); ax.legend(); ax.grid(alpha=.3)
    fig.suptitle('Official synthetic IsoFLOPs fixture (line extrapolates beyond observed budgets)')
    fig.tight_layout(rect=(0,0,1,.93)); fig.savefig(output/'isoflops.png',dpi=160); plt.close(fig)
    return fit,predictions
