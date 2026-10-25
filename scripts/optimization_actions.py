"""Validation-only family selection, adaptive DPO sizing, and final held-out tests."""
import argparse,json,math,os
from pathlib import Path
from cs336_basics.training import atomic_json
ROOT=Path('runs/optimization-v1')


def select(family,names):
    candidates=[]
    for name in names:
        run=ROOT/name;complete=json.loads((run/'complete.json').read_text());best=json.loads((run/'best.json').read_text())
        if complete['exported_step']!=best['step']:raise ValueError('Export is not the selected checkpoint')
        candidates.append({'run':name,'score':best['value'],'step':best['step']})
    winner=max(candidates,key=lambda row:row['score']);model=str(ROOT/winner['run']/'model')
    value={'family':family,'winner':winner,'model':model,'candidates':candidates,'selection':'strict reward on full 256 validation questions; ties favor first planned configuration'}
    path=ROOT/f'selected-{family}.json'
    if path.exists() and json.loads(path.read_text())!=value:raise ValueError('Cannot change an already published family selection')
    alias=ROOT/f'selected-{family}';alias.mkdir(exist_ok=True);link=alias/'model'
    if link.is_symlink():
        if link.resolve()!=Path(model).resolve():raise ValueError('Selected model link changed')
    elif link.exists():raise FileExistsError(link)
    else:link.symlink_to(Path(model).resolve(),target_is_directory=True)
    atomic_json(path,value);print('SELECTED',json.dumps(value),flush=True)


def test(family):
    import torch
    from cs336_alignment.hf_workflow import load_model,load_rows,evaluate
    selection=json.loads((ROOT/f'selected-{family}.json').read_text());torch.set_num_threads(4);torch.manual_seed(42)
    model,tok=load_model(selection['model'],'cuda','float32',False)
    result=evaluate(model,tok,load_rows('data/gsm8k-formal/test.jsonl'),ROOT/f'test-{family}'/'eval.jsonl','r1_zero',512,4)
    print('FINAL_TEST',family,json.dumps(result),flush=True)


def configure_dpo():
    manifest=json.loads(Path('data/optimization-v1/preferences/manifest.json').read_text())
    for name in ['dpo-lr1e-7','dpo-lr3e-7']:
        path=ROOT/'configs'/f'{name}.json';cfg=json.loads(path.read_text());cfg['steps']=math.ceil(manifest['pairs']/cfg['accum'])
        cfg['eval_every']=max(1,cfg['steps']//2);cfg['checkpoint_every']=cfg['eval_every'];cfg['warmup_steps']=max(1,int(cfg['steps']*.05))
        atomic_json(path,cfg)


def a4_summary():
    measurements={}
    for name in ['raw','rules','random42','random43','random44','ranked']:
        measurements[name]=json.loads((ROOT/('a4-small-'+name)/'best.json').read_text())
    atomic_json(ROOT/'a4-small-analysis.json',{'measurements':measurements,'interpretation':'fixed 600 update budget; matched token quantity for ranked/random; whole frozen validation stream'})


def summary():
    lines=['# Optimization v1 results','', 'Models were selected using validation only; final held-out tests run once per family.','']
    for family in ['sft','dpo','rl']:
        selection=json.loads((ROOT/f'selected-{family}.json').read_text());result=json.loads((ROOT/f'test-{family}/eval.summary.json').read_text())
        lines.extend([f'## {family}', '', '```json',json.dumps({'selection':selection,'test':result},indent=2),'```',''])
    (ROOT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['select','test','configure-dpo','a4-summary','summary']);p.add_argument('--family');p.add_argument('--runs',nargs='+');a=p.parse_args()
    if a.action=='select':select(a.family,a.runs)
    elif a.action=='test':test(a.family)
    elif a.action=='configure-dpo':configure_dpo()
    elif a.action=='a4-summary':a4_summary()
    else:summary()


if __name__=='__main__':main()
