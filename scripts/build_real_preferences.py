"""Resumable SFT-policy rollouts on training questions only; verified real errors."""
import argparse,hashlib,json,os,random
from pathlib import Path
import torch
from cs336_basics.training import atomic_json,run_lock
from cs336_basics.experiment import atomic_torch
from cs336_alignment.experiment_state import rng_state,set_rng
from cs336_alignment.hf_workflow import load_model,load_rows,format_prompt,sample,reward_function


def main():
    p=argparse.ArgumentParser();p.add_argument('--selection',required=True);p.add_argument('--output',required=True)
    p.add_argument('--target-pairs',type=int,default=1000);p.add_argument('--max-questions',type=int,default=2048);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    with run_lock(out):
        if (out/'manifest.json').exists():return
        selection=json.loads(Path(a.selection).read_text());source='data/gsm8k-formal/train.jsonl'
        identity={'selection_sha256':hashlib.sha256(Path(a.selection).read_bytes()).hexdigest(),
                  'train_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),'target':a.target_pairs,'max_questions':a.max_questions,'seed':a.seed}
        identity_path=out/'identity.json'
        if identity_path.exists() and json.loads(identity_path.read_text())!=identity:raise ValueError('Preference generation identity changed')
        atomic_json(identity_path,identity)
        rows=load_rows(source);random.Random(a.seed).shuffle(rows);rows=rows[:a.max_questions]
        shards=out/'questions';shards.mkdir(exist_ok=True)
        model,tok=load_model(selection['model'],'cuda','float32',False)
        torch.manual_seed(a.seed);random.seed(a.seed);pairs=[];last=None;start=0
        for index in range(len(rows)):
            path=shards/f'{index:05d}.pt'
            if not path.exists():break
            last=torch.load(path,map_location='cpu',weights_only=False);pairs.extend(last['pairs']);start=index+1
        if last:set_rng(last['rng'])
        reward=reward_function('r1_zero')
        for index,row in enumerate(rows[start:],start):
            if len(pairs)>=a.target_pairs:break
            prompt=format_prompt(tok,row['question'],'r1_zero')
            responses,lengths=sample(model,tok,[prompt]*4,512,4,'r1_zero')
            candidates=[]
            for response,length in zip(responses,lengths):
                score=reward(response,row['answer'])
                ids=tok.encode(prompt,add_special_tokens=False)+tok.encode(response+tok.eos_token,add_special_tokens=False)
                if length<512 and len(ids)<=1025 and score['format_reward']==1:
                    candidates.append({'response':response,'length':length,'correct':score['answer_reward']==1})
            positive=[r for r in candidates if r['correct']];negative=[r for r in candidates if not r['correct']]
            possible=[(c,r) for c in positive for r in negative if max(c['length'],r['length'])<=3*max(1,min(c['length'],r['length']))]
            additions=[]
            if possible:
                chosen,rejected=min(possible,key=lambda pair:abs(pair[0]['length']-pair[1]['length']))
                additions=[{'question':row['question'],'chosen':chosen['response'],'rejected':rejected['response'],'source_id':row['id'],
                    'preference_origin':'four sampled SFT policy responses; verified correct versus actual incorrect answer',
                    'chosen_tokens':chosen['length'],'rejected_tokens':rejected['length']}]
            # Each immutable question commits its sampled candidates and RNG together.
            atomic_torch(shards/f'{index:05d}.pt',{'source_id':row['id'],'pairs':additions,'responses':responses,'lengths':lengths,'rng':rng_state()})
            pairs.extend(additions);atomic_json(out/'progress.json',{'questions_done':index+1,'pairs':len(pairs),'target':a.target_pairs})
            print('REAL_PREFERENCES',index+1,len(pairs),flush=True)
        if len(pairs)<200:raise RuntimeError('Fewer than 200 verified preference pairs; no DPO training will start')
        path=out/'pairs.jsonl';temporary=path.with_suffix('.partial')
        with temporary.open('w',encoding='utf-8') as f:
            for row in pairs[:a.target_pairs]:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            f.flush();os.fsync(f.fileno())
        os.replace(temporary,path)
        atomic_json(out/'manifest.json',{'pairs':min(len(pairs),a.target_pairs),'target_met':len(pairs)>=a.target_pairs,
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'model':selection['model'],'identity':identity,
            'scope':'train questions only; no validation/test generation used for pair selection; minimum 200 verified pairs'})


if __name__=='__main__':main()
