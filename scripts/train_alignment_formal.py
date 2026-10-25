"""Resumable full-parameter SFT/DPO/RLVR on independently prepared real data."""
import argparse,hashlib,json,math,random,time
from pathlib import Path
import numpy as np
import torch
from cs336_basics.training import atomic_json,run_lock
from cs336_alignment.experiment_state import ExperimentState,rng_state,set_rng
from cs336_alignment.hf_workflow import load_model,load_rows,format_prompt,sample,old_scores,reward_function,evaluate
from cs336_alignment.core import tokenize_prompt_and_output,get_response_log_probs,compute_per_instance_dpo_loss,grpo_train_step
from cs336_basics.experiment import EpochSampler,BestModel
from cs336_basics.optimizer import get_lr_cosine_schedule


def dpo_loss(model,reference,tok,prompt,chosen,rejected,beta,max_length):
    def score(network,response):
        batch=tokenize_prompt_and_output([prompt],[response+tok.eos_token],tok)
        if batch['input_ids'].shape[1]>max_length:raise ValueError('Overlength prepared DPO row')
        batch={k:v.to(next(network.parameters()).device) for k,v in batch.items()}
        values=get_response_log_probs(network,batch['input_ids'],batch['labels'])['log_probs']
        return (values*batch['response_mask']).sum()
    delta=score(model,chosen)-score(model,rejected)
    with torch.no_grad():ref_delta=score(reference,chosen)-score(reference,rejected)
    return -torch.nn.functional.logsigmoid(beta*(delta-ref_delta))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--resume');p.add_argument('--stop-after',type=int)
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text());out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)
    for key in ['train_data','val_data']:
        if key in cfg:cfg[key+'_sha256']=hashlib.sha256(Path(cfg[key]).read_bytes()).hexdigest()
    with run_lock(out):
        random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.set_num_threads(4)
        model,tok=load_model(cfg['model'],cfg.get('device','cuda'),cfg.get('dtype','float32'),True)
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],betas=(.9,.95),weight_decay=cfg.get('weight_decay',0.),foreach=False)
        reference=None
        if cfg['method']=='dpo' or (cfg['method']=='rl' and cfg.get('kl_beta',0)>0):
            reference,_=load_model(cfg.get('reference',cfg['model']),cfg.get('device','cuda'),cfg.get('dtype','float32'),False)
            reference.requires_grad_(False)
        state=ExperimentState(cfg,model,optimizer,a.resume)
        rows=load_rows(cfg['train_data']);val=load_rows(cfg['val_data'])[:cfg.get('eval_examples',64)]
        sampler=EpochSampler(len(rows),cfg['seed'],state.extra.get('sampler')) if cfg.get('sampling')=='epochs' else None
        if sampler and cfg.get('epochs') and cfg['steps']!=math.ceil(cfg['epochs']*len(rows)/cfg.get('accum',8)):
            raise ValueError('Epoch schedule does not match dataset size')
        selection=BestModel(out,'max',state.extra.get('selection')) if cfg.get('track_best') else None
        def validation(step):
            saved=rng_state();random.seed(cfg['seed']+100000);torch.manual_seed(cfg['seed']+100000)
            try:result=evaluate(model,tok,val,out/f'eval-{step}.jsonl',cfg.get('prompt','r1_zero'),cfg.get('max_new_tokens',512),cfg.get('rollout_batch_size',4))
            finally:set_rng(saved)
            print('VALIDATION',json.dumps(result),flush=True)
            if selection:
                selection.update(result['reward'],step,model,cfg.get('selection_min_delta',0.))
                state.extra['selection']=selection.state
                state.extra['validated_step']=step
            return result
        limit=min(cfg['steps'],a.stop_after) if a.stop_after else cfg['steps']
        with state.signals():
            for step in range(state.step,limit):
                t0=time.perf_counter()
                if cfg.get('warmup_steps') is not None:
                    lr=get_lr_cosine_schedule(step,cfg['lr'],cfg.get('min_lr',cfg['lr']*.1),cfg['warmup_steps'],cfg['steps'])
                    for group in optimizer.param_groups:group['lr']=lr
                if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
                if cfg['method']=='rl':
                    chosen=random.sample(rows,cfg['prompts_per_step']);group=cfg['group_size'];count=len(chosen)*group
                    prompts=[format_prompt(tok,r['question'],cfg.get('prompt','r1_zero')) for r in chosen for _ in range(group)]
                    answers=[r['answer'] for r in chosen for _ in range(group)]
                    generated=sample(model,tok,prompts,cfg.get('max_new_tokens',512),cfg.get('rollout_batch_size',4),cfg.get('prompt','r1_zero'),cfg.get('rollout_token_ids',False))
                    responses,lengths=generated[:2];encoded=generated[2] if len(generated)==3 else None
                    old=old_scores(model,tok,prompts,responses,1,encoded) if cfg.get('importance_reweighting_method','none')!='none' else None
                    _,stats=grpo_train_step(model,tok,optimizer,count,1.,reward_function(cfg.get('prompt','r1_zero')),prompts,responses,answers,group,
                        cfg.get('baseline','mean'),1e-6,cfg.get('advantage_normalizer','std'),cfg.get('importance_reweighting_method','none'),old,.2,
                        reference_model=reference,kl_beta=cfg.get('kl_beta',0.),encoded_rollouts=encoded)
                    stats['mean_generated_tokens']=sum(lengths)/len(lengths)
                else:
                    optimizer.zero_grad(set_to_none=True);loss_value=0.
                    count=cfg.get('accum',8)
                    if sampler and cfg.get('epochs'):count=min(count,cfg['epochs']*len(rows)-sampler.seen)
                    if count<=0:raise ValueError('steps exceeds configured epoch sample count')
                    indices=sampler.take(count) if sampler else [random.randrange(len(rows)) for _ in range(count)]
                    for index in indices:
                        row=rows[index]
                        if cfg['method']=='sft':
                            prompt=format_prompt(tok,row['question'],cfg.get('prompt','r1_zero'))
                            batch=tokenize_prompt_and_output([prompt],[row['response']+tok.eos_token],tok)
                            if batch['input_ids'].shape[1]>cfg.get('max_length',1024):raise ValueError('Overlength prepared SFT row')
                            batch={k:v.to(next(model.parameters()).device) for k,v in batch.items()}
                            score=get_response_log_probs(model,batch['input_ids'],batch['labels'])['log_probs']
                            loss=-(score*batch['response_mask']).sum()/batch['response_mask'].sum().clamp_min(1)
                        else:
                            prefix=format_prompt(tok,row['question'],cfg.get('prompt','r1_zero'))
                            loss=dpo_loss(model,reference,tok,prefix,row['chosen'],row['rejected'],cfg.get('beta',.1),cfg.get('max_length',1024))
                        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite alignment loss')
                        (loss/count).backward();loss_value+=loss.detach().item()/count
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                    stats={'loss':loss_value,'gradient_norm':float(norm)}
                    if sampler:
                        state.extra['sampler']=sampler.state_dict()
                        stats.update(sample_ids=[rows[i].get('id',rows[i].get('source_id',i)) for i in indices],samples_seen=sampler.seen)
                if torch.cuda.is_available():torch.cuda.synchronize();stats['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
                if not all(math.isfinite(float(stats[k])) for k in ['loss'] if k in stats):raise FloatingPointError('Nonfinite alignment statistics')
                stats.update(step=step+1,step_seconds=time.perf_counter()-t0,lr=optimizer.param_groups[0]['lr']);state.record(stats)
                due=(step+1)%cfg.get('eval_every',100)==0 or step+1==cfg['steps']
                if selection and due and not state.stopping:
                    validation(step+1)
                    state.save('signal' if state.stopping else 'validation')
                    if state.stopping:return 143
                else:
                    if state.boundary():return 143
                    if due:validation(step+1)
            if state.step<cfg['steps']:
                state.save('partial');return 0
            # Downstream phases wait for complete.json, never a partially written export.
            if not (out/f'eval-{cfg["steps"]}.summary.json').exists() or (selection and state.extra.get('validated_step')!=cfg['steps']):
                validation(cfg['steps'])
                if selection:state.save('validation')
            if selection:selection.load(model)
            model.save_pretrained(out/'model');tok.save_pretrained(out/'model')
            atomic_json(out/'complete.json',{'step':state.step,'method':cfg['method'],'time_unix':time.time(),
                        'exported_step':selection.state['step'] if selection else state.step,'selection':selection.state if selection else None})
        return 0


if __name__=='__main__':raise SystemExit(main())
