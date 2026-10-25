"""Reproducible single-GPU A2-A5 plan; full-B200/multi-GPU protocols are excluded."""
import json,math
from pathlib import Path
from cs336_basics.training import atomic_json


def main():
    root=Path('runs/formal/remaining');configs=root/'configs';configs.mkdir(parents=True,exist_ok=True);jobs=[]
    def job(name,command,**extra):
        jobs.append({'id':name,'command':command,'gpu':True,**extra})
    def config(name,cfg):
        path=configs/f'{name}.json';atomic_json(path,cfg);return str(path)
    job('a2-numerics',['scripts/check_flash_numerics.py','--output',str(root/'a2/flash-numerics.json')])
    for dim in [64,128]:
        for backward in [False,True]:
            for impl in ['naive','triton','sdpa']:
                name=f'a2-{impl}-d{dim}-'+('fwd-bwd' if backward else 'fwd')
                command=['scripts/benchmark_attention.py','--impl',impl,'--head-dim',str(dim),'--batch','4',
                         '--lengths','256','1024','4096','--warmup','10','--repeats','30','--causal','--bf16',
                         '--output',str(root/'a2'/f'{name}.json')]
                if impl=='sdpa':command+=['--sdpa-backend','flash']
                if backward:command+=['--backward']
                job(name,command)
    base=json.loads(Path('runs/formal/a1-controller/resolved-config.json').read_text())
    for variant,flags in [('eager',['--profile']),('checkpoint',['--checkpointing']),('compile',['--compile'])]:
        job(f'a2-model-{variant}',['scripts/benchmark_model.py','--config','configs/a1-formal-tinystories.json',
            '--warmup','5','--repeats','20','--output',str(root/'a2'/f'model-{variant}.json'),*flags])
    tiny={'model':{'vocab_size':256,'context_length':32,'d_model':32,'num_layers':1,'num_heads':2,'d_ff':64},'micro_batch_size':4}
    cpuconfig=config('a2-cpu',tiny)
    for i,method in enumerate(['naive','flat','ddp','zero1','fsdp']):
        job('a2-gloo-'+method,['scripts/benchmark_distributed.py','--config',cpuconfig,'--method',method,'--backend','gloo',
            '--cpu-workers','2','--warmup','2','--repeats','5','--port',str(29640+i),'--output',str(root/'a2'/f'gloo-{method}.json')],gpu=False)
    accounting=[]
    for budget in [1e15,3e15,1e16]:
        for d,l,h,f in [(256,4,4,704),(384,6,6,1024),(512,8,8,1344)]:
            arch={'vocab_size':10000,'context_length':256,'d_model':d,'num_layers':l,'num_heads':h,'d_ff':f}
            n=2*10000*d+l*(4*d*d+3*d*f+2*d)+d;k=64*2*256;steps=int(budget/(6*n*k))
            name=f'a3-C{budget:.0e}-N{n}';cfg=base|{'model':arch,'steps':steps,'warmup_steps':max(1,int(.05*steps)),
                'micro_batch_size':64,'gradient_accumulation_steps':2,'eval_every':max(1,steps//5),'eval_batches':32,
                'checkpoint_every':50,'output_dir':str(root/'a3'/name)}
            path=config(name,cfg);job(name,['scripts/train_lm.py','--config',path],training_config=path)
            accounting.append({'id':name,'compute_budget':budget,'parameters':n,'steps':steps,'tokens':steps*k,'actual_flops':6*n*steps*k,'output_dir':cfg['output_dir']})
    atomic_json(root/'a3-accounting.json',accounting)
    job('a3-fit',['scripts/fit_remaining_scaling.py'],gpu=False)
    for variant in ['raw','clean']:
        name='a4-'+variant
        cfg={'seed':42,'architecture':'a4_reference','model':{'vocab_size':50257,'context_length':512,'d_model':512,'num_layers':8,'num_heads':8,'d_ff':1536,'rope_theta':10000.},
             'train_tokens':f'data/a4-formal/{variant}.bin','val_tokens':'data/a4-formal/val.bin','output_dir':str(root/name),
             'lr':6e-4,'min_lr':6e-5,'warmup_steps':60,'steps':2000,'micro_batch_size':8,'gradient_accumulation_steps':8,
             'eval_every':100,'eval_batch_size':8,'eval_batches':32,'bf16':True,'checkpoint_every':50,'checkpoint_minutes':10,'log_every':10}
        path=config(name,cfg);job(name,['scripts/train_lm.py','--config',path],training_config=path,requires=['data/a4-formal/manifest.json'])
    def evaluation(name,model,requires):
        job(name,['scripts/evaluate_alignment.py','--model',model,'--data','data/gsm8k-formal/test.jsonl','--prompt','r1_zero',
                  '--dtype','float32','--batch-size','4','--max-new-tokens','512','--seed','42','--output',str(root/name/'eval.jsonl')],requires=requires)
    data=['data/gsm8k-formal/manifest.json'];source='models/olmo-2-0425-1b'
    probe={'seed':42,'method':'sft','model':source,'device':'cuda','dtype':'float32','lr':1e-5,'steps':2,'accum':1,
           'train_data':'data/gsm8k-formal/sft.jsonl','val_data':'data/gsm8k-formal/val.jsonl','prompt':'r1_zero','max_length':1024,
           'max_new_tokens':64,'rollout_batch_size':2,'eval_examples':4,'checkpoint_every':25,'eval_every':2,
           'output_dir':str(root/'a5-resume-probe')}
    probe_path=config('a5-resume-probe',probe)
    job('a5-resume-probe-initial',['scripts/train_alignment_formal.py','--config',probe_path,'--stop-after','1'],training_config=probe_path,requires=data)
    job('a5-resume-probe-restored',['scripts/train_alignment_formal.py','--config',probe_path],training_config=probe_path,requires=data)
    evaluation('a5-base-test',source,data)
    common={'seed':42,'device':'cuda','dtype':'float32','prompt':'r1_zero','max_new_tokens':512,'max_length':1024,'rollout_batch_size':4,
            'val_data':'data/gsm8k-formal/val.jsonl','eval_examples':64,'checkpoint_minutes':30}
    phases=[('sft',{'model':source,'train_data':'data/gsm8k-formal/sft.jsonl','steps':800,'lr':1e-5,'accum':8,'eval_every':200,'checkpoint_every':100}),
            ('rl',{'model':str(root/'a5-sft/model'),'train_data':'data/gsm8k-formal/train.jsonl','steps':200,'lr':1e-6,
                   'prompts_per_step':4,'group_size':4,'importance_reweighting_method':'none','baseline':'mean','advantage_normalizer':'std','eval_every':50,'checkpoint_every':25}),
            ('dpo',{'model':str(root/'a5-sft/model'),'reference':str(root/'a5-sft/model'),'train_data':'data/gsm8k-formal/dpo.jsonl',
                    'steps':200,'lr':3e-7,'beta':.1,'accum':4,'eval_every':100,'checkpoint_every':50})]
    for method,settings in phases:
        name='a5-'+method;cfg=common|settings|{'method':method,'output_dir':str(root/name)};path=config(name,cfg)
        requirements=data+([str(root/'a5-resume-probe/complete.json')] if method=='sft' else [str(root/'a5-sft/complete.json')])
        job(name,['scripts/train_alignment_formal.py','--config',path],training_config=path,requires=requirements)
        evaluation(name+'-test',str(root/name/'model'),[str(root/name/'complete.json')])
    jobs=[j for j in jobs if not j['id'].startswith('a4-')]+[j for j in jobs if j['id'].startswith('a4-')]
    atomic_json(root/'plan.json',{'jobs':jobs,'limitations':['single GPU; Gloo results are correctness/smoke timings, not NCCL scaling',
                'A3 local estimated IsoFLOPs, not official B200 API','A4 scaled reference architecture and 65.536M tokens per arm, not official 8.59B-token protocol',
                'DPO constructed math consistency preferences, not a safety benchmark']})
    print('PLANNED',len(jobs),'jobs')


if __name__=='__main__':main()
