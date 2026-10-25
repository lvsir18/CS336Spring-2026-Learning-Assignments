"""Ordered v1 optimization plan; immutable once generated on the server."""
import math
from pathlib import Path
from cs336_basics.training import atomic_json

ROOT=Path('runs/optimization-v1')


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'plan.json').exists():return
    (ROOT/'configs').mkdir(exist_ok=True);jobs=[]
    def job(name,command,gpu=False,**extra):jobs.append({'id':name,'command':command,'gpu':gpu,**extra})
    def config(name,cfg):
        path=ROOT/'configs'/f'{name}.json';atomic_json(path,cfg|{'output_dir':str(ROOT/name)});return str(path)
    def train(name,cfg):
        path=config(name,cfg);job(name,['scripts/train_alignment_formal.py','--config',path],True,training_config=path)
    def select(family,names):job('select-'+family,['scripts/optimization_actions.py','select','--family',family,'--runs',*names])
    def test(family):job('test-'+family,['scripts/optimization_actions.py','test','--family',family],True)
    job('prepare-small',['scripts/prepare_optimization_data.py','--part','small'])
    common={'seed':42,'device':'cuda','dtype':'float32','prompt':'r1_zero','max_new_tokens':512,'max_length':1024,
            'rollout_batch_size':4,'val_data':'data/gsm8k-formal/val.jsonl','eval_examples':256,'checkpoint_minutes':30,'track_best':True}
    source='models/olmo-2-0425-1b'
    probe=common|{'method':'sft','model':source,'train_data':'data/gsm8k-formal/sft.jsonl','steps':2,'lr':1e-5,'accum':1,
                  'sampling':'epochs','max_new_tokens':32,'eval_examples':4,'eval_every':2,'checkpoint_every':25}
    path=config('resume-probe',probe)
    for suffix,flags in [('initial',['--stop-after','1']),('restored',[])]:
        job('resume-probe-'+suffix,['scripts/train_alignment_formal.py','--config',path,*flags],True,training_config=path)
    previous='runs/formal/remaining/a5-sft/model'
    train('rl-kl-probe',common|{'method':'rl','model':previous,'reference':previous,'train_data':'data/gsm8k-formal/train.jsonl',
        'steps':2,'lr':1e-6,'prompts_per_step':2,'group_size':2,'importance_reweighting_method':'grpo',
        'rollout_token_ids':True,'baseline':'mean','advantage_normalizer':'std','kl_beta':.01,
        'max_new_tokens':64,'eval_examples':4,'eval_every':2,'checkpoint_every':25})
    pretrain={'seed':42,'architecture':'a4_reference','model':{'vocab_size':50257,'context_length':512,'d_model':512,'num_layers':8,'num_heads':8,'d_ff':1536,'rope_theta':10000.},
              'lr':6e-4,'min_lr':6e-5,'micro_batch_size':8,'gradient_accumulation_steps':8,'eval_batch_size':8,'bf16':True,
              'track_best':True,'validation_mode':'sequential','checkpoint_minutes':15,'log_every':10}
    names=['raw','rules','random42','random43','random44','ranked']
    for name in names:
        path=config('a4-small-'+name,pretrain|{'train_tokens':f'data/optimization-v1/a4-small/{name}.bin',
            'val_tokens':'data/optimization-v1/a4-small/val.bin','steps':600,'warmup_steps':30,'eval_every':50,'checkpoint_every':100})
        job('a4-small-'+name,['scripts/train_lm.py','--config',path],True,training_config=path)
    job('a4-small-report',['scripts/optimization_actions.py','a4-summary'])
    job('expand-web',['scripts/prepare_optimization_data.py','--part','expanded','--target-tokens','100000000','--max-shards','64'])
    for name in ['raw','rules']:
        path=config('a4-expanded-'+name,pretrain|{'train_tokens':f'data/optimization-v1/a4-expanded/{name}.bin',
            'val_tokens':'data/optimization-v1/a4-expanded/val.bin','steps':2000,'warmup_steps':60,'eval_every':100,'checkpoint_every':100,'early_stopping_patience':5})
        job('a4-expanded-'+name,['scripts/train_lm.py','--config',path],True,training_config=path)
    sft=[]
    for name,lr in [('sft-lr5e-6',5e-6),('sft-lr1e-5',1e-5)]:
        sft.append(name);train(name,common|{'method':'sft','model':source,'train_data':'data/gsm8k-formal/sft.jsonl',
            'sampling':'epochs','epochs':2,'steps':math.ceil(7217*2/8),'accum':8,'lr':lr,'min_lr':lr*.1,'warmup_steps':90,'eval_every':450,'checkpoint_every':225})
    select('sft',sft);test('sft')
    job('real-preferences',['scripts/build_real_preferences.py','--selection',str(ROOT/'selected-sft.json'),
                          '--output','data/optimization-v1/preferences','--target-pairs','1000','--max-questions','2048'],True)
    dpo=[]
    for name,lr in [('dpo-lr1e-7',1e-7),('dpo-lr3e-7',3e-7)]:
        dpo.append(name);config(name,common|{'method':'dpo','model':str(ROOT/'selected-sft/model'),'reference':str(ROOT/'selected-sft/model'),
            'train_data':'data/optimization-v1/preferences/pairs.jsonl','sampling':'epochs','epochs':1,'steps':1,'accum':4,'lr':lr,
            'min_lr':lr*.1,'warmup_steps':10,'eval_every':125,'checkpoint_every':125,'beta':.1})
    job('configure-dpo',['scripts/optimization_actions.py','configure-dpo'])
    for name in dpo:
        path=str(ROOT/'configs'/f'{name}.json');job(name,['scripts/train_alignment_formal.py','--config',path],True,training_config=path)
    select('dpo',dpo);test('dpo')
    rl=[]
    # Equal rollout budgets; KL controls hold P/G fixed, group comparison holds beta fixed.
    for name,p,g,beta in [('rl-p8g4-kl0',8,4,0.),('rl-p8g4-kl001',8,4,.001),('rl-p8g4-kl01',8,4,.01),('rl-p4g8-kl01',4,8,.01)]:
        rl.append(name);train(name,common|{'method':'rl','model':str(ROOT/'selected-sft/model'),'reference':str(ROOT/'selected-sft/model'),
            'train_data':'data/gsm8k-formal/train.jsonl','steps':200,'lr':1e-6,'prompts_per_step':p,'group_size':g,
            'importance_reweighting_method':'grpo','rollout_token_ids':True,'baseline':'mean','advantage_normalizer':'std','kl_beta':beta,
            'eval_every':50,'checkpoint_every':50})
    select('rl',rl);test('rl')
    job('summary',['scripts/optimization_actions.py','summary'])
    atomic_json(ROOT/'plan.json',{'version':1,'jobs':jobs,'policy':{'allocator_gib':42,'minimum_free_gib':49,
        'validation':'all 256 GSM8K validation examples; select model/checkpoint by validation only; one test per family',
        'rl':'PPO-style clipped group-relative objective, mu=1, sampled token IDs preserved, frozen selected SFT reference for beta>0',
        'expanded_data':'WET; at least 100M independent retained rules tokens or explicit preparation failure; up to 64 shards'}})
    print('OPTIMIZATION_PLANNED',len(jobs),flush=True)


if __name__=='__main__':main()
