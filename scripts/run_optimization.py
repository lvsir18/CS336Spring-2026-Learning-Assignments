"""Detached serial optimization queue. A failed stage stops dependent work explicitly."""
import argparse,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
from cs336_basics.training import atomic_json,run_lock


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');a=p.parse_args()
    os.chdir(Path(__file__).resolve().parents[1]);root=Path('runs/optimization-v1');root.mkdir(parents=True,exist_ok=True)
    with run_lock(root):
        child=None;stopping=False
        def stop(number,frame):
            nonlocal stopping
            stopping=True
            if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM)
        for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,stop)
        (root/'controller.pid').write_text(str(os.getpid())+'\n')
        subprocess.run([sys.executable,'scripts/plan_optimization.py'],check=True)
        plan=json.loads((root/'plan.json').read_text());result_path=root/'results.json';results=json.loads(result_path.read_text()) if result_path.exists() else {}
        if results and not a.resume:raise FileExistsError('Use --resume for an existing optimization queue')
        env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',CS336_PROGRESS='1',TOKENIZERS_PARALLELISM='false')
        def state(phase,**extra):atomic_json(root/'state.json',{'phase':phase,'controller_pid':os.getpid(),'time_unix':time.time(),**extra})
        for job in plan['jobs']:
            name=job['id']
            if results.get(name,{}).get('status')=='complete':continue
            if stopping:state('stopped');return 143
            command=list(job['command'])
            if job.get('training_config'):
                path=Path(job['training_config']);cfg=json.loads(path.read_text());out=Path(cfg['output_dir'])
                if 'train_tokens' in cfg:
                    cfg['data_identity']={key:digest(cfg[key]) for key in ['train_tokens','val_tokens']};atomic_json(path,cfg)
                if isinstance(cfg.get('model'),str) and cfg['model'].endswith('selected-sft/model'):
                    cfg['model_identity']=digest(root/'selected-sft.json');atomic_json(path,cfg)
                if any((out/n).exists() for n in ['checkpoint.pt','checkpoint.previous.pt']):command+=['--resume','auto']
                elif (out/'metrics.jsonl').exists():(out/'metrics.jsonl').rename(out/f'metrics.uncommitted-{time.time_ns()}.jsonl')
            if job['gpu']:
                while True:
                    free=float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])/1024
                    if free>=49:break
                    state('waiting_for_vram',job=name,free_gib=free)
                    if stopping:state('stopped');return 143
                    time.sleep(10)
                (root/'memory').mkdir(exist_ok=True)
                command=['scripts/gpu_budget.py','--allocator-gib','42','--min-free-gib','48','--report',str(root/'memory'/f'{name}-{time.time_ns()}.json'),'--',*command]
            commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();start=time.time()
            state('running',job=name,command=command,git_commit=commit)
            with (root/(name+'.log')).open('a') as log:
                child=subprocess.Popen([sys.executable,'-u',*command],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                atomic_json(root/'active-child.json',{'job':name,'pid':child.pid,'started':start})
                code=child.wait();child=None
            results[name]={'status':'complete' if code==0 else ('stopped' if stopping else 'failed'),'exit_code':code,'elapsed_seconds':time.time()-start,'git_commit':commit}
            atomic_json(result_path,results)
            if code!=0:
                state('stopped' if stopping else 'failed',job=name,exit_code=code);return code
        state('complete',jobs=len(results));return 0


if __name__=='__main__':raise SystemExit(main())
