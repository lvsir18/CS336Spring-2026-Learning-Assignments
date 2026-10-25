"""Persistent serial GPU queue; independent CPU data preparation and safe resume."""
import argparse,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
from cs336_basics.training import atomic_json,run_lock


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');a=p.parse_args()
    os.chdir(Path(__file__).resolve().parents[1]);root=Path('runs/formal/remaining');root.mkdir(parents=True,exist_ok=True)
    with run_lock(root):
        (root/'controller.pid').write_text(str(os.getpid())+'\n');stopping=False;child=None;prep=None
        env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',TORCHINDUCTOR_COMPILE_THREADS='4',
                 TORCHINDUCTOR_CACHE_DIR=str(Path('.cache/torchinductor').resolve()),TRITON_CACHE_DIR=str(Path('.cache/triton').resolve()),CS336_PROGRESS='1')
        def stop(number,frame):
            nonlocal stopping
            stopping=True
            if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM)
            if prep is not None and prep.poll() is None:os.killpg(prep.pid,signal.SIGTERM)
        for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,stop)
        subprocess.run([sys.executable,'scripts/plan_remaining.py'],check=True)
        plan=json.loads((root/'plan.json').read_text());results_path=root/'results.json'
        results=json.loads(results_path.read_text()) if results_path.exists() else {}
        if results and not a.resume:raise FileExistsError('Existing queue; use --resume')
        # Downloads run outside the GPU queue; their initial launcher is already active.
        helper="""import subprocess,sys,time
from pathlib import Path
started_download=False
def ensure(predicate):
 global started_download
 while not predicate():
  file=Path('data/formal-assets/downloader.pid')
  pid=file.read_text().strip() if file.exists() else '0'
  process=Path('/proc/'+pid+'/cmdline')
  try:alive=process.exists() and b'fetch_formal_assets.py' in process.read_bytes()
  except OSError:alive=False
  if not alive:
   if predicate():return
   if started_download:raise RuntimeError('Asset download exited before required files were complete; inspect download.log')
   subprocess.Popen([sys.executable,'-u','scripts/fetch_formal_assets.py'])
   started_download=True
  time.sleep(5)
needed=['models/olmo-2-0425-1b/model-00001-of-00002.safetensors','models/olmo-2-0425-1b/model-00002-of-00002.safetensors',
        'models/olmo-2-0425-1b/config.json','models/olmo-2-0425-1b/tokenizer.json','models/olmo-2-0425-1b/tokenizer_config.json',
        'data/formal-assets/gsm8k/train-00000-of-00001.parquet','data/formal-assets/gsm8k/test-00000-of-00001.parquet']
ensure(lambda:all(Path(x).exists() for x in needed))
subprocess.run([sys.executable,'-u','scripts/prepare_remaining_data.py','--part','gsm8k'],check=True)
ensure(lambda:Path('data/formal-assets/manifest.json').exists())
subprocess.run([sys.executable,'-u','scripts/prepare_remaining_data.py','--part','web'],check=True)
"""
        helper_path=root/'prepare-helper.py';helper_path.write_text(helper)
        with (root/'preparation.log').open('a') as f:
            prep=subprocess.Popen([sys.executable,'-u',str(helper_path)],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        def state(phase,**extra):
            atomic_json(root/'state.json',{'phase':phase,'controller_pid':os.getpid(),'preparation_pid':prep.pid,'time_unix':time.time(),**extra})
        for job in plan['jobs']:
            name=job['id']
            if results.get(name,{}).get('status')=='complete':continue
            if stopping:state('stopped');return 143
            missing=[x for x in job.get('requires',[]) if not Path(x).exists()]
            while missing and prep.poll() is None and all(x.startswith('data/') for x in missing):
                state('waiting_for_data',job=name,missing=missing)
                if stopping:state('stopped');return 143
                time.sleep(5);missing=[x for x in job.get('requires',[]) if not Path(x).exists()]
            if missing:
                results[name]={'status':'blocked','missing':missing};atomic_json(results_path,results);continue
            command=[*job['command']]
            if job.get('training_config'):
                path=Path(job['training_config']);cfg=json.loads(path.read_text());out=Path(cfg['output_dir'])
                if 'train_tokens' in cfg:
                    cfg['data_identity']={key:sha(cfg[key]) for key in ['train_tokens','val_tokens']}
                    atomic_json(path,cfg)
                checkpoint=any((out/x).exists() for x in ['checkpoint.pt','checkpoint.previous.pt'])
                if checkpoint:command+=['--resume','auto']
                elif (out/'metrics.jsonl').exists():
                    (out/'metrics.jsonl').rename(out/f'metrics.uncommitted-{time.time_ns()}.jsonl')
            if job['gpu']:
                while True:
                    free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
                    if float(free.splitlines()[0])/1024>=49:break
                    state('waiting_for_vram',job=name,free_mib=float(free.splitlines()[0]))
                    if stopping:state('stopped');return 143
                    time.sleep(10)
                command=['scripts/gpu_budget.py','--allocator-gib','42','--min-free-gib','48','--report',str(root/'memory'/f'{name}-{time.time_ns()}.json'),'--',*command]
            state('running',job=name,command=command);start=time.time()
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
            with (root/f'{name}.log').open('a') as log:
                child=subprocess.Popen([sys.executable,'-u',*command],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                atomic_json(root/'active-child.json',{'job':name,'pid':child.pid,'started':start})
                code=child.wait();child=None
            results[name]={'status':'complete' if code==0 else ('stopped' if stopping else 'failed'),'exit_code':code,'elapsed_seconds':time.time()-start,
                           'git_commit':source_commit}
            atomic_json(results_path,results)
            if stopping:state('stopped',job=name);return 143
        # Keep the queue state explicit if some independent stage failed.
        failures={k:v for k,v in results.items() if v['status']!='complete'}
        subprocess.run([sys.executable,'scripts/summarize_remaining.py'],check=True)
        state('complete' if not failures else 'complete_with_failures',failures=failures)
        return 1 if failures else 0


if __name__=='__main__':raise SystemExit(main())
