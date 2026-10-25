"""One independent server controller: full data preparation, verification, training.

No time cutoff, no blind retries, no external service installation. See docs/operations.md.
"""
import argparse, hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path
from cs336_basics.training import atomic_json,run_lock


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''): value.update(chunk)
    return value.hexdigest()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--resume',action='store_true')
    p.add_argument('--config',default='configs/a1-formal-tinystories.json'); p.add_argument('--endpoint',default='https://hf-mirror.com')
    args=p.parse_args(); os.chdir(Path(__file__).resolve().parents[1])
    controller=Path('runs/formal/a1-controller'); controller.mkdir(parents=True,exist_ok=True)
    with run_lock(controller):
        (controller/'controller.pid').write_text(str(os.getpid())+'\n')
        child=None; stopped=False
        def state(phase,**extra):
            row={'phase':phase,'controller_pid':os.getpid(),'time_unix':time.time(),**extra}
            atomic_json(controller/'state.json',row); print(json.dumps(row),flush=True)
        def stop(number,frame):
            nonlocal stopped
            stopped=True
            if child is not None and child.poll() is None: child.send_signal(signal.SIGTERM)
        for sig in [signal.SIGTERM,signal.SIGINT]: signal.signal(sig,stop)
        def execute(command,phase):
            nonlocal child
            if stopped: raise InterruptedError('Operator requested stop')
            state(phase,command=command)
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL)
            atomic_json(controller/'active-child.json',{'pid':child.pid,'phase':phase,'time_unix':time.time()})
            code=child.wait(); child=None
            if stopped: raise InterruptedError('Operator requested stop')
            if code: raise RuntimeError(f'{phase} exited {code}; inspect log; no automatic retry')
        try:
            manifest_path=Path('data/tinystories/manifest.json')
            if not manifest_path.exists():
                execute([sys.executable,'-u','scripts/prepare_formal_a1.py','--endpoint',args.endpoint],'preparing_data')
            state('verifying_data')
            manifest=json.loads(manifest_path.read_text())
            paths={'tokenizer':('data/tinystories/tokenizer.json',manifest['tokenizer']['sha256']),
                   'train':('data/tinystories/train.bin',manifest['processed']['train']['token_sha256']),
                   'val':('data/tinystories/val.bin',manifest['processed']['val']['token_sha256'])}
            for name,(path,sha) in paths.items():
                if digest(path)!=sha: raise ValueError(f'{name} data hash mismatch; task not started')
            metadata_hashes={}
            for split in ['train','val']:
                path=Path(f'data/tinystories/{split}.bin.json'); meta=json.loads(path.read_text())
                expected=manifest['processed'][split]
                if meta['dtype']!=expected['dtype'] or meta['tokens']!=expected['tokens'] or meta['vocab_size']!=manifest['tokenizer']['vocab_size']:
                    raise ValueError(f'{split} token metadata differs from manifest')
                metadata_hashes[f'{split}_metadata_sha256']=digest(path)
            cfg=json.loads(Path(args.config).read_text()); cfg['data_identity']={'manifest_sha256':digest(manifest_path),
                **{name:sha for name,(path,sha) in paths.items()},**metadata_hashes}
            resolved=controller/'resolved-config.json'
            if resolved.exists() and json.loads(resolved.read_text())!=cfg:
                raise ValueError('Existing resolved config differs; create a separate formal experiment')
            atomic_json(resolved,cfg)
            run=Path(cfg['output_dir']); has_checkpoint=any((run/x).exists() for x in ['checkpoint.pt','checkpoint.previous.pt'])
            if args.resume and not has_checkpoint: raise FileNotFoundError('No formal checkpoint to resume')
            if not args.resume and (has_checkpoint or (run/'metrics.jsonl').exists()): raise FileExistsError('Use --resume for existing run')
            report=controller/f'memory-{time.time_ns()}.json'
            command=[sys.executable,'-u','scripts/gpu_budget.py','--allocator-gib','42','--min-free-gib','48',
                     '--report',str(report),'--','scripts/train_lm.py','--config',str(resolved)]
            if args.resume: command+=['--resume','auto']
            execute(command,'training')
            summary=json.loads((run/'status.json').read_text())
            if summary['step']!=cfg['steps']: raise RuntimeError('Training exited before the configured final step')
            state('complete',steps=summary['step'],checkpoint=summary['checkpoint'],memory_report=str(report))
            return 0
        except InterruptedError as exc:
            state('stopped',message=str(exc)); return 143
        except Exception as exc:
            state('failed',error=f'{type(exc).__name__}: {exc}'); raise


if __name__=='__main__': raise SystemExit(main())
