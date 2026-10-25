"""Run one Python script in-process with a bounded PyTorch CUDA allocator.

The allocator limit is not a whole-device reservation or a limit on non-PyTorch
CUDA allocations. Do not use this to wrap scripts that spawn GPU subprocesses.
"""
import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import threading
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allocator-gib',type=float,default=36)
    parser.add_argument('--min-free-gib',type=float,default=42)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command: parser.error('Provide a Python script after --')
    if args.allocator_gib<=0 or args.min_free_gib<args.allocator_gib:
        parser.error('Need 0 < allocator-gib <= min-free-gib')
    report=args.report.resolve();script=Path(command[0]).resolve()
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    free,total=torch.cuda.mem_get_info()
    if free/2**30<args.min_free_gib:
        raise RuntimeError(f'Free VRAM {free/2**30:.2f} GiB below required {args.min_free_gib:.2f}; task not started')
    if args.allocator_gib>=total/2**30:
        raise ValueError('Allocator limit must be smaller than device total')
    torch.cuda.set_per_process_memory_fraction(args.allocator_gib*2**30/total)
    torch.cuda.reset_peak_memory_stats()
    stop=threading.Event();sampled=[];allocated_peaks=[];reserved_peaks=[]
    def monitor():
        while not stop.is_set():
            allocated_peaks.append(torch.cuda.max_memory_allocated())
            reserved_peaks.append(torch.cuda.max_memory_reserved())
            try:
                rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=5)
                for row in rows.splitlines():
                    pid,memory=[part.strip() for part in row.split(',')]
                    if pid==str(os.getpid()): sampled.append(float(memory))
            except (OSError,ValueError,subprocess.SubprocessError): pass
            stop.wait(1)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    start=time.monotonic();exit_code=0;error=None
    original_argv=sys.argv;original_cwd=Path.cwd()
    try:
        sys.argv=[str(script),*command[1:]]
        sys.path.insert(0,str(script.parent))
        runpy.run_path(str(script),run_name='__main__')
    except SystemExit as exc:
        exit_code=exc.code if isinstance(exc.code,int) else (0 if exc.code is None else 1)
        if exit_code: error=str(exc)
        raise
    except BaseException as exc:
        exit_code=1;error=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        sys.argv=original_argv;os.chdir(original_cwd)
        stop.set();thread.join(timeout=6)
        row={'pid':os.getpid(),'command':command,'allocator_limit_gib':args.allocator_gib,
             'min_free_gib':args.min_free_gib,'free_before_gib':free/2**30,
             'elapsed_seconds':time.monotonic()-start,'exit_code':exit_code,'error':error,
             'peak_allocated_gib':max([torch.cuda.max_memory_allocated(),*allocated_peaks])/2**30,
             'peak_reserved_gib':max([torch.cuda.max_memory_reserved(),*reserved_peaks])/2**30,
             'max_sampled_process_mib':max(sampled) if sampled else None,
             'scope':'single process PyTorch allocator; shared GPU; sampling can miss brief peaks if target resets counters; also inspect per-step metrics'}
        report.parent.mkdir(parents=True,exist_ok=True)
        report.write_text(json.dumps(row,indent=2)+'\n',encoding='utf-8')
        print('RESOURCE_REPORT',json.dumps(row),flush=True)


if __name__=='__main__':main()
