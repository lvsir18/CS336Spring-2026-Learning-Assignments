import json
import platform
import sys
import torch

if __name__=='__main__':
    result={'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'cuda_available':torch.cuda.is_available()}
    if torch.cuda.is_available():
        result['gpus']=[{'name':torch.cuda.get_device_name(i),'memory_gib':torch.cuda.get_device_properties(i).total_memory/2**30,
                         'compute_capability':torch.cuda.get_device_capability(i)} for i in range(torch.cuda.device_count())]
        x=torch.randn(128,128,device='cuda');y=x@x.T;torch.cuda.synchronize();result['cuda_matmul_ok']=bool(y.isfinite().all())
        try:
            import triton
            result['triton']=triton.__version__
        except ImportError:result['triton']=None
    print(json.dumps(result,indent=2))
