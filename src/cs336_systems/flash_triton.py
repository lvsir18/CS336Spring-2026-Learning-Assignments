"""Independent forward/backward kernels. Requires Linux CUDA + Triton.

Correctness-first implementation; no GPU speedup claim is made before measurement.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _forward(Q,K,V,O,L,NQ:tl.constexpr,NK:tl.constexpr,D:tl.constexpr,BD:tl.constexpr,B:tl.constexpr,CAUSAL:tl.constexpr):
    row = tl.program_id(0)*B+tl.arange(0,B)
    batch = tl.program_id(1)
    d = tl.arange(0,BD)
    q = tl.load(Q+batch*NQ*D+row[:,None]*D+d[None,:],(row[:,None]<NQ)&(d[None,:]<D),0)
    m = tl.full((B,),float('-inf'),tl.float32)
    l = tl.zeros((B,),tl.float32)
    acc = tl.zeros((B,BD),tl.float32)
    for start in range(tl.cdiv(NK,B)):
        col = start*B+tl.arange(0,B)
        k = tl.load(K+batch*NK*D+col[None,:]*D+d[:,None],(col[None,:]<NK)&(d[:,None]<D),0)
        v = tl.load(V+batch*NK*D+col[:,None]*D+d[None,:],(col[:,None]<NK)&(d[None,:]<D),0)
        s = tl.dot(q,k,input_precision='ieee')*(D**-0.5)
        valid = (col[None,:]<NK)&(row[:,None]<NQ)
        if CAUSAL:
            valid = valid&(row[:,None]>=col[None,:])
        s = tl.where(valid,s,float('-inf'))
        mn = tl.maximum(m,tl.max(s,1))
        # Invalid padded query rows are never stored, but keep their arithmetic finite.
        mn = tl.where(row<NQ,mn,0.)
        alpha = tl.exp(m-mn)
        p = tl.exp(s-mn[:,None])
        acc = acc*alpha[:,None]+tl.dot(p.to(v.dtype),v,input_precision='ieee')
        l = l*alpha+tl.sum(p,1)
        m = mn
    tl.store(O+batch*NQ*D+row[:,None]*D+d[None,:],acc/l[:,None],(row[:,None]<NQ)&(d[None,:]<D))
    tl.store(L+batch*NQ+row,m+tl.log(l),row<NQ)


@triton.jit
def _delta(O,DO,DELTA,N:tl.constexpr,D:tl.constexpr,BD:tl.constexpr,B:tl.constexpr):
    row = tl.program_id(0)*B+tl.arange(0,B)
    d = tl.arange(0,BD)
    o = tl.load(O+row[:,None]*D+d[None,:],(row[:,None]<N)&(d[None,:]<D),0).to(tl.float32)
    do = tl.load(DO+row[:,None]*D+d[None,:],(row[:,None]<N)&(d[None,:]<D),0).to(tl.float32)
    tl.store(DELTA+row,tl.sum(o*do,1),row<N)


@triton.jit
def _backward_q(Q,K,V,DO,L,DELTA,DQ,NQ:tl.constexpr,NK:tl.constexpr,D:tl.constexpr,BD:tl.constexpr,B:tl.constexpr,CAUSAL:tl.constexpr):
    row = tl.program_id(0)*B+tl.arange(0,B)
    batch = tl.program_id(1)
    d = tl.arange(0,BD)
    q = tl.load(Q+batch*NQ*D+row[:,None]*D+d[None,:],(row[:,None]<NQ)&(d[None,:]<D),0)
    do = tl.load(DO+batch*NQ*D+row[:,None]*D+d[None,:],(row[:,None]<NQ)&(d[None,:]<D),0)
    l = tl.load(L+batch*NQ+row,row<NQ,0)
    delta = tl.load(DELTA+batch*NQ+row,row<NQ,0)
    dq = tl.zeros((B,BD),tl.float32)
    for start in range(tl.cdiv(NK,B)):
        col = start*B+tl.arange(0,B)
        k = tl.load(K+batch*NK*D+col[:,None]*D+d[None,:],(col[:,None]<NK)&(d[None,:]<D),0)
        v = tl.load(V+batch*NK*D+col[:,None]*D+d[None,:],(col[:,None]<NK)&(d[None,:]<D),0)
        s = tl.dot(q,tl.trans(k),input_precision='ieee')*(D**-0.5)
        valid = (col[None,:]<NK)&(row[:,None]<NQ)
        if CAUSAL:
            valid = valid&(row[:,None]>=col[None,:])
        p = tl.exp(tl.where(valid,s-l[:,None],float('-inf')))
        dp = tl.dot(do,tl.trans(v),input_precision='ieee')
        ds = p*(dp-delta[:,None])
        dq += tl.dot(ds.to(k.dtype),k,input_precision='ieee')*(D**-0.5)
    tl.store(DQ+batch*NQ*D+row[:,None]*D+d[None,:],dq,(row[:,None]<NQ)&(d[None,:]<D))


@triton.jit
def _backward_kv(Q,K,V,DO,L,DELTA,DK,DV,NQ:tl.constexpr,NK:tl.constexpr,D:tl.constexpr,BD:tl.constexpr,B:tl.constexpr,CAUSAL:tl.constexpr):
    col = tl.program_id(0)*B+tl.arange(0,B)
    batch = tl.program_id(1)
    d = tl.arange(0,BD)
    k = tl.load(K+batch*NK*D+col[:,None]*D+d[None,:],(col[:,None]<NK)&(d[None,:]<D),0)
    v = tl.load(V+batch*NK*D+col[:,None]*D+d[None,:],(col[:,None]<NK)&(d[None,:]<D),0)
    dk,dv = tl.zeros((B,BD),tl.float32),tl.zeros((B,BD),tl.float32)
    for start in range(tl.cdiv(NQ,B)):
        row = start*B+tl.arange(0,B)
        q = tl.load(Q+batch*NQ*D+row[:,None]*D+d[None,:],(row[:,None]<NQ)&(d[None,:]<D),0)
        do = tl.load(DO+batch*NQ*D+row[:,None]*D+d[None,:],(row[:,None]<NQ)&(d[None,:]<D),0)
        l = tl.load(L+batch*NQ+row,row<NQ,0)
        delta = tl.load(DELTA+batch*NQ+row,row<NQ,0)
        s = tl.dot(q,tl.trans(k),input_precision='ieee')*(D**-0.5)
        valid = (row[:,None]<NQ)&(col[None,:]<NK)
        if CAUSAL:
            valid = valid&(row[:,None]>=col[None,:])
        p = tl.exp(tl.where(valid,s-l[:,None],float('-inf')))
        ds = p*(tl.dot(do,tl.trans(v),input_precision='ieee')-delta[:,None])
        dk += tl.dot(tl.trans(ds).to(q.dtype),q,input_precision='ieee')*(D**-0.5)
        dv += tl.dot(tl.trans(p).to(do.dtype),do,input_precision='ieee')
    tl.store(DK+batch*NK*D+col[:,None]*D+d[None,:],dk,(col[:,None]<NK)&(d[None,:]<D))
    tl.store(DV+batch*NK*D+col[:,None]*D+d[None,:],dv,(col[:,None]<NK)&(d[None,:]<D))


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,is_causal=False):
        if not all(t.is_cuda and t.device==q.device and t.dtype==q.dtype for t in (q,k,v)) or q.ndim<2 or q.shape[:-2]!=k.shape[:-2] or q.shape[-1] != k.shape[-1] or k.shape != v.shape:
            raise ValueError('Expected CUDA Q/K/V with equal head dimensions and K/V shapes')
        if q.shape[-1] < 16 or q.shape[-1] > 128:
            raise ValueError('This kernel supports head dimensions 16..128')
        q,k,v = q.contiguous(),k.contiguous(),v.contiguous()
        nq,nk,d = q.shape[-2],k.shape[-2],q.shape[-1]
        batches = q.numel()//(nq*d)
        o = torch.empty_like(q)
        l = torch.empty(q.shape[:-1],device=q.device,dtype=torch.float32)
        # D128 FP32 can exceed Blackwell SM120's per-block shared-memory
        # limit with the default pipeline. Keep its tile pipeline shallow.
        options={'num_stages':1} if d>=128 else {}
        _forward[(triton.cdiv(nq,32),batches)](q,k,v,o,l,nq,nk,d,triton.next_power_of_2(d),32,is_causal,**options)
        ctx.save_for_backward(q,k,v,o,l)
        ctx.is_causal = is_causal
        return o
    @staticmethod
    def backward(ctx,do):
        q,k,v,o,l = ctx.saved_tensors
        do = do.contiguous()
        nq,nk,d = q.shape[-2],k.shape[-2],q.shape[-1]
        batches,bd = q.numel()//(nq*d),triton.next_power_of_2(d)
        delta = torch.empty_like(l)
        dq,dk,dv = torch.empty_like(q),torch.empty_like(k),torch.empty_like(v)
        _delta[(triton.cdiv(batches*nq,32),)](o,do,delta,batches*nq,d,bd,32)
        options={'num_stages':1} if d>=128 else {}
        _backward_q[(triton.cdiv(nq,32),batches)](q,k,v,do,l,delta,dq,nq,nk,d,bd,32,ctx.is_causal,**options)
        _backward_kv[(triton.cdiv(nk,32),batches)](q,k,v,do,l,delta,dk,dv,nq,nk,d,bd,32,ctx.is_causal,**options)
        return dq,dk,dv,None
