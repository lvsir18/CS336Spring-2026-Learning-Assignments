"""Tiled exact attention: no sequence-by-sequence score matrix is retained."""
import math
import torch


class FlashAttentionPyTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,is_causal=False):
        scale = 1/math.sqrt(q.shape[-1])
        out = torch.empty_like(q)
        lse = torch.empty(q.shape[:-1],device=q.device,dtype=torch.float32)
        for i in range(0,q.shape[-2],32):
            qi = q[...,i:i+32,:].float()
            m = torch.full(qi.shape[:-1],-float('inf'),device=q.device)
            l = torch.zeros_like(m)
            acc = torch.zeros_like(qi)
            for j in range(0,k.shape[-2],32):
                kj,vj = k[...,j:j+32,:].float(),v[...,j:j+32,:].float()
                scores = (qi @ kj.transpose(-1,-2))*scale
                if is_causal:
                    mask = torch.arange(i,i+qi.shape[-2],device=q.device)[:,None] >= torch.arange(j,j+kj.shape[-2],device=q.device)[None,:]
                    scores = scores.masked_fill(~mask,-float('inf'))
                m_new = torch.maximum(m,scores.max(-1).values)
                correction = (m-m_new).exp()
                p = (scores-m_new[...,None]).exp()
                l = correction*l+p.sum(-1)
                acc = correction[...,None]*acc+p @ vj
                m = m_new
            out[...,i:i+32,:] = (acc/l[...,None]).to(q.dtype)
            lse[...,i:i+32] = m+l.log()
        ctx.save_for_backward(q,k,v,out,lse)
        ctx.is_causal = is_causal
        return out

    @staticmethod
    def backward(ctx,do):
        q,k,v,out,lse = ctx.saved_tensors
        scale = 1/math.sqrt(q.shape[-1])
        dq,dk,dv = torch.zeros_like(q,dtype=torch.float32),torch.zeros_like(k,dtype=torch.float32),torch.zeros_like(v,dtype=torch.float32)
        delta = (do.float()*out.float()).sum(-1)
        for i in range(0,q.shape[-2],32):
            qi,doi = q[...,i:i+32,:].float(),do[...,i:i+32,:].float()
            for j in range(0,k.shape[-2],32):
                kj,vj = k[...,j:j+32,:].float(),v[...,j:j+32,:].float()
                scores = (qi @ kj.transpose(-1,-2))*scale
                if ctx.is_causal:
                    mask = torch.arange(i,i+qi.shape[-2],device=q.device)[:,None] >= torch.arange(j,j+kj.shape[-2],device=q.device)[None,:]
                    scores = scores.masked_fill(~mask,-float('inf'))
                p = (scores-lse[...,i:i+32,None]).exp()
                ds = p*((doi @ vj.transpose(-1,-2))-delta[...,i:i+32,None])
                dq[...,i:i+32,:] += (ds @ kj)*scale
                dk[...,j:j+32,:] += (ds.transpose(-1,-2) @ qi)*scale
                dv[...,j:j+32,:] += p.transpose(-1,-2) @ doi
        return dq.to(q.dtype),dk.to(k.dtype),dv.to(v.dtype),None
