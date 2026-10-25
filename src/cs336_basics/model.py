"""Pre-norm decoder Transformer; names match the official A1 state dictionaries."""
import math
import torch
from torch import nn


class Linear(nn.Module):
    def __init__(self, d_in, d_out, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in, device=device, dtype=dtype))
        std = math.sqrt(2 / (d_in + d_out))
        nn.init.trunc_normal_(self.weight, std=std, a=-3*std, b=3*std)
    def forward(self, x):
        return x.to(self.weight.dtype) @ self.weight.T


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, std=1, a=-3, b=3)
    def forward(self, ids):
        return self.weight[ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight).to(x.dtype)


def silu(x):
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, **kwargs):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, **kwargs)
        self.w2 = Linear(d_ff, d_model, **kwargs)
        self.w3 = Linear(d_model, d_ff, **kwargs)
    def forward(self, x):
        return self.w2(silu(self.w1(x)) * self.w3(x))


class SiLUFFN(nn.Module):
    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.w1 = Linear(d_model, 4*d_model, **kwargs)
        self.w2 = Linear(4*d_model, d_model, **kwargs)
    def forward(self, x):
        return self.w2(silu(self.w1(x)))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        if d_k % 2:
            raise ValueError('RoPE head dimension must be even')
        inv_freq = theta ** (-torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k)
        self.register_buffer('inv_freq', inv_freq, persistent=False)
    def forward(self, x, token_positions):
        angles = token_positions.to(device=x.device, dtype=torch.float32)[..., None] * self.inv_freq
        # Broadcast positions across attention heads, including per-batch positions.
        while angles.ndim < x.ndim:
            angles = angles.unsqueeze(-3)
        cos, sin = angles.cos(), angles.sin()
        even, odd = x[..., 0::2].float(), x[..., 1::2].float()
        return torch.stack((even*cos - odd*sin, even*sin + odd*cos), -1).flatten(-2).to(x.dtype)


def softmax(x, dim=-1):
    shifted = x - x.max(dim=dim, keepdim=True).values
    exps = shifted.exp()
    return exps / exps.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(q, k, v, mask=None):
    scores = (q @ k.transpose(-1,-2)) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    return softmax(scores, -1) @ v


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, rope=None, **kwargs):
        super().__init__()
        if d_model % num_heads:
            raise ValueError('d_model must be divisible by num_heads')
        self.num_heads, self.d_head, self.rope = num_heads, d_model//num_heads, rope
        self.q_proj = Linear(d_model,d_model,**kwargs)
        self.k_proj = Linear(d_model,d_model,**kwargs)
        self.v_proj = Linear(d_model,d_model,**kwargs)
        self.output_proj = Linear(d_model,d_model,**kwargs)
    def forward(self, x, token_positions=None):
        def split(proj):
            return proj(x).unflatten(-1,(self.num_heads,self.d_head)).transpose(-3,-2)
        q,k,v = split(self.q_proj),split(self.k_proj),split(self.v_proj)
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(x.shape[-2], device=x.device)
            q,k = self.rope(q,token_positions), self.rope(k,token_positions)
        mask = torch.ones(x.shape[-2],x.shape[-2],device=x.device,dtype=torch.bool).tril()
        out = scaled_dot_product_attention(q,k,v,mask)
        return self.output_proj(out.transpose(-3,-2).flatten(-2))


class TransformerBlock(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,max_seq_len,theta=10000., norm='pre',rope=True,ffn='swiglu',**kwargs):
        super().__init__()
        self.norm = norm
        self.ln1 = RMSNorm(d_model,**kwargs) if norm != 'none' else nn.Identity()
        self.ln2 = RMSNorm(d_model,**kwargs) if norm != 'none' else nn.Identity()
        r = RotaryPositionalEmbedding(theta,d_model//num_heads,max_seq_len,kwargs.get('device')) if rope else None
        self.attn = MultiHeadSelfAttention(d_model,num_heads,r,**kwargs)
        self.ffn = SwiGLU(d_model,d_ff,**kwargs) if ffn == 'swiglu' else SiLUFFN(d_model,**kwargs)
    def forward(self,x):
        if self.norm == 'post':
            x = self.ln1(x + self.attn(x))
            return self.ln2(x + self.ffn(x))
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class TransformerLM(nn.Module):
    def __init__(self,vocab_size,context_length,d_model,num_layers,num_heads,d_ff,rope_theta=10000.,norm='pre',rope=True,ffn='swiglu',**kwargs):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size,d_model,**kwargs)
        self.layers = nn.ModuleList([TransformerBlock(d_model,num_heads,d_ff,context_length,rope_theta,norm,rope,ffn,**kwargs) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model,**kwargs) if norm != 'none' else nn.Identity()
        self.lm_head = Linear(d_model,vocab_size,**kwargs)
    def forward(self,ids):
        if ids.shape[-1] > self.context_length:
            raise ValueError('Input exceeds configured context length')
        x = self.token_embeddings(ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.ln_final(x))
