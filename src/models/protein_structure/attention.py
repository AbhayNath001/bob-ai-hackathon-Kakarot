import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange
from typing import Optional
from .utils import exists, apply_rotary_pos_emb


class Attention(nn.Module):
    # Multi-head self-attention যা rotary positional embedding সাপোর্ট করে
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        # single linear projects to QKV ট্রিপল
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, rotary_pos: Optional[torch.Tensor] = None):
        b, n, _ = x.shape
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        # যদি rotary_pos প্রদান করা হয়, q ও k-তে প্রয়োগ করা হয়
        if exists(rotary_pos):
            q = apply_rotary_pos_emb(rotary_pos, q)
            k = apply_rotary_pos_emb(rotary_pos, k)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        if exists(mask):
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, "b j -> b 1 1 j")
            dots = dots.masked_fill(~mask, mask_value)

        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class CrossAttention(nn.Module):
    # Context-aware attention (query comes from x, key/value from context)
    def __init__(self, dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else dim

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, mask=None, context_mask=None, rotary_pos: Optional[torch.Tensor] = None):
        h = self.heads

        q = self.to_q(x)
        context = context if context is not None else x
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        if exists(rotary_pos):
            q = apply_rotary_pos_emb(rotary_pos, q)
            k = apply_rotary_pos_emb(rotary_pos, k)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        if exists(mask) or exists(context_mask):
            mask = mask if mask is not None else torch.ones(x.shape[:2], device=x.device).bool()
            context_mask = context_mask if context_mask is not None else torch.ones(context.shape[:2], device=context.device).bool()
            mask_combined = rearrange(mask, "b i -> b 1 i 1") * rearrange(context_mask, "b j -> b 1 1 j")
            mask_value = -torch.finfo(dots.dtype).max
            dots = dots.masked_fill(~mask_combined, mask_value)

        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out