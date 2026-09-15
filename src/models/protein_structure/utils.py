import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange

# এই ফাইলটিতে সাধারণ হেলপার ফাংশন ও ছোট ক্লাসগুলো রাখা হয়েছে
# উদ্দেশ্য: বোঝানো হবে কেন এগুলো প্রয়োজন ও কোথায় ব্যবহার হবে


def exists(val):
    # ভ্যালুটি None নয় কি না চেক করে
    return val is not None


def default(val, d):
    # যদি val থাকে তা রিটার্ন করবে, না থাকলে d রিটার্ন করবে
    return val if exists(val) else d


def cast_tuple(val, length=1):
    # একটি ভ্যারিয়েবল টিউপলে কাস্ট করে নির্দিষ্ট দৈর্ঘ্যে
    return val if isinstance(val, tuple) else (val,) * length


class Mish(nn.Module):
    # Mish অ্যাক্টিভেশন: মোল্ডারিং করা 'softplus' ব্যবহার করে নিরবশিষ্টতা বাড়ায়
    def forward(self, x):
        # Mish ফর্মুলা: x * tanh(softplus(x))
        return x * torch.tanh(F.softplus(x))


class RotaryEmbedding(nn.Module):
    # Head-dimension-aware Rotary positional embedding
    # উদ্দেশ্য: Q/K-তে RoPE প্রয়োগ করার জন্য ব্যবহারযোগ্য ফ্রিকোয়েন্সি টেনসর তৈরি করা
    def __init__(self, dim_head: int):
        super().__init__()
        if dim_head % 2 != 0:
            raise ValueError("dim_head must be an even integer for rotary embedding.")
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim_head // 2).float() / dim_head))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int, device):
        # seq_len অনুযায়ী (seq_len, dim_head) আকারে এমবেডিং রিটার্ন করে
        seq = torch.arange(seq_len, device=device).type(self.inv_freq.dtype)
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb


def rotate_half(x):
    # শেষ ডাইমেনশনকে জোড়া করে রোটেট করে
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def apply_rotary_pos_emb(pos, t):
    # pos: (seq_len, dim_head)
    # t: (..., seq_len, dim_head)
    # RoPE প্রয়োগ করে q/k-কে পরিবর্তন করে রিটার্ন করে
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())