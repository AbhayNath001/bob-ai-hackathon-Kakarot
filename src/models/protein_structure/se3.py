import torch
import torch.nn as nn
from .attention import Attention
from .utils import RotaryEmbedding

# SE(3)-সমতুল্য ব্লক — সরল, কিন্তু স্ট্রাকচারাল ফিচার ও কোঅর্ডিনেট আপডেট করে
# লক্ষ্য: একটি সাধারন, পরীক্ষণযোগ্য ব্লক দেওয়া যা পরে আরও ফিজিক্যালি-রিগোরাস
# (vector/tensor features, invariant ops) দিয়ে প্রতিস্থাপন করা যাবে।
# মন্তব্যগুলো প্রতিটি ধাপ কেন করা হচ্ছে তা বাংলায় ব্যাখ্যা করে।

class SE3EquivariantBlock(nn.Module):
    def __init__(self, hidden_dim=256, heads=8, dim_head=64, dropout=0.0):
        """
        Args:
            hidden_dim: ফিচার স্পেস (প্রতিটি রেসিডুতে থাকা ফিচার ডাইমেনশন)
            heads: multi-head_attention-এ হেড সংখ্যা
            dim_head: প্রতি হেডের ডাইমেনশন (dim_head * heads == inner_dim)
            dropout: attention/dropout হাইপারপারামিটার
        নোট: dim_head অবশ্যই যতোটা ব্যবহার করবেন RotaryEmbedding-এ সেটি যেন জোড় সংখ্যা হয়।
        """
        super().__init__()

        # ক্লাস অ্যাট্রিবিউটস সংরক্ষণ
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dim_head = dim_head

        # কোঅর্ডিনেট-আপডেট প্রেডিক্টর: hidden features থেকে সরাসরি (dx,dy,dz) প্রেডিক্ট করি
        # সরলভাবে লিনিয়ার ম্যাপ - পরে এটা আরও জ্যামিতিকভাবে সংবেদনশীল মডিউলে বদলানো যেতে পারে
        self.coord_proj = nn.Linear(hidden_dim, 3)

        # ফিচার-অ্যাটেনশন: hidden_dim স্পেস-এ attention চালানো হবে
        # এখানে Attention expects dim == hidden_dim
        self.attention = Attention(hidden_dim, heads=heads, dim_head=dim_head, dropout=dropout)

        # ছোট feed-forward ব্লক — attention এর পরে ফিচার রিফাইন করার জন্য
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # রোটারি positional embedding (head-dimension aware)
        # উদ্দেশ্য: attention-এ RoPE প্রয়োগ করে পজিশনাল-সংকেত যোগ করা
        self.rotary_emb = RotaryEmbedding(dim_head)

        # ছোট স্কেল ফ্যাক্টর যাতে কোঅর্ডিনেট আপডেট খুব বড় না হয় (প্রায়ই প্রয়োজন)
        # এটি ট্রেনেবলও করা যেতে পারে; এখানে একটি স্থির ছোট স্কেল ব্যবহার করছি
        self.coord_update_scale = 0.1

    def forward(self, coords: torch.Tensor, feats: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            coords: (b, n, 3) — প্রতিটি বেচে n রেসিডুর (x,y,z)
            feats:  (b, n, hidden_dim) — প্রতিটি রেসিডুতে ফিচার ভেক্টর
            mask:   (b, n) boolean mask (optional) — True যেখানে টোকেন/রেসিডু উপস্থিত
        Returns:
            new_coords: (b, n, 3) — আপডেটেড কোঅর্ডিনেট
            new_feats:  (b, n, hidden_dim) — আপডেটেড ফিচার ভেক্টর
        নোট: এই ব্লকটি পূর্ণ SE(3)-ইক্যুইভেরিয়ান নয়; বরং একটি পথ-প্রমাণ ধারণা যা
        পরবর্তী ধাপে আরও জটিল ভেক্টর-টেনসর অপারেশন দ্বারা প্রতিস্থাপিত হতে পারে।
        """
        b, n, _ = coords.shape

        # 1) রোটারি positional embedding তৈরি করা (n, dim_head)
        #    RotaryEmbedding seq length ও ডিভাইস অনুযায়ী নির্ধারণ করে
        rotary_pos = self.rotary_emb(n, device=coords.device)

        # 2) ফিচার-অ্যাটেনশন প্রয়োগ (rotary_pos পাস করা হয়)
        #    Attention মডিউল q/k-তে RoPE প্রয়োগ করে এবং ফিচার আপডেট রিটার্ন করে
        feats_attn = self.attention(feats, mask=mask, rotary_pos=rotary_pos)
        feats = feats + feats_attn  # residual যোগ

        # 3) ফিড-ফরওয়ার্ড রিপ-ফাইন
        feats = feats + self.ff(feats)

        # 4) কোঅর্ডিনেট আপডেট অনুমান
        #    সহজ লিনিয়ার প্রেডিকশন: প্রতিটি রেসিডুর জন্য (dx,dy,dz)
        coord_updates = self.coord_proj(feats)  # (b, n, 3)

        # 5) আপডেট স্কেল করে যোগ করা — সরল এবং স্থিতিশীল রাখার জন্য
        #    ট্রেনিং শুরুতে বড় ধাক্কা এড়াতে ছোট স্কেল ব্যবহার করা ভালো
        new_coords = coords + coord_updates * self.coord_update_scale

        return new_coords, feats