import torch
import torch.nn as nn
from einops import rearrange
from .attention import Attention, CrossAttention
from .utils import exists

# Pairformer ব্লক: single ও pair প্রাস্টকে হ্যান্ডল করে এবং ligand-এ ক্রস-সংযোগ দেয়

class FeedForward(nn.Module):
    # ছোট ফিডফরওয়ার্ড ব্লক ব্যবহার করা হয়েছে এখানে (সহজত্বের জন্য)
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class PairformerBlock(nn.Module):
    def __init__(self, dim, pair_dim, heads=8, dim_head=64, dropout=0.0, ligand_context_dim=512):
        super().__init__()
        self.single_pre_norm = nn.LayerNorm(dim)
        self.pair_pre_norm = nn.LayerNorm(pair_dim)

        self.single_attention = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.single_ff = FeedForward(dim, dropout=dropout)

        self.pair_attention = Attention(pair_dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.pair_ff = FeedForward(pair_dim, dropout=dropout)

        self.ligand_cross_attention = CrossAttention(
            pair_dim, context_dim=ligand_context_dim, heads=heads, dim_head=dim_head, dropout=dropout
        )

        self.single_to_pair = nn.Linear(dim, pair_dim)
        self.pair_to_single = nn.Linear(pair_dim, dim)

    def forward(self, singles, pairs, ligand_emb=None, mask=None, rotary_pos=None):
        # singles: (b, i, dim)
        # pairs: (b, i, j, pair_dim)

        # 1) single pathway: pre-norm -> attention -> ff
        singles_norm = self.single_pre_norm(singles)
        singles_attn = self.single_attention(singles_norm, mask=mask, rotary_pos=rotary_pos)
        singles = singles + singles_attn
        singles = singles + self.single_ff(singles)

        # 2) pair pathway: আমরা pair টেনসরকে flatten করে attention চালাই (memory-সন্তুষ্টি জন্য)
        pairs_norm = self.pair_pre_norm(pairs)
        b, i, j, d = pairs_norm.shape
        pairs_flat = rearrange(pairs_norm, "b i j d -> b (i j) d")
        pairs_attn_flat = self.pair_attention(pairs_flat, mask=None, rotary_pos=rotary_pos)
        pairs_attn = rearrange(pairs_attn_flat, "b (i j) d -> b i j d", i=i, j=j)
        pairs = pairs + pairs_attn

        # 3) যদি ligand এমবেডিং থাকে, pair থেকে ligand কনটেক্সট-এ ক্রস-অ্যাটেনশন প্রয়োগ করো
        if exists(ligand_emb):
            pairs_flat = rearrange(pairs_norm, "b i j d -> b (i j) d")
            pairs_cross = self.ligand_cross_attention(pairs_flat, context=ligand_emb, mask=None, context_mask=None, rotary_pos=rotary_pos)
            pairs_cross = rearrange(pairs_cross, "b (i j) d -> b i j d", i=i, j=j)
            pairs = pairs + pairs_cross

        pairs = pairs + self.pair_ff(pairs)

        # 4) single <-> pair কমিউনিকেশন
        single_contrib = self.single_to_pair(singles)
        pair_contrib = self.pair_to_single(pairs.mean(dim=2))

        pairs = pairs + single_contrib.unsqueeze(1) + single_contrib.unsqueeze(2)
        singles = singles + pair_contrib

        return singles, pairs