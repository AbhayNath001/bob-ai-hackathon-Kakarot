import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat
import math
from typing import Optional, Tuple, Union

# হেল্পার ফাংশনসমূহ
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else (val,) * length

class Mish(nn.Module):
    def forward(self, x):
        # Mish সক্রিয়করণ
        return x * torch.tanh(F.softplus(x))

# রোটারি positional embedding (head-dimension aware)
class RotaryEmbedding(nn.Module):
    def __init__(self, dim_head: int):
        # dim_head অবশ্যই জোড় সংখ্যা হওয়া উচিত
        super().__init__()
        if dim_head % 2 != 0:
            raise ValueError("dim_head must be an even integer for rotary embedding.")
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim_head // 2).float() / dim_head))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int, device):
        # একটি (seq_len, dim_head) টেনসর রিটার্ন করে যেখানে
        # রোটারি ফ্রিকোয়েন্সিগুলো জোড়ভাগে সাজানো আছে
        seq = torch.arange(seq_len, device=device).type(self.inv_freq.dtype)
        freqs = einsum("i , j -> i j", seq, self.inv_freq)  # (seq_len, dim_head//2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim_head)
        return emb  # ব্যবহার করতে হবে যেমন: rotary_pos (n, dim_head)

def rotate_half(x):
    # শেষ ডাইমেনশনকে জোড় জোড় করে ঘোরানো: (..., d*2) -> (..., d*2)
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")

def apply_rotary_pos_emb(pos, t):
    # pos: (seq_len, dim_head), t: (..., seq_len, dim_head) বা (..., seq_len, dim_head)
    # RoPE প্রয়োগ করে q ও k উভয়ের সাথে
    # pos.cos() ও pos.sin() broadcast-able হওয়া উচিত t-র সাথে
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())

# মূল উপাদানসমূহ
class Residual(nn.Module):
    def __init__(self, fn):
        # রেসিডুয়াল র‌্যাপার
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        # লেয়ারনর্মের আগে ফাংশন (PreNorm)
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        # MLP ব্লক
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            Mish(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        # সাধারণ মাল্টি-হেড self-attention, এখন rotary সমর্থন করে
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, rotary_pos: Optional[torch.Tensor] = None):
        # x: (b, n, dim)
        # rotary_pos: None বা (n, dim_head) - আগেই head-dimension অনুযায়ী তৈরি করা
        b, n, _ = x.shape
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        # প্রতিটি q/k/v কোষকে (b, h, n, d_head) এ রূপান্তর
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        # রোটারি যদি আছে, q এবং k-এ প্রয়োগ করো
        if exists(rotary_pos):
            # rotary_pos: (n, dim_head) -> expand to (1, 1, n, dim_head) স্বয়ংক্রিয়ভাবে
            q = apply_rotary_pos_emb(rotary_pos, q)
            k = apply_rotary_pos_emb(rotary_pos, k)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        if exists(mask):
            # mask: (b, j) বা (b, n) — আমরা mask-কে key-dimension (j) জন্য ব্যবহার করছি
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
    def __init__(self, dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        # ক্রস-অ্যাটেনশন যেখানে কনটেক্সটে ভিন্ন ডায়ম রয়েছে
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, dim)

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, mask=None, context_mask=None, rotary_pos: Optional[torch.Tensor] = None):
        # x: (b, n_x, dim)
        # context: (b, n_ctx, context_dim)
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        # রোটারি প্রয়োগ করা হলে q ও k-তে প্রয়োগ করো
        if exists(rotary_pos):
            q = apply_rotary_pos_emb(rotary_pos, q)
            k = apply_rotary_pos_emb(rotary_pos, k)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        if exists(mask) or exists(context_mask):
            mask = default(mask, torch.ones(x.shape[:2], device=x.device).bool())
            context_mask = default(context_mask, torch.ones(context.shape[:2], device=context.device).bool())
            mask_combined = rearrange(mask, "b i -> b 1 i 1") * rearrange(context_mask, "b j -> b 1 1 j")
            mask_value = -torch.finfo(dots.dtype).max
            dots = dots.masked_fill(~mask_combined, mask_value)

        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out

# Pairformer ব্লক — এখন pair-to-ligand cross-attention সঠিকভাবে flatten/reshape করে
class PairformerBlock(nn.Module):
    def __init__(self, dim, pair_dim, heads=8, dim_head=64, dropout=0.0, ligand_context_dim=512):
        super().__init__()
        self.single_pre_norm = nn.LayerNorm(dim)
        self.pair_pre_norm = nn.LayerNorm(pair_dim)

        self.single_attention = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.single_ff = FeedForward(dim, dropout=dropout)

        self.pair_attention = Attention(pair_dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.pair_ff = FeedForward(pair_dim, dropout=dropout)

        # pair -> ligand ক্রস-অ্যাটেনশন (কনটেক্সট ডাইম প্যারাম)
        self.ligand_cross_attention = CrossAttention(
            pair_dim, context_dim=ligand_context_dim, heads=heads, dim_head=dim_head, dropout=dropout
        )

        # single <-> pair রূপান্তর
        self.single_to_pair = nn.Linear(dim, pair_dim)
        self.pair_to_single = nn.Linear(pair_dim, dim)

    def forward(self, singles, pairs, ligand_emb=None, mask=None, rotary_pos: Optional[torch.Tensor] = None):
        # singles: (b, i, dim)
        # pairs: (b, i, j, pair_dim)
        # ligand_emb: None বা (b, L, context_dim)
        # mask: (b, n) or None
        singles_norm = self.single_pre_norm(singles)
        singles_attn = self.single_attention(singles_norm, mask=mask, rotary_pos=rotary_pos)
        singles = singles + singles_attn
        singles = singles + self.single_ff(singles)

        pairs_norm = self.pair_pre_norm(pairs)
        # pair_attention কাজ করে প্রতিটি (i, j) উপর পর্যায়ক্রমে (batch-wise)
        b, i, j, d = pairs_norm.shape
        pairs_flat = rearrange(pairs_norm, "b i j d -> b (i j) d")  # (b, i*j, d)
        pairs_attn_flat = self.pair_attention(pairs_flat, mask=None, rotary_pos=rotary_pos)
        pairs_attn = rearrange(pairs_attn_flat, "b (i j) d -> b i j d", i=i, j=j)
        pairs = pairs + pairs_attn

        # যদি ligand এমবেডিং থাকে, তাহলে pair-->ligand ক্রস-অ্যাটেনশন প্রয়োগ করো
        if exists(ligand_emb):
            # ligand_emb প্রত্যাশিত আকার (b, L, context_dim)
            # CrossAttention প্রত্যাশা করে x: (b, n, pair_dim) এবং context: (b, L, context_dim)
            pairs_flat = rearrange(pairs_norm, "b i j d -> b (i j) d")
            # ligand_emb সরাসরি দিয়ে ক্রস-অ্যাটেনশন করো
            pairs_cross = self.ligand_cross_attention(
                pairs_flat, context=ligand_emb, mask=None, context_mask=None, rotary_pos=rotary_pos
            )
            pairs_cross = rearrange(pairs_cross, "b (i j) d -> b i j d", i=i, j=j)
            pairs = pairs + pairs_cross

        pairs = pairs + self.pair_ff(pairs)

        # single <-> pair তথ্য শেয়ারিং
        single_contrib = self.single_to_pair(singles)  # (b, i, pair_dim)
        pair_contrib = self.pair_to_single(pairs.mean(dim=2))  # (b, i, dim) - আমরা j-তে গড় নিয়ে নিলাম

        # broadcasting মেকানিজম ব্যবহার করে যোগ করা
        pairs = pairs + single_contrib.unsqueeze(1) + single_contrib.unsqueeze(2)
        singles = singles + pair_contrib

        return singles, pairs

# SE(3)-সমতুল্য ব্লক (সরল, কিন্তু rotary + coordinate updates সহ)
class SE3EquivariantBlock(nn.Module):
    def __init__(self, hidden_dim=256, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        # hidden_dim: ফিচার স্পেস
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dim_head = dim_head

        # কোঅর্ডিনেট আপডেট প্রেডিক্টর ( সরল লিনিয়ার )
        self.coord_proj = nn.Linear(hidden_dim, 3)
        # attention ব্লক যা hidden_dim অংশকে প্রক্রিয়া করবে
        self.attention = Attention(hidden_dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ff = FeedForward(hidden_dim, dropout=dropout)

        # রোটারি এমবেডিং (head-dimension ব্যবহার)
        self.rotary_emb = RotaryEmbedding(dim_head)

    def forward(self, coords, feats, mask=None):
        # coords: (b, n, 3), feats: (b, n, hidden_dim)
        b, n, _ = coords.shape

        # rotary positional embeddings তৈরি করা (নির্ভরশীল: seq_len n এবং ডিভাইস)
        rotary_pos = self.rotary_emb(n, device=coords.device)  # (n, dim_head)

        # সেল্ফ-অ্যাটেনশন — rotary_pos পাস করাও হয়
        feats = feats + self.attention(feats, mask=mask, rotary_pos=rotary_pos)
        feats = feats + self.ff(feats)

        # কোঅর্ডিনেট আপডেট অনুমান
        coord_updates = self.coord_proj(feats)

        # কোঅর্ডিনেট আপডেট প্রয়োগ (বসিক ব্যাসিক পদক্ষেপ)
        new_coords = coords + coord_updates

        return new_coords, feats

# প্রধান EnhancedAlphaFold মডিউল - আপডেটসহ
class EnhancedAlphaFold(nn.Module):
    def __init__(
        self,
        num_amino_acids=20,
        dim=512,
        pair_dim=256,
        depth=6,  # ডেমো উদ্দেশ্যে depth কমানো (পাঠযোগ্যতার জন্য)
        heads=8,
        dim_head=64,
        diffusion_steps=6,
        equivariant_hidden_dim=256,
        dropout=0.0,
        ligand_context_dim=512,
    ):
        super().__init__()
        self.dim = dim
        self.pair_dim = pair_dim
        self.diffusion_steps = diffusion_steps
        self.heads = heads
        self.dim_head = dim_head
        self.ligand_context_dim = ligand_context_dim

        # আমিনো অ্যাসিড এমবেডিং
        self.aa_embedding = nn.Embedding(num_amino_acids, dim)

        # MSA প্রোফাইল এমবেডিং (প্রতি রেসিডু তে প্রয়োগযোগ্য)
        self.msa_embedding = nn.Linear(num_amino_acids, dim)

        # pair representation initialization
        self.pair_embedding = nn.Linear(dim * 2, pair_dim)

        # ligand embedding (SMILES / features -> context vectors)
        self.ligand_embedding = nn.Sequential(
            nn.Linear(128, 256),  # ইনপুট ligand ফিচার 128
            Mish(),
            nn.Linear(256, ligand_context_dim),
            Mish()
        )

        # ligand -> pocket projection (ডিটারমিনিস্টিক পকেট কেন্দ্র তৈরির জন্য)
        self.ligand_to_pocket = nn.Sequential(
            nn.Linear(ligand_context_dim, 128),
            Mish(),
            nn.Linear(128, 3)
        )

        # Pairformer ব্লকসমূহ
        self.pairformer_blocks = nn.ModuleList([
            PairformerBlock(
                dim=dim,
                pair_dim=pair_dim,
                heads=heads,
                dim_head=dim_head,
                dropout=dropout,
                ligand_context_dim=ligand_context_dim
            ) for _ in range(depth)
        ])

        # Structure module - SE3 equivariant blocks
        self.structure_module = nn.ModuleList([
            SE3EquivariantBlock(
                hidden_dim=equivariant_hidden_dim,
                heads=heads,
                dim_head=dim_head,
                dropout=dropout
            ) for _ in range(diffusion_steps)
        ])

        # প্রাথমিক কোঅর্ডিনেট প্রেডিকশন
        self.init_coords = nn.Linear(dim, 3)

        # single + pair -> structure_feats প্রজেকশন
        self.to_structure_feats = nn.Sequential(
            nn.Linear(dim + pair_dim, equivariant_hidden_dim),
            Mish(),
            nn.Linear(equivariant_hidden_dim, equivariant_hidden_dim)
        )

        # নিয়মীকারক লস (MSE ব্যবহার)
        self.bond_length_penalty = nn.MSELoss()
        self.angle_penalty = nn.MSELoss()

    def forward(
        self,
        aa_seq,
        msa_profile=None,
        ligand_feats=None,
        mask=None,
        return_trajectory=False
    ):
        # aa_seq: (b, n) - ইন্টেজার টোকেনস
        # msa_profile: (b, n, num_amino_acids) অথবা None
        # ligand_feats: (b, 128) বা (b, L, 128) অথবা None

        # আমিনো অ্যাসিড এমবেড করা
        aa_emb = self.aa_embedding(aa_seq)  # (b, n, dim)

        # MSA প্রোফাইল থাকলে যোগ করা
        if exists(msa_profile):
            msa_emb = self.msa_embedding(msa_profile)  # (b, n, dim)
            aa_emb = aa_emb + msa_emb

        # ligand এমবেডিং: ensure shape (b, L, ligand_context_dim) অথবা None
        ligand_emb = None
        if exists(ligand_feats):
            if ligand_feats.dim() == 2:
                # (b, feat_dim) -> (b, 1, feat_dim)
                ligand_feats_proc = ligand_feats.unsqueeze(1)
            else:
                ligand_feats_proc = ligand_feats
            # এমবেড করো (প্রতিটি ligand token-এ)
            b, L, _ = ligand_feats_proc.shape
            ligand_emb = self.ligand_embedding(ligand_feats_proc.view(b * L, -1))
            ligand_emb = ligand_emb.view(b, L, -1)  # (b, L, ligand_context_dim)

        # pair প্রতিনিধিত্ব আরম্ভ করা
        i_emb = repeat(aa_emb, "b i d -> b i j d", j=aa_emb.shape[1])
        j_emb = repeat(aa_emb, "b j d -> b i j d", i=aa_emb.shape[1])
        pair_emb = torch.cat([i_emb, j_emb], dim=-1)
        pairs = self.pair_embedding(pair_emb)  # (b, i, j, pair_dim)

        # Pairformer ব্লকগুলো চালাও
        singles = aa_emb
        for block in self.pairformer_blocks:
            singles, pairs = block(singles, pairs, ligand_emb=ligand_emb, mask=mask,
                                   rotary_pos=None)  # মূলভাবে rotary_pos=None; structure module এ Rotary ব্যবহৃত হবে

        # প্রাথমিক কোঅর্ডিনেট
        init_coords = self.init_coords(singles)  # (b, n, 3)

        # structure_feats তৈরী করা (single + pair resumen)
        pair_i = pairs.mean(dim=2)  # (b, i, pair_dim)
        structure_feats = torch.cat([singles, pair_i], dim=-1)  # (b, i, dim+pair_dim)
        structure_feats = self.to_structure_feats(structure_feats)  # (b, i, equivariant_hidden_dim)

        # ডিফিউশন-সদৃশ ধাপ (প্রতিটি ধাপে equivariant ব্লক)
        coords = init_coords
        trajectory = [coords] if return_trajectory else None

        for step in range(self.diffusion_steps):
            coords, structure_feats = self.structure_module[step](
                coords, structure_feats, mask=mask
            )
            if return_trajectory:
                trajectory.append(coords)

        # ট্রেইনিং-সময়ে নিয়মীকারক লস প্রয়োগ
        reg_loss = torch.tensor(0.0, device=coords.device)
        if self.training:
            reg_loss = self.apply_regularizers(coords, aa_seq, ligand_emb)

        if return_trajectory:
            return coords, trajectory, reg_loss
        return coords, reg_loss

    def apply_regularizers(self, coords, aa_seq, ligand_emb=None):
        # বন্ড দৈর্ঘ্য ও কোণ শাস্তি গণনা
        bond_loss = self.calc_bond_length_penalty(coords)
        angle_loss = self.calc_bond_angle_penalty(coords)

        pocket_loss = torch.tensor(0.0, device=coords.device)
        if exists(ligand_emb):
            pocket_loss = self.calc_pocket_contact_term(coords, ligand_emb)

        return bond_loss + angle_loss + pocket_loss

    def calc_bond_length_penalty(self, coords):
        # পরপর Cα-Cα মত দূরত্ব অনুমান করে MSE লস দেয়
        diffs = coords[:, 1:] - coords[:, :-1]
        dists = torch.norm(diffs, dim=-1)
        expected_dist = 3.8
        return self.bond_length_penalty(dists, torch.ones_like(dists) * expected_dist)

    def calc_bond_angle_penalty(self, coords):
        # তিনটি ধারাবাহিক পজিশনের ওপর ভিত্তি করে কোণ গণনা করে MSE দেয়
        vec1 = coords[:, 1:-1] - coords[:, :-2]
        vec2 = coords[:, 2:] - coords[:, 1:-1]
        cos_angles = torch.sum(vec1 * vec2, dim=-1) / (
            torch.norm(vec1, dim=-1) * torch.norm(vec2, dim=-1) + 1e-8
        )
        angles = torch.acos(torch.clamp(cos_angles, -1.0, 1.0))
        expected_angle = 2.094  # ~120 degrees in radians
        return self.angle_penalty(angles, torch.ones_like(angles) * expected_angle)

    def calc_pocket_contact_term(self, coords, ligand_emb):
        # ligand_emb: (b, L, ligand_context_dim)
        # প্রোটিনের কেন্দ্র ও ligand projection থেকে পকেট কেন্দ্র বের করা
        protein_center = coords.mean(dim=1)  # (b, 3)
        ligand_center = ligand_emb.mean(dim=1)  # (b, ligand_context_dim)
        pocket_offset = self.ligand_to_pocket(ligand_center)  # (b, 3)
        pocket_center = protein_center + pocket_offset  # (b, 3)

        # প্রতিটি রেসিডু থেকে পকেট কেন্দ্র পর্যন্ত দূরত্ব
        dists = torch.norm(coords - pocket_center.unsqueeze(1), dim=-1)
        threshold = 10.0
        penalty = torch.where(
            dists > threshold,
            (dists - threshold) ** 2,
            torch.zeros_like(dists)
        )
        return penalty.mean()

# উদাহরণ ব্যবহার ও ট্রেনিং সেটআপ
def example_usage():
    # হাইপারপ্যারামিটার (ডেমো)
    batch_size = 32
    seq_len = 64
    num_amino_acids = 20

    # মডেল তৈরি
    model = EnhancedAlphaFold(
        num_amino_acids=num_amino_acids,
        dim=512,
        pair_dim=256,
        depth=6,
        heads=8,
        dim_head=64,
        diffusion_steps=6,
        equivariant_hidden_dim=256,
        dropout=0.1,
        ligand_context_dim=512,
    )

    # উদাহরণ ইনপুট
    aa_seq = torch.randint(0, num_amino_acids, (batch_size, seq_len))
    msa_profile = torch.randn(batch_size, seq_len, num_amino_acids)
    ligand_feats = torch.randn(batch_size, 128)  # (b, feat_dim)

    # ফরওয়ার্ড (ট্রাজেক্টরি)
    coords, trajectory, reg_loss = model(
        aa_seq,
        msa_profile=msa_profile,
        ligand_feats=ligand_feats,
        return_trajectory=True
    )

    print(f"Input sequence shape: {aa_seq.shape}")
    print(f"Output coordinates shape: {coords.shape}")
    print(f"Trajectory length: {len(trajectory)}")
    print(f"Regularization loss: {reg_loss.item()}")

    # ট্রেনার সেটআপ (ডেমো)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-2
    )
    loss_fn = nn.MSELoss()

    model.train()
    optimizer.zero_grad()

    coords, reg_loss = model(aa_seq, msa_profile=msa_profile, ligand_feats=ligand_feats)

    target_coords = torch.randn_like(coords)  # উদাহরণ টার্গেট

    main_loss = loss_fn(coords, target_coords)
    total_loss = main_loss + 0.1 * reg_loss

    total_loss.backward()
    optimizer.step()

    print(f"Main loss: {main_loss.item()}, Total loss: {total_loss.item()}")

if __name__ == "__main__":

    example_usage()
