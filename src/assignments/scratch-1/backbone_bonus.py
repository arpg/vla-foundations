"""
Scratch-1: The Transformer Backbone (AGGRESSIVE ABLATION VERSION)
CSCI 7000 - VLA Foundations

What this version adds (beyond the base backbone):
1) Attention capture (return_attn=True) through Attention -> Block -> Model
2) Attention visualization:
   - Single head heatmaps (causal / no-mask)
   - FULL GRID heatmaps for ALL heads, per layer (causal / no-mask)
3) Quantitative ablation metric:
   - "future attention ratio" = attention mass in upper triangle / total mass
     * should be ~0 for causal, and >0 for no-mask
4) Training loop requirements:
   - teacher forcing
   - gradient clipping (1.0)
   - loss + perplexity logging
   - checkpoints every 1000 steps
5) KV-caching for efficient inference (prefill + decode)
6) RoPE vs Sinusoidal positional encoding ablation (train + compare plot)
7) Inference speed comparison with and without KV-caching (plot)

Run:
  python backbone_bonus.py

Outputs:
  checkpoints/best_model.pt
  checkpoints/*_model_step_*.pt
  images/training_loss.png
  images/causal_vs_nomask_loss.png
  images/attention_causal_layer0_head0.png
  images/attention_nomask_layer0_head0.png
  images/attn_grid_causal_layer{L}.png   (all layers)
  images/attn_grid_no_mask_layer{L}.png  (all layers)
  images/rope_vs_sinusoidal_loss.png
  images/inference_speed_kvcache.png
  images/attention_sinusoidal_layer0_head0.png
  images/attn_grid_sinusoidal_layer{L}.png
"""

import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from generate_data import create_dataloaders


# ----------------------------- Norm -----------------------------
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    RMS(a) = sqrt(mean(a^2) + eps)
    y = (a / RMS(a)) * g
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms_inv = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms_inv * self.scale.view(1, 1, -1)


# ----------------------------- Sinusoidal PE -----------------------------
class SinusoidalPositionalEmbedding(nn.Module):
    """
    Classic absolute sinusoidal positional embeddings (Vaswani et al. 2017):
      PE[pos, 2i]   = sin(pos / 10000^(2i/dim))
      PE[pos, 2i+1] = cos(pos / 10000^(2i/dim))

    Returned shape: (1, T, D) so it broadcasts over batch.
    """
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        pe = self._build(max_seq_len, dim)  # (T, D)
        self.register_buffer("pe", pe, persistent=False)

    @staticmethod
    def _build(T: int, D: int) -> torch.Tensor:
        pos = torch.arange(T).float().unsqueeze(1)  # (T, 1)
        i = torch.arange(D).float().unsqueeze(0)    # (1, D)
        div = torch.pow(10000.0, (2 * (i // 2)) / D)
        angles = pos / div
        pe = torch.zeros(T, D)
        pe[:, 0::2] = torch.sin(angles[:, 0::2])
        pe[:, 1::2] = torch.cos(angles[:, 1::2])
        return pe

    def forward(self, T: int, start_pos: int = 0) -> torch.Tensor:
        if start_pos + T > self.pe.size(0):
            raise ValueError(f"Sinusoidal PE max_len={self.pe.size(0)} too small for start_pos+T={start_pos+T}")
        return self.pe[start_pos:start_pos + T].unsqueeze(0)  # (1, T, D)


# ----------------------------- RoPE -----------------------------
class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) applied to (B, H, T, head_dim)

    IMPORTANT for KV-caching:
    - we support start_pos offset, so we can apply RoPE correctly for incremental decode.
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)  # (T,)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (T, dim//2)
        emb = torch.cat([freqs, freqs], dim=-1)            # (T, dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, start_pos: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        # q,k: (B, H, T, head_dim)
        T = q.shape[2]
        end_pos = start_pos + T
        if end_pos > self.cos_cached.shape[0]:
            # Expand cache if needed (rare for this assignment, but safe)
            self._build_cache(max(end_pos, self.cos_cached.shape[0] * 2))

        cos = self.cos_cached[start_pos:end_pos, ...]  # (T, D)
        sin = self.sin_cached[start_pos:end_pos, ...]  # (T, D)

        # Broadcast (T, D) over (B, H, T, D)
        q_rot = (q * cos) + (self.rotate_half(q) * sin)
        k_rot = (k * cos) + (self.rotate_half(k) * sin)
        return q_rot, k_rot


# ----------------------------- Attention -----------------------------
KVCache = Tuple[torch.Tensor, torch.Tensor]  # (k, v), each (B, H, T, hd)

class CausalSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention with:
      - optional causal masking
      - optional RoPE (for RoPE mode)
      - KV-caching support for inference
    """
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1, rope_max_seq_len: int = 2048):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len=rope_max_seq_len)

    def forward(
        self,
        x: torch.Tensor,                      # (B, T, D)
        mask: Optional[torch.Tensor] = None,  # (T_total, T_total) or None
        return_attn: bool = False,
        kv_cache: Optional[KVCache] = None,
        start_pos: int = 0,
        use_rope: bool = True,
        return_kv: bool = False,
    ):
        B, T, _ = x.shape

        qkv = self.qkv_proj(x)  # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, T, H, hd) -> (B, H, T, hd)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE only in RoPE mode
        if use_rope:
            q, k = self.rope(q, k, start_pos=start_pos)

        # KV caching: append to past
        if kv_cache is not None:
            k_past, v_past = kv_cache
            # concat along sequence dim (-2)
            k = torch.cat([k_past, k], dim=-2)
            v = torch.cat([v_past, v], dim=-2)

        # Scores over total keys
        scores = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, T_total)

        if mask is not None:
            # mask should be broadcastable to (B,H,T,T_total); here it's (T_total,T_total)
            # but our query length T might be < T_total, so slice last T queries:
            T_total = scores.size(-1)
            if mask.shape != (T_total, T_total):
                raise ValueError(f"Mask shape {mask.shape} does not match (T_total,T_total)=({T_total},{T_total})")
            scores = scores.masked_fill(mask[-T:, :] == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)  # (B, H, T, T_total)
        attn = self.attn_dropout(attn)

        out = attn @ v  # (B, H, T, hd)
        out = out.transpose(1, 2).contiguous().view(B, T, self.dim)
        out = self.out_proj(out)
        out = self.resid_dropout(out)

        new_kv = (k, v) if return_kv else None

        if return_attn and return_kv:
            return out, attn, new_kv
        if return_attn:
            return out, attn
        if return_kv:
            return out, new_kv
        return out


# ----------------------------- FFN -----------------------------
class FeedForward(nn.Module):
    """
    Position-wise FFN with SiLU
    """
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# ----------------------------- Block -----------------------------
class TransformerBlock(nn.Module):
    """
    Pre-norm decoder block:
      x = x + Attn(RMSNorm(x))
      x = x + FFN(RMSNorm(x))
    """
    def __init__(self, dim: int, num_heads: int, ff_hidden_dim: int, dropout: float = 0.1, rope_max_seq_len: int = 2048):
        super().__init__()
        self.attention = CausalSelfAttention(dim, num_heads, dropout, rope_max_seq_len=rope_max_seq_len)
        self.feed_forward = FeedForward(dim, ff_hidden_dim, dropout)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
        kv_cache: Optional[KVCache] = None,
        start_pos: int = 0,
        use_rope: bool = True,
        return_kv: bool = False,
    ):
        if return_attn and return_kv:
            attn_out, attn, new_kv = self.attention(
                self.norm1(x),
                mask=mask,
                return_attn=True,
                kv_cache=kv_cache,
                start_pos=start_pos,
                use_rope=use_rope,
                return_kv=True,
            )
            x = x + attn_out
            x = x + self.feed_forward(self.norm2(x))
            return x, attn, new_kv

        if return_attn:
            attn_out, attn = self.attention(
                self.norm1(x),
                mask=mask,
                return_attn=True,
                kv_cache=kv_cache,
                start_pos=start_pos,
                use_rope=use_rope,
                return_kv=False,
            )
            x = x + attn_out
            x = x + self.feed_forward(self.norm2(x))
            return x, attn

        if return_kv:
            attn_out, new_kv = self.attention(
                self.norm1(x),
                mask=mask,
                return_attn=False,
                kv_cache=kv_cache,
                start_pos=start_pos,
                use_rope=use_rope,
                return_kv=True,
            )
            x = x + attn_out
            x = x + self.feed_forward(self.norm2(x))
            return x, new_kv

        x = x + self.attention(self.norm1(x), mask=mask, use_rope=use_rope)
        x = x + self.feed_forward(self.norm2(x))
        return x


# ----------------------------- Model -----------------------------
class DecoderOnlyTransformer(nn.Module):
    """
    Decoder-only Transformer for next-token prediction.

    Positional modes:
      - pos_encoding="rope"       => RoPE in attention (your current behavior)
      - pos_encoding="sinusoidal" => absolute sinusoidal added to token embeddings; no RoPE in attention
    """
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_layers: int,
        num_heads: int,
        ff_hidden_dim: int,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        pos_encoding: str = "rope",
    ):
        super().__init__()
        assert pos_encoding in ("rope", "sinusoidal")
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.pos_encoding = pos_encoding

        self.token_embedding = nn.Embedding(vocab_size, dim)

        self.sin_pe = None
        if self.pos_encoding == "sinusoidal":
            self.sin_pe = SinusoidalPositionalEmbedding(dim, max_seq_len=max_seq_len)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, ff_hidden_dim, dropout, rope_max_seq_len=max_seq_len)
            for _ in range(num_layers)
        ])
        self.norm_final = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _build_causal_mask(self, T_total: int, device: torch.device) -> torch.Tensor:
        return torch.tril(torch.ones(T_total, T_total, device=device))

    def forward(
        self,
        input_ids: torch.Tensor,                       # (B, T)
        targets: Optional[torch.Tensor] = None,        # (B, T)
        return_attn: bool = False,
        use_causal_mask: bool = True,
        kv_cache: Optional[List[Optional[KVCache]]] = None,  # per-layer cache
        start_pos: int = 0,
        return_kv: bool = False,
    ):
        """
        If kv_cache is provided, you can do:
          - prefill: input_ids is the full context, kv_cache=None, return_kv=True
          - decode:  input_ids is only the NEW tokens (usually shape (B,1)),
                    kv_cache is a list of per-layer caches from prefill/decode,
                    start_pos is where these new tokens begin in the global sequence.
        """
        B, T = input_ids.shape
        x = self.token_embedding(input_ids)  # (B, T, D)

        # Sinusoidal adds absolute PE to embeddings (offset-aware for KV caching)
        if self.pos_encoding == "sinusoidal":
            assert self.sin_pe is not None
            x = x + self.sin_pe(T, start_pos=start_pos).to(x.device)

        use_rope = (self.pos_encoding == "rope")

        # If training (no KV cache), build standard mask over T
        # If caching, total length differs; we handle masking carefully:
        #  - For prefill (kv_cache is None), T_total = T.
        #  - For decode (kv_cache present), we can skip mask because there's no future in keys.
        mask = None
        if use_causal_mask and (kv_cache is None):
            mask = self._build_causal_mask(T, x.device)

        attn_maps: Optional[List[torch.Tensor]] = [] if return_attn else None
        new_cache: Optional[List[KVCache]] = [] if return_kv else None

        for layer_idx, block in enumerate(self.blocks):
            layer_cache = None if kv_cache is None else kv_cache[layer_idx]

            if return_attn and return_kv:
                x, attn, layer_new_kv = block(
                    x,
                    mask=mask,
                    return_attn=True,
                    kv_cache=layer_cache,
                    start_pos=start_pos,
                    use_rope=use_rope,
                    return_kv=True,
                )
                attn_maps.append(attn)
                new_cache.append(layer_new_kv)

            elif return_attn:
                x, attn = block(
                    x,
                    mask=mask,
                    return_attn=True,
                    kv_cache=layer_cache,
                    start_pos=start_pos,
                    use_rope=use_rope,
                    return_kv=False,
                )
                attn_maps.append(attn)

            elif return_kv:
                x, layer_new_kv = block(
                    x,
                    mask=mask,
                    return_attn=False,
                    kv_cache=layer_cache,
                    start_pos=start_pos,
                    use_rope=use_rope,
                    return_kv=True,
                )
                new_cache.append(layer_new_kv)

            else:
                x = block(
                    x,
                    mask=mask,
                    return_attn=False,
                    kv_cache=layer_cache,
                    start_pos=start_pos,
                    use_rope=use_rope,
                    return_kv=False,
                )

        x = self.norm_final(x)
        logits = self.lm_head(x)  # (B, T, V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )

        if return_attn and return_kv:
            return logits, loss, attn_maps, new_cache
        if return_attn:
            return logits, loss, attn_maps
        if return_kv:
            return logits, loss, new_cache
        return logits, loss

    @torch.no_grad()
    def generate_no_cache(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Baseline generation (no KV caching): recompute full forward each step.
        """
        self.eval()
        for _ in range(max_new_tokens):
            ctx = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]
            logits, _ = self.forward(ctx, targets=None, use_causal_mask=True, kv_cache=None, start_pos=0, return_kv=False)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    @torch.no_grad()
    def generate_kv_cache(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        KV-cached generation:
          1) prefill with full context to build per-layer KV caches
          2) decode one token at a time, appending to caches (O(1) per layer per token)
        """
        self.eval()

        # Trim context if longer than max_seq_len
        if input_ids.size(1) > self.max_seq_len:
            input_ids = input_ids[:, -self.max_seq_len:]

        B, T0 = input_ids.shape

        # 1) PREFILL
        logits, _, kv_cache = self.forward(
            input_ids,
            targets=None,
            use_causal_mask=True,
            kv_cache=None,
            start_pos=0,
            return_kv=True,
        )
        # sample first new token from last position
        next_logits = logits[:, -1, :] / temperature
        if top_k is not None:
            values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < values[:, [-1]]] = -float("inf")
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        # 2) DECODE LOOP (each step feeds only the newest token)
        for step in range(1, max_new_tokens):
            # start_pos is position index of this new token in the global sequence
            start_pos = T0 + step - 1  # because we already appended 1 token
            new_token = input_ids[:, -1:]  # (B,1)

            logits_step, _, kv_cache = self.forward(
                new_token,
                targets=None,
                use_causal_mask=True,
                kv_cache=kv_cache,
                start_pos=start_pos,
                return_kv=True,
            )

            next_logits = logits_step[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < values[:, [-1]]] = -float("inf")
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


# ----------------------------- Training -----------------------------
def train_epoch(
    model: DecoderOnlyTransformer,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    global_step: int,
    ckpt_dir: Path,
    use_causal_mask: bool = True,
    log_every: int = 100,
    ckpt_every: int = 1000,
    tag: str = "model",
) -> Tuple[float, int]:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for _, actions in dataloader:
        actions = actions.to(device)  # (B, T)
        input_ids = actions[:, :-1]   # teacher forcing
        targets = actions[:, 1:].contiguous()

        _, loss = model(input_ids, targets, use_causal_mask=use_causal_mask)
        if loss is None:
            raise RuntimeError("Loss is None during training.")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        global_step += 1
        total_loss += loss.item()
        num_batches += 1

        if global_step % log_every == 0:
            ppl = math.exp(min(loss.item(), 20.0))
            mode = "causal" if use_causal_mask else "no-mask"
            print(f"[{tag} | {mode}] epoch {epoch+1} step {global_step} loss {loss.item():.4f} ppl {ppl:.2f}")

        if global_step % ckpt_every == 0:
            ckpt_path = ckpt_dir / f"{tag}_step_{global_step}.pt"
            torch.save(model.state_dict(), ckpt_path)

    return total_loss / max(1, num_batches), global_step


# ----------------------------- Visualization -----------------------------
@torch.no_grad()
def save_attention_map_single(
    model: DecoderOnlyTransformer,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    out_path: str,
    layer_idx: int = 0,
    head_idx: int = 0,
    use_causal_mask: bool = True,
):
    model.eval()
    _, actions = next(iter(dataloader))
    actions = actions.to(device)

    input_ids = actions[:1, :-1]
    targets = actions[:1, 1:]

    _, _, attn_maps = model(
        input_ids,
        targets,
        return_attn=True,
        use_causal_mask=use_causal_mask,
    )

    attn = attn_maps[layer_idx][0, head_idx].detach().cpu().numpy()  # (T, T)

    plt.figure()
    plt.imshow(attn, aspect="auto")
    plt.title(f"Attention Map ({model.pos_encoding}) | layer={layer_idx} head={head_idx}")
    plt.xlabel("Key position (attended-to)")
    plt.ylabel("Query position (attending-from)")
    plt.colorbar()
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


@torch.no_grad()
def save_attention_grids(
    model: DecoderOnlyTransformer,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    out_dir: str = "images",
    layers_to_plot=None,
    heads_to_plot=None,
    use_causal_mask: bool = True,
    max_cols: int = 4,
):
    model.eval()
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    _, actions = next(iter(dataloader))
    actions = actions.to(device)

    input_ids = actions[:1, :-1]
    targets = actions[:1, 1:]

    _, _, attn_maps = model(
        input_ids,
        targets,
        return_attn=True,
        use_causal_mask=use_causal_mask,
    )

    num_layers = len(attn_maps)
    num_heads = attn_maps[0].shape[1]

    if layers_to_plot is None:
        layers_to_plot = list(range(num_layers))
    if heads_to_plot is None:
        heads_to_plot = list(range(num_heads))

    for layer_idx in layers_to_plot:
        attn = attn_maps[layer_idx][0].detach().cpu().numpy()  # (H, T, T)
        heads = heads_to_plot

        n = len(heads)
        cols = min(max_cols, n)
        rows = int(np.ceil(n / cols))

        plt.figure(figsize=(4 * cols, 4 * rows))
        for i, h in enumerate(heads):
            ax = plt.subplot(rows, cols, i + 1)
            ax.imshow(attn[h], aspect="auto")
            ax.set_title(f"head {h}")
            ax.set_xlabel("key")
            ax.set_ylabel("query")

        plt.suptitle(f"Attention Grid ({model.pos_encoding}) | layer={layer_idx}", y=0.98)
        plt.tight_layout()
        out_path = Path(out_dir) / f"attn_grid_{model.pos_encoding}_layer{layer_idx}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()


@torch.no_grad()
def compute_future_attention_ratio(
    model: DecoderOnlyTransformer,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    use_causal_mask: bool = True,
    num_batches: int = 10,
) -> float:
    model.eval()
    total_ratio = 0.0
    count = 0

    for b, (_, actions) in enumerate(dataloader):
        if b >= num_batches:
            break

        actions = actions.to(device)
        input_ids = actions[:, :-1]
        targets = actions[:, 1:].contiguous()

        _, _, attn_maps = model(
            input_ids,
            targets,
            return_attn=True,
            use_causal_mask=use_causal_mask,
        )

        attn = torch.stack(attn_maps, dim=0)  # (L, B, H, T, T)
        T = attn.shape[-1]
        upper = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        lower = torch.tril(torch.ones(T, T, device=device), diagonal=0).bool()

        future_mass = attn[..., upper].sum(dim=-1)  # (L,B,H)
        past_mass = attn[..., lower].sum(dim=-1)
        total_mass = future_mass + past_mass

        ratio = (future_mass / (total_mass + 1e-9)).mean().item()
        total_ratio += ratio
        count += 1

    return total_ratio / max(1, count)


def plot_loss_curve(losses, out_path, title):
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_two_loss_curves(loss_a, loss_b, out_path, label_a, label_b, title):
    plt.figure()
    plt.plot(range(1, len(loss_a) + 1), loss_a, marker="o", label=label_a)
    plt.plot(range(1, len(loss_b) + 1), loss_b, marker="o", label=label_b)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_speed_bars(speed_dict: Dict[str, float], out_path: str, title: str):
    """
    speed_dict: label -> tokens/sec
    """
    labels = list(speed_dict.keys())
    values = [speed_dict[k] for k in labels]

    plt.figure(figsize=(10, 4))
    plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels, rotation=20, ha="right")
    plt.ylabel("Tokens / second")
    plt.title(title)
    plt.grid(axis="y")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


# ----------------------------- Benchmarking -----------------------------
@torch.no_grad()
def benchmark_inference_speed(
    model: DecoderOnlyTransformer,
    device: torch.device,
    prompt_len: int = 50,
    gen_tokens: int = 200,
    iters: int = 10,
    top_k: Optional[int] = None,
) -> Dict[str, float]:
    """
    Returns tokens/sec for:
      - no_cache
      - kv_cache
    """
    model.eval()

    # fixed random prompt (so both modes see identical input)
    torch.manual_seed(0)
    prompt = torch.randint(low=0, high=model.vocab_size, size=(1, prompt_len), device=device)

    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Warmup
    _ = model.generate_no_cache(prompt.clone(), max_new_tokens=5, top_k=top_k)
    _ = model.generate_kv_cache(prompt.clone(), max_new_tokens=5, top_k=top_k)
    _sync()

    speeds = {}

    # No-cache timing
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model.generate_no_cache(prompt.clone(), max_new_tokens=gen_tokens, top_k=top_k)
    _sync()
    t1 = time.perf_counter()
    total_tokens = iters * gen_tokens
    speeds["no_cache"] = total_tokens / (t1 - t0 + 1e-9)

    # KV-cache timing
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model.generate_kv_cache(prompt.clone(), max_new_tokens=gen_tokens, top_k=top_k)
    _sync()
    t1 = time.perf_counter()
    speeds["kv_cache"] = total_tokens / (t1 - t0 + 1e-9)

    return speeds


# ----------------------------- Main -----------------------------
def main():
    # Hyperparameters
    vocab_size = 256
    dim = 256
    num_layers = 4
    num_heads = 8
    ff_hidden_dim = 1024
    max_seq_len = 50
    bench_gen_tokens = 200
    pe_max_len = max_seq_len + bench_gen_tokens + 5  # safe margin
    
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 10

    # Logging/checkpoints
    log_every = 100
    ckpt_every = 1000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    data_path = Path("data/trajectories.pkl")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run `python generate_data.py` first.")
    with open(data_path, "rb") as f:
        dataset = pickle.load(f)

    train_loader, val_loader = create_dataloaders(dataset, batch_size=batch_size, train_split=0.9)

    ckpt_dir = Path("checkpoints/")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    img_dir = Path("images/")
    img_dir.mkdir(parents=True, exist_ok=True)

    # ------------------ Train CAUSAL model (RoPE) ------------------
    model_rope = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_hidden_dim=ff_hidden_dim,
        max_seq_len=pe_max_len,
        dropout=0.1,
        pos_encoding="rope",
    ).to(device)

    opt_rope = torch.optim.AdamW(model_rope.parameters(), lr=learning_rate)

    loss_rope = []
    global_step = 0
    print("\n=== Training (CAUSAL, RoPE) ===")
    for epoch in range(num_epochs):
        train_loss, global_step = train_epoch(
            model=model_rope,
            dataloader=train_loader,
            optimizer=opt_rope,
            device=device,
            epoch=epoch,
            global_step=global_step,
            ckpt_dir=ckpt_dir,
            use_causal_mask=True,
            log_every=log_every,
            ckpt_every=ckpt_every,
            tag="causal_rope",
        )
        print(f"[causal_rope] Epoch {epoch+1}/{num_epochs} - Avg Loss: {train_loss:.4f}")
        loss_rope.append(train_loss)

    torch.save(model_rope.state_dict(), ckpt_dir / "best_model_rope.pt")
    print("Saved RoPE causal model to checkpoints/best_model_rope.pt")
    plot_loss_curve(loss_rope, str(img_dir / "training_loss_rope.png"), "Training Loss (Causal, RoPE)")

    # Attention maps (RoPE)
    save_attention_map_single(
        model_rope, val_loader, device,
        out_path=str(img_dir / "attention_rope_layer0_head0.png"),
        layer_idx=0, head_idx=0, use_causal_mask=True,
    )
    save_attention_grids(
        model_rope, val_loader, device,
        out_dir=str(img_dir),
        layers_to_plot=None,
        heads_to_plot=None,
        use_causal_mask=True,
        max_cols=4,
    )

    # Causal mask audit (RoPE)
    r_rope = compute_future_attention_ratio(model_rope, val_loader, device, use_causal_mask=True)
    print(f"Future-attention ratio (RoPE causal): {r_rope:.8f} (expected ~0)")

    # ------------------ Train CAUSAL model (Sinusoidal) [Mastery Ablation] ------------------
    model_sin = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_hidden_dim=ff_hidden_dim,
        max_seq_len=pe_max_len,
        dropout=0.1,
        pos_encoding="sinusoidal",
    ).to(device)

    opt_sin = torch.optim.AdamW(model_sin.parameters(), lr=learning_rate)

    loss_sin = []
    global_step_sin = 0
    print("\n=== Training (CAUSAL, Sinusoidal) [ABLATION] ===")
    for epoch in range(num_epochs):
        train_loss, global_step_sin = train_epoch(
            model=model_sin,
            dataloader=train_loader,
            optimizer=opt_sin,
            device=device,
            epoch=epoch,
            global_step=global_step_sin,
            ckpt_dir=ckpt_dir,
            use_causal_mask=True,
            log_every=log_every,
            ckpt_every=ckpt_every,
            tag="causal_sinusoidal",
        )
        print(f"[causal_sinusoidal] Epoch {epoch+1}/{num_epochs} - Avg Loss: {train_loss:.4f}")
        loss_sin.append(train_loss)

    torch.save(model_sin.state_dict(), ckpt_dir / "best_model_sinusoidal.pt")
    print("Saved Sinusoidal causal model to checkpoints/best_model_sinusoidal.pt")
    plot_loss_curve(loss_sin, str(img_dir / "training_loss_sinusoidal.png"), "Training Loss (Causal, Sinusoidal)")

    # RoPE vs Sinusoidal loss comparison image (Mastery)
    plot_two_loss_curves(
        loss_rope, loss_sin,
        out_path=str(img_dir / "rope_vs_sinusoidal_loss.png"),
        label_a="RoPE",
        label_b="Sinusoidal",
        title="Loss Comparison: RoPE vs Sinusoidal (Causal)",
    )

    # Attention maps (Sinusoidal) [optional but useful for report]
    save_attention_map_single(
        model_sin, val_loader, device,
        out_path=str(img_dir / "attention_sinusoidal_layer0_head0.png"),
        layer_idx=0, head_idx=0, use_causal_mask=True,
    )
    save_attention_grids(
        model_sin, val_loader, device,
        out_dir=str(img_dir),
        layers_to_plot=None,
        heads_to_plot=None,
        use_causal_mask=True,
        max_cols=4,
    )

    # ------------------ Inference speed comparison (with/without KV cache) ------------------
    print("\n=== Benchmarking inference speed (tokens/sec) ===")
    speeds_rope = benchmark_inference_speed(model_rope, device=device, prompt_len=max_seq_len, gen_tokens=bench_gen_tokens, iters=10)
    speeds_sin  = benchmark_inference_speed(model_sin,  device=device, prompt_len=max_seq_len, gen_tokens=bench_gen_tokens, iters=10)



    speed_plot = {
        "RoPE no_cache": speeds_rope["no_cache"],
        "RoPE kv_cache": speeds_rope["kv_cache"],
        "Sin no_cache": speeds_sin["no_cache"],
        "Sin kv_cache": speeds_sin["kv_cache"],
    }
    print("Speed (tokens/sec):")
    for k, v in speed_plot.items():
        print(f"  {k:14s}: {v:.2f}")

    plot_speed_bars(
        speed_plot,
        out_path=str(img_dir / "inference_speed_kvcache.png"),
        title="Inference Speed: With vs Without KV-Caching",
    )

    print("\nDone.")
    print(f"- RoPE loss curve:             {img_dir / 'training_loss_rope.png'}")
    print(f"- Sinusoidal loss curve:       {img_dir / 'training_loss_sinusoidal.png'}")
    print(f"- RoPE vs Sin loss compare:    {img_dir / 'rope_vs_sinusoidal_loss.png'}")
    print(f"- Inference speed plot:        {img_dir / 'inference_speed_kvcache.png'}")
    print(f"- Attn single RoPE:            {img_dir / 'attention_rope_layer0_head0.png'}")
    print(f"- Attn single Sinusoidal:      {img_dir / 'attention_sinusoidal_layer0_head0.png'}")
    print(f"- Attn grids RoPE:             {img_dir / 'attn_grid_rope_layer*.png'}")
    print(f"- Attn grids Sinusoidal:       {img_dir / 'attn_grid_sinusoidal_layer*.png'}")
    print(f"- RoPE ckpt:                   {ckpt_dir / 'best_model_rope.pt'}")
    print(f"- Sinusoidal ckpt:             {ckpt_dir / 'best_model_sinusoidal.pt'}")
    print(f"- Step ckpts:                  {ckpt_dir}/*_step_*.pt")


if __name__ == "__main__":
    main()