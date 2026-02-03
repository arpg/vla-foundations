"""
Scratch-1: The Transformer Backbone
CSCI 7000 - VLA Foundations

Template for implementing a decoder-only Transformer from scratch.
Students must complete the TODO sections for:
1. CausalSelfAttention
2. RMSNorm
3. Training loop
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
import pickle
from pathlib import Path
from generate_data import create_dataloaders
import matplotlib.pyplot as plt
import time

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization

    Paper: https://arxiv.org/abs/1910.07467
    Used in: Llama-3, Grok, PaLM

    Formula:
        a_bar_i = (a_i / RMS(a)) * g_i
        where RMS(a) = sqrt(mean(a^2) + eps)
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Initialize learnable scale parameter 'g' (gamma)
        # Hint: Use nn.Parameter with torch.ones
        ####self.scale = None  # REPLACE THIS LINE
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
        Returns:
            Normalized tensor of same shape
        """
        # Implement RMSNorm
        # Step 1: Compute RMS (root mean square) along the last dimension
        # Step 2: Normalize by dividing x by RMS
        # Step 3: Apply learnable scale parameter

        # HINT: Use torch.mean, torch.rsqrt for efficiency
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x_norm = x * rms
        return x_norm * self.scale
        
        raise NotImplementedError("Implement RMSNorm forward pass")


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)

    Paper: RoFormer (Su et al., 2021) - https://arxiv.org/abs/2104.09864
    Used in: Llama-3, PaLM-E, GPT-NeoX

    Key Idea: Rotate pairs of dimensions by an angle proportional to position
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute rotation frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos and sin for all positions
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Precompute cos and sin values for all positions"""
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (seq_len, dim//2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Helper function to rotate tensor by swapping and negating half the dimensions"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_start_pos: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to query and key tensors

        Args:
            q: Query tensor (batch, num_heads, seq_len, head_dim)
            k: Key tensor (batch, num_heads, seq_len, head_dim)
            seq_start_pos: Starting position of the sequence (for KV caching)
        Returns:
            Rotated (q, k) tensors
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # When using KV cache, seq_len of q might be 1, but we need position indices corresponding to the global position in the sequence.
        # Had to add this to make sure it works with KV caching
        
        # Get cached cos/sin values for the relevant positions
        positions = torch.arange(seq_start_pos, seq_start_pos + seq_len, device=q.device)
        
        cos = self.cos_cached[positions]  # (seq_len, dim)
        sin = self.sin_cached[positions]  # (seq_len, dim)
        
        # Expand for broadcasting: (1, 1, seq_len, dim) to match (batch, heads, seq_len, dim)
        cos = cos.view(1, 1, seq_len, head_dim)
        sin = sin.view(1, 1, seq_len, head_dim)

        # Apply rotation: q_rot = q * cos + rotate_half(q) * sin
        q_rot = (q * cos) + (self.rotate_half(q) * sin)
        k_rot = (k * cos) + (self.rotate_half(k) * sin)

        return q_rot, k_rot


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention with RoPE
    
    Key constraints:
    - Token at position t can only attend to positions <= t
    - Uses RoPE instead of absolute positional embeddings
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Linear projections for Q, K, V
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Rotary embeddings
        self.rope = RotaryPositionalEmbedding(self.head_dim)

    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Optional attention mask (seq_len, seq_len)
            past_key_value: Optional tuple of (key, value) from previous step
        Returns:
            Output tensor (batch, seq_len, dim)
            present_key_value: Tuple of (key, value) for next step
        """
        batch_size, seq_len, _ = x.shape

        # Step 1: Project input to Q, K, V
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3*dim)
        
        # Split into Q, K, V and reshape for multi-head attention
        # Hint: Use .view() and .transpose() to get shape (batch, num_heads, seq_len, head_dim)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Determine starting position for RoPE
        seq_start_pos = 0
        if past_key_value is not None:
            # past_key_value[0] shape: (batch, num_heads, past_seq_len, head_dim)
            seq_start_pos = past_key_value[0].shape[2]

        # Step 2: Apply RoPE to Q and K
        q, k = self.rope(q, k, seq_start_pos=seq_start_pos)

        # Step 2++: KV Caching
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            
        present_key_value = (k, v)

        # Step 3: Compute attention scores
        # scores = (Q @ K^T) / sqrt(d_k)
        # scores: (batch, n_head, q_len, k_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Step 4: Apply causal mask
        # Only apply mask if dimensions permit or broadcast
        if mask is not None:
            mask = mask.to(device=x.device, dtype=torch.bool)
            if mask.shape[-2:] == scores.shape[-2:]:
                 scores = scores.masked_fill(
                    ~mask.unsqueeze(0).unsqueeze(0),
                    torch.finfo(scores.dtype).min,
                )
            elif seq_len > 1:
                # Fallback for prefill if mask is (seq_len, seq_len)
                 scores = scores.masked_fill(
                    ~mask.view(1, 1, seq_len, seq_len),
                    torch.finfo(scores.dtype).min,
                )


        # Step 5: Apply softmax and dropout
        # attn_weights = F.softmax(scores, dim=-1)
        # attn_weights = self.attn_dropout(attn_weights)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Step 6: Apply attention to values
        # out = attn_weights @ V
        out = torch.matmul(attn_weights, v)  # (batch, num_heads, seq_len, head_dim)


        # Step 7: Reshape and project back
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        
        return out, present_key_value


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network with SwiGLU activation

    Used in modern LLMs for better performance than standard ReLU
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
        Returns:
            Output tensor (batch, seq_len, dim)
        """
        x = self.fc1(x)
        x = F.silu(x)  # SwiGLU activation (SiLU = Swish)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    A single Transformer decoder block

    Architecture:
        x = x + Attention(RMSNorm(x))
        x = x + FeedForward(RMSNorm(x))
    """

    def __init__(self, dim: int, num_heads: int, ff_hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = CausalSelfAttention(dim, num_heads, dropout)
        self.feed_forward = FeedForward(dim, ff_hidden_dim, dropout)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Optional attention mask
            past_key_value: Optional tuple of (key, value) from previous step
        Returns:
            Output tensor (batch, seq_len, dim)
            present_key_value: Tuple of (key, value) for next step
        """
        # Pre-norm architecture (norm before attention/FF)
        attn_out, present_key_value = self.attention(self.norm1(x), mask, past_key_value)
        x = x + attn_out
        x = x + self.feed_forward(self.norm2(x))
        return x, present_key_value


class DecoderOnlyTransformer(nn.Module):
    """
    Decoder-only Transformer for next-token prediction on robot trajectories

    Architecture similar to GPT, but for robotic control sequences
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
        causal_mask: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.causal_mask = causal_mask

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # Final norm and projection to vocabulary
        self.norm_final = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights following standard Transformer initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            input_ids: Input token indices (batch, seq_len)
            targets: Target token indices for training (batch, seq_len)
            past_key_values: List of (k, v) tuples for each layer
        Returns:
            logits: Output logits (batch, seq_len, vocab_size)
            loss: Cross-entropy loss if targets provided, else None
            current_key_values: List of (k, v) tuples for next step
        """
        batch_size, seq_len = input_ids.shape

        # Embed tokens
        x = self.token_embedding(input_ids)  # (batch, seq_len, dim)

        # Create causal mask (lower triangular)
        # Only needed if seq_len > 1 (prefill) and we are not just attending to *all* past
        
        mask = None
        if self.causal_mask and seq_len > 1:
             mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))

        current_key_values = []
        
        # Apply transformer blocks
        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, present_kv = block(x, mask, past_kv)
            current_key_values.append(present_kv)

        # Final norm and projection
        x = self.norm_final(x)
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Flatten for cross-entropy
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1,  # Ignore padding tokens
            )

        return logits, loss, current_key_values

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        Autoregressive generation

        Args:
            input_ids: Starting tokens (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (if None, use full distribution)
            use_cache: Whether to use KV caching
        Returns:
            Generated sequence (batch, seq_len + max_new_tokens)
        """
        past_key_values = None
        
        for _ in range(max_new_tokens):
            if use_cache:
                if past_key_values is None:
                    # First step: prefill with full context
                    logits, _, past_key_values = self.forward(input_ids)
                    # We only care about the last token's logits
                    logits = logits[:, -1, :]
                else:
                    # Subsequent steps: process only the last generated token
                    logits, _, past_key_values = self.forward(
                        input_ids[:, -1:], 
                        past_key_values=past_key_values
                    )
                    logits = logits[:, -1, :]
            else:
                # No cache: process full context every time
                # Crop context if too long
                input_context = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]
                logits, _, _ = self.forward(input_context)
                logits = logits[:, -1, :]

            # Apply temperature
            logits = logits / temperature

            # Optional top-k sampling
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float('Inf')

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


def benchmark_inference(
    vocab_size=256,
    dim=256,
    num_layers=4,
    num_heads=8,
    ff_hidden_dim=1024,
    max_seq_len=2048,
    batch_size=1,
    prompt_len=500,
    gen_len=1000,
):
    """
    Benchmark inference with and without KV caching
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on {device}")

    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_hidden_dim=ff_hidden_dim,
        max_seq_len=max_seq_len,
        causal_mask=True
    ).to(device)
    model.eval()

    # Create dummy input
    input_ids = torch.randint(0, vocab_size, (batch_size, prompt_len)).to(device)

    # Warmup
    print("Warming up...")
    model.generate(input_ids, max_new_tokens=5, use_cache=True)
    model.generate(input_ids, max_new_tokens=5, use_cache=False)

    # Benchmark without cache
    print(f"Generating {gen_len} tokens WITHOUT cache...")
    start_time = time.time()
    _ = model.generate(input_ids, max_new_tokens=gen_len, use_cache=False)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    time_no_cache = end_time - start_time
    tokens_per_sec_no_cache = (gen_len * batch_size) / time_no_cache
    print(f"Time: {time_no_cache:.4f}s, Speed: {tokens_per_sec_no_cache:.2f} tokens/sec")

    # Benchmark with cache
    print(f"Generating {gen_len} tokens WITH cache...")
    start_time = time.time()
    _ = model.generate(input_ids, max_new_tokens=gen_len, use_cache=True)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    time_cache = end_time - start_time
    tokens_per_sec_cache = (gen_len * batch_size) / time_cache
    print(f"Time: {time_cache:.4f}s, Speed: {tokens_per_sec_cache:.2f} tokens/sec")

    speedup = tokens_per_sec_cache / tokens_per_sec_no_cache
    print(f"Speedup: {speedup:.2f}x")


    # Need to make sure that output tokens are the same
    print("Verifying correctness...")
    # Seed for reproducibility of sampling (though we might just check logits)
    torch.manual_seed(45)
    # Generate 1 token
    out_no_cache = model.generate(input_ids, max_new_tokens=5, use_cache=False)
    
    torch.manual_seed(45)
    out_cache = model.generate(input_ids, max_new_tokens=5, use_cache=True)
    
    match = torch.equal(out_no_cache, out_cache)
    print(f"Outputs match: {match}")
    if not match:
        print("Warning: Outputs do not match!")
        print(f"No cache: {out_no_cache}")
        print(f"Cache:    {out_cache}")


def main():
    benchmark_inference()

if __name__ == "__main__":
    main()
