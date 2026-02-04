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
from typing import Optional, Tuple    

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
        # TODO: Initialize learnable scale parameter 'g' (gamma)
        # Hint: Use nn.Parameter with torch.ones
        self.scale = nn.Parameter(torch.ones(dim))  # REPLACE THIS LINE

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
        Returns:
            Normalized tensor of same shape
        """
        # TODO: Implement RMSNorm
        # Step 1: Compute RMS (root mean square) along the last dimension
        # Step 2: Normalize by dividing x by RMS
        # Step 3: Apply learnable scale parameter

        # HINT: Use torch.mean, torch.rsqrt for efficiency
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.scale * rms


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

    def forward(self, q: torch.Tensor, k: torch.Tensor, start_pos=0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to query and key tensors

        Args:
            q: Query tensor (batch, num_heads, seq_len, head_dim)
            k: Key tensor (batch, num_heads, seq_len, head_dim)
            start_pos: The position of the current token, needed for KV cache
        Returns:
            Rotated (q, k) tensors
        """
        seq_len = q.shape[2]

        # Get cached cos/sin values
        cos = self.cos_cached[start_pos:start_pos+seq_len, ...]
        sin = self.sin_cached[start_pos:start_pos+seq_len, ...]

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
        
        # KV cache
        self.k_cache = None
        self.v_cache = None
        self.index = 0

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, use_cache=False, reset_cache=False) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Optional attention mask (seq_len, seq_len)
        Returns:
            Output tensor (batch, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape
        if reset_cache:
            self.reset_cache()

        # Step 1: Project input to Q, K, V
        # qkv = self.qkv_proj(x)  # (batch, seq_len, 3*dim)
        # Split into Q, K, V and reshape for multi-head attention
        is_full_attention = (not use_cache) or (self.k_cache is None) # always used when no cache, only used the first time when cached

        if is_full_attention:
            qkv = self.qkv_proj(x)
            qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # (batch, seq_len, dim)

            # Step 2: Apply RoPE to Q and K
            q, k = self.rope(q, k)
               
        # initializing the kv caches for the first time
        if use_cache and self.k_cache is None:
            self.k_cache = torch.zeros(batch_size, self.num_heads, seq_len * 2, self.head_dim).to(x.device)
            self.v_cache = torch.zeros(batch_size, self.num_heads, seq_len * 2, self.head_dim).to(x.device)
            self.k_cache[:, :, :seq_len] = k.clone()
            self.v_cache[:, :, :seq_len] = v.clone()
            self.index = seq_len
        elif use_cache:
            if self.index + 1 > self.k_cache.size(2):
                new_cap = self.k_cache.size(2) * 2
                k_new_cache = torch.zeros(batch_size, self.num_heads, new_cap, self.head_dim,
                                        device=x.device, dtype=self.k_cache.dtype)
                v_new_cache = torch.zeros(batch_size, self.num_heads, new_cap, self.head_dim,
                                        device=x.device, dtype=self.v_cache.dtype)
                k_new_cache[:, :, :self.index] = self.k_cache[:, :, :self.index]
                v_new_cache[:, :, :self.index] = self.v_cache[:, :, :self.index]
                self.k_cache = k_new_cache
                self.v_cache = v_new_cache
            
            # compute the QKV for the last element only
            qkv = self.qkv_proj(x[:, -1].unsqueeze(1))
            qkv = qkv.view(batch_size, 1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k_new, v_new = qkv[0], qkv[1], qkv[2]  # (batch, 1, dim)
            q, k_new = self.rope(q, k_new, self.index)
            self.k_cache[:, :, self.index:self.index+1] = k_new
            self.v_cache[:, :, self.index:self.index+1] = v_new
            self.index += 1
            k = self.k_cache[:, :, :self.index]
            v = self.v_cache[:, :, :self.index]


        # Step 3: Compute attention scores
        # scores = (Q @ K^T) / sqrt(d_k)
        # Shape should be (batch, num_heads, seq_len, seq_len)
        score = (q @ k.transpose(-1, -2)) * self.scale

        # Step 4: Apply causal mask
        # The mask should prevent position i from attending to positions > i
        # Set masked positions to -inf BEFORE softmax
        
        # comment out to not use mask
        
        if not use_cache:
            score = score.masked_fill_(mask == 0, -torch.inf)

        # Step 5: Apply softmax and dropout
        attn_weights = F.softmax(score, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Step 6: Apply attention to values
        out = attn_weights @ v

        # Step 7: Reshape and project back
        # Concatenate heads and apply output projection
        out = out.permute(0, 2, 1, 3).flatten(-2)
        
        return self.resid_dropout(self.out_proj(out))
    
    def reset_cache(self):
        self.k_cache = None
        self.v_cache = None
        self.index = 0
        
class CausalLatentAttention(nn.Module):
    """
    Multi-Head Latent Causal Self-Attention with RoPE

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
        self.ldim = dim // 4
        self.head_ldim = self.ldim // num_heads # the latent dimension where attention is done
        self.scale = self.head_ldim ** -0.5

        # Linear projections for Q, K, V
        self.qkv_proj = nn.Linear(dim, 3 * self.ldim, bias=False)
        self.w_o = nn.Linear(self.ldim, dim, bias=False) # projects attention stuff back into normal dimensions
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Rotary embeddings
        self.rope = RotaryPositionalEmbedding(self.head_ldim)
        
        # KV cache
        self.k_cache = None  # normally MLA doesn't need both caches but since we are applying RoPE, we cache them separately.
        self.v_cache = None
        self.index = 0

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, use_cache=False, reset_cache=False) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Optional attention mask (seq_len, seq_len)
        Returns:
            Output tensor (batch, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape
        if reset_cache:
            self.reset_cache()

        # Step 1: Project input to Q, K, V
        # qkv = self.qkv_proj(x)  # (batch, seq_len, 3*dim)
        # Split into Q, K, V and reshape for multi-head attention
        is_full_attention = (not use_cache) or (self.k_cache is None) # always used when no cache, only used the first time when cached

        if is_full_attention:
            qkv = self.qkv_proj(x)
            qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_ldim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # (batch, seq_len, num_heads, head_ldim)

            # Step 2: Apply RoPE to Q and K
            q, k = self.rope(q, k)
               
        # initializing the kv caches for the first time
        if use_cache and self.k_cache is None:
            self.k_cache = torch.zeros(batch_size, self.num_heads, seq_len * 2, self.head_ldim).to(x.device)
            self.v_cache = torch.zeros(batch_size, self.num_heads, seq_len * 2, self.head_ldim).to(x.device)
            self.k_cache[:, :, :seq_len] = k.clone()
            self.v_cache[:, :, :seq_len] = v.clone()
            self.index = seq_len
        elif use_cache:
            if self.index + 1 > self.k_cache.size(2):
                new_cap = self.k_cache.size(2) * 2
                k_new_cache = torch.zeros(batch_size, self.num_heads, new_cap, self.head_ldim,
                                        device=x.device, dtype=self.k_cache.dtype)
                v_new_cache = torch.zeros(batch_size, self.num_heads, new_cap, self.head_ldim,
                                        device=x.device, dtype=self.v_cache.dtype)
                k_new_cache[:, :, :self.index] = self.k_cache[:, :, :self.index]
                v_new_cache[:, :, :self.index] = self.v_cache[:, :, :self.index]
                self.k_cache = k_new_cache
                self.v_cache = v_new_cache
            
            # compute the QKV for the last element only
            qkv = self.qkv_proj(x[:, -1].unsqueeze(1))
            qkv = qkv.view(batch_size, 1, 3, self.num_heads, self.head_ldim).permute(2, 0, 3, 1, 4)
            q, k_new, v_new = qkv[0], qkv[1], qkv[2]  # (batch, 1, dim)
            q, k_new = self.rope(q, k_new, self.index)
            self.k_cache[:, :, self.index:self.index+1] = k_new
            self.v_cache[:, :, self.index:self.index+1] = v_new
            self.index += 1
            k = self.k_cache[:, :, :self.index]
            v = self.v_cache[:, :, :self.index]


        # Step 3: Compute attention scores
        # scores = (Q @ K^T) / sqrt(d_k)
        score = (q @ k.transpose(-1, -2)) * self.scale

        # Step 4: Apply causal mask
        # The mask should prevent position i from attending to positions > i
        if is_full_attention:
            score = score.masked_fill_(mask == 0, -torch.inf)

        # Step 5: Apply softmax and dropout
        attn_weights = F.softmax(score, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Step 6: Apply attention to values
        out = attn_weights @ v

        # Step 7: Reshape and project back
        # Concatenate heads and apply output projection
        out = out.permute(0, 2, 1, 3).flatten(-2)
        out = self.w_o(out)
        
        return self.resid_dropout(self.out_proj(out))
    
    def reset_cache(self):
        self.k_cache = None
        self.v_cache = None
        self.index = 0


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

    def __init__(self, dim: int, num_heads: int, ff_hidden_dim: int, dropout: float = 0.1, use_mla=False):
        super().__init__()
        if use_mla:
            self.attention = CausalLatentAttention(dim, num_heads, dropout)
        else:
            self.attention = CausalSelfAttention(dim, num_heads, dropout)
        self.feed_forward = FeedForward(dim, ff_hidden_dim, dropout)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, use_cache=False, reset_cache=False) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Optional attention mask
        Returns:
            Output tensor (batch, seq_len, dim)
        """
        # Pre-norm architecture (norm before attention/FF)
        x = x + self.attention(self.norm1(x), mask, use_cache, reset_cache)
        x = x + self.feed_forward(self.norm2(x))
        return x


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
        use_mla=False
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, ff_hidden_dim, dropout, use_mla)
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: Input token indices (batch, seq_len)
            targets: Target token indices for training (batch, seq_len)
        Returns:
            logits: Output logits (batch, seq_len, vocab_size)
            loss: Cross-entropy loss if targets provided, else None
        """
        batch_size, seq_len = input_ids.shape

        # Embed tokens
        x = self.token_embedding(input_ids)  # (batch, seq_len, dim)

        # Create causal mask (lower triangular)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)

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

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation

        Args:
            input_ids: Starting tokens (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (if None, use full distribution)
        Returns:
            Generated sequence (batch, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop context if too long
            input_context = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]

            # Forward pass
            logits, _ = self.forward(input_context)
            logits = logits[:, -1, :] / temperature  # Get last token logits

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
    
    @torch.no_grad()
    def forward_cached(
        self,
        input_ids: torch.Tensor,
        reset_cache: bool = False,
    ) -> torch.Tensor:
        self.eval()
        _, seq_len = input_ids.shape

        # Embed
        x = self.token_embedding(input_ids) 

        # Only need a causal mask for prefill
        mask = None
        if reset_cache and seq_len > 1:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))

        for block in self.blocks:
            x = block(x, mask, use_cache=True, reset_cache=reset_cache)

        x = self.norm_final(x)
        logits = self.lm_head(x) 
        return logits


    @torch.no_grad()
    def generate_cached(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        KV-cached generation:
        1) Prefill caches with the full prompt (one full forward)
        2) Incrementally decode by feeding ONLY the last token each step
        """
        self.eval()
        assert input_ids.dim() == 2, "input_ids must be (batch, seq_len)"
        assert max_new_tokens >= 0

        logits = self.forward_cached(input_ids, reset_cache=True)
        next_logits = logits[:, -1, :] / temperature

        if top_k is not None:
            values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < values[:, [-1]]] = -float("inf")

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        # decoding, only input the last token at each step
        for _ in range(max_new_tokens - 1):
            last_token = input_ids[:, -1:]
            logits = self.forward_cached(last_token, reset_cache=False)
            next_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < values[:, [-1]]] = -float("inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


def train_epoch(
    model: DecoderOnlyTransformer,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """
    Train for one epoch

    Args:
        model: The transformer model
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        _, loss = model(y[:, :-1], y[:, 1:].contiguous())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.detach().item()
        num_batches += 1
        if num_batches % 100 == 0:
            print(f"Epoch {epoch}: Batch {num_batches}, Loss {total_loss / num_batches}")
        
    return total_loss / num_batches

def main():
    from generate_data import create_dataloaders
    import pickle
    from matplotlib import pyplot as plt
    """
    Main training script

    Usage:
        python backbone.py
    """
    # Hyperparameters
    vocab_size = 256  # Discretized action space
    dim = 512  # Model dimension
    num_layers = 4  # Number of transformer blocks
    num_heads = 8  # Number of attention heads
    ff_hidden_dim = 1024  # Feed-forward hidden dimension
    max_seq_len = 1024  # Maximum sequence length
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 10
    use_mla = False # toggle true to train a MLA model

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # TODO: Load dataset
    # Use the generate_data.py script to create synthetic trajectories
    # Load from data/trajectories.pkl
    with open("data/trajectories.pkl", 'rb') as f:
        dataset = pickle.load(f)
    train_loader, _ = create_dataloaders(dataset, batch_size, 1.0)

    model = DecoderOnlyTransformer(vocab_size, dim, num_layers, num_heads, ff_hidden_dim, max_seq_len, use_mla=use_mla).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    losses = []
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        losses.append(train_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}")
    plt.plot(losses)    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Model Loss" if not use_mla else "MLA Model Loss")
    plt.savefig("model_loss.png" if not use_mla else "mla_model_loss.png")

    torch.save(model.state_dict(), "checkpoints/best_model.pt" if not use_mla else "checkpoints/best_model_mla.pt")


if __name__ == "__main__":
    main()
