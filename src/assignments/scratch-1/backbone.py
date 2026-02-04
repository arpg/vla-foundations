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
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
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
        self.scale = nn.Parameter(torch.ones(dim))

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
        return x * rms * self.scale


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
            seq_start_pos: Starting position index for the sequence
        Returns:
            Rotated (q, k) tensors
        """
        seq_len = q.shape[2]
        end_pos = seq_start_pos + seq_len

        # Get cached cos/sin values for the specific positions
        cos = self.cos_cached[seq_start_pos:end_pos, ...]
        sin = self.sin_cached[seq_start_pos:end_pos, ...]

        # Expand for batch and num_heads: (seq_len, head_dim) -> (1, 1, seq_len, head_dim)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]

        # Apply rotation: q_rot = q * cos + rotate_half(q) * sin
        q_rot = (q * cos) + (self.rotate_half(q) * sin)
        k_rot = (k * cos) + (self.rotate_half(k) * sin)

        return q_rot, k_rot


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Standard Sinusoidal Positional Embedding
    """

    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_seq_len, dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
        Returns:
            x + positional_encoding
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :]


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention with RoPE

    Key constraints:
    - Token at position t can only attend to positions <= t
    - Uses RoPE instead of absolute positional embeddings
    """

    def __init__(
        self, dim: int, num_heads: int, dropout: float = 0.1, use_rope: bool = True
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.use_rope = use_rope

        # Linear projections for Q, K, V
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Rotary embeddings
        if self.use_rope:
            self.rope = RotaryPositionalEmbedding(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Optional attention mask (seq_len, seq_len)
            past_kv: Key/Value cache from previous step
        Returns:
            Output tensor (batch, seq_len, dim)
            present_kv: New KV cache
        """
        batch_size, seq_len, _ = x.shape

        # Step 1: Project input to Q, K, V
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3*dim)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Handle KV Cache
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v)

        # Step 2: Apply RoPE to Q and K
        if self.use_rope:
            # If using cache, we need to offset the position for RoPE
            seq_start_pos = past_kv[0].shape[2] if past_kv is not None else 0
            q, k = self.rope(q, k, seq_start_pos=seq_start_pos)

        # Step 3: Compute attention scores
        # q: (batch, n_heads, seq_len_q, head_dim)
        # k: (batch, n_heads, seq_len_k, head_dim)
        # For cached inference: seq_len_q=1, seq_len_k=context_len
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Step 4: Apply causal mask
        if mask is not None:
            # During generation with cache:
            # scores shape: (batch, heads, 1, total_seq_len)
            # mask shape should align.
            # If standard forward pass (no cache logic or full sequence), standard mask works.
            # For 1-token generation with cache, we attend to all previous tokens (mask is all 1s effectively, usually handled by shape)

            # If mask is provided and shapes mismatch (e.g. during generation), we might need to slice
            if mask.shape[-2] == seq_len and mask.shape[-1] == k.shape[2]:
                scores = scores.masked_fill(mask == 0, float("-inf"))
            elif (
                past_kv is None
            ):  # Only apply standard triangular mask during training/full forward
                scores = scores.masked_fill(mask == 0, float("-inf"))

        # Step 5: Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Step 6: Apply attention to values
        out = attn_weights @ v

        # Step 7: Reshape and project back
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        out = self.resid_dropout(self.out_proj(out))

        return out, present_kv


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

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_hidden_dim: int,
        dropout: float = 0.1,
        use_rope: bool = True,
    ):
        super().__init__()
        self.attention = CausalSelfAttention(dim, num_heads, dropout, use_rope=use_rope)
        self.feed_forward = FeedForward(dim, ff_hidden_dim, dropout)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Optional attention mask
            past_kv: Previous KV cache
        Returns:
            Output tensor (batch, seq_len, dim)
            present_kv: New KV cache
        """
        # Pre-norm architecture (norm before attention/FF)
        norm_x = self.norm1(x)
        attn_out, present_kv = self.attention(norm_x, mask, past_kv)
        x = x + attn_out
        x = x + self.feed_forward(self.norm2(x))
        return x, present_kv


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
        use_rope: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, dim)

        # Positional embedding (if not using RoPE)
        if not use_rope:
            self.pos_embedding = SinusoidalPositionalEmbedding(dim, max_seq_len)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim, num_heads, ff_hidden_dim, dropout, use_rope=use_rope
                )
                for _ in range(num_layers)
            ]
        )

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
        elif isinstance(module, SinusoidalPositionalEmbedding):
            pass  # No weights to initialize

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_kv: Optional[list] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[list]]:
        """
        Args:
            input_ids: Input token indices (batch, seq_len)
            targets: Target token indices for training (batch, seq_len)
            past_kv: List of KV caches for each layer
            use_cache: whether to use KV caching
        Returns:
            logits: Output logits (batch, seq_len, vocab_size)
            loss: Cross-entropy loss if targets provided, else None
            present_kv: List of new KV caches
        """
        batch_size, seq_len = input_ids.shape

        # Embed tokens
        x = self.token_embedding(input_ids)  # (batch, seq_len, dim)

        # Add positional embeddings if not using RoPE
        if not self.use_rope:
            x = self.pos_embedding(x)

        # Create causal mask (lower triangular)
        # If using cache (generating one token), we don't need a mask usually or it's implicitly handled
        mask = None
        if not use_cache or past_kv is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))

        # Apply transformer blocks
        present_kv = []
        for i, block in enumerate(self.blocks):
            layer_past = past_kv[i] if past_kv is not None else None
            x, layer_present = block(x, mask, past_kv=layer_past)
            if use_cache:
                present_kv.append(layer_present)

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

        return logits, loss, present_kv if use_cache else None

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        use_cache: bool = True,
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
        past_kv = None

        for _ in range(max_new_tokens):
            # Crop context if too long (only if NOT using cache or if cache refill logic handles it)
            # If using cache, we only pass in the last token anyway, so context length limits are handled
            # by the cache size implicitly or we essentially just run indefinitely until OOM/end.
            # But RoPE has limit `max_seq_len`.

            if use_cache and past_kv is not None:
                # Only pass the last token
                input_context = input_ids[:, -1:]
            else:
                # Standard full context
                input_context = (
                    input_ids
                    if input_ids.size(1) <= self.max_seq_len
                    else input_ids[:, -self.max_seq_len :]
                )

            # Forward pass
            logits, _, past_kv = self.forward(
                input_context, past_kv=past_kv, use_cache=use_cache
            )
            logits = logits[:, -1, :] / temperature  # Get last token logits

            # Optional top-k sampling
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("Inf")

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


def train_epoch(
    model: DecoderOnlyTransformer,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Tuple[float, list]:
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
    step_losses = []

    # TODO: Implement training loop
    # For each batch:
    for batch_idx, batch in enumerate(dataloader):

        #   1. Move data to device
        sequences = batch.to(device)
        inputs = sequences[:, :-1].contiguous()
        targets = sequences[:, 1:].contiguous()

        #   2. Forward pass (get logits and loss)
        optimizer.zero_grad()
        logits, loss, _ = model(inputs, targets)

        #   3. Backward pass
        loss.backward()

        #   4. Gradient clipping (max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        #   5. Optimizer step
        optimizer.step()

        #   6. Zero gradients
        # (already done above before forward pass)

        #   7. Accumulate loss
        total_loss += loss.item()
        step_losses.append(loss.item())
        num_batches += 1

        # Save checkpoint every 1000 steps
        step = epoch * len(dataloader) + batch_idx + 1
        if step % 1000 == 0:
            torch.save(model.state_dict(), f"checkpoints/step_{step}.pt")

        # Hint: Use torch.nn.utils.clip_grad_norm_ for gradient clipping
        # Hint: Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            avg_loss = total_loss / num_batches
            perplexity = math.exp(avg_loss)
            print(
                f"Epoch {epoch}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}"
            )

    return total_loss / num_batches, step_losses


def main():
    """
    Main training script

    Usage:
        python backbone.py
    """
    # Hyperparameters
    vocab_size = 256  # Discretized action space
    dim = 256  # Model dimension
    num_layers = 4  # Number of transformer blocks
    num_heads = 8  # Number of attention heads
    ff_hidden_dim = 1024  # Feed-forward hidden dimension
    max_seq_len = 50  # Maximum sequence length
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 10

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # TODO: Load dataset
    # Use the generate_data.py script to create synthetic trajectories
    # Load from data/trajectories.pkl
    with open("data/trajectories.pkl", "rb") as f:
        trajectories = pickle.load(f)
    dataset = trajectories["actions"].clone().detach().long()
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    # TODO: Create model
    # model = DecoderOnlyTransformer(...)
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_hidden_dim=ff_hidden_dim,
        max_seq_len=max_seq_len,
    ).to(device)

    # TODO: Create optimizer
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # TODO: Training loop
    # for epoch in range(num_epochs):
    #     train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
    #     print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}")
    os.makedirs("checkpoints", exist_ok=True)
    all_losses = []
    for epoch in range(num_epochs):
        train_loss, step_losses = train_epoch(
            model, dataloader, optimizer, device, epoch
        )
        all_losses.extend(step_losses)
        perplexity = math.exp(train_loss)
        print(
            f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}, Perplexity: {perplexity:.4f}"
        )
        torch.save(model.state_dict(), f"checkpoints/epoch_{epoch+1}.pt")

    # Save losses for plotting
    with open("checkpoints/losses.pkl", "wb") as f:
        pickle.dump(all_losses, f)

    # TODO: Save checkpoint
    # torch.save(model.state_dict(), "checkpoints/best_model.pt")
    torch.save(model.state_dict(), "checkpoints/best_model.pt")
    print("Training complete")

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(all_losses)
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig("training_loss.png")


if __name__ == "__main__":
    main()
