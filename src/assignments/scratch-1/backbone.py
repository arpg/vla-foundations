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
import time
import pickle
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
        # Learnable scale parameter 'g' (gamma)
        self.scale = nn.Parameter(torch.ones(dim))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
        Returns:
            Normalized tensor of same shape
        """
        rms = torch.rsqrt(torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)) # permit broadcasting on last dim
        x_normed = x * rms
        x_scaled = x_normed * self.scale

        return x_scaled


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

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to query and key tensors

        Args:
            q: Query tensor (batch, num_heads, seq_len, head_dim)
            k: Key tensor (batch, num_heads, seq_len, head_dim)
        Returns:
            Rotated (q, k) tensors
        """
        seq_len = q.shape[2]

        # Get cached cos/sin values
        cos = self.cos_cached[:seq_len, ...]
        sin = self.sin_cached[:seq_len, ...]

        # Apply rotation: q_rot = q * cos + rotate_half(q) * sin
        q = (q * cos) + (self.rotate_half(q) * sin)
        k = (k * cos) + (self.rotate_half(k) * sin)

        # ALTERNATIVE: Absolute sinusoidal
        # q[::2]  = (q + cos)[::2]
        # q[1::2] = (q + sin)[1::2]
        # k[::2]  = (k + cos)[::2]
        # k[1::2] = (k + sin)[1::2]
        
        return q, k


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
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # KV cache
        self.kvcache = None
        self.cache_enabled = False

        # Rotary embeddings
        self.rope = RotaryPositionalEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Optional attention mask (seq_len, seq_len)
        Returns:
            Output tensor (batch, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape


        # Project input to q, k, v
        # Recall that `reshape` does `view` automatically if it's possible.


        if self.cache_enabled and self.kvcache is not None:
            q = self.q_proj(x).reshape((batch_size, seq_len, self.num_heads, self.head_dim)).transpose(1, 2)

            # Get k,v for only most recent token
            x_recent = x[:, -1:, :]
            kv_recent = self.kv_proj(x_recent)
            kv_recent = kv_recent.reshape((batch_size, 1, self.num_heads, self.head_dim, 2)).transpose(1, 2)

            # Shorten the cache if necessary
            # kv is of shape (batch, seq_len, dim, 2)
            n = x.shape[1]
            kv = torch.cat((self.kvcache, kv_recent), dim=2)[:, :, :seq_len, :, :]
        else:
            q = self.q_proj(x).reshape((batch_size, seq_len, self.num_heads, self.head_dim)).transpose(1, 2)  
            kv = self.kv_proj(x).reshape((batch_size, seq_len, self.num_heads, self.head_dim, 2)).transpose(1, 2)

        if self.cache_enabled:
            self.kvcache = kv
        
        k = kv[:, :, :, :, 0]
        v = kv[:, :, :, :, 1]
        # Now (q, k, v) should all be of shape (batch, num_heads, seq_len, head_dim)


        # Apply RoPE to Q and K
        q, k = self.rope(q, k)

        # Compute attention scores
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim) # (batch, num_heads, seq_len, seq_len)

        # Apply causal mask
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # print(torch.exp(scores))

        # Softmax
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)

        # Dropout
        attn_weights = self.attn_dropout(attn_weights)

        # Apply values
        out = attn_weights @ v # (batch, num_heads, seq_len, head_dim)

        # Concatenate heads and apply output projection
        # Heads may or may not be contiguous - reshape should handle both cases
        return out.transpose(1, 2).reshape((batch_size, seq_len, self.dim))

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

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Optional attention mask
        Returns:
            Output tensor (batch, seq_len, dim)
        """
        # Pre-norm architecture (norm before attention/FF)
        x = x + self.attention(self.norm1(x), mask)
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
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len

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
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
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
        use_cache: bool = False
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

        # KV caching: Use only when autoregressively generating (ie, right here)
        #   Also, clear the cache on every call to `generate`.
        if use_cache:
            for block in self.blocks:
                block.attention.cache_enabled = True
                block.attention.kvcache = None

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

        if use_cache:
            for block in self.blocks:
                block.attention.cache_enabled = False

        return input_ids


def train_epoch(
    model: DecoderOnlyTransformer,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
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

    for data in dataloader:
        batch = data[0].to(device)
        logits, loss = model(batch[:, 0:48], batch[:, 1:49])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss
        num_batches += 1

    return total_loss / num_batches


def test_performance(
    model: DecoderOnlyTransformer,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> float:
    """
    Test loss on a test dataset

    Args:
        model: The transformer model
        dataloader: Test data loader
        device: Device to test on
    Returns:
        Average loss for the test data
    """

    model.eval()
    total_loss = 0.0
    num_batches = 0

    for data in dataloader:
        batch = data[0].to(device)
        logits, loss = model(batch[:, 0:48], batch[:, 1:49])

        total_loss += loss
        num_batches += 1
    return total_loss / num_batches


def test_inference_time(
    model: DecoderOnlyTransformer,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    use_cache: bool = False
) -> float:
    """
    Test inference time on the test dataset

    Args:
        model: The transformer model
        dataloader: test data loader
        device: Device to test on
    Returns:
        Average inference time for 10 steps of generation (with batching allowed) 
    """

    total_time = 0
    num_samples = 0

    for data in dataloader:
        batch = data[0].to(device)

        ago = time.perf_counter()
        model.generate(batch, 100, use_cache=use_cache)
        total_time += time.perf_counter() - ago
        num_samples += 10

    return total_time / num_samples


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
    batch_size = 64 # "if you have enough VRAM"
    learning_rate = 1e-4
    num_epochs = 40

    # Set up device (CUDA used in development).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load synthetic data from data/trajectories.pkl
    with open("data/trajectories.pkl", 'rb') as f:
        train_dataset = torch.utils.data.TensorDataset(pickle.load(f)['actions'][:9000, :])

    with open("data/trajectories.pkl", 'rb') as f:
        test_dataset = torch.utils.data.TensorDataset(pickle.load(f)['actions'][9000:, :])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, shuffle=True)

    model = DecoderOnlyTransformer(
        vocab_size,
        dim,
        num_layers,
        num_heads,
        ff_hidden_dim,
        max_seq_len = max_seq_len,
        dropout = 0.1,
    ).to(device)

    # If needed: Load model parameters from checkpoint
    # model.load_state_dict(torch.load("checkpoints/epoch_40.pt", weights_only=False))

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Train for many epochs
    for epoch in range(num_epochs):
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)

        print(f"Epoch {epoch+1}/{num_epochs} - Training loss: {train_loss:.4f}")

        torch.save(model.state_dict(), f"checkpoints/epoch_{epoch+1}.pt")

    # Test: Loss and inference time
    model.eval()
    test_loss = test_performance(model, test_loader, device)
    print(f"Testing loss: {test_loss:.4f}")
    test_time = test_inference_time(model, test_loader, device, use_cache=True)
    print(f"Inference time: {test_time:.4f}")


if __name__ == "__main__":
    main()
