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
from pathlib import Path
from typing import Optional, Tuple, List    

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
        # Set this to ones so that RMSNorm is the identity operation at initialization
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
        Returns:
            Normalized tensor of same shape
        """
        # Use rsqrt here instead of 1/sqrt because it's numerically optimal/efficient
        # Take the mean across the last dimension (mean of squared values for each token position)
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
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Optional attention mask (seq_len, seq_len) or (seq_len, total_seq_len) when using cache
            kv_cache: Optional tuple of (cached_keys, cached_values) from previous tokens
                      Shape: (batch, num_heads, cached_seq_len, head_dim)
            use_cache: If True, use and update KV cache (for generation optimization)
        Returns:
            Output tensor (batch, seq_len, dim)
            Updated kv_cache: (new_keys, new_values) with shape (batch, num_heads, total_seq_len, head_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Step 1: Project input to Q, K, V
        # qkv = self.qkv_proj(x)  # (batch, seq_len, 3*dim)
        # Split into Q, K, V and reshape for multi-head attention
        # View allows us to reshape a tensor without allocating new memory for the data
        qkv = self.qkv_proj(x)                                    # (batch, seq, 3*dim)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)        # (batch, seq, 3, heads, head_dim)
        # Use permute instead of transpose to swap all dimensions at once
        qkv = qkv.permute(2, 0, 3, 1, 4)                          # (3, batch, heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]                          # Each: (batch, heads, seq, head_dim)

        # Step 2: Apply RoPE to Q and K
        q, k = self.rope(q, k)

        # Step 3: Concatenate with cache if using cache (for KV cache optimization during generation)
        if use_cache and kv_cache is not None:
            cached_k, cached_v = kv_cache
            # cached_k, cached_v: (batch, num_heads, cached_seq_len, head_dim)
            # k, v: (batch, num_heads, seq_len, head_dim) - usually seq_len=1 for new token
            k = torch.cat([cached_k, k], dim=2)  # Concatenate along sequence dimension
            v = torch.cat([cached_v, v], dim=2)
        
        # Store updated cache for return
        updated_cache = (k, v)

        # Step 4: Compute attention scores
        # Original shape for q, k: batch, num_heads, seq_len, head_dim)
        # Shape of k.transpose(-2, -1): batch, num_heads, head_dim, seq_len)
        # Final shape after the multiplication: (batch, num_heads, seq_len, seq_len)
        # q: (batch, heads, seq_len, head_dim) - usually seq_len=1 during generation with cache
        # k: (batch, heads, total_seq_len, head_dim) - includes cached tokens if cache used
        total_seq_len = k.shape[2]
        scores = (q @ k.transpose(-2, -1)) * self.scale  # (batch, heads, seq_len, total_seq_len)

        # Step 5: Apply causal mask
        # The mask should prevent position i from attending to positions > i
        # Hint: Create a lower-triangular matrix using torch.tril
        # Set masked positions to -inf BEFORE softmax
        # Example: scores = scores.masked_fill(mask == 0, float('-inf'))
        if mask is None:
            if use_cache:
                # Generation mode with cache: new token can attend to all cached positions
                mask = torch.ones(seq_len, total_seq_len, device=q.device)
            else:
                # Training mode: create full causal mask
                # mask is an Optional param so create it if it's not provided
                mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # Step 6: Apply softmax and dropout
        # Apply the softmax across the last dimension to give us the probability distribution over all possible keys
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Step 7: Apply attention to values to get the weighted sum of value vectors
        # Original shape for attn_weights: batch, num_heads, seq_len, seq_len
        # Original shape for v: batch, num_heads, seq_len, head_dim
        out = attn_weights @ v
        # Resulting shape: batch, num_heads, seq_len, head_dim

        # Step 8: Reshape and project back
        # This first step concatenates the heads back together
        # Transpose does not guarantee contiguous memory, but we need that for view to work, so call contiguous first then view
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        # Apply the output projection to get the final output
        out = self.out_proj(out)
        # Apply the residual dropout
        out = self.resid_dropout(out)
        
        return out, updated_cache


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
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Optional attention mask
            kv_cache: Optional KV cache from previous tokens
            use_cache: If True, use and update KV cache
        Returns:
            Output tensor (batch, seq_len, dim)
            Updated kv_cache
        """
        # Pre-norm architecture (norm before attention/FF)
        attn_out, kv_cache = self.attention(self.norm1(x), mask, kv_cache=kv_cache, use_cache=use_cache)
        x = x + attn_out
        x = x + self.feed_forward(self.norm2(x))
        return x, kv_cache


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
        kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ):
        """
        Args:
            input_ids: Input token indices (batch, seq_len)
            targets: Target token indices for training (batch, seq_len)
            kv_cache: Optional list of KV caches, one per layer
                      Each cache is (cached_k, cached_v) with shape (batch, num_heads, cached_seq_len, head_dim)
            use_cache: If True, use and update KV cache (for generation optimization)
        Returns:
            If use_cache is False (training mode):
                (logits, loss) where loss is None if targets not provided
            If use_cache is True (generation mode):
                (logits, loss, updated_kv_cache) where loss is always None during generation
        """
        batch_size, seq_len = input_ids.shape

        # Embed tokens
        x = self.token_embedding(input_ids)  # (batch, seq_len, dim)

        # Create causal mask (only needed for training, not during cached generation)
        mask = None
        if not use_cache:
            # Training mode - use full causal mask
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))

        # Apply transformer blocks
        updated_caches = []
        for idx, block in enumerate(self.blocks):
            if use_cache:
                # Generation mode with cache
                layer_cache = kv_cache[idx] if kv_cache and idx < len(kv_cache) else None
                x, updated_cache = block(x, mask, kv_cache=layer_cache, use_cache=use_cache)
                updated_caches.append(updated_cache)
            else:
                # Training mode - no cache (block still returns tuple for consistency)
                x, _ = block(x, mask, kv_cache=None, use_cache=False)

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

        if use_cache:
            return logits, loss, updated_caches
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
        Autoregressive generation with KV cache optimization

        Args:
            input_ids: Starting tokens (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (if None, use full distribution)
        Returns:
            Generated sequence (batch, seq_len + max_new_tokens)
        """
        # Initialize KV cache as None (will be created on first forward pass)
        kv_cache = None
        
        for _ in range(max_new_tokens):
            # For first token, use full context; for subsequent tokens, only use last token
            if kv_cache is None:
                # First iteration: process entire input context (up to max_seq_len)
                input_context = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]
            else:
                # Subsequent iterations: only process the last generated token
                input_context = input_ids[:, -1:]  # Shape: (batch, 1)
            
            # Forward pass with cache enabled
            logits, _, kv_cache = self.forward(input_context, kv_cache=kv_cache, use_cache=True)
            
            # Get logits for last token
            logits = logits[:, -1, :] / temperature

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

    # Implement the training loop
    for batch_idx, (states, actions) in enumerate(dataloader):
        actions = actions.to(device)

        # We're training for "given everything so far, what's the next action?"
        # input_ids is all batches, include everything except the last token
        # Use .contiguous() because slicing creates non-contiguous tensors
        input_ids = actions[:, :-1].contiguous()

        # For every batch element, take all tokens except the first one
        targets = actions[:, 1:].contiguous()

        logits, loss = model(input_ids, targets)

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Step
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 100 == 0:
            print(
                f"Epoch {epoch+1} | Batch {batch_idx} | Loss {loss.item():.4f}"
            )

    return total_loss / num_batches


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
    max_seq_len = 2048  # Maximum sequence length (matches RoPE default for KV cache benchmarking)
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 10

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use the generate_data.py script to create synthetic trajectories
    # Load from data/trajectories.pkl
    with open('data/trajectories.pkl', 'rb') as f:
        dataset = pickle.load(f)
    
    from generate_data import create_dataloaders
    train_loader, val_loader = create_dataloaders(dataset, batch_size=batch_size)

    model = DecoderOnlyTransformer(vocab_size, dim, num_layers, num_heads, ff_hidden_dim, max_seq_len, dropout=0.1).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    training_loss = []
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}")
        training_loss.append(train_loss)

    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/best_model.pt")
    
    # Save training loss history for visualization
    with open("checkpoints/training_loss.pkl", "wb") as f:
        pickle.dump(training_loss, f)
    print(f"Training loss history saved to checkpoints/training_loss.pkl")


if __name__ == "__main__":
    main()
