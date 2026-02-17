"""
Scratch-1: The Transformer Backbone
CSCI 7000 - VLA Foundations

Template for implementing a decoder-only Transformer from scratch.
Students must complete the TODO sections for:
1. CausalSelfAttention
2. RMSNorm
3. Training loop
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Union

from generate_data import create_dataloaders, generate_dataset

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
        # Learnable scale parameter (gamma)
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
        Returns:
            Normalized tensor of same shape
        """
        # RMS = sqrt(mean(x^2) + eps); use rsqrt for efficiency: 1/sqrt(...)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # Normalize: x / RMS, then apply learned scale
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

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, position_offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to query and key tensors

        Args:
            q: Query tensor (batch, num_heads, seq_len, head_dim)
            k: Key tensor (batch, num_heads, seq_len, head_dim)
            position_offset: For KV-cache: position of first token in q/k (default 0)
        Returns:
            Rotated (q, k) tensors
        """
        seq_len = q.shape[2]
        # Get cached cos/sin for the correct positions (supports KV-cache incremental decoding)
        cos = self.cos_cached[position_offset : position_offset + seq_len, ...]
        sin = self.sin_cached[position_offset : position_offset + seq_len, ...]
        q_rot = (q * cos) + (self.rotate_half(q) * sin)
        k_rot = (k * cos) + (self.rotate_half(k) * sin)
        return q_rot, k_rot


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embeddings (Vaswani et al., 2017).
    Used in original Transformer; add to input embeddings.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_seq_len, dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        """Add sinusoidal PE to input (batch, seq_len, dim). Use position_offset for KV-cache."""
        return x + self.pe[:, position_offset : position_offset + x.size(1), :]


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention with RoPE

    Key constraints:
    - Token at position t can only attend to positions <= t
    - Uses RoPE instead of absolute positional embeddings
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1, use_rope: bool = True):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_rope = use_rope

        # Linear projections for Q, K, V
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Rotary embeddings (only when use_rope=True)
        self.rope = RotaryPositionalEmbedding(self.head_dim) if use_rope else None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]],
    ]:
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Optional attention mask (seq_len, seq_len)
            return_attn_weights: If True, return (output, attn_weights)
            past_key_value: Cached (k, v) from previous step for KV-cache (batch, num_heads, past_len, head_dim)
            use_cache: If True and past_key_value provided, return updated cache
        Returns:
            output; or (output, attn_weights); or (output, present_key_value) when use_cache
        """
        batch_size, seq_len, _ = x.shape
        past_len = past_key_value[0].shape[2] if past_key_value is not None else 0
        position_offset = past_len

        # Step 1: Project input to Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Step 2: Apply RoPE to Q and K (with position offset for incremental decoding)
        if self.rope is not None:
            q, k = self.rope(q, k, position_offset=position_offset)

        # KV-cache: concatenate past K, V with new K, V
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_key_value = (k, v) if use_cache else None
        total_len = k.shape[2]

        # Step 3: Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Step 4: Apply causal mask
        if mask is not None:
            causal_mask = mask
        else:
            if past_len > 0:
                # Incremental: single query attends to all keys (all in past)
                causal_mask = torch.ones(seq_len, total_len, device=x.device)
            else:
                causal_mask = torch.tril(torch.ones(seq_len, total_len, device=x.device))
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))

        # Step 5: Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Step 6: Apply attention to values
        out = torch.matmul(attn_weights, v)

        # Step 7: Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        out = self.resid_dropout(self.out_proj(out))

        if return_attn_weights:
            return out, attn_weights.detach()
        if use_cache:
            return out, present_key_value
        return out


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
        self, dim: int, num_heads: int, ff_hidden_dim: int, dropout: float = 0.1, use_rope: bool = True
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
        return_attn_weights: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]],
    ]:
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Optional attention mask
            return_attn_weights: If True, return (output, attn_weights)
            past_key_value: KV-cache from previous step
            use_cache: If True, return (output, present_key_value)
        """
        attn_out = self.attention(
            self.norm1(x), mask,
            return_attn_weights=return_attn_weights,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        if use_cache:
            attn_out, present_key_value = attn_out
        elif return_attn_weights:
            attn_out, attn_weights = attn_out
        x = x + attn_out
        x = x + self.feed_forward(self.norm2(x))
        if use_cache:
            return x, present_key_value
        if return_attn_weights:
            return x, attn_weights
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
        position_encoding: str = "rope",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.position_encoding = position_encoding
        use_rope = position_encoding == "rope"

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, dim)

        # Positional encoding (sinusoidal added to input; RoPE is applied in attention)
        self.pos_embedding = (
            SinusoidalPositionalEmbedding(dim, max_seq_len)
            if position_encoding == "sinusoidal"
            else None
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, ff_hidden_dim, dropout, use_rope=use_rope)
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
        return_attn_weights: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor],
    ]:
        """
        Args:
            input_ids: Input token indices (batch, seq_len)
            targets: Target token indices for training (batch, seq_len)
            return_attn_weights: If True, return attention from first block
        Returns:
            logits, loss; or (logits, loss, attn_weights) if return_attn_weights
        """
        batch_size, seq_len = input_ids.shape

        # Embed tokens
        x = self.token_embedding(input_ids)  # (batch, seq_len, dim)
        if self.pos_embedding is not None:
            x = self.pos_embedding(x)

        # Create causal mask (lower triangular)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))

        # Apply transformer blocks
        attn_weights = None
        for i, block in enumerate(self.blocks):
            get_attn = return_attn_weights and (i == 0)
            out = block(x, mask, return_attn_weights=get_attn)
            if get_attn:
                x, attn_weights = out
            else:
                x = out

        # Final norm and projection
        x = self.norm_final(x)
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )

        if return_attn_weights:
            return logits, loss, attn_weights
        return logits, loss

    def forward_with_cache(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with KV-cache for efficient autoregressive generation.

        Args:
            input_ids: Token indices (batch, seq_len) - seq_len=1 when using cache
            past_key_values: List of (k, v) per layer from previous step
        Returns:
            logits: (batch, seq_len, vocab_size)
            past_key_values: Updated cache for next step
        """
        batch_size, seq_len = input_ids.shape
        past_len = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        x = self.token_embedding(input_ids)
        if self.pos_embedding is not None:
            x = self.pos_embedding(x, position_offset=past_len)
        mask = None if past_key_values is not None else torch.tril(
            torch.ones(seq_len, seq_len, device=x.device)
        )
        new_past_key_values: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, present_kv = block(x, mask, return_attn_weights=False,
                                  past_key_value=past_kv, use_cache=True)
            new_past_key_values.append(present_kv)

        x = self.norm_final(x)
        logits = self.lm_head(x)
        return logits, new_past_key_values

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
        Autoregressive generation. Supports KV-cache for efficient inference.

        Args:
            input_ids: Starting tokens (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (if None, use full distribution)
            use_cache: If True, use KV-cache for faster generation
        Returns:
            Generated sequence (batch, seq_len + max_new_tokens)
        """
        if use_cache:
            return self._generate_with_cache(input_ids, max_new_tokens, temperature, top_k)
        return self._generate_without_cache(input_ids, max_new_tokens, temperature, top_k)

    def _generate_without_cache(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: Optional[int],
    ) -> torch.Tensor:
        """Generate without KV-cache (recomputes full context each step)."""
        for _ in range(max_new_tokens):
            input_context = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]
            logits, _ = self.forward(input_context)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids

    def _generate_with_cache(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: Optional[int],
    ) -> torch.Tensor:
        """Generate with KV-cache (efficient incremental decoding)."""
        input_context = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]
        logits, past_key_values = self.forward_with_cache(input_context)
        logits = logits[:, -1, :]
        for _ in range(max_new_tokens):
            if logits.dim() == 2:
                logits = logits.unsqueeze(1)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if _ < max_new_tokens - 1:
                logits, past_key_values = self.forward_with_cache(next_token, past_key_values)
        return input_ids


def train_epoch(
    model: DecoderOnlyTransformer,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    global_step: int = 0,
    checkpoint_dir: Optional[Path] = None,
) -> Tuple[float, int]:
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
    step_losses: list = []

    for batch_idx, batch in enumerate(dataloader):
        # Batch is (states, actions) from create_dataloaders
        _, actions = batch
        actions = actions.to(device)

        # Next-token prediction: at position t predict token t+1
        # input_ids = full sequence, targets = shifted (targets[t] = input[t+1]), last position ignored
        input_ids = actions
        targets = torch.cat([
            actions[:, 1:],
            torch.full((actions.shape[0], 1), -1, device=device, dtype=torch.long)
        ], dim=1)

        optimizer.zero_grad()
        _, loss = model(input_ids, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        step_losses.append((global_step + 1, loss.item()))

        global_step += 1
        if checkpoint_dir is not None and global_step % 1000 == 0:
            ckpt_path = checkpoint_dir / f"checkpoint_step_{global_step}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved checkpoint at step {global_step}")

        if (batch_idx + 1) % 100 == 0:
            avg_loss = total_loss / num_batches
            perplexity = math.exp(min(avg_loss, 20))
            print(f"  Epoch {epoch+1} - Batch {batch_idx+1}/{len(dataloader)} - Loss: {loss.item():.4f} - Avg: {avg_loss:.4f} - Perplexity: {perplexity:.2f}")

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss, global_step, step_losses


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

    # Load or generate dataset
    data_path = Path(__file__).parent / "data" / "trajectories.pkl"
    if data_path.exists():
        print(f"Loading dataset from {data_path}")
        with open(data_path, "rb") as f:
            dataset = pickle.load(f)
    else:
        print("Dataset not found, generating...")
        dataset = generate_dataset(num_trajectories=10000, seq_length=50, seed=42)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(data_path, "wb") as f:
            pickle.dump(dataset, f)

    train_loader, val_loader = create_dataloaders(
        dataset, batch_size=batch_size, train_split=0.9
    )

    # Create model
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_hidden_dim=ff_hidden_dim,
        max_seq_len=max_seq_len,
        dropout=0.1,
    )
    model = model.to(device)

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Checkpoint directory
    checkpoint_dir = Path(__file__).parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Training loop
    global_step = 0
    best_loss = float("inf")
    all_step_losses: list = []
    for epoch in range(num_epochs):
        train_loss, global_step, step_losses = train_epoch(
            model, train_loader, optimizer, device, epoch,
            global_step=global_step, checkpoint_dir=checkpoint_dir
        )
        all_step_losses.extend(step_losses)
        perplexity = math.exp(min(train_loss, 20))
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f} - Perplexity: {perplexity:.2f}")

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
            print(f"  Saved best model (loss: {best_loss:.4f})")

    # Save training log for visualization
    log_path = Path(__file__).parent / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(
            [{"step": s, "loss": l} for s, l in all_step_losses],
            f, indent=2
        )
    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to {checkpoint_dir}")
    print(f"Training log saved to {log_path}")


if __name__ == "__main__":
    main()
