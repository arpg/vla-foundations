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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# When SINUSOIDAL is False, RoPE embeddings are used and when SINUSOIDAL = True, absolute sinusoidal embeddings are passed to the model.
SINUSOIDAL = False
# Flag to turn ON/OFF the causal masking. when CAUSAL_MASKING = True causal masking is applied.
CAUSAL_MASKING = True

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
        # Initialize a learnable scale parameter of length dim with ones
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
        Returns:
            Normalized tensor of same shape
        """
        # Implementation of RMSNorm
        # Step 1: Compute RMS (root mean square) along the last dimension
        # Step 2: Normalize by dividing x by RMS
        # Step 3: Apply learnable scale parameter
        rms = torch.rsqrt(x.pow(2).mean(dim = -1, keepdim = True) + self.eps)
        xnorm = x* rms
        return xnorm * self.scale

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

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_len=2048):
        super().__init__()
        # Create a positional encoding matrix of shape (max_len, dim)
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)

        # Compute the frequency scaling factor for each even dimension 
        # This controls how fast the sinusoids oscillate at different dimensions
        div_term = torch.exp(
            torch.arange(0, dim, 2) * (-math.log(10000.0) / dim)
        )

        # Apply sine to even and cos to odd indices in the embedding dimension
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # Add positional embeddings to token embeddings
        return x + self.pe[:x.size(1)]

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
        if not SINUSOIDAL:
            self.rope = RotaryPositionalEmbedding(self.head_dim)

    def forward(self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            use_cache: bool = False,
        ):
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Optional attention mask (seq_len, seq_len)
        Returns:
            Output tensor (batch, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape

        # Implementation of Causal Self-Attention

        # Step 1: Project input to Q, K, V
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3*dim)
        # Split into Q, K, V and reshape for multi-head attention
        qkv = qkv.view((batch_size, seq_len, 3, self.num_heads, self.head_dim))
        qkv = qkv.permute(2, 0, 3, 1, 4)

        Q, K, V = qkv[0], qkv[1], qkv[2]

        # # Step 2: Apply RoPE to Q and K
        # Apply RoPE only to current tokens
        if not SINUSOIDAL:
            Q, K = self.rope(Q, K)

        # KV caching
        if past_kv is not None:
            past_K, past_V = past_kv
            K = torch.cat([past_K, K], dim=2)
            V = torch.cat([past_V, V], dim=2)

        present_kv = (K, V) if use_cache else None


        # Step 3: Compute attention scores
        # scores = (Q @ K^T) / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Step 4: Apply causal mask
        # The mask should prevent position i from attending to positions > i
        # Check if cache is used, if not used, then create mask 
        if not use_cache and CAUSAL_MASKING:
            if mask is None:
                mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(mask == 0, -1e9)
        

        # Step 5: Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        try:
            self.last_attn = attn_weights.detach().cpu()
        except Exception:
            self.last_attn = None

        # Step 6: Apply attention to values
        # out = attn_weights @ V
        out = torch.matmul(attn_weights, V)

        # Step 7: Reshape and project back
        # Concatenate heads and apply output projection
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        out = self.out_proj(out)
        out = self.resid_dropout(out)

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

    def __init__(self, dim: int, num_heads: int, ff_hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = CausalSelfAttention(dim, num_heads, dropout)
        self.feed_forward = FeedForward(dim, ff_hidden_dim, dropout)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x, mask=None, past_kv=None, use_cache=False):
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Optional attention mask
        Returns:
            Output tensor (batch, seq_len, dim)
        """
        attn_out, present_kv = self.attention(
            self.norm1(x),
            mask=mask,
            past_kv=past_kv,
            use_cache=use_cache,
        )
        # Pre-norm architecture (norm before attention/FF)
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
        if SINUSOIDAL:
            self.pos_embedding = SinusoidalPositionalEmbedding(dim, max_seq_len)


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
        if SINUSOIDAL:
            x = self.pos_embedding(x)

        # Create causal mask (lower triangular)
        if CAUSAL_MASKING:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        else:
            mask = None

        # Apply transformer blocks
        for block in self.blocks:
            x, _ = block(x, mask)

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
    ) -> torch.Tensor:
        """
        Autoregressive generation with KV caching

        Args:
            input_ids: Starting tokens (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (if None, use full distribution)
        Returns:
            Generated sequence (batch, seq_len + max_new_tokens)
        """
        batch_size = input_ids.size(0)

        # One KV cache per transformer layer
        past_kvs = [None] * len(self.blocks)

        for _ in range(max_new_tokens):
            # Only embed the *new* token
            x = self.token_embedding(input_ids[:, -1:])  # (batch, 1, dim)

            new_past_kvs = []

            # Pass through transformer blocks with KV caching
            for i, block in enumerate(self.blocks):
                x, present_kv = block(
                    x,
                    past_kv=past_kvs[i],
                    use_cache=True,
                )
                new_past_kvs.append(present_kv)

            past_kvs = new_past_kvs

            # Final norm + LM head
            x = self.norm_final(x)
            logits = self.lm_head(x[:, -1, :]) / temperature  # (batch, vocab)

            # Optional top-k sampling
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")

            # Sample next token
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

    # Implementation of training loop
    # For each batch:
    for batch_idx, batch in enumerate(dataloader):
        states, actions = batch

        #   1. Move data to device
        actions = actions.to(device)

        input_ids = actions[:, :-1].long().to(device)
        targets = actions[:, 1:].long().to(device)

        #   2. Forward pass (get logits and loss)
        optimizer.zero_grad(set_to_none = True)
        _, loss = model(input_ids, targets)

        #   3. Backward pass
        loss.backward()

        #   4. Gradient clipping (max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)

        #   5. Optimizer step
        optimizer.step()

        #   6. Zero gradients
        optimizer.zero_grad(set_to_none=True)
        #   7. Accumulate loss
        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % 100 == 0:
            avg = total_loss / num_batches if num_batches > 0 else float('nan')
            print(f"Epoch {epoch + 1} Batch {batch_idx + 1}: avg_loss = {avg:.6f}")

    return total_loss/num_batches if num_batches> 0 else 0.0

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
    num_epochs = 20

    project_root = Path(__file__).resolve().parents[3]
    outputs_dir = project_root / "content" / "course" / "submissions" / "scratch-1" / "images"

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    df = pd.read_pickle('data/trajectories.pkl')
    states = df['states']
    actions = df['actions']
    if isinstance(states, np.ndarray):
        states = torch.tensor(states, dtype=torch.float32)
    if isinstance(actions, np.ndarray):
        actions = torch.tensor(actions, dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(states, actions)
    num_samples = len(dataset)
    num_train = int(0.9 * num_samples)
    num_val = num_samples - num_train
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [num_train, num_val])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Create model
    model = DecoderOnlyTransformer(
        vocab_size,
        dim,
        num_layers,
        num_heads,
        ff_hidden_dim,
        max_seq_len
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    tr_loss_list = []

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}")
        tr_loss_list.append(train_loss)

    # Save checkpoint
    torch.save(model.state_dict(), "checkpoints/best_model.pt")

    # Loss curve plotting
    epochs = list(range(1, len(tr_loss_list) + 1))
    plt.figure()
    plt.plot(epochs, tr_loss_list, label="train")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    loss_path = str(outputs_dir / "loss_curve_sinusoidal.png")
    plt.savefig(loss_path, dpi=150)
    print(f"Saved loss curve to {loss_path}")
    plt.close()

    # Attention visualization
    try:
        model.eval()
        sample_batch = next(iter(val_loader))
        _, sample_actions = sample_batch
        sample_actions = sample_actions.to(device)
        input_ids = sample_actions[:, :-1].long()
        with torch.no_grad():
            _logits, _ = model(input_ids)

        # pick layer and head to visualize
        layer_idx = 0
        head_idx = 0
        attn_tensor = None
        if 0 <= layer_idx < len(model.blocks):
            attn_module = model.blocks[layer_idx].attention
            attn_tensor = getattr(attn_module, "last_attn", None)

        if attn_tensor is None:
            print("No attention stored for visualization.")
        else:
            # attn_tensor shape: (batch, num_heads, seq_len, seq_len)
            attn_img = attn_tensor[0, head_idx].numpy()  # visualize first sample, chosen head
            plt.figure(figsize=(6,6))
            plt.imshow(attn_img, aspect='auto', cmap='viridis')
            plt.colorbar()
            plt.xlabel("Key position")
            plt.ylabel("Query position")
            plt.title(f"Attention heatmap")
            attn_path = str(outputs_dir / "attention_maps_sinusoidal.png")
            plt.tight_layout()
            plt.savefig(attn_path, dpi=150)
            plt.close()
            print(f"Saved attention heatmap to {attn_path}")
    except Exception as e:
        print(f"Could not create attention visualization: {e}")
    
if __name__ == "__main__":
    main()
