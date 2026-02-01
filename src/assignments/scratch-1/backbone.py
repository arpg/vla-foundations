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
import pickle
import math
from typing import Optional, Tuple

from torch.utils.data import DataLoader, Dataset

from visualizer import Visualizer
from pathlib import Path


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
        # rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * rms
        x = x * self.scale
        return x

        # raise NotImplementedError("TODO: Implement RMSNorm forward pass")


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
    """
    Classic Sinusoidal Positional Embedding from "Attention Is All You Need"
    
    Paper: Vaswani et al., 2017 - https://arxiv.org/abs/1706.03762
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        pe = torch.zeros(max_seq_len, dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(base) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_seq_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token embeddings (batch, seq_len, dim)
        Returns:
            x + positional encoding (batch, seq_len, dim)
        """
        return x + self.pe[:, :x.size(1), :]



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

        # Linear projections for Q, K, V
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Rotary embeddings
        self.rope = RotaryPositionalEmbedding(self.head_dim) if use_rope else None

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Optional attention mask (seq_len, seq_len)
            return_attention: Whether to return attention weights
        Returns:
            Output tensor (batch, seq_len, dim)
            Attention weights (batch, num_heads, seq_len, seq_len) if return_attention=True
        """
        batch_size, seq_len, _ = x.shape

        # TODO: Implement Causal Self-Attention

        # Step 1: Project input to Q, K, V
        # qkv = self.qkv_proj(x)  # (batch, seq_len, 3*dim)
        # Split into Q, K, V and reshape for multi-head attention
        # Hint: Use .view() and .transpose() to get shape (batch, num_heads, seq_len, head_dim)

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1) # q -> (batch, seq_len, dim)...
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim) # (batch, seq_len, num_heads, head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2) # (batch, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Step 2: Apply RoPE to Q and K
        if self.rope is not None:
            q, k = self.rope(q, k)

        # Step 3: Compute attention scores
        # scores = (Q @ K^T) / sqrt(d_k)
        # Hint: Use torch.matmul or @ operator
        # Shape should be (batch, num_heads, seq_len, seq_len)
        scores = (q @ k.transpose(-2, -1)) * self.scale # (batch, num_heads, seq_len, seq_len)

        # Step 4: Apply causal mask
        # The mask should prevent position i from attending to positions > i
        # Hint: Create a lower-triangular matrix using torch.tril
        # Set masked positions to -inf BEFORE softmax
        # Example: scores = scores.masked_fill(mask == 0, float('-inf'))
        tril = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        scores = scores.masked_fill(tril == 0, float('-inf'))

        # Step 5: Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights_dropped = self.attn_dropout(attn_weights)

        # Step 6: Apply attention to values
        out = attn_weights_dropped @ v

        # Step 7: Reshape and project back
        # Concatenate heads and apply output projection
        # Hint: Use .transpose() and .contiguous().view() to reshape
        # out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        out = out.transpose(1, 2) # (batch, seq_len, num_heads, head_dim)
        out = out.contiguous().view(batch_size, seq_len, self.dim) # (batch, seq_len, dim)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        
        if return_attention:
            return out, attn_weights
        return out, None


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

    def __init__(self, dim: int, num_heads: int, ff_hidden_dim: int, dropout: float = 0.1, use_rope: bool = True):
        super().__init__()
        self.attention = CausalSelfAttention(dim, num_heads, dropout, use_rope)
        self.feed_forward = FeedForward(dim, ff_hidden_dim, dropout)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Optional attention mask
            return_attention: Whether to return attention weights
        Returns:
            Output tensor (batch, seq_len, dim)
            Attention weights if return_attention=True
        """
        # Pre-norm architecture (norm before attention/FF)
        attn_out, attn_weights = self.attention(self.norm1(x), mask, return_attention)
        x = x + attn_out
        x = x + self.feed_forward(self.norm2(x))
        return x, attn_weights


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

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, dim)

        # Matery: Sinudoidal Positional Embedding Ablation
        if not use_rope:
            self.positional_embedding = SinusoidalPositionalEmbedding(dim, max_seq_len)
        else:
            self.positional_embedding = None

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, ff_hidden_dim, dropout, use_rope)
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
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[list]]:
        """
        Args:
            input_ids: Input token indices (batch, seq_len)
            targets: Target token indices for training (batch, seq_len)
            return_attention: Whether to return attention weights from all layers
        Returns:
            logits: Output logits (batch, seq_len, vocab_size)
            loss: Cross-entropy loss if targets provided, else None
            attention_weights: List of attention weights from each layer if return_attention=True
        """
        batch_size, seq_len = input_ids.shape

        # Embed tokens
        x = self.token_embedding(input_ids)  # (batch, seq_len, dim)

        # Matery: Sinudoidal Positional Embedding Ablation
        if self.positional_embedding is not None:
            x = self.positional_embedding(x) # (batch, seq_len, dim)

        # Create causal mask (lower triangular)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))

        # Apply transformer blocks
        all_attention_weights = [] if return_attention else None
        for block in self.blocks:
            x, attn_weights = block(x, mask, return_attention)
            if return_attention and attn_weights is not None:
                all_attention_weights.append(attn_weights)

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
                ignore_index=-1,
            )

        return logits, loss, all_attention_weights

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
            logits, _, _ = self.forward(input_context)
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

class TrajectoryDataset(Dataset):
    def __init__(self, path: str):
        self.samples = pickle.load(open(path, "rb"))
        self.states = self.samples['states']
        self.actions = self.samples['actions']

    def __len__(self):
        return self.actions.shape[0]
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

def create_dataloaders(dataset: TrajectoryDataset, batch_size: int = 32, train_split: float = 0.9):
    num_samples = len(dataset)
    indices = torch.randperm(num_samples)
    split = int(num_samples * train_split)
    train_indices = indices[:split]
    val_indices = indices[split:]
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

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

    # TODO: Implement training loop
    # For each batch:
    #   1. Move data to device
    #   2. Forward pass (get logits and loss)
    #   3. Backward pass
    #   4. Gradient clipping (max_norm=1.0)
    #   5. Optimizer step
    #   6. Zero gradients
    #   7. Accumulate loss

    for batch in dataloader:
        actions = batch[1]
        input_ids = actions[:, :-1].to(device) # Gets the preceding actions
        targets = actions[:, 1:].to(device) # Gets the subsequent actions
        # print(input_ids.shape, targets.shape)
        logits, loss, _ = model(input_ids, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step() # update model parameters
        optimizer.zero_grad() # reset gradients
        total_loss += loss.item()
        num_batches += 1

    # Hint: Use torch.nn.utils.clip_grad_norm_ for gradient clipping
    # Hint: Print progress every 100 batches
    return total_loss / num_batches if num_batches else 0.0

    # raise NotImplementedError("TODO: Implement training loop")


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
    learning_rate = 1e-3
    num_epochs = 10
    dropout = 0.1
    use_rope = False

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create checkpoint directory
    Path("checkpoints").mkdir(exist_ok=True)
    
    # Initialize visualizer
    visualizer = Visualizer(save_dir="visualizations")

    # Load dataset
    dataset = TrajectoryDataset(path="data/trajectories.pkl")
    train_loader, val_loader = create_dataloaders(dataset, batch_size=batch_size, train_split=0.9)

    # TODO: Create model
    # model = DecoderOnlyTransformer(...)
    model = DecoderOnlyTransformer(vocab_size=vocab_size, dim=dim, num_layers=num_layers, num_heads=num_heads, ff_hidden_dim=ff_hidden_dim, max_seq_len=max_seq_len, dropout=dropout, use_rope=use_rope)
    model.to(device)

    # TODO: Create optimizer
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01) # Added weight decay to prevent overfitting

    # TODO: Training loop
    # for epoch in range(num_epochs):
    #     train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
    #     print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}")
    best_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        visualizer.add_loss(train_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}")
        
        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), "checkpoints/best_model.pt")
            print(f"  → Saved best model (loss: {best_loss:.4f})")

    # Save final checkpoint
    # TODO: Save checkpoint
    torch.save(model.state_dict(), "checkpoints/final_model.pt")
    
    # Generate and save loss curve
    visualizer.visualize_loss()
    print("\nLoss curve saved to visualizations/loss_curve.png")
    print("Run 'python visualize_attention.py' to visualize attention maps")


if __name__ == "__main__":
    main()
