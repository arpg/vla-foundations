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
from torch.utils.data import DataLoader, TensorDataset

import os
import pickle
import csv
import argparse  

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

        mean_square = x.pow(2).mean(dim = -1, keepdim = True)
        inv_rms = torch.rsqrt(mean_square + self.eps)

        # raise NotImplementedError("TODO: Implement RMSNorm forward pass")
        return x * inv_rms * self.scale


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
        cos = self.cos_cached[:seq_len, :]
        sin = self.sin_cached[:seq_len, :]

        # Explicit broadcast to (1, 1, T, D)
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1,1,T,D)
        sin = sin.unsqueeze(0).unsqueeze(0)

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

        self.last_attn = None

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Optional attention mask (seq_len, seq_len)
        Returns:
            Output tensor (batch, seq_len, dim)
        """
        B, T, C = x.shape

        # TODO: Implement Causal Self-Attention

        # Step 1: Project input to Q, K, V
        # qkv = self.qkv_proj(x)  # (batch, seq_len, 3*dim)
        # Split into Q, K, V and reshape for multi-head attention
        # Hint: Use .view() and .transpose() to get shape (batch, num_heads, seq_len, head_dim)
        qkv = self.qkv_proj(x)

        q, k, v = qkv.chunk(3, dim = -1)

        H, D = self.num_heads, self.head_dim
        q = q.view(B, T, H, D).transpose(1, 2)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        # Step 2: Apply RoPE to Q and K
        # q, k = self.rope(q, k)
        q, k = self.rope(q, k)

        # Step 3: Compute attention scores
        # scores = (Q @ K^T) / sqrt(d_k)
        # Hint: Use torch.matmul or @ operator
        # Shape should be (batch, num_heads, seq_len, seq_len)
        scores = (q @ k.transpose(-2, -1)) * self.scale

        # Step 4: Apply causal mask
        # The mask should prevent position i from attending to positions > i
        # Hint: Create a lower-triangular matrix using torch.tril
        # Set masked positions to -inf BEFORE softmax
        # Example: scores = scores.masked_fill(mask == 0, float('-inf'))
        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.bool()

            # (T,T) -> (1,1,T,T)
            mask = mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(~mask, float("-inf"))

        # Step 5: Apply softmax and dropout
        # attn_weights = F.softmax(scores, dim=-1)
        # attn_weights = self.attn_dropout(attn_weights)
        attn = F.softmax(scores, dim = -1)
        attn = self.attn_dropout(attn)

        # Save attention map for visualization (no grad)
        self.last_attn = attn.detach()

        # Step 6: Apply attention to values
        # out = attn_weights @ V
        out = attn @ v

        # Step 7: Reshape and project back
        # Concatenate heads and apply output projection
        # Hint: Use .transpose() and .contiguous().view() to reshape
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.out_proj(out)
        out = self.resid_dropout(out)

        return out

        # raise NotImplementedError("TODO: Implement CausalSelfAttention forward pass")


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

    def forward(self, 
                input_ids, 
                targets=None, *, 
                use_causal_mask: bool = True):

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
        mask = None
        if use_causal_mask:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))

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


def train_epoch(
    model,
    dataloader,
    optimizer,
    device,
    epoch: int,
    *,
    ckpt_dir: str = "checkpoints",
    global_step: int = 0,
    save_every: int = 1000,
    log_every: int = 100,
    use_causal_mask: bool = True,
    loss_csv: str = "checkpoints/step_loss.csv",
) -> tuple[float, int]:

    model.train()
    os.makedirs(ckpt_dir, exist_ok=True)

    # create step loss csv header once
    if global_step == 0 and (not os.path.exists(loss_csv)):
        os.makedirs(os.path.dirname(loss_csv), exist_ok=True)
        with open(loss_csv, "w", newline="") as f:
            csv.writer(f).writerow(["step", "loss"])

    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        tokens = batch[0].to(device) if isinstance(batch, (tuple, list)) else batch.to(device)

        x = tokens[:, :-1].contiguous()
        y = tokens[:, 1:].contiguous()

        logits, loss = model(x, targets=None, use_causal_mask=use_causal_mask)

        if loss is None:
            B, Tm1, V = logits.shape
            loss = F.cross_entropy(logits.view(B * Tm1, V), y.view(B * Tm1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        global_step += 1

        # append step loss
        with open(loss_csv, "a", newline="") as f:
            csv.writer(f).writerow([global_step, loss.item()])

        if global_step % log_every == 0:
            ppl = math.exp(loss.item()) if loss.item() < 20 else float("inf")
            print(f"epoch {epoch} | step {global_step} | loss {loss.item():.4f} | ppl {ppl:.2f}")

        if global_step % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"step_{global_step:07d}.pt")
            torch.save(
                {
                    "global_step": global_step,
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "loss": loss.item(),
                },
                ckpt_path,
            )
            print(f"[checkpoint] saved: {ckpt_path}")

    return total_loss / max(1, num_batches), global_step


def main():
    # Hyperparameters
    vocab_size = 256
    dim = 256
    num_layers = 4
    num_heads = 8
    ff_hidden_dim = 1024
    max_seq_len = 50
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 10

    parser = argparse.ArgumentParser()
    parser.add_argument("--no_causal_mask", action="store_true")
    args = parser.parse_args()

    use_causal_mask = not args.no_causal_mask
    mode = "no_mask" if args.no_causal_mask else "causal"

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load dataset
    data_path = os.path.join("data", "trajectories.pkl")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Run: python generate_data.py --output data/trajectories.pkl"
        )

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    tokens = data["tokenized"]
    tokens = tokens.long() if torch.is_tensor(tokens) else torch.tensor(tokens, dtype=torch.long)

    assert tokens.dim() == 2, f"Expected tokens shape (N, T), got {tokens.shape}"
    assert tokens.size(1) == max_seq_len, f"Expected seq_len {max_seq_len}, got {tokens.size(1)}"

    dataset = TensorDataset(tokens)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Loaded dataset: {tokens.size(0)} sequences, seq_len={tokens.size(1)}")

    # Model
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_hidden_dim=ff_hidden_dim,
        max_seq_len=max_seq_len,
        dropout=0.1,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {num_params/1e6:.2f}M")
    print(f"Audit mode: {mode} (use_causal_mask={use_causal_mask})")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Logs
    os.makedirs("checkpoints", exist_ok=True)
    loss_csv_path = os.path.join("checkpoints", f"step_loss_{mode}.csv")
    log_path = os.path.join("checkpoints", f"train_log_{mode}.csv")

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss"])

    global_step = 0
    best_loss = float("inf")

    for epoch in range(num_epochs):
        train_loss, global_step = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            ckpt_dir="checkpoints",
            global_step=global_step,
            use_causal_mask=use_causal_mask,
            loss_csv=loss_csv_path,
            save_every=1000,
            log_every=100,
        )

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, train_loss])

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), os.path.join("checkpoints", f"best_model_{mode}.pt"))
            print(f"Saved best checkpoint (loss={best_loss:.4f}) -> best_model_{mode}.pt")

    print("Training complete.")

if __name__ == "__main__":
    main()
