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
import logging
import re
import matplotlib.pyplot as plt
from typing import Optional, Tuple    
import pickle
from torch.utils.data import TensorDataset, DataLoader
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
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
        Returns:
            Normalized tensor of same shape
        """
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
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Optional attention mask (seq_len, seq_len)
        Returns:
            Output tensor (batch, seq_len, dim)
        """
        B, T, D = x.shape
        H, Hd = self.num_heads, self.head_dim
        # Step 1: Project input to Q, K, V
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3*dim)
        #Split into Q, K, V and reshape for multi-head attention
        # Hint: Use .view() and .transpose() to get shape (batch, num_heads, seq_len, head_dim)
        q, k, v = qkv.chunk(3, dim=-1)         # each (B, T, D)

        q = q.view(B, T, H, Hd).transpose(1, 2)  # (B, H, T, Hd)
        k = k.view(B, T, H, Hd).transpose(1, 2)  # (B, H, T, Hd)
        v = v.view(B, T, H, Hd).transpose(1, 2)  # (B, H, T, Hd)

        # Step 2: Apply RoPE to Q and K
        q, k = self.rope(q, k)

        # Step 3: Compute attention scores
        # scores = (Q @ K^T) / sqrt(d_k)
        # Hint: Use torch.matmul or @ operator
        # Shape should be (batch, num_heads, seq_len, seq_len)
        scores = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        # Step 4: Apply causal mask
        # The mask should prevent position i from attending to positions > i
        # Hint: Create a lower-triangular matrix using torch.tril
        # Set masked positions to -inf BEFORE softmax
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Step 5: Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Step 6: Apply attention to values
        out = attn_weights @ v

        # Step 7: Reshape and project back
        # Concatenate heads and apply output projection
        # Hint: Use .transpose() and .contiguous().view() to reshape
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)

        if return_attn:
            return out, attn_weights
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
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Optional attention mask
        Returns:
            Output tensor (batch, seq_len, dim)
        """
        # Pre-norm architecture (norm before attention/FF)
        if return_attn:
            attn_out, attn = self.attention(self.norm1(x), mask, return_attn=True)
            x = x + attn_out
        else:
            x = x + self.attention(self.norm1(x), mask)
        x = x + self.feed_forward(self.norm2(x))
        if return_attn:
            return x, attn
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
        use_causal_mask: bool = True,
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
    def forward_with_attn(
        self,
        input_ids: torch.Tensor,
        use_causal_mask: bool = True,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass that also returns attention maps per layer.
        """
        _, seq_len = input_ids.shape

        x = self.token_embedding(input_ids)  # (batch, seq_len, dim)
        mask = None
        if use_causal_mask:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))

        attn_maps: list[torch.Tensor] = []
        for block in self.blocks:
            x, attn = block(x, mask, return_attn=True)
            attn_maps.append(attn)

        x = self.norm_final(x)
        logits = self.lm_head(x)
        return logits, attn_maps

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
    model: DecoderOnlyTransformer,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    epoch: int,
    use_causal_mask: bool = True,
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

    # For each batch:
    #   1. Move data to device
    #   2. Forward pass (get logits and loss)
    #   3. Backward pass
    #   4. Gradient clipping (max_norm=1.0)
    #   5. Optimizer step
    #   6. Zero gradients
    #   7. Accumulate loss

    for batch_idx, (input_ids, targets) in enumerate(dataloader):
        inputs_ids = input_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits, loss = model(inputs_ids, targets, use_causal_mask=use_causal_mask)
        if loss is None:
            raise RuntimeError("Loss is None during training")
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / num_batches
    return avg_loss


def plot_loss_curve(
    train_losses: list[float],
    val_losses: list[float],
    output_path: str = "images/loss_curve.png",
) -> None:
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)


def plot_loss_from_log(
    log_path: str,
    output_path: str = "images/loss_curve_from_log.png",
) -> None:
    train_losses: list[float] = []
    val_losses: list[float] = []

    pattern = re.compile(r"Loss:\s+([0-9]*\.?[0-9]+)")
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if "Train Loss" in line:
                match = pattern.search(line)
                if match:
                    train_losses.append(float(match.group(1)))
            elif "Val Loss" in line:
                match = pattern.search(line)
                if match:
                    val_losses.append(float(match.group(1)))

    if not train_losses or not val_losses:
        logging.getLogger(__name__).warning(
            "No train/val losses found in %s",
            log_path,
        )
        return

    plot_loss_curve(train_losses, val_losses, output_path=output_path)


def plot_attention_maps(
    model: DecoderOnlyTransformer,
    input_ids: torch.Tensor,
    output_path: str = "images/attention_maps.png",
    use_causal_mask: bool = True,
) -> None:
    model.eval()
    if input_ids.size(1) > model.max_seq_len:
        input_ids = input_ids[:, -model.max_seq_len:]

    _, attn_maps = model.forward_with_attn(input_ids, use_causal_mask=use_causal_mask)

    num_layers = len(attn_maps)
    cols = min(4, num_layers)
    rows = (num_layers + cols - 1) // cols

    plt.figure(figsize=(4 * cols, 4 * rows))
    for idx, attn in enumerate(attn_maps):
        attn_avg = attn[0].mean(dim=0).detach().cpu().numpy()
        ax = plt.subplot(rows, cols, idx + 1)
        ax.imshow(attn_avg, aspect="auto", origin="lower")
        ax.set_title(f"Layer {idx + 1}")
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
    plt.tight_layout()
    plt.savefig(output_path)


@torch.no_grad()
def eval_epoch(model, dataloader, device, use_causal_mask: bool = True):
    model.eval()
    total, n = 0.0, 0
    for input_ids, targets in dataloader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        _, loss = model(input_ids=input_ids, targets=targets, use_causal_mask=use_causal_mask)
        total += loss.item()
        n += 1
    return total / max(n, 1)

class WarmupCosine:
    """
    Learning rate scheduler with linear warmup and cosine decay
    """
    def __init__(self, warmup_steps: int, total_steps: int):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
def main():
    """
    Main training script

    Usage:
        python backbone.py
    """
    # Hyperparameters
    vocab_size = 256  # Discretized action space
    dim = 384  # Model dimension
    num_layers = 6  # Number of transformer blocks
    num_heads = 8  # Number of attention heads
    ff_hidden_dim = 1536  # Feed-forward hidden dimension
    max_seq_len = 50  # Maximum sequence length
    batch_size = 128
    learning_rate = 3e-4
    num_epochs = 20
    use_causal_mask = False

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("training.log"),
        ],
    )
    logger = logging.getLogger(__name__)

    # Use the generate_data.py script to create synthetic trajectories
    # Load from data/trajectories.pkl
    with open("data/trajectories.pkl", "rb") as f:
        dataset = pickle.load(f)

    actions = dataset["actions"]          # (N, T), torch.long
    # states = data["states"]          # (N, T, 10), unused for action LM

    # Build input/target by shifting
    input_ids = actions[:, :-1].contiguous()   # (N, T-1)
    targets   = actions[:, 1:].contiguous()    # (N, T-1)

    # Split train/val over trajectories
    N = input_ids.size(0)
    num_train = int(0.9 * N)
    perm = torch.randperm(N)
    train_idx = perm[:num_train]
    val_idx   = perm[num_train:]

    train_ds = TensorDataset(input_ids[train_idx], targets[train_idx])
    val_ds   = TensorDataset(input_ids[val_idx], targets[val_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = DecoderOnlyTransformer(vocab_size=vocab_size,
                                    dim=dim,
                                    num_layers=num_layers,
                                    num_heads=num_heads,
                                    ff_hidden_dim=ff_hidden_dim,
                                    max_seq_len=max_seq_len,
                                    dropout=0.05)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=learning_rate,
                                  betas=(0.9, 0.95),
                                  weight_decay=0.1)

    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(0.05 * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=WarmupCosine(warmup_steps, total_steps))

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            epoch,
            use_causal_mask=use_causal_mask,
        )
        train_losses.append(train_loss)
        train_ppl = math.exp(train_loss)
        logger.info(
            "Epoch %d/%d - Train Loss: %.4f - Train PPL: %.4f",
            epoch + 1,
            num_epochs,
            train_loss,
            train_ppl,
        )

        val_loss = eval_epoch(model, val_loader, device, use_causal_mask=use_causal_mask)
        val_losses.append(val_loss)
        val_ppl = math.exp(val_loss)
        logger.info(
            "Epoch %d/%d - Val Loss: %.4f - Val PPL: %.4f",
            epoch + 1,
            num_epochs,
            val_loss,
            val_ppl,
        )
        
    torch.save(model.state_dict(), "checkpoints/best_model.pt")

    images_dir = Path("images")
    images_dir.mkdir(parents=True, exist_ok=True)
    plot_path = images_dir / "loss_curve.png"
    plot_loss_curve(train_losses, val_losses, output_path=str(plot_path))
    logger.info("Saved loss curve to %s", plot_path)

    attn_filename = "attention_maps.png" if use_causal_mask else "attention_maps_no_causal.png"
    attn_path = images_dir / attn_filename
    sample_input = input_ids[:1].to(device)
    plot_attention_maps(
        model,
        sample_input,
        output_path=str(attn_path),
        use_causal_mask=use_causal_mask,
    )
    logger.info("Saved attention maps to %s", attn_path)

def sample_plots(use_causal_mask: bool = True):
    # Hyperparameters (must match training)
    vocab_size = 256
    dim = 384
    num_layers = 6
    num_heads = 8
    ff_hidden_dim = 1536
    max_seq_len = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("data/trajectories.pkl", "rb") as f:
        dataset = pickle.load(f)

    actions = dataset["actions"]
    input_ids = actions[:, :-1].contiguous()

    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_hidden_dim=ff_hidden_dim,
        max_seq_len=max_seq_len,
        dropout=0.05,
    )
    model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device))
    model.to(device)

    images_dir = Path("images")
    images_dir.mkdir(parents=True, exist_ok=True)

    sample_input = input_ids[:1].to(device)
    attn_filename = "attention_maps.png" if use_causal_mask else "attention_maps_no_causal.png"
    attn_path = images_dir / attn_filename
    plot_attention_maps(
        model,
        sample_input,
        output_path=str(attn_path),
        use_causal_mask=use_causal_mask,
    )
    print(f"Saved {attn_path}")
if __name__ == "__main__":
    main()
    sample_plots()
