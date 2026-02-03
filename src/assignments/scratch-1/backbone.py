"""
Scratch-1: The Transformer Backbone
CSCI 7000 - VLA Foundations

Template for implementing a decoder-only Transformer from scratch.
Students must complete the TODO sections for:
1. CausalSelfAttention
2. RMSNorm
3. Training loop
"""
import os
import math
import argparse
import pickle
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



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
        # rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        rms_inv = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # (B, T, 1)
        x_norm = x * rms_inv
        return x_norm * self.scale  # broadcast (D,)
        #raise NotImplementedError("TODO: Implement RMSNorm forward pass")


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)

    Paper: RoFormer (Su et al., 2021) - https://arxiv.org/abs/2104.09864
    Used in: Llama-3, PaLM-E, GPT-NeoX

    Key Idea: Rotate pairs of dimensions by an angle proportional to position
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "RoPE head_dim must be even to rotate pairs"
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
        if seq_len > self.cos_cached.size(0):
            # auto-extend cache to avoid OOB
            self._build_cache(seq_len)

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
        # for visualization/debug
        self.last_attn: Optional[torch.Tensor] = None  # (B, H, T, T)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        disable_causal_mask: bool = False,
        store_attn: bool = False,
    ) -> torch.Tensor:
        """
        x: (B, T, D)
        mask: optional bool or 0/1 tensor (T, T), True means allowed.
        disable_causal_mask: if True, do not apply causal mask (audit).
        store_attn: if True, store attention weights into self.last_attn.
        """
        B, T, _ = x.shape

        qkv = self.qkv_proj(x)          # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, T, D)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, Hd)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = self.rope(q, k)

        scores = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        if not disable_causal_mask:
            if mask is None:
                # default causal mask (lower triangular)
                mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            else:
                mask = mask.to(device=x.device)
                if mask.dtype != torch.bool:
                    mask = mask != 0
            scores = scores.masked_fill(~mask[None, None, :, :], float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        if store_attn:
            self.last_attn = attn.detach()

        out = attn @ v  # (B, H, T, Hd)
        out = out.transpose(1, 2).contiguous().view(B, T, self.dim)  # (B, T, D)

        out = self.out_proj(out)
        out = self.resid_dropout(out)
        return out


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network with SwiGLU activation

    Used in modern LLMs for better performance than standard ReLU
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
        Returns:
            Output tensor (batch, seq_len, dim)
        """
        gate = F.silu(self.w1(x))
        up = self.w2(x)
        x = gate * up
        x = self.dropout(x)
        x = self.w3(x)
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
        disable_causal_mask: bool = False,
        store_attn: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Optional attention mask
        Returns:
            Output tensor (batch, seq_len, dim)
        """
        # Pre-norm architecture (norm before attention/FF)
        x = x + self.attention(self.norm1(x), mask, disable_causal_mask=disable_causal_mask, store_attn=store_attn)
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

        # cache a max-length causal mask buffer (bool)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)),
            persistent=False,
        )

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
        input_ids: torch.Tensor,                 # (B, T)
        targets: Optional[torch.Tensor] = None,   # (B, T)
        disable_causal_mask: bool = False,
        store_attn: bool = False,                # store attn of block0 by default if True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_ids.shape
        if T > self.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}")

        x = self.token_embedding(input_ids)  # (B, T, D)

        # NOTE: use cached mask (keeps your variable name 'mask')
        mask = self.causal_mask[:T, :T].to(device=x.device)  # bool (T, T)

        for i, block in enumerate(self.blocks):
            # If store_attn, store only on the first block to keep it simple
            x = block(
                x,
                mask,
                disable_causal_mask=disable_causal_mask,
                store_attn=(store_attn and i == 0),
            )

        x = self.norm_final(x)
        logits = self.lm_head(x)  # (B, T, vocab)

        loss = None
        if targets is not None:
            # Flatten for cross-entropy
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=-1,  # Ignore padding tokens (keep as-is)
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

#Dataset class for loading trajectories
class TrajectoryDataset(Dataset):
    """
    Expects a pickle file containing:
      {
        'trajectories': torch.Tensor (N, 50, 10),
        'tokenized':   torch.LongTensor (N, 50)
      }
    Returns: (states, actions)
    """

    def __init__(self, pkl_path: str):
        super().__init__()
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        # Accept both formats:
        # 1) generate_data.py: {'states', 'actions'}
        # 2) older/template:  {'trajectories', 'tokenized'}
        if ("states" in data) and ("actions" in data):
            self.states = data["states"]      # (N, T, 10)
            self.actions = data["actions"]    # (N, T)
        elif ("trajectories" in data) and ("tokenized" in data):
            self.states = data["trajectories"]  # (N, T, 10)
            self.actions = data["tokenized"]    # (N, T)
        else:
            raise ValueError("PKL must contain either keys {'states','actions'} or {'trajectories','tokenized'}")

        if not torch.is_tensor(self.states):
            self.states = torch.tensor(self.states)
        if not torch.is_tensor(self.actions):
            self.actions = torch.tensor(self.actions, dtype=torch.long)

        if self.actions.dtype != torch.long:
            self.actions = self.actions.long()

        if self.states.size(0) != self.actions.size(0):
            raise ValueError("trajectories and tokenized must have same N")

    def __len__(self) -> int:
        return self.actions.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.states[idx], self.actions[idx]

    
#For plotting loss curve
def save_loss_curve(history: Dict[str, List[float]], out_path: str) -> None:
    steps = history["step"]
    losses = history["loss"]
    plt.figure()
    plt.plot(steps, losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_attention_map(attn: torch.Tensor, out_path: str, layer: int = 0, head: int = 0) -> None:
    """
    attn: (B, H, T, T)
    Saves heatmap for the first sample in batch, selected head.
    """
    if attn is None:
        return
    a = attn[0, head].cpu().numpy()  # (T, T)
    plt.figure()
    plt.imshow(a, aspect="auto")
    plt.colorbar()
    plt.xlabel("Key position")
    plt.ylabel("Query position")
    plt.title(f"Attention Map (layer {layer}, head {head})")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def quick_attention_viz(
    model: DecoderOnlyTransformer,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: "TrainConfig",
    out_path: str,
    layer: int = 0,
    head: int = 0,
) -> None:
    """
    Runs one forward pass (store_attn=True) and saves first-block attention heatmap.
    """
    model.eval()
    states, actions = next(iter(dataloader))
    actions = actions.to(device)

    input_ids = actions[:, :-1]
    _logits, _loss = model(
        input_ids=input_ids,
        targets=None,
        disable_causal_mask=cfg.disable_causal_mask,
        store_attn=True,
    )

    attn = model.blocks[0].attention.last_attn
    if attn is not None:
        save_attention_map(attn, out_path, layer=layer, head=head)


# Training
@dataclass
class TrainConfig:
    data: str
    vocab_size: int = 256
    dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    ff_hidden_dim: int = 1024
    max_seq_len: int = 50
    dropout: float = 0.1
    batch_size: int = 32
    lr: float = 1e-4
    epochs: int = 10
    num_workers: int = 0
    ckpt_dir: str = "checkpoints"
    run_dir: str = "runs"
    disable_causal_mask: bool = False
    device: str = "auto"

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to data/trajectories.pkl")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--ff_hidden_dim", type=int, default=1024)
    parser.add_argument("--max_seq_len", type=int, default=50)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--run_dir", type=str, default="runs")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--disable_causal_mask", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    return TrainConfig(
        data=args.data,
        vocab_size=256,  # keep same variable meaning
        dim=args.dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_hidden_dim=args.ff_hidden_dim,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        num_workers=args.num_workers,
        ckpt_dir=args.ckpt_dir,
        run_dir=args.run_dir,
        disable_causal_mask=args.disable_causal_mask,
        device=args.device,
    )

#GPT generated training config introduction

def train_epoch(
    model: DecoderOnlyTransformer,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    global_step: int,
    cfg: TrainConfig,
    history: Dict[str, List[float]],
) -> int:
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
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    
    # TODO: Implement training loop
    # For each batch:
    #   1. Move data to device
    #   2. Forward pass (get logits and loss)
    #   3. Backward pass
    #   4. Gradient clipping (max_norm=1.0)
    #   5. Optimizer step
    #   6. Zero gradients
    #   7. Accumulate loss

    # Hint: Use torch.nn.utils.clip_grad_norm_ for gradient clipping
    # Hint: Print progress every 100 batches

    for batch_idx, (states, actions) in enumerate(dataloader):
        # 1) Move data to device (states may be unused in this assignment)
        actions = actions.to(device)  # (B, T)

        # Teacher forcing / next-token setup
        input_ids = actions[:, :-1]   # (B, T-1)
        targets   = actions[:, 1:]    # (B, T-1)

        # 2) Forward pass
        logits, loss = model(
            input_ids=input_ids,
            targets=targets,
            disable_causal_mask=cfg.disable_causal_mask,
            store_attn=False,
        )
        assert loss is not None

        # 6) Zero gradients
        optimizer.zero_grad(set_to_none=True)

        # 3) Backward pass
        loss.backward()

        # 4) Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 5) Optimizer step
        optimizer.step()

        global_step += 1

        # 7) Accumulate loss (history for loss curve)
        history["step"].append(float(global_step))
        history["loss"].append(float(loss.item()))

        # Progress
        if (global_step) % 100 == 0:
            ppl = math.exp(min(loss.item(), 20.0))
            print(
                f"Epoch {epoch} | Step {global_step} "
                f"| loss {loss.item():.4f} | ppl {ppl:.2f}"
            )

        # Save checkpoints every 1000 steps
        if (global_step) % 1000 == 0:
            ckpt_path = os.path.join(cfg.ckpt_dir, f"model_step_{global_step}.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": global_step,
                    "epoch": epoch,
                    "config": cfg.__dict__,
                },
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")

    return global_step



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

    # Parse args into cfg (keeps your cfg.xxx usage without renaming your variables)
    global cfg
    cfg = parse_args()

    # Device
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)
    print(f"Using device: {device}")


    # TODO: Load dataset
    # Use the generate_data.py script to create synthetic trajectories
    # Load from data/trajectories.pkl
    dataset = TrajectoryDataset(cfg.data)
    train_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
    )

    # TODO: Create model
    # model = DecoderOnlyTransformer(...)
    model = DecoderOnlyTransformer(
        vocab_size=cfg.vocab_size,
        dim=cfg.dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        ff_hidden_dim=cfg.ff_hidden_dim,
        max_seq_len=cfg.max_seq_len,
        dropout=cfg.dropout,
    ).to(device)

    # TODO: Create optimizer
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    # TODO: Training loop
    # for epoch in range(num_epochs):
    #     train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
    #     print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}")
    os.makedirs(cfg.run_dir, exist_ok=True)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    history = {"step": [], "loss": []}
    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        global_step = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            global_step=global_step,
            cfg=cfg,
            history=history,
        )

        # save loss curve each epoch
        loss_curve_path = os.path.join(cfg.run_dir, "loss_curve.png")
        save_loss_curve(history, loss_curve_path)

        # save a sample attention map each epoch (from block 0)
        attn_path = os.path.join(cfg.run_dir, "attn_map_layer0_head0.png")
        quick_attention_viz(
            model=model,
            dataloader=train_loader,
            device=device,
            cfg=cfg,
            out_path=attn_path,
            layer=0,
            head=0,
        )

        # save history tensor for report
        hist_path = os.path.join(cfg.run_dir, "loss_history.pt")
        torch.save(history, hist_path)

        # also save an epoch checkpoint
        epoch_ckpt = os.path.join(cfg.ckpt_dir, f"model_epoch_{epoch}.pt")
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": global_step, "epoch": epoch},
            epoch_ckpt,
        )
        print(f"Epoch {epoch}/{cfg.epochs} done. Saved: {epoch_ckpt}")

    # TODO: Save checkpoint
    # torch.save(model.state_dict(), "checkpoints/best_model.pt")

    print("Training complete.")
    print(f"Loss curve saved to: {os.path.join(cfg.run_dir, 'loss_curve.png')}")
    print(f"Attention map saved to: {os.path.join(cfg.run_dir, 'attn_map_layer0_head0.png')}")
    if cfg.disable_causal_mask:
        print("NOTE: Causal mask was DISABLED (audit run). Expect artificially lower training loss, worse generation.")


if __name__ == "__main__":
    main()
