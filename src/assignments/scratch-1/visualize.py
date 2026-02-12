"""
Visualization script for Scratch-1 assignment.

Generates loss curve and attention map plots for the submission report.

Usage:
    python visualize.py --checkpoint checkpoints/best_model.pt
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from backbone import DecoderOnlyTransformer
from generate_data import generate_dataset


def plot_loss_curve(log_path: Path, output_path: Path) -> None:
    """Plot training loss over steps."""
    with open(log_path) as f:
        data = json.load(f)

    steps = [d["step"] for d in data]
    losses = [d["loss"] for d in data]

    # Downsample if too many points for cleaner plot
    if len(steps) > 500:
        stride = len(steps) // 500
        steps = steps[::stride]
        losses = losses[::stride]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, losses, alpha=0.8, linewidth=0.8)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Training Loss Curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved loss curve to {output_path}")


def plot_attention_maps(
    checkpoint_path: Path,
    output_path: Path,
    num_heads: int = 8,
    seq_len: int = 30,
) -> None:
    """Extract and visualize attention patterns from the trained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 256
    dim = 256
    num_layers = 4
    ff_hidden_dim = 1024

    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_hidden_dim=ff_hidden_dim,
        max_seq_len=50,
        dropout=0.0,
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt)
    model = model.eval().to(device)

    # Get a sample trajectory
    dataset = generate_dataset(num_trajectories=10, seq_length=50, seed=123)
    actions = dataset["actions"]
    input_ids = actions[:1, :seq_len].to(device)

    with torch.no_grad():
        logits, loss, attn_weights = model(input_ids, return_attn_weights=True)

    # attn_weights: (batch, num_heads, seq_len, seq_len)
    attn = attn_weights[0].cpu().numpy()

    # Plot grid: 2x4 or similar for 8 heads
    n_heads = min(8, attn.shape[0])
    n_cols = 4
    n_rows = (n_heads + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = axes.flatten() if n_heads > 1 else [axes]

    for i in range(n_heads):
        im = axes[i].imshow(
            attn[i],
            aspect="auto",
            cmap="Blues",
            vmin=0,
            vmax=attn[i].max(),
        )
        axes[i].set_title(f"Head {i+1}")
        axes[i].set_xlabel("Key Position")
        axes[i].set_ylabel("Query Position")

    for j in range(n_heads, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Causal Attention Patterns (Layer 0)", fontsize=14, y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved attention maps to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).parent / "checkpoints" / "best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent.parent / "content" / "course" / "submissions" / "scratch-1" / "images",
        help="Directory to save output images",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    log_path = script_dir / "training_log.json"

    if log_path.exists():
        plot_loss_curve(log_path, args.output_dir / "loss_curve.png")
    else:
        print(f"Training log not found at {log_path}. Run backbone.py to completion for loss curve.")

    if args.checkpoint.exists():
        plot_attention_maps(args.checkpoint, args.output_dir / "attention_maps.png")
    else:
        print(f"Checkpoint not found at {args.checkpoint}. Run backbone.py first.")


if __name__ == "__main__":
    main()
