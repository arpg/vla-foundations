"""
Visualization script for trained Transformer models on robot trajectories.

Usage:
    python visualize.py                          # Use defaults
    python visualize.py --sample_idx 5           # Visualize sample 5
    python visualize.py --checkpoint path/to.pt  # Use specific checkpoint
"""

import torch
from pathlib import Path
import pickle
import argparse
from typing import Optional, Tuple
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from backbone import (
    DecoderOnlyTransformer,
    plot_end_effector_trajectory,
)


def visualize_attention(
    model: DecoderOnlyTransformer,
    input_ids: torch.Tensor,
    device: torch.device,
    layer_idx: int = 0,
    head_idx: Optional[int] = None,
    save_path: Optional[Path] = None,
):
    """
    Visualize attention patterns for a sample trajectory

    Args:
        model: The transformer model
        input_ids: Input token ids (1, seq_len) - single trajectory
        device: Device
        layer_idx: Which layer to visualize (0 to num_layers-1)
        head_idx: Which head to visualize (None = average all heads)
        save_path: Path to save the figure (optional)
    """
    model.eval()
    input_ids = input_ids.to(device)

    # Ensure batch dimension
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    with torch.no_grad():
        # Forward pass with attention weights
        _, _, all_attention_weights, _ = model(input_ids, return_attention=True)

    # Get attention weights for specified layer
    # Shape: (batch, num_heads, seq_len, seq_len)
    attn_weights = all_attention_weights[layer_idx][0]  # Remove batch dim

    # Either use specific head or average across heads
    if head_idx is not None:
        attn_weights = attn_weights[head_idx]  # (seq_len, seq_len)
        title = f'Attention Pattern - Layer {layer_idx}, Head {head_idx}'
    else:
        attn_weights = attn_weights.mean(dim=0)  # Average across heads
        title = f'Attention Pattern - Layer {layer_idx} (averaged across heads)'

    # Move to CPU for plotting
    attn_weights = attn_weights.cpu().numpy()
    seq_len = attn_weights.shape[0]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attn_weights, cmap='viridis', aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight')

    # Labels
    ax.set_xlabel('Key Position (attending to)')
    ax.set_ylabel('Query Position (attending from)')
    ax.set_title(title)

    # Add tick marks if sequence is not too long
    if seq_len <= 50:
        ax.set_xticks(range(0, seq_len, max(1, seq_len // 10)))
        ax.set_yticks(range(0, seq_len, max(1, seq_len // 10)))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention visualization to {save_path}")

    plt.show()

    return attn_weights


def visualize_all_heads(
    model: DecoderOnlyTransformer,
    input_ids: torch.Tensor,
    device: torch.device,
    layer_idx: int = 0,
    save_path: Optional[Path] = None,
):
    """
    Visualize attention patterns for all heads in a layer

    Args:
        model: The transformer model
        input_ids: Input token ids (1, seq_len)
        device: Device
        layer_idx: Which layer to visualize
        save_path: Path to save the figure (optional)
    """
    model.eval()
    input_ids = input_ids.to(device)

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    with torch.no_grad():
        _, _, all_attention_weights, _ = model(input_ids, return_attention=True)

    # Get attention weights for specified layer (batch, num_heads, seq_len, seq_len)
    attn_weights = all_attention_weights[layer_idx][0].cpu().numpy()  # (num_heads, seq_len, seq_len)
    num_heads = attn_weights.shape[0]

    # Create subplot grid
    cols = 4
    rows = (num_heads + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten()

    for head_idx in range(num_heads):
        ax = axes[head_idx]
        im = ax.imshow(attn_weights[head_idx], cmap='viridis', aspect='auto')
        ax.set_title(f'Head {head_idx}')
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')

    # Hide empty subplots
    for idx in range(num_heads, len(axes)):
        axes[idx].axis('off')

    fig.suptitle(f'All Attention Heads - Layer {layer_idx}', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved all-heads visualization to {save_path}")

    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize trained Transformer attention patterns")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/causal_mask_removed_best_model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--data", type=str, default="data/trajectories.pkl",
                        help="Path to trajectory data")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Index of trajectory to visualize")
    parser.add_argument("--output_dir", type=str, default="images",
                        help="Directory to save visualizations")
    parser.add_argument("--layer", type=int, default=None,
                        help="Specific layer to visualize (default: all layers)")
    parser.add_argument("--head", type=int, default=None,
                        help="Specific head to visualize (default: average all heads)")
    parser.add_argument("--no_save", action="store_true",
                        help="Don't save images, just display")
    args = parser.parse_args()

    # Model hyperparameters (must match training)
    vocab_size = 256
    dim = 256
    num_layers = 4
    num_heads = 8
    ff_hidden_dim = 1024
    max_seq_len = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_hidden_dim=ff_hidden_dim,
        max_seq_len=max_seq_len,
        dropout=0.1,
    )

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded successfully ({sum(p.numel() for p in model.parameters()):,} parameters)")

    # Load data
    print(f"Loading data from {args.data}...")
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    with open(data_path, "rb") as f:
        trajectory_data = pickle.load(f)

    num_trajectories = len(trajectory_data['actions'])
    print(f"Loaded {num_trajectories} trajectories")

    if args.sample_idx >= num_trajectories:
        raise ValueError(f"Sample index {args.sample_idx} out of range (max: {num_trajectories - 1})")

    # Get sample trajectory
    sample_state = trajectory_data['states'][args.sample_idx]
    sample_actions = trajectory_data['actions'][args.sample_idx]
    sample_input = sample_actions[:-1]  # Exclude last token for next-token prediction format

    print(f"Visualizing trajectory {args.sample_idx} (sequence length: {len(sample_input)})")

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot end effector trajectory
    print("\n--- End Effector Trajectory ---")
    save_path = None if args.no_save else output_dir / f"trajectory_{args.sample_idx}_end_effector.png"
    plot_end_effector_trajectory(
        states=sample_state,
        save_path=save_path,
        title=f"Trajectory {args.sample_idx} - End Effector Position",
        multi_view=True
    )

    # Visualize attention patterns
    print("\n--- Attention Patterns ---")

    if args.layer is not None:
        # Visualize specific layer
        layers_to_viz = [args.layer]
    else:
        # Visualize all layers
        layers_to_viz = list(range(num_layers))

    for layer_idx in layers_to_viz:
        print(f"Layer {layer_idx}...")
        save_path = None if args.no_save else output_dir / f"trajectory_{args.sample_idx}_attention_layer_{layer_idx}.png"
        visualize_attention(
            model=model,
            input_ids=sample_input,
            device=device,
            layer_idx=layer_idx,
            head_idx=args.head,
            save_path=save_path
        )

    # Visualize all heads for first layer (or specified layer)
    print("\n--- All Attention Heads ---")
    layer_for_heads = args.layer if args.layer is not None else 0
    save_path = None if args.no_save else output_dir / f"trajectory_{args.sample_idx}_all_heads_layer_{layer_for_heads}.png"
    visualize_all_heads(
        model=model,
        input_ids=sample_input,
        device=device,
        layer_idx=layer_for_heads,
        save_path=save_path
    )

    print(f"\nDone! Visualizations saved to {output_dir}/" if not args.no_save else "\nDone!")


if __name__ == "__main__":
    main()
