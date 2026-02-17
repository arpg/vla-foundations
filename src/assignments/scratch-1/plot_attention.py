"""
Code generated using ChatGPT

plot_attention.py

Plot attention maps for your Scratch-1 DecoderOnlyTransformer checkpoint.

This script:
1) Loads a single token sequence from trajectories.pkl (dataset["actions"])
2) Loads the model + checkpoint
3) Runs ONE forward pass
4) Captures attention weights from each layer (all heads) via a temporary wrapper
5) Saves plots (single head or per-layer grids)

Supports turning causal masking OFF via --no-causal-mask
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List

import torch


def load_one_sequence(
    data_path: Path,
    batch_size: int,
    batch_index: int,
    sample_index: int,
    max_seq_len: int,
) -> torch.Tensor:
    with open(data_path, "rb") as f:
        dataset = pickle.load(f)

    actions = dataset["actions"]  # expected shape: (N, T) integer tokens
    if not torch.is_tensor(actions):
        actions = torch.tensor(actions)

    start = batch_index * batch_size
    end = start + batch_size
    if end > actions.shape[0]:
        raise IndexError(
            f"batch_index {batch_index} out of range for N={actions.shape[0]} and batch_size={batch_size}"
        )

    batch = actions[start:end]  # (batch_size, T)

    if not (0 <= sample_index < batch.shape[0]):
        raise IndexError(
            f"sample_index {sample_index} out of range for batch_size={batch.shape[0]}"
        )

    seq = batch[sample_index]  # (T,)
    seq = seq[:max_seq_len]
    return seq.unsqueeze(0)  # (1, T)


def choose_head(attn: torch.Tensor, head: str) -> int:
    """
    attn: (H, T, T) for a single sample and layer
    head: "auto" or integer as string
    """
    if head != "auto":
        h = int(head)
        if not (0 <= h < attn.shape[0]):
            raise IndexError(f"head {h} out of range for num_heads={attn.shape[0]}")
        return h

    # pick head with lowest mean entropy (most focused)
    probs = attn.clamp_min(1e-9)
    entropy = -(probs * probs.log()).sum(dim=-1)  # (H, Q)
    mean_entropy = entropy.mean(dim=-1)  # (H,)
    return int(mean_entropy.argmin().item())


def wrap_attention_modules(model) -> Dict[str, List[torch.Tensor]]:
    """
    Temporarily wraps each block.attention.forward to capture attention weights.
    Returns a dict that will be filled during forward:
      captured["attn"] = [attn_layer0, attn_layer1, ...]
    where each attn_layer is shape (B, H, T, T) on CPU.
    """
    captured: Dict[str, List[torch.Tensor]] = {"attn": []}

    for layer_idx, block in enumerate(model.blocks):
        attn_mod = block.attention
        original_forward = attn_mod.forward

        def make_wrapped_forward(original_fwd):
            def wrapped_forward(x, mask=None):
                B, T, _ = x.shape

                H = attn_mod.num_heads
                D = attn_mod.head_dim
                scale = attn_mod.scale

                qkv = attn_mod.qkv_proj(x)
                q, k, v = qkv.chunk(3, dim=-1)

                q = q.view(B, T, H, D).transpose(1, 2)  # (B,H,T,D)
                k = k.view(B, T, H, D).transpose(1, 2)
                v = v.view(B, T, H, D).transpose(1, 2)

                q, k = attn_mod.rope(q, k)

                scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B,H,T,T)

                # Your model uses a float mask with 0/1 values, shape (T,T)
                if mask is None:
                    mask_local = torch.tril(torch.ones((T, T), device=x.device))
                else:
                    mask_local = mask

                allowed = (mask_local != 0)
                scores = scores.masked_fill(~allowed.view(1, 1, T, T), float("-inf"))

                attn_weights = torch.softmax(scores, dim=-1)  # (B,H,T,T)

                # Store a CPU copy
                captured["attn"].append(attn_weights.detach().cpu())

                # Run the real attention forward to preserve exact behavior
                return original_fwd(x, mask)

            return wrapped_forward

        attn_mod._original_forward_for_viz = original_forward
        attn_mod.forward = make_wrapped_forward(original_forward)

    return captured


def restore_attention_modules(model):
    for block in model.blocks:
        attn_mod = block.attention
        if hasattr(attn_mod, "_original_forward_for_viz"):
            attn_mod.forward = attn_mod._original_forward_for_viz
            delattr(attn_mod, "_original_forward_for_viz")


def save_single(attn_layer: torch.Tensor, layer_idx: int, head_idx: int, out_path: Path):
    """
    attn_layer: (H, T, T) on CPU
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    attn_map = attn_layer[head_idx].numpy()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(attn_map, origin="upper", aspect="auto", cmap="viridis")
    ax.set_title(f"Layer {layer_idx} Head {head_idx}")
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    fig.colorbar(im, ax=ax)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def save_per_layer(attn_layer: torch.Tensor, layer_idx: int, out_path: Path, ncols: int = 4):
    """
    attn_layer: (H, T, T) on CPU
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H = attn_layer.shape[0]
    nrows = (H + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows), squeeze=False
    )

    vmax = float(attn_layer.max().item())
    for h in range(H):
        ax = axes[h // ncols][h % ncols]
        attn_map = attn_layer[h].numpy()
        ax.imshow(
            attn_map,
            origin="upper",
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_title(f"H{h}")
        ax.set_xlabel("Key")
        ax.set_ylabel("Query")

    for h in range(H, nrows * ncols):
        axes[h // ncols][h % ncols].axis("off")

    fig.suptitle(f"Layer {layer_idx}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/causal_mask_on/best_model.pt",
    )
    parser.add_argument("--data", type=str, default="data/trajectories.pkl")

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--sample-index", type=int, default=0)

    parser.add_argument("--mode", type=str, default="per-layer", choices=["single", "per-layer"])
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--head", type=str, default="auto")  # integer string or "auto"

    parser.add_argument("--out", type=str, default="images/attention")
    parser.add_argument("--max-seq-len", type=int, default=50)

    # NEW: disable causal mask
    parser.add_argument(
        "--no-causal-mask",
        action="store_true",
        help="Disable causal mask (allow attending to future tokens)",
    )

    # model hyperparams (must match training)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ff-hidden-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)

    args = parser.parse_args()

    data_path = Path(args.data)
    ckpt_path = Path(args.checkpoint)
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    # Import your model definition
    from backbone import DecoderOnlyTransformer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input_ids = load_one_sequence(
        data_path=data_path,
        batch_size=args.batch_size,
        batch_index=args.batch_index,
        sample_index=args.sample_index,
        max_seq_len=args.max_seq_len,
    ).to(device)

    model = DecoderOnlyTransformer(
        vocab_size=args.vocab_size,
        dim=args.dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_hidden_dim=args.ff_hidden_dim,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # NEW: flip the model's mask switch
    if args.no_causal_mask:
        model.mask_future = False
        print("Causal mask: OFF (full attention allowed)")
    else:
        model.mask_future = True
        print("Causal mask: ON (causal / triangular)")

    captured = wrap_attention_modules(model)

    with torch.no_grad():
        _logits, _loss = model(input_ids)

    restore_attention_modules(model)

    # captured["attn"] is list length = num_layers, each (B,H,T,T) on CPU
    attn_per_layer = [a[0] for a in captured["attn"]]  # each is (H,T,T)

    out_arg = Path(args.out)
    if args.mode == "single":
        if not (0 <= args.layer < len(attn_per_layer)):
            raise IndexError(
                f"layer {args.layer} out of range for captured layers={len(attn_per_layer)}"
            )
        layer_attn = attn_per_layer[args.layer]
        head_idx = choose_head(layer_attn, args.head)

        if out_arg.suffix:
            out_path = out_arg
        else:
            out_path = out_arg / f"attention_layer{args.layer}_head{head_idx}.png"

        save_single(layer_attn, args.layer, head_idx, out_path)
    else:
        # per-layer grids
        if out_arg.suffix:
            base = out_arg
            out_dir = base.parent
            stem = base.stem
            suffix = base.suffix
            for layer_idx, layer_attn in enumerate(attn_per_layer):
                out_path = out_dir / f"{stem}_layer{layer_idx}{suffix}"
                save_per_layer(layer_attn, layer_idx, out_path)
        else:
            out_dir = out_arg
            for layer_idx, layer_attn in enumerate(attn_per_layer):
                out_path = out_dir / f"attention_layer{layer_idx}.png"
                save_per_layer(layer_attn, layer_idx, out_path)


if __name__ == "__main__":
    main()
