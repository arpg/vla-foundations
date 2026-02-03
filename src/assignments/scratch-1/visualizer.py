"""
Visualize attention maps from a saved checkpoint.
"""

import argparse
import pickle
from pathlib import Path

import torch

from backbone import DecoderOnlyTransformer


def load_batch_actions(actions: torch.Tensor, batch_size: int, batch_index: int) -> torch.Tensor:
    start = batch_index * batch_size
    end = start + batch_size
    if end > actions.shape[0]:
        raise IndexError(
            f"Batch index {batch_index} out of range for batch_size={batch_size} "
            f"and total samples={actions.shape[0]}."
        )
    return actions[start:end]


def select_head(attn_weights: torch.Tensor, head: str) -> int:
    """
    Select a head index.
    attn_weights: (batch, num_heads, seq_len, seq_len)
    head: "auto" or integer as string
    """
    if head != "auto":
        return int(head)

    # Auto-pick head with lowest mean entropy (most focused)
    probs = attn_weights.clamp_min(1e-9)
    entropy = -(probs * probs.log()).sum(dim=-1)  # (batch, heads, seq_len)
    mean_entropy = entropy.mean(dim=(0, 2))  # (heads,)
    return int(mean_entropy.argmin().item())


@torch.no_grad()
def attention_for_layer(
    model: DecoderOnlyTransformer,
    input_ids: torch.Tensor,
    layer_idx: int,
    use_causal_mask: bool,
) -> torch.Tensor:
    """
    Return attention weights for a specific layer.
    Output shape: (batch, num_heads, seq_len, seq_len)
    """
    model.eval()
    x = model.token_embedding(input_ids)
    batch_size, seq_len, _ = x.shape
    mask = None
    if use_causal_mask:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))

    for i, block in enumerate(model.blocks):
        if i == layer_idx:
            x_norm = block.norm1(x)
            attn = block.attention

            qkv = attn.qkv_proj(x_norm)
            q, k, _v = qkv.chunk(3, dim=-1)

            q = q.view(batch_size, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)

            q, k = attn.rope(q, k)

            scores = torch.matmul(q, k.transpose(-2, -1)) * attn.scale
            if mask is not None:
                scores = scores.masked_fill(
                    ~mask.view(1, 1, seq_len, seq_len),
                    torch.finfo(scores.dtype).min,
                )
            attn_weights = torch.softmax(scores, dim=-1)
            return attn_weights

        x = x + block.attention(block.norm1(x), mask)
        x = x + block.feed_forward(block.norm2(x))

    raise IndexError(f"layer_idx {layer_idx} out of range (num_layers={len(model.blocks)})")


def _resolve_out_path(out_arg: str, layer_idx: int) -> Path:
    out_base = Path(out_arg)
    if out_base.suffix:
        out_path = out_base.with_name(out_base.stem + f"_layer{layer_idx}" + out_base.suffix)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return out_path

    out_base.mkdir(parents=True, exist_ok=True)
    return out_base / f"attention_layer{layer_idx}.png"


def main():
    parser = argparse.ArgumentParser(description="Visualize attention maps from a checkpoint")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--data", type=str, default="data/trajectories.pkl")
    parser.add_argument("--mode", type=str, default="per-layer", choices=["single", "per-layer"])
    parser.add_argument("--layer", type=int, default=0, help="Layer index to visualize (single mode)")
    parser.add_argument("--head", type=str, default="0", help='Head index or "auto" (single mode)')
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--out", type=str, default=None, help="Output image path (png)")
    parser.add_argument("--no-causal-mask", action="store_true", help="Disable causal mask")

    # Model hyperparameters (must match training)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ff-hidden-dim", type=int, default=1024)
    parser.add_argument("--max-seq-len", type=int, default=50)

    args = parser.parse_args()
    use_causal_mask = not args.no_causal_mask

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    with open(data_path, "rb") as f:
        dataset = pickle.load(f)

    actions = dataset["actions"]
    batch = load_batch_actions(actions, args.batch_size, args.batch_index)

    if not (0 <= args.sample_index < batch.shape[0]):
        raise IndexError(
            f"sample_index {args.sample_index} out of range for batch_size={batch.shape[0]}"
        )

    input_ids = batch[args.sample_index].unsqueeze(0)
    input_ids = input_ids[:, : args.max_seq_len]

    model = DecoderOnlyTransformer(
        vocab_size=args.vocab_size,
        dim=args.dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_hidden_dim=args.ff_hidden_dim,
        max_seq_len=args.max_seq_len,
        dropout=0.0,
        causal_mask=False,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("matplotlib is required for visualization") from e

    if args.mode == "single":
        attn_weights = attention_for_layer(model, input_ids, args.layer, use_causal_mask)
        head_idx = select_head(attn_weights, args.head)

        attn_map = attn_weights[0, head_idx].cpu().numpy()
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(attn_map, origin="upper", aspect="auto", cmap="viridis")
        ax.set_title(f"Layer {args.layer} Head {head_idx}")
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")
        fig.colorbar(im, ax=ax)

        out_path = args.out or f"attention_layer{args.layer}_head{head_idx}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        print(f"Saved attention map to {out_path}")
        return

    # Per-layer mode: plot all heads per layer
    for layer_idx in range(args.num_layers):
        attn_weights = attention_for_layer(model, input_ids, layer_idx, use_causal_mask)
        num_heads = attn_weights.shape[1]

        ncols = 4
        nrows = (num_heads + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows), squeeze=False)

        vmax = float(attn_weights[0].max().item())
        im = None
        for h in range(num_heads):
            ax = axes[h // ncols][h % ncols]
            attn_map = attn_weights[0, h].cpu().numpy()
            im = ax.imshow(attn_map, origin="upper", aspect="auto", cmap="viridis", vmin=0.0, vmax=vmax)
            ax.set_title(f"H{h}")
            ax.set_xlabel("Key")
            ax.set_ylabel("Query")

        for h in range(num_heads, nrows * ncols):
            axes[h // ncols][h % ncols].axis("off")

        fig.suptitle(f"Layer {layer_idx}")
        #if im is not None:
            #fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)

        if args.out:
            out_path = _resolve_out_path(args.out, layer_idx)
        else:
            out_path = Path(f"attention_layer{layer_idx}.png")

        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        print(f"Saved attention map to {out_path}")


if __name__ == "__main__":
    main()
