import argparse
import os
import pickle

import torch
import matplotlib.pyplot as plt

from backbone import DecoderOnlyTransformer
from generate_data import create_dataloaders


@torch.no_grad()
def compute_attention_map(
    model: DecoderOnlyTransformer,
    input_ids: torch.Tensor,   # (1, T)
    layer_idx: int,
    head_idx: int,
) -> torch.Tensor:
    """
    Recompute the attention probabilities for a specific layer/head.

    Returns:
        attn: (T, T) attention map for the chosen head (causal).
    """
    model.eval()
    assert input_ids.dim() == 2 and input_ids.size(0) == 1, "input_ids must be (1, T)"
    T = input_ids.size(1)

    # Embed tokens
    x = model.token_embedding(input_ids)  # (1, T, D)

    # Run up to the chosen layer, but stop before applying that layer's attention
    for i in range(layer_idx):
        mask = torch.tril(torch.ones(T, T, device=x.device))
        x = model.blocks[i](x, mask)

    block = model.blocks[layer_idx]
    attn_mod = block.attention

    # This block uses pre-norm: attention(norm1(x))
    x_norm = block.norm1(x)  # (1, T, D)

    # QKV projection (reuse your module)
    B, T, D = x_norm.shape
    qkv = attn_mod.qkv_proj(x_norm)  # (1, T, 3D)

    # reshape to (3, B, H, T, Hd)
    qkv = qkv.view(B, T, 3, attn_mod.num_heads, attn_mod.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, T, Hd)

    # Apply RoPE (reuse your RoPE module)
    q, k = attn_mod.rope(q, k, start_pos=0)

    # Scaled dot-product attention scores: (B, H, T, T)
    scores = (q @ k.transpose(-1, -2)) * attn_mod.scale

    # Causal mask
    causal = torch.tril(torch.ones(T, T, device=x.device)).bool()
    scores = scores.masked_fill(~causal, float("-inf"))

    # Softmax -> attention probs
    attn = torch.softmax(scores, dim=-1)  # (B, H, T, T)

    # Select head and squeeze batch
    attn_head = attn[0, head_idx]  # (T, T)
    return attn_head


def load_example_from_dataset(path: str, sample_idx: int, max_len: int) -> torch.Tensor:
    """
    Loads a single trajectory from data/trajectories.pkl.
    Tries to be robust to common dataset formats:
      - list of 1D tensors
      - list of lists/arrays of ints
      - a torch Dataset-like object holding sequences
    """
    with open(path, "rb") as f:
        dataset = pickle.load(f)

    seq = dataset[sample_idx]

    if isinstance(seq, torch.Tensor):
        seq_ids = seq.clone().detach().long().flatten()
    else:
        seq_ids = torch.tensor(seq, dtype=torch.long).flatten()

    if max_len is not None and seq_ids.numel() > max_len:
        seq_ids = seq_ids[:max_len]

    return seq_ids.unsqueeze(0)  # (1, T)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to model .pt (state_dict) checkpoint")
    p.add_argument("--vocab-size", type=int, default=256)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--ff-hidden-dim", type=int, default=1024)
    p.add_argument("--max-seq-len", type=int, default=16)

    p.add_argument("--layer", type=int, default=2, help="Which transformer block index to visualize")
    p.add_argument("--head", type=int, default=0, help="Which attention head index to visualize")

    p.add_argument("--seq-len", type=int, default=64, help="Prompt length if not using dataset")
    p.add_argument("--use-dataset", default=True, help="Load a real example from data/trajectories.pkl")
    p.add_argument("--dataset-path", type=str, default="data/trajectories.pkl")
    p.add_argument("--sample-idx", type=int, default=0)
    p.add_argument("--max-len", type=int, default=128, help="Max tokens to take from dataset sequence")

    p.add_argument("--out", type=str, default="attention.png", help="Optional path to save the figure (png)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model with the same hyperparams as training
    model = DecoderOnlyTransformer(
        vocab_size=args.vocab_size,
        dim=args.dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_hidden_dim=args.ff_hidden_dim,
        max_seq_len=args.max_seq_len,
        dropout=0.0,  # disable dropout for deterministic viz
    ).to(device)

    # Load checkpoint
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Prepare input
    if args.use_dataset:
        if not os.path.exists(args.dataset_path):
            raise FileNotFoundError(f"Dataset not found at {args.dataset_path}")
        with open(args.dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        train_loader, _ = create_dataloaders(dataset, 1, 1.0)
        _, input_ids = next(iter(train_loader))
        input_ids = input_ids.to(device)
    else:
        # random prompt
        input_ids = torch.randint(0, args.vocab_size, (1, args.seq_len), device=device)

    # Sanity checks
    if not (0 <= args.layer < args.num_layers):
        raise ValueError(f"--layer must be in [0, {args.num_layers-1}]")
    if not (0 <= args.head < args.num_heads):
        raise ValueError(f"--head must be in [0, {args.num_heads-1}]")

    attn = compute_attention_map(model, input_ids, args.layer, args.head)  # (T, T)
    attn_cpu = attn.detach().float().cpu()

    # Plot
    plt.figure(figsize=(7, 6))
    plt.imshow(attn_cpu, aspect="auto")
    plt.colorbar()
    plt.title(f"Attention map | layer={args.layer} head={args.head} | T={attn_cpu.shape[0]}")
    plt.xlabel("Key position (attended-to)")
    plt.ylabel("Query position (attending-from)")

    if args.out is not None:
        plt.tight_layout()
        plt.savefig(args.out, dpi=200)
        print(f"Saved attention visualization to: {args.out}")

    plt.show()


if __name__ == "__main__":
    main()
