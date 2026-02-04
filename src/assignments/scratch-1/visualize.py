"""
Visualize results for Scratch-1 Transformer Backbone

1) Plot GT vs Prediction tokens using best_model.pt
2) Plot training loss curve from checkpoints/step_loss.csv (Step vs Loss)
3) Save attention maps from last layer (head0 + avg heads)
4) Save figures as PNG
"""

import argparse
import os
import pickle
import csv
import torch
import matplotlib.pyplot as plt

from backbone import DecoderOnlyTransformer


def load_step_loss(path):
    steps, losses = [], []
    if not os.path.exists(path):
        return steps, losses

    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            if "step" not in row or "loss" not in row:
                continue
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
    return steps, losses


def moving_average(values, window: int):
    """Simple moving average for smoothing."""
    if window is None or window <= 1:
        return list(values)
    out = []
    s = 0.0
    q = []
    for v in values:
        q.append(v)
        s += v
        if len(q) > window:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


def trim_to_last_run(steps, losses):
    """
    If steps reset (e.g., training restarted and appended to same CSV),
    keep only the last run segment.
    """
    if not steps:
        return steps, losses

    cut = 0
    for i in range(1, len(steps)):
        if steps[i] < steps[i - 1]:
            cut = i
    return steps[cut:], losses[cut:]


def sort_by_step(steps, losses):
    pairs = sorted(zip(steps, losses), key=lambda x: x[0])
    if not pairs:
        return [], []
    s, l = zip(*pairs)
    return list(s), list(l)


def save_attention_heatmap(attn: torch.Tensor, out_path: str, title: str, *, head: int = 0):
    """
    attn: (B, H, T, T) attention weights
    Saves a heatmap for a single head of the first sample in batch.
    """
    attn_2d = attn[0, head].detach().to("cpu")  # (T, T)

    plt.figure(figsize=(6, 5))
    plt.imshow(attn_2d, aspect="auto")
    plt.colorbar()
    plt.xlabel("Key position (attended-to)")
    plt.ylabel("Query position (attending-from)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_attention_heatmap_avg_heads(attn: torch.Tensor, out_path: str, title: str):
    """
    attn: (B, H, T, T)
    Save heatmap averaged over heads: (T, T)
    """
    attn_avg = attn.mean(dim=1)  # (B, T, T)
    attn_2d = attn_avg[0].detach().to("cpu")

    plt.figure(figsize=(6, 5))
    plt.imshow(attn_2d, aspect="auto")
    plt.colorbar()
    plt.xlabel("Key position (attended-to)")
    plt.ylabel("Query position (attending-from)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--data", type=str, default="data/trajectories.pkl")
    parser.add_argument("--num_tokens", type=int, default=30)
    parser.add_argument("--out_dir", type=str, default="figures")

    # step-level loss curve
    parser.add_argument("--step_loss_csv", type=str, default="checkpoints/step_loss.csv")
    parser.add_argument("--smooth", type=int, default=0, help="moving average window (e.g., 50). 0 = no smoothing")

    # optional audit (two curves)
    parser.add_argument("--audit_causal_csv", type=str, default="", help="e.g., checkpoints/step_loss_causal.csv")
    parser.add_argument("--audit_nomask_csv", type=str, default="", help="e.g., checkpoints/step_loss_no_mask.csv")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Device (Mac: MPS)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # -----------------------------
    # 1) Load dataset + sample
    # -----------------------------
    with open(args.data, "rb") as f:
        data = pickle.load(f)

    tokens = data["tokenized"].long()  # (N, T)
    print(f"Loaded dataset with shape: {tokens.shape}")
    sample = tokens[0].unsqueeze(0).to(device)  # (1, T)

    # -----------------------------
    # 2) Load model + checkpoint
    # -----------------------------
    model = DecoderOnlyTransformer(
        vocab_size=256,
        dim=256,
        num_layers=4,
        num_heads=8,
        ff_hidden_dim=1024,
        max_seq_len=50,
        dropout=0.0,
    ).to(device)

    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # -----------------------------
    # 3) Forward + predictions + attention maps
    # -----------------------------
    with torch.no_grad():
        x = sample[:, :-1]  # (1, T-1)
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        preds = logits.argmax(dim=-1)  # (1, T-1)

        # attention maps
        last_attn = None
        layer0_attn = None
        try:
            last_attn = model.blocks[-1].attention.last_attn
            layer0_attn = model.blocks[0].attention.last_attn
        except Exception:
            pass

        if layer0_attn is not None:
            save_attention_heatmap(
                layer0_attn,
                os.path.join(args.out_dir, "attention_layer0_head0.png"),
                title="Attention Map (Layer 0, Head 0)",
                head=0,
            )
            save_attention_heatmap_avg_heads(
                layer0_attn,
                os.path.join(args.out_dir, "attention_layer0_avg_heads.png"),
                title="Attention Map (Layer 0, Avg Heads)",
            )

        if last_attn is None:
            print("No attention map found. Did you store attention weights to self.last_attn in your attention module?")
        else:
            attn_path_head0 = os.path.join(args.out_dir, "attention_last_layer_head0.png")
            save_attention_heatmap(last_attn, attn_path_head0, "Attention Map (Last Layer, Head 0)", head=0)
            print(f"Saved: {attn_path_head0}")

            attn_path_avg = os.path.join(args.out_dir, "attention_last_layer_avg_heads.png")
            save_attention_heatmap_avg_heads(last_attn, attn_path_avg, "Attention Map (Last Layer, Avg Heads)")
            print(f"Saved: {attn_path_avg}")

    # -----------------------------
    # 4) GT vs Prediction plot
    # -----------------------------
    gt = sample[:, 1:]  # (1, T-1)
    n = min(args.num_tokens, preds.shape[1])
    timesteps = list(range(n))

    plt.figure(figsize=(12, 4))
    plt.plot(timesteps, gt[0, :n].detach().cpu(), label="Ground Truth", marker="o")
    plt.plot(timesteps, preds[0, :n].detach().cpu(), label="Prediction", marker="x")
    plt.xlabel("Timestep")
    plt.ylabel("Action Token")
    plt.title("Next-Token Prediction (GT vs Prediction)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    pred_fig_path = os.path.join(args.out_dir, "best_model_predictions.png")
    plt.savefig(pred_fig_path, dpi=200)
    plt.close()
    print(f"Saved: {pred_fig_path}")

    # -----------------------------
    # 5) Step Loss curve plot (single CSV)
    # -----------------------------
    steps, losses = load_step_loss(args.step_loss_csv)
    if len(steps) == 0:
        print(f"Step-loss CSV not found or empty: {args.step_loss_csv}. Skipping step loss curve.")
    else:
        steps, losses = trim_to_last_run(steps, losses)
        steps, losses = sort_by_step(steps, losses)

        plt.figure(figsize=(7, 5))
        plt.plot(steps, losses, linewidth=1.0, label="Loss")
        if args.smooth and args.smooth > 1:
            smooth_losses = moving_average(losses, args.smooth)
            plt.plot(steps, smooth_losses, linewidth=1.5, label=f"MA({args.smooth})")

        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Loss (Step vs Loss)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(args.out_dir, "step_loss_curve.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved: {out_path}")

    # -----------------------------
    # 6) Optional audit: causal vs no-mask
    # -----------------------------
    if args.audit_causal_csv and args.audit_nomask_csv:
        steps1, loss1 = load_step_loss(args.audit_causal_csv)
        steps2, loss2 = load_step_loss(args.audit_nomask_csv)

        if len(steps1) == 0 or len(steps2) == 0:
            print("Audit CSV missing/empty. Skipping audit plot.")
        else:
            steps1, loss1 = sort_by_step(*trim_to_last_run(steps1, loss1))
            steps2, loss2 = sort_by_step(*trim_to_last_run(steps2, loss2))

            plt.figure(figsize=(7, 5))
            plt.plot(steps1, loss1, linewidth=1.0, label="Causal Mask")
            plt.plot(steps2, loss2, linewidth=1.0, label="No Mask (Cheating)")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Audit: Causal Mask vs No Mask")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            out_path = os.path.join(args.out_dir, "audit_loss_curve.png")
            plt.savefig(out_path, dpi=200)
            plt.close()
            print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
