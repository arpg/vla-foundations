import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
import numpy as np
import time

from backbone import DecoderOnlyTransformer, train_epoch

PARAMS = dict(
    vocab_size=256,
    dim=256,
    num_layers=4,
    num_heads=8,
    ff_hidden_dim=1024,
    max_seq_len=50,
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_and_eval(name, model, train_loader, val_loader, epochs=3):
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    step_losses = []
    val_losses = []
    perplexities = []

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            inputs = batch[:, :-1].contiguous()
            targets = batch[:, 1:].contiguous()

            optimizer.zero_grad()
            _, loss, _ = model(inputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step_losses.append(loss.item())

        # Validation
        model.eval()
        val_loss_sum = 0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                inputs = batch[:, :-1].contiguous()
                targets = batch[:, 1:].contiguous()
                _, loss, _ = model(inputs, targets)
                val_loss_sum += loss.item()
                val_batches += 1

        avg_val_loss = val_loss_sum / val_batches
        val_losses.append(avg_val_loss)
        ppl = math.exp(avg_val_loss)
        perplexities.append(ppl)
        print(f"{name} Epoch {epoch}, Val Loss: {avg_val_loss:.4f}, Perplexity: {ppl:.4f}")

    return step_losses, val_losses, perplexities


def run_ablation():
    print("RoPE vs Sinusoidal")
    with open("data/trajectories.pkl", "rb") as f:
        trajectories = pickle.load(f)
    full_dataset = trajectories["actions"].clone().detach().long()

    N = 2000
    full_dataset = full_dataset[:N]
    train_size = int(0.8 * N)
    val_size = N - train_size
    train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=False)

    model_rope = DecoderOnlyTransformer(**PARAMS, use_rope=True).to(DEVICE)
    rope_steps, rope_val, rope_ppl = train_and_eval(
        "RoPE", model_rope, train_loader, val_loader
    )

    model_sin = DecoderOnlyTransformer(**PARAMS, use_rope=False).to(DEVICE)
    sin_steps, sin_val, sin_ppl = train_and_eval(
        "Sinusoidal", model_sin, train_loader, val_loader
    )

    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    def smooth(data, window=10):
        return np.convolve(data, np.ones(window) / window, mode="valid")

    ax1.plot(smooth(rope_steps), label="RoPE", alpha=0.8)
    ax1.plot(smooth(sin_steps), label="Sinusoidal", alpha=0.8)
    ax1.set_title("Training Loss (Step-wise)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    epochs = range(len(rope_val))
    ax2.plot(epochs, rope_val, label="RoPE", marker="o")
    ax2.plot(epochs, sin_val, label="Sinusoidal", marker="x")
    ax2.set_title("Validation Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.plot(epochs, rope_ppl, label="RoPE", marker="o")
    ax3.plot(epochs, sin_ppl, label="Sinusoidal", marker="x")
    ax3.set_title("Validation Perplexity")
    ax3.set_yscale("log")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("rope_vs_sinusoidal_ablation.png")


def run_benchmark():
    print("KV-Cache vs Native")
    model = DecoderOnlyTransformer(**PARAMS).to(DEVICE)
    model.eval()

    input_ids = torch.randint(0, 256, (1, 10)).to(DEVICE)
    max_new_tokens = 50
    num_runs = 5
    model.generate(input_ids, max_new_tokens=5, use_cache=True)
    model.generate(input_ids, max_new_tokens=5, use_cache=False)

    times_cache = []
    for _ in range(num_runs):
        start = time.time()
        model.generate(input_ids, max_new_tokens=max_new_tokens, use_cache=True)
        times_cache.append(time.time() - start)

    avg_cache = sum(times_cache) / num_runs
    speed_cache = max_new_tokens / avg_cache
    print(f"With Cache: {speed_cache:.2f} tok/s")

    times_no_cache = []
    for _ in range(num_runs):
        start = time.time()
        model.generate(input_ids, max_new_tokens=max_new_tokens, use_cache=False)
        times_no_cache.append(time.time() - start)

    avg_no_cache = sum(times_no_cache) / num_runs
    speed_no_cache = max_new_tokens / avg_no_cache
    print(f"No Cache: {speed_no_cache:.2f} tok/s")

    plt.figure()
    plt.bar(
        ["Without Cache", "With Cache"],
        [speed_no_cache, speed_cache],
        color=["red", "blue"],
    )
    plt.ylabel("Tokens / Second")
    plt.title("Inference Speed Comparison")
    plt.savefig("kv_cache_vs_native_benchmark.png")


def run_audit():
    print("Removing Causal Mask")

    with open("data/trajectories.pkl", "rb") as f:
        trajectories = pickle.load(f)
    dataset = trajectories["actions"].clone().detach().long()[:1000]
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    class CheatingTransformer(DecoderOnlyTransformer):
        def forward(self, input_ids, targets=None, past_kv=None, use_cache=False):
            batch_size, seq_len = input_ids.shape
            x = self.token_embedding(input_ids)
            if not self.use_rope:
                x = self.pos_embedding(x)
            mask = torch.ones(seq_len, seq_len, device=x.device)

            for block in self.blocks:
                x, _ = block(x, mask)

            x = self.norm_final(x)
            logits = self.lm_head(x)

            loss = None
            if targets is not None:
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=-1
                )
            return logits, loss, None

    model = CheatingTransformer(**PARAMS, use_rope=True).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    print("Training Cheating Model (Bidirectional Attention)...")
    for epoch in range(1):
        loss, _ = train_epoch(model, dataloader, optimizer, DEVICE, epoch)
        print(f"Epoch {epoch}, Loss: {loss:.4f} (Expected << Initial Loss)")


def run_visualization():
    model = DecoderOnlyTransformer(**PARAMS).to(DEVICE)

    if os.path.exists("checkpoints/best_model.pt"):
        try:
            state = torch.load("checkpoints/best_model.pt", map_location=DEVICE)
            model_keys = model.state_dict().keys()
            filtered_state = {
                k: v
                for k, v in state.items()
                if k in model_keys and v.size() == model.state_dict()[k].size()
            }
            model.load_state_dict(filtered_state, strict=False)
            print("Loaded pretrained weights.")
        except:
            print("Using random weights.")

    model.eval()
    seq_len = 20
    x = torch.randint(0, 256, (1, seq_len)).to(DEVICE)
    layer = model.blocks[0].attention
    qkv = layer.qkv_proj(model.token_embedding(x))
    qkv = qkv.view(1, seq_len, 3, layer.num_heads, layer.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    if layer.use_rope:
        q, k = layer.rope(q, k)

    scores = (q @ k.transpose(-2, -1)) / math.sqrt(layer.head_dim)
    mask = torch.tril(torch.ones(seq_len, seq_len, device=DEVICE))
    scores = scores.masked_fill(mask == 0, float("-inf"))
    attn_weights = torch.softmax(scores, dim=-1)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for h in range(4):
        sns.heatmap(
            attn_weights[0, h].detach().cpu().numpy(),
            ax=axes[h],
            cmap="viridis",
            square=True,
            cbar=False,
        )
        axes[h].set_title(f"Head {h}")

    plt.suptitle(f"Layer 0 Attention Patterns")
    plt.savefig("attention_maps.png")


if __name__ == "__main__":
    run_ablation()
    run_benchmark()
    run_audit()
    run_visualization()
