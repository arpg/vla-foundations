"""
Mastery-level benchmarks for Scratch-1.

1. Ablation study: RoPE vs Sinusoidal positional encodings
2. Inference speed comparison: with vs without KV-caching
"""

import json
import pickle
import time
from pathlib import Path
from typing import Optional

import torch

from backbone import DecoderOnlyTransformer
from generate_data import create_dataloaders, generate_dataset


def run_ablation_study(
    num_epochs: int = 2,
    data_path: Optional[Path] = None,
) -> dict:
    """
    Compare RoPE vs Sinusoidal positional encodings.
    Returns dict with final loss for each.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if data_path is None:
        data_path = Path(__file__).parent / "data" / "trajectories.pkl"

    with open(data_path, "rb") as f:
        dataset = pickle.load(f)
    train_loader, _ = create_dataloaders(dataset, batch_size=32, train_split=0.9)

    config = dict(
        vocab_size=256, dim=256, num_layers=4, num_heads=8,
        ff_hidden_dim=1024, max_seq_len=50, dropout=0.1,
    )
    results = {}

    for enc_name, enc_type in [("rope", "rope"), ("sinusoidal", "sinusoidal")]:
        model = DecoderOnlyTransformer(**config, position_encoding=enc_type).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        model.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            n = 0
            for _, actions in train_loader:
                actions = actions.to(device)
                input_ids = actions
                targets = torch.cat([
                    actions[:, 1:],
                    torch.full((actions.shape[0], 1), -1, device=device, dtype=torch.long),
                ], dim=1)
                opt.zero_grad()
                _, loss = model(input_ids, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total_loss += loss.item()
                n += 1
        avg_loss = total_loss / n if n else 0
        results[enc_name] = {"final_loss": avg_loss}
        print(f"  {enc_name}: final_loss = {avg_loss:.4f}")

    return results


def run_inference_benchmark(
    checkpoint_path: Path,
    num_trials: int = 5,
    prompt_len: int = 10,
    num_new_tokens: int = 40,
) -> dict:
    """
    Compare inference speed with and without KV-caching.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DecoderOnlyTransformer(
        vocab_size=256, dim=256, num_layers=4, num_heads=8,
        ff_hidden_dim=1024, max_seq_len=50, dropout=0.0,
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt)
    model = model.eval().to(device)

    prompt = torch.randint(0, 256, (1, prompt_len), device=device)

    times_no_cache = []
    times_with_cache = []
    for _ in range(num_trials):
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()
        model.generate(prompt, max_new_tokens=num_new_tokens, use_cache=False)
        torch.cuda.synchronize() if device.type == "cuda" else None
        times_no_cache.append(time.perf_counter() - t0)

        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()
        model.generate(prompt, max_new_tokens=num_new_tokens, use_cache=True)
        torch.cuda.synchronize() if device.type == "cuda" else None
        times_with_cache.append(time.perf_counter() - t0)

    return {
        "no_cache_sec_mean": sum(times_no_cache) / len(times_no_cache),
        "with_cache_sec_mean": sum(times_with_cache) / len(times_with_cache),
        "speedup": (sum(times_no_cache) / len(times_no_cache)) / (sum(times_with_cache) / len(times_with_cache)),
        "prompt_len": prompt_len,
        "num_new_tokens": num_new_tokens,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).parent / "checkpoints" / "best_model.pt")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent.parent.parent.parent / "content" / "course" / "submissions" / "scratch-1")
    parser.add_argument("--skip-ablation", action="store_true", help="Skip ablation (needs training)")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip inference benchmark")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    if not args.skip_ablation:
        print("Running ablation: RoPE vs Sinusoidal")
        all_results["ablation"] = run_ablation_study()
        with open(output_dir / "ablation_results.json", "w") as f:
            json.dump(all_results["ablation"], f, indent=2)

    if not args.skip_benchmark and args.checkpoint.exists():
        print("Running inference speed benchmark")
        all_results["inference"] = run_inference_benchmark(args.checkpoint)
        print(f"  No cache: {all_results['inference']['no_cache_sec_mean']:.4f}s")
        print(f"  With cache: {all_results['inference']['with_cache_sec_mean']:.4f}s")
        print(f"  Speedup: {all_results['inference']['speedup']:.2f}x")
        with open(output_dir / "inference_benchmark.json", "w") as f:
            json.dump(all_results["inference"], f, indent=2)

    # Generate benchmark plot if matplotlib available
    if "inference" in all_results and output_dir:
        try:
            import matplotlib.pyplot as plt
            inf = all_results["inference"]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(["No KV-cache", "With KV-cache"], [
                inf["no_cache_sec_mean"],
                inf["with_cache_sec_mean"],
            ], color=["#e74c3c", "#2ecc71"])
            ax.set_ylabel("Time (seconds)")
            ax.set_title(f"Inference Speed (prompt={inf['prompt_len']}, gen={inf['num_new_tokens']} tokens)\nSpeedup: {inf['speedup']:.2f}x")
            fig.tight_layout()
            (output_dir / "images").mkdir(exist_ok=True)
            fig.savefig(output_dir / "images" / "kv_cache_benchmark.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("Saved kv_cache_benchmark.png")
        except ImportError:
            pass

    print("Done. Results saved to", output_dir)


if __name__ == "__main__":
    main()
