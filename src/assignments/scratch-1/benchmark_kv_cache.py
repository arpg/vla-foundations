"""
Benchmark script for comparing inference speed with and without KV-caching.

This script measures the time difference between:
1. Normal inference: Recomputes full attention for entire sequence each step
2. KV-cached inference: Caches key-value pairs, only computes attention for new token

Usage:
    python benchmark_kv_cache.py
"""

import torch
import time
from pathlib import Path
import matplotlib.pyplot as plt
from backbone import DecoderOnlyTransformer


def benchmark_generation(
    model: DecoderOnlyTransformer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    use_kv_cache: bool,
    num_runs: int = 5,
    warmup_runs: int = 2,
) -> float:
    """
    Benchmark generation speed.

    Args:
        model: The transformer model
        input_ids: Starting tokens (batch, seq_len)
        max_new_tokens: Number of tokens to generate
        use_kv_cache: Whether to use KV caching
        num_runs: Number of timed runs to average
        warmup_runs: Number of warmup runs before timing

    Returns:
        Average time per run in seconds
    """
    model.eval()

    # Warmup runs
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model.generate(
                input_ids.clone(),
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                use_kv_cache=use_kv_cache,
            )

    # Synchronize CUDA before timing
    if input_ids.is_cuda:
        torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()

        with torch.no_grad():
            _ = model.generate(
                input_ids.clone(),
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                use_kv_cache=use_kv_cache,
            )

        # Synchronize CUDA after generation
        if input_ids.is_cuda:
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append(end - start)

    return sum(times) / len(times)


def run_benchmark(
    model: DecoderOnlyTransformer,
    device: torch.device,
    prompt_lengths: list[int],
    generation_lengths: list[int],
    num_runs: int = 5,
):
    """
    Run comprehensive benchmark comparing KV-cache vs no-cache.

    Args:
        model: The transformer model
        device: Device to run on
        prompt_lengths: List of different prompt lengths to test
        generation_lengths: List of different generation lengths to test
        num_runs: Number of runs per configuration

    Returns:
        Dictionary with benchmark results
    """
    results = {
        'prompt_lengths': prompt_lengths,
        'generation_lengths': generation_lengths,
        'with_cache': {},
        'without_cache': {},
        'speedup': {},
    }

    for prompt_len in prompt_lengths:
        for gen_len in generation_lengths:
            key = (prompt_len, gen_len)

            # Create random input prompt
            input_ids = torch.randint(0, model.vocab_size, (1, prompt_len), device=device)

            print(f"\nBenchmarking: prompt_len={prompt_len}, gen_len={gen_len}")

            # Benchmark with KV-cache
            time_with_cache = benchmark_generation(
                model, input_ids, gen_len, use_kv_cache=True, num_runs=num_runs
            )
            results['with_cache'][key] = time_with_cache
            print(f"  With KV-cache:    {time_with_cache*1000:.2f} ms")

            # Benchmark without KV-cache
            time_without_cache = benchmark_generation(
                model, input_ids, gen_len, use_kv_cache=False, num_runs=num_runs
            )
            results['without_cache'][key] = time_without_cache
            print(f"  Without KV-cache: {time_without_cache*1000:.2f} ms")

            # Calculate speedup
            speedup = time_without_cache / time_with_cache
            results['speedup'][key] = speedup
            print(f"  Speedup:          {speedup:.2f}x")

    return results


def plot_results(results: dict, save_path: Path = None):
    """
    Plot benchmark results.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    prompt_lengths = results['prompt_lengths']
    generation_lengths = results['generation_lengths']

    # Plot 1: Time comparison for different generation lengths (fixed prompt)
    ax1 = axes[0]
    prompt_len = prompt_lengths[0]
    times_cache = [results['with_cache'][(prompt_len, g)] * 1000 for g in generation_lengths]
    times_no_cache = [results['without_cache'][(prompt_len, g)] * 1000 for g in generation_lengths]

    ax1.plot(generation_lengths, times_cache, 'o-', label='With KV-cache', color='green')
    ax1.plot(generation_lengths, times_no_cache, 's-', label='Without KV-cache', color='red')
    ax1.set_xlabel('Generation Length (tokens)')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title(f'Inference Time vs Generation Length\n(prompt_len={prompt_len})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Time comparison for different prompt lengths (fixed generation)
    ax2 = axes[1]
    gen_len = generation_lengths[-1]  # Use longest generation
    times_cache = [results['with_cache'][(p, gen_len)] * 1000 for p in prompt_lengths]
    times_no_cache = [results['without_cache'][(p, gen_len)] * 1000 for p in prompt_lengths]

    ax2.plot(prompt_lengths, times_cache, 'o-', label='With KV-cache', color='green')
    ax2.plot(prompt_lengths, times_no_cache, 's-', label='Without KV-cache', color='red')
    ax2.set_xlabel('Prompt Length (tokens)')
    ax2.set_ylabel('Time (ms)')
    ax2.set_title(f'Inference Time vs Prompt Length\n(gen_len={gen_len})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Speedup heatmap
    ax3 = axes[2]
    speedup_matrix = [[results['speedup'][(p, g)] for g in generation_lengths] for p in prompt_lengths]
    im = ax3.imshow(speedup_matrix, cmap='Greens', aspect='auto')
    ax3.set_xticks(range(len(generation_lengths)))
    ax3.set_xticklabels(generation_lengths)
    ax3.set_yticks(range(len(prompt_lengths)))
    ax3.set_yticklabels(prompt_lengths)
    ax3.set_xlabel('Generation Length')
    ax3.set_ylabel('Prompt Length')
    ax3.set_title('KV-Cache Speedup Factor')

    # Add speedup values as text
    for i in range(len(prompt_lengths)):
        for j in range(len(generation_lengths)):
            text = ax3.text(j, i, f'{speedup_matrix[i][j]:.1f}x',
                           ha='center', va='center', color='black', fontsize=10)

    plt.colorbar(im, ax=ax3, label='Speedup (x)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved benchmark plot to {save_path}")

    plt.show()


def main():
    # Model configuration - use larger max_seq_len for extended benchmark
    vocab_size = 256
    dim = 256
    num_layers = 4
    num_heads = 8
    ff_hidden_dim = 1024
    max_seq_len = 50  # Increased for longer sequence testing

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_hidden_dim=ff_hidden_dim,
        max_seq_len=max_seq_len,
        dropout=0.0,  # No dropout for inference
    )
    model.to(device)
    model.eval()

    # Try to load trained weights if available
    checkpoint_path = Path("checkpoints/casual_mask_removed_best_model.pt")
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        print(f"Loaded trained model from {checkpoint_path}")
    else:
        print("No checkpoint found, using randomly initialized model")

    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    # Benchmark configurations - include longer sequences to show KV-cache benefits
    prompt_lengths = [5, 10, 25, 50]
    generation_lengths = [5, 10, 25, 45]

    # Run benchmark
    print("\n" + "="*60)
    print("KV-Cache Inference Speed Benchmark")
    print("="*60)

    results = run_benchmark(
        model=model,
        device=device,
        prompt_lengths=prompt_lengths,
        generation_lengths=generation_lengths,
        num_runs=5,
    )

    # Summary statistics
    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    speedups = list(results['speedup'].values())
    print(f"Average speedup: {sum(speedups)/len(speedups):.2f}x")
    print(f"Min speedup:     {min(speedups):.2f}x")
    print(f"Max speedup:     {max(speedups):.2f}x")

    # Plot results
    plots_dir = Path("images")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / "kv_cache_benchmark.png"
    plot_results(results, save_path=plot_path)


if __name__ == "__main__":
    main()
