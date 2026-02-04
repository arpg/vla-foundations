import argparse
import time
from typing import Optional, List
import pickle
from generate_data import create_dataloaders

import torch
import torch.nn as nn
import numpy as np

import backbone

def reset_all_kv_caches(model: nn.Module):
    for block in model.blocks:
        block.attention.reset_cache()

@torch.no_grad()
def baseline_timing(
    model: nn.Module,
    input_ids: torch.Tensor,
    gen_len: int,
    temperature: float,
    top_k: Optional[int],
) -> float:
    """
    Baseline path: full recompute each step.
    """
    reset_all_kv_caches(model)
    model.eval()

    t0 = time.perf_counter()

    _ = model.generate(
        input_ids=input_ids.clone(),
        max_new_tokens=gen_len,
        temperature=temperature,
        top_k=top_k,
    )

    t1 = time.perf_counter()
    
    model_size = model_kv_cache_report(model)
    return t1 - t0, model_size

@torch.no_grad()
def cached_timing(
    model: nn.Module,
    prompt: torch.Tensor,
    gen_len: int,
    temperature: float,
    top_k: Optional[int],
) -> float:
    """
    Cached path: KV cached decoding.
    """
    reset_all_kv_caches(model)
    model.eval()

    t0 = time.perf_counter()

    _ = model.generate_cached(
        input_ids=prompt.clone(),
        max_new_tokens=gen_len,
        temperature=temperature,
        top_k=top_k,
    )

    t1 = time.perf_counter()
    
    model_size = model_kv_cache_report(model)
    return t1 - t0, model_size


@torch.no_grad()
def benchmark_kv_cache(
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    input_ids: torch.Tensor,
    gen_len: int,
    temperature: float,
    top_k: Optional[int],
    iters: int,
):
    model.eval()

    # Measure
    t_base = []
    t_cache = []
    size_cache = []
    for _ in range(iters):
        t, _ = baseline_timing(model, input_ids, gen_len, temperature, top_k)
        t_base.append(t)
        t, size = cached_timing(model, input_ids, gen_len, temperature, top_k)
        t_cache.append(t)
        size_cache.append(size)

    t_base_t = torch.tensor(t_base)
    t_cache_t = torch.tensor(t_cache)

    tokens = batch_size * gen_len  # count only generated tokens
    tps_base = tokens / t_base_t
    tps_cache = tokens / t_cache_t

    print("============================================================")
    print(f"Device: {device}")
    print(f"batch_size={batch_size}, prompt_len={input_ids.shape[1]}, gen_len={gen_len}")
    print(f"temperature={temperature}, top_k={top_k}")
    print("")
    print(f"baseline (no cache)  : {tps_base.mean().item():.2f} tok/s (std {tps_base.std(unbiased=False).item():.2f})")
    print(f"generate_cached (KV) : {tps_cache.mean().item():.2f} tok/s (std {tps_cache.std(unbiased=False).item():.2f})")
    print(f"speedup              : {(tps_cache.mean() / tps_base.mean()).item():.2f}x")
    print(f"cache size           : {np.mean(size_cache)/1024**2:.2f} MiB")
    print("============================================================\n")

def kv_cache_size_bytes(attn):
    k = getattr(attn, "k_cache", None)
    v = getattr(attn, "v_cache", None)
    idx = getattr(attn, "index", None)

    if k is None or v is None or idx is None:
        return 0

    k_used = k[:, :, :idx]
    v_used = v[:, :, :idx]
    return (k_used.numel() * k_used.element_size()) + (v_used.numel() * v_used.element_size())

def kv_cache_size_mib(module):
    return kv_cache_size_bytes(module) / (1024 ** 2)

def model_kv_cache_report(model):
    total_bytes = 0

    for i, block in enumerate(model.blocks):
        attn = block.attention
        b = kv_cache_size_bytes(attn)
        total_bytes += b

    return total_bytes


def main():
    vocab_size = 256
    dim = 512
    layers = 4
    heads = 8
    ff_hidden = 1024
    max_seq_len = 1024
    dropout = 0.1

    batch_size = 8
    gen_len = 768
    temperature = 1.0
    iters = 10
    device = 'cuda' if torch.cuda.is_available() else "cpu"

    with open("data/trajectories.pkl", 'rb') as f:
        dataset = pickle.load(f)
    train_loader, _ = create_dataloaders(dataset, batch_size, 1.0)
    _, input_ids = next(iter(train_loader))
    input_ids = input_ids.to(device)
    print(f"Using device: {device}")

    model = backbone.DecoderOnlyTransformer(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=layers,
        num_heads=heads,
        ff_hidden_dim=ff_hidden,
        max_seq_len=max_seq_len,
        dropout=dropout,
    ).to(device)

    sd = torch.load("checkpoints/best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(sd)

    benchmark_kv_cache(
        model=model,
        device=device,
        batch_size=batch_size,
        input_ids=input_ids,
        gen_len=gen_len,
        temperature=temperature,
        top_k=None,
        iters=iters,
    )
    
    model = backbone.DecoderOnlyTransformer(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=layers,
        num_heads=heads,
        ff_hidden_dim=ff_hidden,
        max_seq_len=max_seq_len,
        dropout=dropout,
        use_mla=True
    ).to(device)
    
    sd = torch.load("checkpoints/best_model_mla.pt", map_location=device, weights_only=True)
    model.load_state_dict(sd)

    benchmark_kv_cache(
        model=model,
        device=device,
        batch_size=batch_size,
        input_ids=input_ids,
        gen_len=gen_len,
        temperature=temperature,
        top_k=None,
        iters=iters,
    )


if __name__ == "__main__":
    main()
