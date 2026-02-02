"""
Scratch-1: The Transformer Backbone
CSCI 7000 - VLA Foundations

Template for implementing a decoder-only Transformer from scratch.
Students must complete the TODO sections for:
1. CausalSelfAttention
2. RMSNorm
3. Training loop
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import os
import matplotlib.pyplot as plt
import pickle
import time

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization

    Paper: https://arxiv.org/abs/1910.07467
    Used in: Llama-3, Grok, PaLM

    Formula:
        a_bar_i = (a_i / RMS(a)) * g_i
        where RMS(a) = sqrt(mean(a^2) + eps)
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # TODO: Initialize learnable scale parameter 'g' (gamma)
        # Hint: Use nn.Parameter with torch.ones
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
        Returns:
            Normalized tensor of same shape
        """
        # TODO: Implement RMSNorm
        # Step 1: Compute RMS (root mean square) along the last dimension
        # Step 2: Normalize by dividing x by RMS
        # Step 3: Apply learnable scale parameter

        # HINT: Use torch.mean, torch.rsqrt for efficiency
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        norm = x * rms
        scaled = norm * self.scale
        # raise NotImplementedError("TODO: Implement RMSNorm forward pass")
        return scaled

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)

    Paper: RoFormer (Su et al., 2021) - https://arxiv.org/abs/2104.09864
    Used in: Llama-3, PaLM-E, GPT-NeoX

    Key Idea: Rotate pairs of dimensions by an angle proportional to position
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute rotation frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos and sin for all positions
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Precompute cos and sin values for all positions"""
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (seq_len, dim//2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Helper function to rotate tensor by swapping and negating half the dimensions"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, pos_offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to query and key tensors

        Args:
            q: Query tensor (batch, num_heads, seq_len, head_dim)
            k: Key tensor (batch, num_heads, seq_len, head_dim)
            pos_offset: Offset so we start at the latest position for RoPE instead of rotating all
        Returns:
            Rotated (q, k) tensors
        """
        seq_len = q.shape[2]

        # Get cached cos/sin values
        # Note: for kv caching, we need to offset by the current position so we know to rotate tokens correctly
        # i.e. with kv caching we only process latest token so seq_len is always passed in as 1 instead of
        # whatever full length it would be w/o kv caching
        cos = self.cos_cached[pos_offset:pos_offset + seq_len, ...]
        sin = self.sin_cached[pos_offset:pos_offset + seq_len, ...]

        # Apply rotation: q_rot = q * cos + rotate_half(q) * sin
        q_rot = (q * cos) + (self.rotate_half(q) * sin)
        k_rot = (k * cos) + (self.rotate_half(k) * sin)

        return q_rot, k_rot


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention with RoPE

    Key constraints:
    - Token at position t can only attend to positions <= t
    - Uses RoPE instead of absolute positional embeddings
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Linear projections for Q, K, V
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Rotary embeddings
        self.rope = RotaryPositionalEmbedding(self.head_dim)

        # Storage ~ add this new variable to store last attn weights for attention map visualization later
        self.last_attn_weights = None

    # modify input args to allow past_kv, and modify return to pass through kvs
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Optional attention mask (seq_len, seq_len)
            past_kv: Optional cached kvs from prev step
        Returns:
            Output tensor (batch, seq_len, dim)
            Tuple of current k and v for caching
        """
        batch_size, seq_len, _ = x.shape

        # TODO: Implement Causal Self-Attention

        # Step 1: Project input to Q, K, V
        # qkv = self.qkv_proj(x)  # (batch, seq_len, 3*dim)
        # Split into Q, K, V and reshape for multi-head attention
        # Hint: Use .view() and .transpose() to get shape (batch, num_heads, seq_len, head_dim)
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1) # split into 3, each are size (batch, seq len, dim)
        # view to reshape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        # transpose to reorder seq_len and num_heads
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Step 2: Apply RoPE to Q and K
        pos_offset = 0
        past_k, past_v = None, None
        if past_kv  is not None:
            past_k, past_v = past_kv # we just need to extract 1 to get the current pos offset
            pos_offset = past_k.shape[2] # the seq len dimension
        q, k = self.rope(q, k, pos_offset=pos_offset)

        # check if we have any past_kv passed into fxn
        # if we do, concat it with the current to skip recomputing all previous
        if past_kv is not None:
            k = torch.cat([past_k, k], dim=2) # make sure concat along seq_len
            v = torch.cat([past_v, v], dim=2)
        # store and pass kv into next iter
        curr_kv = (k, v)

        # Step 3: Compute attention scores
        # scores = (Q @ K^T) / sqrt(d_k)
        # Hint: Use torch.matmul or @ operator
        # Shape should be (batch, num_heads, seq_len, seq_len)
        scores = (q @ k.transpose(-2, -1)) * self.scale # only transpose last 2 dims, keep batch and num_heads the same

        # Step 4: Apply causal mask
        # The mask should prevent position i from attending to positions > i
        # Hint: Create a lower-triangular matrix using torch.tril
        # Set masked positions to -inf BEFORE softmax
        # Example: scores = scores.masked_fill(mask == 0, float('-inf'))
        # Edit: for kv caching, we need to change mask size for cached sequence
        # ^ e.g. previously, k is shape (batch, heads, seq_len=1, dim)
        # but now its (batch, heads, seq_len=1+cache, dim)
        cached_seq_len = k.shape[2]
        mask = torch.tril(torch.ones(cached_seq_len, cached_seq_len, device=x.device))
        # For kv caching apply mask to the new positions only
        mask = mask[-seq_len:, :] # positions go down by row, we take the last seq_len out of seq_len+cache
        scores = scores.masked_fill(mask == 0, float('-inf'))
        # ^ remove the above 2 lines to see results w/o causal mask

        # Step 5: Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Step 6: Apply attention to values
        self.last_attn_weights = attn_weights # store so we can make attention map visualization later...
        out = attn_weights @ v

        # Step 7: Reshape and project back
        # Concatenate heads and apply output projection
        # Hint: Use .transpose() and .contiguous().view() to reshape
        # ++ Output tensor (batch, seq_len, dim)
        out = out.transpose(2, 1)
        out = out.contiguous().view(batch_size, seq_len, -1)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        return out, curr_kv

        # raise NotImplementedError("TODO: Implement CausalSelfAttention forward pass")


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network with SwiGLU activation

    Used in modern LLMs for better performance than standard ReLU
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
        Returns:
            Output tensor (batch, seq_len, dim)
        """
        x = self.fc1(x)
        x = F.silu(x)  # SwiGLU activation (SiLU = Swish)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    A single Transformer decoder block

    Architecture:
        x = x + Attention(RMSNorm(x))
        x = x + FeedForward(RMSNorm(x))
    """

    def __init__(self, dim: int, num_heads: int, ff_hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = CausalSelfAttention(dim, num_heads, dropout)
        self.feed_forward = FeedForward(dim, ff_hidden_dim, dropout)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, past_kv: Optional[Tuple] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Optional attention mask
            past_kv: Optional kv from prev step
        Returns:
            Output tensor (batch, seq_len, dim)
            Current kv cache
        """
        # Pre-norm architecture (norm before attention/FF)

        # KV caching edit: split into two steps so we can extract the curr kv to pass through
        attn, curr_kv = self.attention(self.norm1(x), mask, past_kv)
        x = x + attn
        x = x + self.feed_forward(self.norm2(x))
        return x, curr_kv


class DecoderOnlyTransformer(nn.Module):
    """
    Decoder-only Transformer for next-token prediction on robot trajectories

    Architecture similar to GPT, but for robotic control sequences
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_layers: int,
        num_heads: int,
        ff_hidden_dim: int,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # Final norm and projection to vocabulary
        self.norm_final = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

        # Storage for kv cache
        self.kv_cache = None

    def _init_weights(self, module):
        """Initialize weights following standard Transformer initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_cache: Optional[list] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: Input token indices (batch, seq_len)
            targets: Target token indices for training (batch, seq_len)
            kv_cache: Optional list of all previous KVs
        Returns:
            logits: Output logits (batch, seq_len, vocab_size)
            loss: Cross-entropy loss if targets provided, else None
            kvs: Updated list of all previous kvs
        """
        batch_size, seq_len = input_ids.shape

        # Embed tokens
        x = self.token_embedding(input_ids)  # (batch, seq_len, dim)

        # Create causal mask (lower triangular)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))

        # storage for kvs
        kvs = []

        # Apply transformer blocks
        for i, block in enumerate(self.blocks):
            if kv_cache is not None:
                past_kv = kv_cache[i]
            else:
                past_kv = None
            x, curr_kv = block(x, mask, past_kv)
            kvs.append(curr_kv)

        # Final norm and projection
        x = self.norm_final(x)
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Flatten for cross-entropy
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1,  # Ignore padding tokens
            )

        return logits, loss, kvs

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        use_kv_cache: bool = False,
    ) -> torch.Tensor:
        """
        Autoregressive generation

        Args:
            input_ids: Starting tokens (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (if None, use full distribution)
        Returns:
            Generated sequence (batch, seq_len + max_new_tokens)
        """
        # cache past kvs
        self.kv_cache = None # reset cache for each generate call
        
        for _ in range(max_new_tokens):
            # Crop context if too long
            # ^ Edit: process entire context for first
            # then for subsequent since we have the cache we only need to process latest
            if self.kv_cache is None or not use_kv_cache:
                input_context = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]
            else:
                input_context = input_ids[:, -1:] # just process last token since we already have cache

            # Forward pass
            # modified forward that takes and returns the kv cache
            logits, _, self.kv_cache = self.forward(input_context, kv_cache=self.kv_cache if use_kv_cache else None)
            logits = logits[:, -1, :] / temperature  # Get last token logits

            # Optional top-k sampling
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float('Inf')

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


def train_epoch(
    model: DecoderOnlyTransformer,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    step_counter: int,
) -> Tuple[float, int, list]:
    """
    Train for one epoch

    Args:
        model: The transformer model
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    batch_losses = [] # save loss per step not epoch!!

    # TODO: Implement training loop
    # For each batch:
    #   1. Move data to device
    #   2. Forward pass (get logits and loss)
    #   3. Backward pass
    #   4. Gradient clipping (max_norm=1.0)
    #   5. Optimizer step
    #   6. Zero gradients
    #   7. Accumulate loss

    # note: inputs and targets are token inputs and correct token output in sequence
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits, loss, _ = model(inputs, targets) # forward pass ~ kv cache not used during training, just leave as _ for now

        loss.backward() # backward pass ~ refer to gpt dev for how this works...

        # max_norm is max allowed l2 norm of gradient...use 1.0 as specified
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step() # step optimizer

        optimizer.zero_grad() # zero out grads ~ refer to gpt dev for example again

        total_loss += loss.item() # add to loss!
        num_batches += 1 # increment num batches so we can average

        # add new counter so we can track every 1000 save a checkpoint
        # also add it to fxn signature, and change return to have loss, step counter, and batch loss for plotting later
        step_counter += 1
        batch_losses.append(loss.item())

        # save checkpoint every 1000
        if step_counter % 1000 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/checkpoint_step_{step_counter}.pt")

        # print progress every 100 batches
        if batch_idx % 100 == 0:
            perplexity = math.exp(loss.item())
            print(f"Epoch {epoch} Batch {batch_idx} | Loss = {loss.item()} Perplexity = {perplexity}")

    return total_loss / num_batches, step_counter, batch_losses

    # Hint: Use torch.nn.utils.clip_grad_norm_ for gradient clipping
    # Hint: Print progress every 100 batches

    # raise NotImplementedError("TODO: Implement training loop")


def main():
    """
    Main training script

    Usage:
        python backbone.py
    """
    # Hyperparameters
    vocab_size = 256  # Discretized action space
    dim = 256  # Model dimension
    num_layers = 4  # Number of transformer blocks
    num_heads = 8  # Number of attention heads
    ff_hidden_dim = 1024  # Feed-forward hidden dimension
    max_seq_len = 50  # Maximum sequence length
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 10

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # TODO: Load dataset
    # Use the generate_data.py script to create synthetic trajectories
    # Load from data/trajectories.pkl
    with open('data/trajectories.pkl', 'rb') as f:
        dataset = pickle.load(f)
    # make a dataloader from dataset, add shuffle so order is randomized in each training iter
    tokenized = dataset["actions"] # dataset labels are states and actions, not trajectories and tokenized...
    inputs = tokenized[:, :-1] # inputs is all tokens in sequence except last
    targets = tokenized[:, 1:] # same but except first ~ i.e. autoregressive, one points to the next, like in gpt from scratch
    train_dataset = torch.utils.data.TensorDataset(inputs, targets)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # TODO: Create model
    # model = DecoderOnlyTransformer(...)
    model = DecoderOnlyTransformer(vocab_size=vocab_size, dim=dim, num_layers=num_layers, num_heads=num_heads, ff_hidden_dim=ff_hidden_dim, max_seq_len=max_seq_len)
    model = model.to(device)

    # TODO: Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # TODO: Training loop
    loss_plot = []
    step_counter = 0
    for epoch in range(num_epochs):
        train_loss, step_counter, batch_losses = train_epoch(model, train_loader, optimizer, device, epoch, step_counter)
        loss_plot.extend(batch_losses)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}")

    # TODO: Save checkpoint
    torch.save(model.state_dict(), "checkpoints/best_model.pt")

    # Plotting stuff

    # plot loss
    plt.figure()
    plt.plot(loss_plot)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.savefig("loss_curve.png")

    # plot attention map
    model.eval() # change to eval mode ~ part of nn.module fxn, like model.train()
    with torch.no_grad():
        sample_input = inputs[0:1].to(device) # get first traj, move to device
        model(sample_input) # run model on the trajectory ^
    # pull out attention weights for first traj
    fig, axes = plt.subplots(4, 8, figsize=(32, 16))
    for layer in range(num_layers):
        for head in range(num_heads):
            # weights is [batch, num_heads, seq_len, seq_len], so pull out first 2 dims for each
            # only 1 traj so batch 0, num_heads index = head
            # move off gpu to cpu, and convert to numpy for plotting
            attn = model.blocks[layer].attention.last_attn_weights[0, head].cpu().numpy()
            attn_plot = axes[layer, head].imshow(attn, cmap='viridis')
            axes[layer, head].set_title(f"L{layer} H{head}")
            axes[layer, head].set_xlabel("Key")
            axes[layer, head].set_ylabel("Query")
    plt.tight_layout()
    fig.colorbar(attn_plot, ax=axes, label='Attention Weight')
    plt.savefig("attention_maps.png", bbox_inches='tight') # add tight to crop extra whitespace

    # print("TODO: Complete the main training script")

    # KV-Caching test
    print("KV-Caching test:")
    model.eval()
    # test generate 100 new tokens w/ no cache
    start = time.time()
    _ = model.generate(inputs[0:1].to(device), max_new_tokens=100, use_kv_cache=False)
    dur_no_cache = time.time() - start
    # test generate 100 new tokens w/ cache
    start = time.time()
    _ = model.generate(inputs[0:1].to(device), max_new_tokens=100, use_kv_cache=True)
    dur_cache = time.time() - start
    print(f"Inference speed for 100 tokens without KV-Caching: {dur_no_cache}s")
    print(f"Inference speed for 100 tokens with KV-Caching: {dur_cache}s")
    print(f"Improvement: {dur_no_cache/dur_cache}")

if __name__ == "__main__":
    main()
