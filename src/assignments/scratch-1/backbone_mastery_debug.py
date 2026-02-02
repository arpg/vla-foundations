"""
Scratch-1: The Transformer Backbone - MASTERY LEVEL IMPLEMENTATION
CSCI 7000 - VLA Foundations

This extends the base implementation with:
1. KV-Caching for efficient inference
2. Sinusoidal positional embeddings (for comparison)
3. Ablation study framework
4. Inference speed benchmarking

MASTERY REQUIREMENTS:
- KV-Caching implementation
- Rigorous derivation of RoPE superiority
- Ablation study comparing RoPE vs Sinusoidal
- Inference speed comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# POSITIONAL ENCODING OPTIONS
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.scale


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)
    
    MATHEMATICAL FOUNDATION:
    -----------------------
    RoPE represents position m as a rotation matrix R_m applied to embeddings.
    
    For a 2D subspace (dimensions 2i, 2i+1):
        R_m(θ_i) = [cos(m·θ_i)  -sin(m·θ_i)]
                   [sin(m·θ_i)   cos(m·θ_i)]
    
    Where θ_i = 10000^(-2i/d) is the rotation frequency for dimension i.
    
    KEY PROPERTY - Relative Position Encoding:
    ------------------------------------------
    For queries q at position m and keys k at position n:
        <R_m·q, R_n·k> = <q, R_(n-m)·k>
    
    This means attention score depends only on (n-m), the relative position!
    
    WHY SUPERIOR FOR SPATIAL DATA:
    -----------------------------
    1. **Relative encoding**: Robot motion cares about relative distances
       (how far between waypoints), not absolute timesteps
    
    2. **Continuous interpolation**: Rotation smoothly interpolates between
       positions, perfect for continuous trajectories
    
    3. **Unbounded extrapolation**: Can handle longer sequences than trained on
       (important for variable-length robot paths)
    
    4. **No additive interference**: Unlike sinusoidal, doesn't add to embeddings,
       so doesn't interfere with semantic content
    
    DERIVATION:
    ----------
    For spatial data like robot trajectories, we care about:
        Δx_t = x_t - x_{t-1}  (motion between consecutive states)
    
    RoPE naturally captures this because:
        Attention(q_t, k_{t-1}) ∝ <R_t·q, R_{t-1}·k> = <q, R_1·k>
    
    The attention score depends on Δt=1, encoding the motion relationship!
    
    Contrast with sinusoidal: sin(t·ω) vs sin((t-1)·ω)
    The difference sin(t·ω) - sin((t-1)·ω) is not constant and depends on t,
    making it harder to learn consistent motion patterns.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute rotation frequencies: θ_i = base^(-2i/d)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Precompute cos and sin for all positions"""
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotation in 2D subspaces: [x0,x1,x2,x3] -> [-x1,x0,-x3,x2]"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to Q and K"""
        seq_len = q.shape[2]
        cos = self.cos_cached[:seq_len, ...]
        sin = self.sin_cached[:seq_len, ...]
        
        q_rot = (q * cos) + (self.rotate_half(q) * sin)
        k_rot = (k * cos) + (self.rotate_half(k) * sin)
        return q_rot, k_rot


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Sinusoidal Positional Embeddings (Original Transformer)
    
    MATHEMATICAL FOUNDATION:
    -----------------------
    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    
    These are ADDED to token embeddings.
    
    WHY LESS SUITABLE FOR SPATIAL DATA:
    ----------------------------------
    1. **Absolute encoding**: Position t gets embedding PE(t), regardless of
       context. Robot motion is inherently relative!
    
    2. **Additive interference**: Adding PE to embeddings can interfere with
       semantic content, especially problematic when embeddings encode
       continuous physical states
    
    3. **Bounded extrapolation**: Fixed wavelengths limit generalization to
       longer sequences
    
    4. **No natural relative property**: Attention score between positions
       t and t-k involves complex trigonometric identities, harder to learn
    
    COMPARISON IN TRAJECTORY PREDICTION:
    -----------------------------------
    Consider predicting motion: x_t = x_{t-1} + Δx
    
    With RoPE: Model learns Δx directly from relative attention
    With Sinusoidal: Model must decode absolute positions t and t-1,
                     then infer Δx - more complex!
    """

    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        
        # Create positional encoding matrix
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        
        pe = torch.zeros(max_seq_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input
        Args:
            x: (batch, seq_len, dim)
        Returns:
            x + positional_encoding
        """
        seq_len = x.shape[1]
        return x + self.pe[:seq_len, :].unsqueeze(0)


# ============================================================================
# CAUSAL SELF-ATTENTION WITH KV-CACHING
# ============================================================================

class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention with KV-Caching
    
    KV-CACHING EXPLANATION:
    ----------------------
    Problem: Autoregressive generation recomputes K, V for all previous tokens
             at each step, causing O(n²) complexity.
    
    Solution: Cache K, V matrices from previous steps!
    
    Without caching (naive):
        Step 1: Compute K,V for token 1
        Step 2: Compute K,V for tokens 1,2  (recomputes token 1!)
        Step 3: Compute K,V for tokens 1,2,3  (recomputes tokens 1,2!)
        ...
        Step n: Compute K,V for all n tokens  (recomputes everything!)
    
    With caching:
        Step 1: Compute K,V for token 1, cache them
        Step 2: Compute K,V for token 2, concatenate with cache
        Step 3: Compute K,V for token 3, concatenate with cache
        ...
        Step n: Compute K,V for token n only, use cached history
    
    SPEEDUP: O(n²) -> O(n), typically 3-10x faster for generation!
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1, pos_encoding: str = 'rope'):
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.pos_encoding = pos_encoding

        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Positional encoding
        if pos_encoding == 'rope':
            self.rope = RotaryPositionalEmbedding(self.head_dim)
        elif pos_encoding == 'sinusoidal':
            pass  # No setup needed in attention layer
        else:
            raise ValueError(f"Unknown pos_encoding: {pos_encoding}")

    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        """
        Forward pass with optional KV-caching
        
        Args:
            x: Input (batch, seq_len, dim)
            mask: Attention mask
            return_attention: Return attention weights for visualization
            past_kv: Cached (K, V) from previous steps
            use_cache: Whether to return K, V for caching
        
        Returns:
            output: (batch, seq_len, dim)
            attention_weights: Optional, for visualization
            present_kv: Optional (K, V) for caching
        """
        batch_size, seq_len, _ = x.shape

        # DEBUG: Time entire attention operation
        if seq_len == 1 and past_kv is not None:
            torch.cuda.synchronize()
            attn_start = time.time()

        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE if selected (after projection)
        if self.pos_encoding == 'rope':
            q, k = self.rope(q, k)

        # KV-CACHING: Concatenate with past if provided
        if past_kv is not None:
            past_k, past_v = past_kv

                # DEBUG: Time the concatenation
            if seq_len == 1:
                torch.cuda.synchronize()  # Ensure previous ops finished
                concat_start = time.time()

            # DEBUG: Check cache concatenation
            # if seq_len == 1:  # Only during generation with cache
            #     print(f"    [ATTENTION] Before concat - k.shape: {k.shape}, past_k.shape: {past_k.shape}")
            k = torch.cat([past_k, k], dim=2)  # Concatenate along seq_len dimension
            v = torch.cat([past_v, v], dim=2)
            if seq_len == 1:
                # print(f"    [ATTENTION] After concat - k.shape: {k.shape}, v.shape: {v.shape}")

                torch.cuda.synchronize()  # Wait for concat to finish
                concat_time = (time.time() - concat_start) * 1000  # ms
                print(f"    [TIMING] Concat took: {concat_time:.3f} ms")
                print(f"    [TIMING] Cache size: {past_k.shape[2]} tokens")
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Adjust mask for cached sequence length
        full_seq_len = k.shape[2]
        if mask is None:
            mask = torch.tril(torch.ones(full_seq_len, full_seq_len, device=x.device))
        
        # For generation with cache, we only need mask for new tokens attending to all tokens
        if past_kv is not None:
            # q is new tokens, k includes history
            # New tokens can attend to all history + themselves
            mask = mask[-seq_len:, :]  # Take last seq_len rows
        
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        out = self.out_proj(out)
        out = self.resid_dropout(out)

        # Prepare cache for next step
        present_kv = (k, v) if use_cache else None
        
        # DEBUG: Verify cache is created
        if use_cache and seq_len == 1:
            print(f"    [ATTENTION] Creating cache - K.shape: {k.shape}, V.shape: {v.shape}")

        # DEBUG: Total attention timing
        if seq_len == 1 and past_kv is not None:
            torch.cuda.synchronize()
            attn_time = (time.time() - attn_start) * 1000
            print(f"    [TIMING] Total attention: {attn_time:.3f} ms")

        if return_attention:
            return out, attn_weights, present_kv
        return out, None, present_kv


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network"""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer decoder block with optional KV-caching"""
    def __init__(self, dim: int, num_heads: int, ff_hidden_dim: int, dropout: float = 0.1, pos_encoding: str = 'rope'):
        super().__init__()
        self.attention = CausalSelfAttention(dim, num_heads, dropout, pos_encoding)
        self.feed_forward = FeedForward(dim, ff_hidden_dim, dropout)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, attn_weights, present_kv = self.attention(
            self.norm1(x), mask, return_attention, past_kv, use_cache
        )
        x = x + attn_out
        x = x + self.feed_forward(self.norm2(x))
        return x, attn_weights, present_kv


class DecoderOnlyTransformer(nn.Module):
    """
    Decoder-only Transformer with KV-caching support
    
    MASTERY FEATURES:
    ----------------
    1. Configurable positional encoding: RoPE or Sinusoidal
    2. KV-caching for efficient autoregressive generation
    3. Inference speed benchmarking capabilities
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
        pos_encoding: str = 'rope'  # 'rope' or 'sinusoidal'
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.pos_encoding = pos_encoding

        self.token_embedding = nn.Embedding(vocab_size, dim)

        if pos_encoding == 'sinusoidal':
            self.sinusoidal_pe = SinusoidalPositionalEmbedding(dim, max_seq_len)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, ff_hidden_dim, dropout, pos_encoding)
            for _ in range(num_layers)
        ])

        self.norm_final = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
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
        return_attention: bool = False,
        past_kvs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[List]]:
        batch_size, seq_len = input_ids.shape

        x = self.token_embedding(input_ids)

        if self.pos_encoding == 'sinusoidal':
            x = self.sinusoidal_pe(x)
        
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        if past_kvs is not None:
            # Adjust mask for cached context
            past_length = past_kvs[0][0].shape[2]
            full_length = past_length + seq_len
            mask = torch.tril(torch.ones(full_length, full_length, device=x.device))

        attention_weights = None
        present_kvs = [] if use_cache else None

        for i, block in enumerate(self.blocks):
            past_kv = past_kvs[i] if past_kvs is not None else None
            return_attn = return_attention and (i == len(self.blocks) - 1)
            
            x, attn, present_kv = block(x, mask, return_attn, past_kv, use_cache)
            
            if attn is not None:
                attention_weights = attn
            if use_cache:
                present_kvs.append(present_kv)

        x = self.norm_final(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=-1,
            )

        return logits, loss, attention_weights, present_kvs

    @torch.no_grad()
    def generate_with_cache(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Generate with KV-caching for speed
        Returns: (generated_tokens, generation_time)
        """
        start_time = time.time()
        
        past_kvs = None
        
        for step in range(max_new_tokens):
            # # DEBUG 1: Check cache status at each step
            # if step < 3:  # Only print first 3 steps to avoid spam
            #     print(f"\n[DEBUG Step {step}]")
            #     print(f"  past_kvs is None: {past_kvs is None}")
            #     if past_kvs is not None:
            #         print(f"  Number of cached layers: {len(past_kvs)}")
            #         print(f"  Cache K shape (layer 0): {past_kvs[0][0].shape}")
            #         print(f"  Cache V shape (layer 0): {past_kvs[0][1].shape}")
            
            # Only process new token if we have cache
            idx_cond = input_ids if past_kvs is None else input_ids[:, -1:]
            
            # # DEBUG 2: Check input shape
            # if step < 3:
            #     print(f"  Input shape to forward: {idx_cond.shape}")
            
            logits, _, _, present_kvs = self.forward(idx_cond, past_kvs=past_kvs, use_cache=True)
            
            # # DEBUG 3: Check if present_kvs is returned
            # if step < 3:
            #     print(f"  present_kvs is None: {present_kvs is None}")
            #     if present_kvs is not None:
            #         print(f"  New cache K shape (layer 0): {present_kvs[0][0].shape}")
            #         print(f"  New cache V shape (layer 0): {present_kvs[0][1].shape}")
            
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            past_kvs = present_kvs
            
            # # DEBUG 4: Verify cache was assigned
            # if step < 3:
            #     print(f"  After assignment - past_kvs is None: {past_kvs is None}")
        
        torch.cuda.synchronize()
        generation_time = time.time() - start_time
        return input_ids, generation_time

    @torch.no_grad()
    def generate_without_cache(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Generate without caching (naive implementation)
        Returns: (generated_tokens, generation_time)
        """
        start_time = time.time()
        
        for _ in range(max_new_tokens):
            logits, _, _, _ = self.forward(input_ids, use_cache=False)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        torch.cuda.synchronize()
        generation_time = time.time() - start_time
        return input_ids, generation_time


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(
    model: DecoderOnlyTransformer,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (states, actions) in enumerate(dataloader):
        actions = actions.to(device)
        input_ids = actions[:, :-1]
        targets = actions[:, 1:]
        
        logits, loss, _, _ = model(input_ids, targets)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    return avg_loss


# ============================================================================
# MASTERY: ABLATION STUDY & BENCHMARKING
# ============================================================================

def run_ablation_study(
    data_path: str = 'data/trajectories.pkl',
    device: torch.device = None,
    num_epochs: int = 10,
    batch_size: int = 32
):
    """
    MASTERY REQUIREMENT: Ablation study comparing RoPE vs Sinusoidal
    
    Trains two identical models with different positional encodings:
    1. RoPE (Rotary Position Embedding)
    2. Sinusoidal (Original Transformer)
    
    Compares:
    - Training loss convergence
    - Final performance
    - Attention pattern quality
    """
    import pickle
    from generate_data import create_dataloaders
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*70)
    print("MASTERY: ABLATION STUDY - RoPE vs Sinusoidal Positional Encoding")
    print("="*70)
    
    # Load data
    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)
    train_loader, val_loader = create_dataloaders(dataset, batch_size, 0.9)
    
    # Model configurations
    model_config = {
        'vocab_size': 256,
        'dim': 256,
        'num_layers': 4,
        'num_heads': 8,
        'ff_hidden_dim': 1024,
        'max_seq_len': 50,
        'dropout': 0.1
    }
    
    results = {}
    
    for pos_encoding in ['rope', 'sinusoidal']:
        print(f"\n{'='*70}")
        print(f"Training with {pos_encoding.upper()} positional encoding")
        print(f"{'='*70}")
        
        # Create model
        model = DecoderOnlyTransformer(**model_config, pos_encoding=pos_encoding).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        
        # Train
        losses = []
        for epoch in range(num_epochs):
            train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
            losses.append(train_loss)
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}")
        
        model.eval()

        results[pos_encoding] = {
            'losses': losses,
            'model': model,
            'final_loss': losses[-1]
        }
        
        # Save checkpoint
        import os
        os.makedirs('checkpoints', exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'losses': losses,
            'pos_encoding': pos_encoding
        }, f'checkpoints/model_{pos_encoding}.pt')
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(results['rope']['losses'], label='RoPE', linewidth=2, marker='o')
    plt.plot(results['sinusoidal']['losses'], label='Sinusoidal', linewidth=2, marker='s')
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.title('Training Loss Comparison', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    improvement = [r['losses'][0] - r['losses'][-1] for r in results.values()]
    plt.bar(['RoPE', 'Sinusoidal'], improvement, color=['#2E86AB', '#A23B72'])
    plt.ylabel('Loss Improvement', fontsize=12, fontweight='bold')
    plt.title('Total Improvement', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('ablation_study.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS")
    print("="*70)
    print(f"\nRoPE:")
    print(f"  Initial loss: {results['rope']['losses'][0]:.4f}")
    print(f"  Final loss:   {results['rope']['losses'][-1]:.4f}")
    print(f"  Improvement:  {results['rope']['losses'][0] - results['rope']['losses'][-1]:.4f}")
    
    print(f"\nSinusoidal:")
    print(f"  Initial loss: {results['sinusoidal']['losses'][0]:.4f}")
    print(f"  Final loss:   {results['sinusoidal']['losses'][-1]:.4f}")
    print(f"  Improvement:  {results['sinusoidal']['losses'][0] - results['sinusoidal']['losses'][-1]:.4f}")
    
    winner = 'RoPE' if results['rope']['final_loss'] < results['sinusoidal']['final_loss'] else 'Sinusoidal'
    print(f"\n✓ Best performer: {winner}")
    print(f"  Advantage: {abs(results['rope']['final_loss'] - results['sinusoidal']['final_loss']):.4f} loss points")
    
    return results


def benchmark_kv_caching(
    checkpoint_path: str = 'checkpoints/best_model.pt',
    device: torch.device = None,
    num_trials: int = 10,
    sequence_lengths: List[int] = [10, 20, 30, 40, 50]
):
    """
    MASTERY REQUIREMENT: Benchmark inference speed with/without KV-caching
    
    Measures generation time for different sequence lengths:
    - With KV-caching (efficient)
    - Without KV-caching (naive)
    
    Expected speedup: 3-10x for longer sequences
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*70)
    print("MASTERY: KV-CACHING INFERENCE SPEED BENCHMARK")
    print("="*70)
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = DecoderOnlyTransformer(
        vocab_size=256,
        dim=256,
        num_layers=4,
        num_heads=8,
        ff_hidden_dim=1024,
        max_seq_len=50,
        dropout=0.1
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    results = {'with_cache': {}, 'without_cache': {}}
    
    for seq_len in sequence_lengths:
        print(f"\nBenchmarking sequence length: {seq_len}")
        
        # Warmup
        input_ids = torch.randint(0, 256, (1, 5), device=device)
        model.generate_with_cache(input_ids, max_new_tokens=5)
        
        # With cache
        times_with = []
        for _ in range(num_trials):
            input_ids = torch.randint(0, 256, (1, 5), device=device)
            _, gen_time = model.generate_with_cache(input_ids, max_new_tokens=seq_len)
            times_with.append(gen_time)
        
        # Without cache
        times_without = []
        for _ in range(num_trials):
            input_ids = torch.randint(0, 256, (1, 5), device=device)
            _, gen_time = model.generate_without_cache(input_ids, max_new_tokens=seq_len)
            times_without.append(gen_time)
        
        avg_with = np.mean(times_with)
        avg_without = np.mean(times_without)
        speedup = avg_without / avg_with
        
        results['with_cache'][seq_len] = avg_with
        results['without_cache'][seq_len] = avg_without
        
        print(f"  With cache:    {avg_with*1000:.2f} ms")
        print(f"  Without cache: {avg_without*1000:.2f} ms")
        print(f"  Speedup:       {speedup:.2f}x")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(sequence_lengths, [results['without_cache'][l]*1000 for l in sequence_lengths], 
             label='Without Cache', marker='o', linewidth=2)
    plt.plot(sequence_lengths, [results['with_cache'][l]*1000 for l in sequence_lengths], 
             label='With Cache', marker='s', linewidth=2)
    plt.xlabel('Sequence Length', fontsize=12, fontweight='bold')
    plt.ylabel('Generation Time (ms)', fontsize=12, fontweight='bold')
    plt.title('KV-Cache Performance', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    speedups = [results['without_cache'][l] / results['with_cache'][l] for l in sequence_lengths]
    plt.plot(sequence_lengths, speedups, marker='o', linewidth=2, color='#2E86AB')
    plt.xlabel('Sequence Length', fontsize=12, fontweight='bold')
    plt.ylabel('Speedup Factor', fontsize=12, fontweight='bold')
    plt.title('KV-Cache Speedup', fontsize=14, fontweight='bold')
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kv_cache_benchmark.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    print(f"\nAverage speedup: {np.mean(speedups):.2f}x")
    print(f"Max speedup:     {np.max(speedups):.2f}x (at length {sequence_lengths[np.argmax(speedups)]})")
    print("\n✓ Benchmark complete! Saved to kv_cache_benchmark.png")
    
    return results


def main():
    """Main training and evaluation script"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Train base model
    print("\n" + "="*70)
    print("1. Training Base Model with RoPE")
    print("="*70)
    
    import pickle
    from generate_data import create_dataloaders
    
    with open('data/trajectories.pkl', 'rb') as f:
        dataset = pickle.load(f)
    train_loader, val_loader = create_dataloaders(dataset, batch_size=32, train_split=0.9)
    
    model = DecoderOnlyTransformer(
        vocab_size=256,
        dim=256,
        num_layers=4,
        num_heads=8,
        ff_hidden_dim=1024,
        max_seq_len=50,
        dropout=0.1,
        pos_encoding='rope'
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    losses = []
    for epoch in range(10):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        losses.append(train_loss)
        print(f"Epoch {epoch+1}/10 - Loss: {train_loss:.4f}")

    model.eval()  # Switch to eval mode after training
    
    # Save checkpoint
    import os
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'losses': losses,
    }, 'checkpoints/best_model.pt')
    
    print("\n✓ Base model training complete!")
    
    # Run mastery evaluations
    print("\n" + "="*70)
    print("2. Running Mastery Evaluations")
    print("="*70)
    
    # Ablation study
    ablation_results = run_ablation_study(device=device)
    
    # KV-cache benchmark
    benchmark_results = benchmark_kv_caching(device=device)
    
    print("\n" + "="*70)
    print("✅ ALL MASTERY REQUIREMENTS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  ✓ ablation_study.png - RoPE vs Sinusoidal comparison")
    print("  ✓ kv_cache_benchmark.png - Inference speed analysis")
    print("  ✓ checkpoints/model_rope.pt")
    print("  ✓ checkpoints/model_sinusoidal.pt")


if __name__ == "__main__":
    main()
