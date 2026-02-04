"""
Scratch-1: The Transformer Backbone - FINAL IMPLEMENTATION WITH VALIDATION
CSCI 7000 - VLA Foundations

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

    def forward(self, q: torch.Tensor, k: torch.Tensor, start_pos: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings with correct position offset for caching"""
        seq_len = q.shape[2]
        # Slice from the correct starting position in the precomputed cache
        cos = self.cos_cached[start_pos : start_pos + seq_len, :].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[start_pos : start_pos + seq_len, :].unsqueeze(0).unsqueeze(0)
        
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
            pass  # Handled at model level
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
        """Forward pass with optional KV-caching"""
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE if selected (with correct start position for caching)
        start_pos = past_kv[0].shape[2] if past_kv is not None else 0
        if self.pos_encoding == 'rope':
            q, k = self.rope(q, k, start_pos=start_pos)

        # KV-CACHING: Concatenate with past if provided
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Only create mask if processing more than one token (training/initial encoding)
        # In generation (seq_len=1), we can skip masking as single token sees everything
        if seq_len > 1:
            if mask is None:
                full_seq_len = k.shape[2]
                mask = torch.tril(torch.ones(full_seq_len, full_seq_len, device=x.device))
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        out = self.out_proj(out)
        out = self.resid_dropout(out)

        # Prepare cache for next step
        present_kv = (k, v) if use_cache else None

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
        pos_encoding: str = 'rope'
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
        
        # Only create a mask if we are processing more than one token (training/encoding)
        # In generation (seq_len=1), we can just pass None and skip the work
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)) if seq_len > 1 else None

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
        """Generate with KV-caching for speed"""
        start_time = time.time()
        
        past_kvs = None
        
        for step in range(max_new_tokens):
            idx_cond = input_ids if past_kvs is None else input_ids[:, -1:]
            logits, _, _, present_kvs = self.forward(idx_cond, past_kvs=past_kvs, use_cache=True)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            past_kvs = present_kvs
        
        if input_ids.is_cuda:
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
        """Generate without caching (naive implementation)"""
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
        
        if input_ids.is_cuda:
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
) -> Tuple[float, float]:
    """Train for one epoch and return (loss, perplexity)"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (states, actions) in enumerate(dataloader):
        global_step = epoch * len(dataloader) + batch_idx
        
        actions = actions.to(device)
        input_ids = actions[:, :-1].contiguous()
        targets = actions[:, 1:].contiguous()
        
        logits, loss, _, _ = model(input_ids, targets)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 100 == 0:
            perplexity = math.exp(loss.item())
            print(f"Epoch {epoch+1} | Step {global_step} | Loss: {loss.item():.4f} | PPL: {perplexity:.4f}")
            
        if global_step > 0 and global_step % 1000 == 0:
            import os
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({'model_state_dict': model.state_dict()}, f"checkpoints/step_{global_step}.pt")
    
    avg_loss = total_loss / num_batches
    avg_ppl = math.exp(avg_loss)
    return avg_loss, avg_ppl


def validate_epoch(
    model: DecoderOnlyTransformer,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate on held-out data and return both Loss and Perplexity"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for states, actions in dataloader:
            actions = actions.to(device)
            input_ids = actions[:, :-1].contiguous()
            targets = actions[:, 1:].contiguous()
            
            _, loss, _, _ = model(input_ids, targets)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_ppl = math.exp(avg_loss)
    
    return avg_loss, avg_ppl


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
    Trains two identical models with different positional encodings:
    1. RoPE (Rotary Position Embedding)
    2. Sinusoidal (Original Transformer)
    
    Compares:
    - Training loss convergence
    - Validation loss (generalization)
    - Final performance
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
    
    # Model configurations (LARGER MODEL for demonstrating KV-cache)
    model_config = {
        'vocab_size': 256,
        'dim': 512,
        'num_layers': 8,
        'num_heads': 16,
        'ff_hidden_dim': 2048,
        'max_seq_len': 2048,
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
        
        # Train with validation
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            # FIXED: Unpack both loss and perplexity
            train_loss, train_ppl = train_epoch(model, train_loader, optimizer, device, epoch)
            val_loss, val_ppl = validate_epoch(model, val_loader, device)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs} - Train: {train_loss:.4f} (PPL: {train_ppl:.2f}), Val: {val_loss:.4f} (PPL: {val_ppl:.2f})")
        
        model.eval()

        results[pos_encoding] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'model': model,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1]
        }
        
        # Save checkpoint
        import os
        os.makedirs('checkpoints', exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'pos_encoding': pos_encoding,
            'config': model_config
        }, f'checkpoints/model_{pos_encoding}_large.pt')
    
    # Plot comparison with train + val curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Training curves
    ax = axes[0]
    ax.plot(results['rope']['train_losses'], label='RoPE Train', linewidth=2, marker='o', linestyle='-')
    ax.plot(results['rope']['val_losses'], label='RoPE Val', linewidth=2, marker='o', linestyle='--')
    ax.plot(results['sinusoidal']['train_losses'], label='Sinusoidal Train', linewidth=2, marker='s', linestyle='-')
    ax.plot(results['sinusoidal']['val_losses'], label='Sinusoidal Val', linewidth=2, marker='s', linestyle='--')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Final performance comparison
    ax = axes[1]
    x = np.arange(2)
    width = 0.35
    ax.bar(x - width/2, [results['rope']['final_train_loss'], results['sinusoidal']['final_train_loss']], 
           width, label='Train Loss', color='#2E86AB')
    ax.bar(x + width/2, [results['rope']['final_val_loss'], results['sinusoidal']['final_val_loss']], 
           width, label='Val Loss', color='#A23B72')
    ax.set_ylabel('Final Loss', fontsize=12, fontweight='bold')
    ax.set_title('Final Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['RoPE', 'Sinusoidal'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Generalization gap (train - val)
    ax = axes[2]
    rope_gap = results['rope']['final_train_loss'] - results['rope']['final_val_loss']
    sin_gap = results['sinusoidal']['final_train_loss'] - results['sinusoidal']['final_val_loss']
    ax.bar(['RoPE', 'Sinusoidal'], [rope_gap, sin_gap], color=['#2E86AB', '#A23B72'])
    ax.set_ylabel('Generalization Gap (Train - Val)', fontsize=12, fontweight='bold')
    ax.set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('ablation_study_large.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS")
    print("="*70)
    print(f"\nRoPE:")
    print(f"  Initial train loss: {results['rope']['train_losses'][0]:.4f}")
    print(f"  Final train loss:   {results['rope']['final_train_loss']:.4f}")
    print(f"  Final val loss:     {results['rope']['final_val_loss']:.4f}")
    print(f"  Generalization gap: {rope_gap:.4f}")
    
    print(f"\nSinusoidal:")
    print(f"  Initial train loss: {results['sinusoidal']['train_losses'][0]:.4f}")
    print(f"  Final train loss:   {results['sinusoidal']['final_train_loss']:.4f}")
    print(f"  Final val loss:     {results['sinusoidal']['final_val_loss']:.4f}")
    print(f"  Generalization gap: {sin_gap:.4f}")
    
    winner = 'RoPE' if results['rope']['final_val_loss'] < results['sinusoidal']['final_val_loss'] else 'Sinusoidal'
    print(f"\n✓ Best performer (by validation loss): {winner}")
    print(f"  Validation advantage: {abs(results['rope']['final_val_loss'] - results['sinusoidal']['final_val_loss']):.4f} loss points")
    
    return results


def benchmark_kv_caching(
    checkpoint_path: str = 'checkpoints/model_rope_large.pt',
    device: torch.device = None,
    num_trials: int = 10,
    sequence_lengths: List[int] = [50, 100, 200, 500, 1000]
):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*70)
    print("MASTERY: KV-CACHING INFERENCE SPEED BENCHMARK")
    print("="*70)
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to load config from checkpoint, fallback to hardcoded
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = {
            'vocab_size': 256,
            'dim': 512,
            'num_layers': 8,
            'num_heads': 16,
            'ff_hidden_dim': 2048,
            'max_seq_len': 2048,
            'dropout': 0.1
        }
    
    model = DecoderOnlyTransformer(**config).to(device)
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
    plt.savefig('kv_cache_benchmark_large.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    print(f"\nAverage speedup: {np.mean(speedups):.2f}x")
    print(f"Max speedup:     {np.max(speedups):.2f}x (at length {sequence_lengths[np.argmax(speedups)]})")
    print("\n✓ Benchmark complete! Saved to kv_cache_benchmark_large.png")
    
    return results


def benchmark_kv_caching_per_token(
    checkpoint_path: str = 'checkpoints/model_rope_large.pt',
    device: torch.device = None,
    max_sequence_length: int = 2000,
    num_trials: int = 3
):
    """
    ENHANCED BENCHMARK: Per-token timing
    
    This measures time for EACH individual token generation, showing the
    true O(n) vs O(n²) behavior without loop overhead masking the difference.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*70)
    print("MASTERY: PER-TOKEN KV-CACHING BENCHMARK")
    print("="*70)
    print(f"\nGenerating {max_sequence_length} tokens with per-token timing...")
    print("This will take a few minutes...")
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = {
            'vocab_size': 256,
            'dim': 512,
            'num_layers': 8,
            'num_heads': 16,
            'ff_hidden_dim': 2048,
            'max_seq_len': 2048,
            'dropout': 0.1
        }
    
    model = DecoderOnlyTransformer(**config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Storage for per-token times
    times_with_cache = []
    times_without_cache = []
    
    for trial in range(num_trials):
        print(f"\nTrial {trial + 1}/{num_trials}")
        
        # WITH CACHE
        print("  Measuring WITH cache...")
        input_ids = torch.randint(0, 256, (1, 10), device=device)
        past_kvs = None
        trial_times_with = []
        
        for step in range(max_sequence_length):
            idx_cond = input_ids if past_kvs is None else input_ids[:, -1:]
            
            # Time this single token
            if device.type == 'cuda':
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            else:
                start_time = time.time()
            
            with torch.no_grad():
                logits, _, _, present_kvs = model(idx_cond, past_kvs=past_kvs, use_cache=True)
            
            if device.type == 'cuda':
                end.record()
                torch.cuda.synchronize()
                token_time = start.elapsed_time(end) / 1000.0  # Convert to seconds
            else:
                token_time = time.time() - start_time
            
            trial_times_with.append(token_time)
            
            # Generate next token
            next_token = torch.multinomial(F.softmax(logits[:, -1, :], dim=-1), num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            past_kvs = present_kvs
            
            if (step + 1) % 500 == 0:
                print(f"    Generated {step + 1}/{max_sequence_length} tokens")
        
        # WITHOUT CACHE
        print("  Measuring WITHOUT cache...")
        input_ids = torch.randint(0, 256, (1, 10), device=device)
        trial_times_without = []
        
        for step in range(max_sequence_length):
            # Time this single token (recomputing all K,V)
            if device.type == 'cuda':
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            else:
                start_time = time.time()
            
            with torch.no_grad():
                logits, _, _, _ = model(input_ids, use_cache=False)
            
            if device.type == 'cuda':
                end.record()
                torch.cuda.synchronize()
                token_time = start.elapsed_time(end) / 1000.0
            else:
                token_time = time.time() - start_time
            
            trial_times_without.append(token_time)
            
            # Generate next token
            next_token = torch.multinomial(F.softmax(logits[:, -1, :], dim=-1), num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            if (step + 1) % 500 == 0:
                print(f"    Generated {step + 1}/{max_sequence_length} tokens")
        
        # Average across trials
        if trial == 0:
            times_with_cache = trial_times_with
            times_without_cache = trial_times_without
        else:
            times_with_cache = [(times_with_cache[i] + trial_times_with[i]) / 2 
                                for i in range(len(trial_times_with))]
            times_without_cache = [(times_without_cache[i] + trial_times_without[i]) / 2 
                                   for i in range(len(trial_times_without))]
    
    positions = list(range(len(times_with_cache)))
    
    # Calculate metrics
    avg_with = np.mean(times_with_cache) * 1000  # Convert to ms
    avg_without = np.mean(times_without_cache) * 1000
    final_with = times_with_cache[-1] * 1000
    final_without = times_without_cache[-1] * 1000
    
    print(f"\n" + "="*70)
    print("PER-TOKEN BENCHMARK RESULTS")
    print("="*70)
    print(f"\nAverage time per token:")
    print(f"  With cache:    {avg_with:.3f} ms/token")
    print(f"  Without cache: {avg_without:.3f} ms/token")
    print(f"  Divergence:    {avg_without / avg_with:.2f}x slower without cache")
    print(f"\nFinal token (position {max_sequence_length}):")
    print(f"  With cache:    {final_with:.2f} ms  (constant O(n))")
    print(f"  Without cache: {final_without:.2f} ms  (quadratic O(n²))")
    print(f"  Divergence:    {final_without / final_with:.2f}x slower without cache")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top plot: Per-token time over generation
    ax = axes[0]
    times_with_ms = [t * 1000 for t in times_with_cache]
    times_without_ms = [t * 1000 for t in times_without_cache]
    
    ax.plot(positions, times_with_ms, label=f'With KV Cache (O(n))', 
            linewidth=1, color='blue', alpha=0.7)
    ax.plot(positions, times_without_ms, label=f'Without KV Cache (O(n²))', 
            linewidth=1, color='red', alpha=0.7)
    
    ax.set_xlabel('Token Position in Generation', fontsize=14, fontweight='bold')
    ax.set_ylabel('Time per Token (ms)', fontsize=14, fontweight='bold')
    ax.set_title('Per-Token Generation Time: KV Cache vs No Cache', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Bottom plot: Per-token time vs total sequence length (shows divergence)
    ax = axes[1]
    
    # Compute cumulative sequence lengths
    seq_lengths = [10 + i for i in positions]  # Started with 10 tokens
    
    ax.plot(seq_lengths, times_with_ms, label='With KV Cache (O(n))',
            linewidth=1, color='blue', alpha=0.7)
    ax.plot(seq_lengths, times_without_ms, label='Without KV Cache (O(n²))',
            linewidth=1, color='red', alpha=0.7)
    
    ax.set_xlabel('Total Sequence Length (input + generated)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Time per Token (ms)', fontsize=14, fontweight='bold')
    ax.set_title('Per-Token Time vs Sequence Length (Shows Divergence)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add vertical line at input length
    ax.axvline(x=10, color='gray', linestyle='--', alpha=0.5, 
               label=f'Input length (10)')
    
    plt.tight_layout()
    plt.savefig('kv_cache_per_token_benchmark.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Per-token benchmark plot saved to: kv_cache_per_token_benchmark.png")
    print("  This clearly shows O(n) vs O(n²) behavior!")
    
    return {
        'positions': positions,
        'times_with_cache': times_with_cache,
        'times_without_cache': times_without_cache,
        'avg_with': avg_with,
        'avg_without': avg_without
    }


def main():
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\nThis will train TWO models:")
    print("  1. RoPE (20M parameters)")
    print("  2. Sinusoidal (20M parameters)")
    print("\nThen run PER-TOKEN KV-caching benchmark")
    print("="*70)
    
    # Run ablation study (trains both models)
    ablation_results = run_ablation_study(device=device)
    
    # Per-token KV-cache benchmark (shows true O(n) vs O(n²))
    print("\n" + "="*70)
    print("Running per-token benchmark (this takes ~5-10 minutes)...")
    print("="*70)
    
    benchmark_results = benchmark_kv_caching_per_token(
        checkpoint_path='checkpoints/model_rope_large.pt',
        device=device,
        max_sequence_length=2000,  # Like your classmate
        num_trials=3
    )


    print("\nGenerated files:")
    print("  ✓ ablation_study_large.png - RoPE vs Sinusoidal with train/val curves")
    print("  ✓ kv_cache_per_token_benchmark.png - Per-token timing (O(n) vs O(n²))")
    print("  ✓ checkpoints/model_rope_large.pt - RoPE model (use for visualizations)")
    print("  ✓ checkpoints/model_sinusoidal_large.pt - Sinusoidal model")
    print("\nNext steps:")
    print("  1. Run generate_visualization_complete.py --all")
    print("  2. Check kv_cache_per_token_benchmark.png - should show clear divergence!")

if __name__ == "__main__":
    main()