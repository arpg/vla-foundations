"""
Scratch-1: NON-CAUSAL Transformer (for comparison with causal version)

This is IDENTICAL to backbone_mastery_final.py EXCEPT:
- Causal masking is DISABLED (model can see future tokens)
- This creates the "cheating" behavior we want to demonstrate

Usage:
    python backbone_no_causal_mask.py
    
This will train a model WITHOUT causal masking and save to:
    checkpoints/model_no_mask.pt
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
# COPY ALL THE CLASSES FROM YOUR ORIGINAL FILE
# (RMSNorm, RoPE, SinusoidalPE, FeedForward, etc.)
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
    """Rotary Position Embedding (RoPE)"""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

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
        """Rotation in 2D subspaces"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, start_pos: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings"""
        seq_len = q.shape[2]
        cos = self.cos_cached[start_pos : start_pos + seq_len, :].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[start_pos : start_pos + seq_len, :].unsqueeze(0).unsqueeze(0)
        
        q_rot = (q * cos) + (self.rotate_half(q) * sin)
        k_rot = (k * cos) + (self.rotate_half(k) * sin)
        return q_rot, k_rot


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


# ============================================================================
# MODIFIED ATTENTION - NO CAUSAL MASK!
# ============================================================================

class NonCausalSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention WITHOUT CAUSAL MASKING
    
    This allows the model to "cheat" by seeing future tokens!
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

        if pos_encoding == 'rope':
            self.rope = RotaryPositionalEmbedding(self.head_dim)

    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass WITHOUT causal masking"""
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE if selected
        start_pos = past_kv[0].shape[2] if past_kv is not None else 0
        if self.pos_encoding == 'rope':
            q, k = self.rope(q, k, start_pos=start_pos)

        # KV-CACHING
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # ⚠️ CRITICAL DIFFERENCE: NO CAUSAL MASKING!
        # Model can attend to ALL positions, including future!
        # This is the "cheating" we want to demonstrate
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        out = self.out_proj(out)
        out = self.resid_dropout(out)

        present_kv = (k, v) if use_cache else None

        if return_attention:
            return out, attn_weights, present_kv
        return out, None, present_kv


class TransformerBlock(nn.Module):
    """Transformer decoder block WITHOUT causal masking"""
    def __init__(self, dim: int, num_heads: int, ff_hidden_dim: int, dropout: float = 0.1, pos_encoding: str = 'rope'):
        super().__init__()
        self.attention = NonCausalSelfAttention(dim, num_heads, dropout, pos_encoding)
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


class NonCausalTransformer(nn.Module):
    """
    Transformer WITHOUT Causal Masking (for comparison)
    
    This model can "cheat" by attending to future tokens during training!
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

        # NO MASK - model can see everything!
        mask = None

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


# ============================================================================
# TRAINING FUNCTIONS (same as original)
# ============================================================================

def train_epoch(
    model: NonCausalTransformer,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (states, actions) in enumerate(dataloader):
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
            print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f} | PPL: {perplexity:.4f}")
    
    avg_loss = total_loss / num_batches
    avg_ppl = math.exp(avg_loss)
    return avg_loss, avg_ppl


def validate_epoch(
    model: NonCausalTransformer,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate"""
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


def main():
    """Train non-causal model for comparison"""
    import pickle
    from generate_data import create_dataloaders
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\n" + "="*70)
    print("TRAINING NON-CAUSAL TRANSFORMER (NO MASKING)")
    print("="*70)
    print("\n⚠️  This model can CHEAT by seeing future tokens!")
    print("   We expect MUCH lower training loss than the causal model\n")
    
    # Load data
    with open('data/trajectories.pkl', 'rb') as f:
        dataset = pickle.load(f)
    train_loader, val_loader = create_dataloaders(dataset, batch_size=32, train_split=0.9)
    
    # Same config as your large model
    model = NonCausalTransformer(
        vocab_size=256,
        dim=512,
        num_layers=8,
        num_heads=16,
        ff_hidden_dim=2048,
        max_seq_len=2048,
        dropout=0.1,
        pos_encoding='rope'
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    train_losses = []
    val_losses = []
    train_ppls = []
    val_ppls = []
    
    num_epochs = 10
    
    for epoch in range(num_epochs):
        train_loss, train_ppl = train_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, val_ppl = validate_epoch(model, val_loader, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ppls.append(train_ppl)
        val_ppls.append(val_ppl)
        
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Train: Loss={train_loss:.4f}, PPL={train_ppl:.2f}")
        print(f"  Val:   Loss={val_loss:.4f}, PPL={val_ppl:.2f}")
    
    # Save checkpoint
    import os
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_ppls': train_ppls,
        'val_ppls': val_ppls,
        'config': {
            'vocab_size': 256,
            'dim': 512,
            'num_layers': 8,
            'num_heads': 16,
            'ff_hidden_dim': 2048,
            'max_seq_len': 2048,
            'dropout': 0.1,
            'pos_encoding': 'rope'
        }
    }, 'checkpoints/model_no_mask.pt')
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nFinal Results (NO CAUSAL MASK):")
    print(f"  Train Loss: {train_losses[-1]:.4f}")
    print(f"  Val Loss:   {val_losses[-1]:.4f}")
    print(f"\n✓ Saved to: checkpoints/model_no_mask.pt")
    print("\nNext: Compare with checkpoints/model_rope_large.pt!")


if __name__ == "__main__":
    main()
