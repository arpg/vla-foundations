"""
Scratch-1: The Transformer Backbone
CSCI 7000 - VLA Foundations

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple    
import pickle
import matplotlib.pyplot as plt
import time
import statistics

LOSS_STEPS = []
LOSS_VALUES = []
GLOBAL_STEP = 0 

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
 
        self.scale = nn.Parameter(torch.ones(dim)) # scalable parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
        Returns:
            Normalized tensor of same shape
        """
        # Implementing RMSNorm
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        norm = x * rms
        return norm * self.scale 



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

    def forward(self, q: torch.Tensor, k: torch.Tensor,position_offset=0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to query and key tensors

        Args:
            q: Query tensor (batch, num_heads, seq_len, head_dim)
            k: Key tensor (batch, num_heads, seq_len, head_dim)
        Returns:
            Rotated (q, k) tensors
        """
        seq_len = q.shape[2]

        # Get cached cos/sin values
        cos = self.cos_cached[position_offset:position_offset+seq_len, ...]
        sin = self.sin_cached[position_offset:position_offset+seq_len, ...]
        cos = cos[None,None,:,:]
        sin = sin[None,None,:,:]
        
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
        self.last_attn = None
        self.max_seq_len = 2048

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,past_kv:Optional[Tuple[torch.Tensor, torch.Tensor]] = None,use_cache=False) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Optional attention mask (seq_len, seq_len)
        Returns:
            Output tensor (batch, seq_len, dim) with no cache
        """
        batch_size, seq_len, _ = x.shape

        # Project input to Q, K, V
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3*dim)
        
        # Split into Q, K, V and reshape for multi-head attention
        q,k,v = qkv.chunk(3,dim=-1)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        if not use_cache:
        # RoPE for full sequence (offset 0)
            q, k = self.rope(q, k, position_offset=0)

            # Attention scores: (B,H,T,T)
            scores = (q @ k.transpose(-2, -1)) * self.scale

            # Causal mask for training: lower triangle
            # (T,T) broadcast to (B,H,T,T)
            causal = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
            if mask is not None:
                scores = scores.masked_fill(mask==0, float("-inf"))
            else:   
                scores = scores.masked_fill(~causal, float("-inf"))

            attn = F.softmax(scores, dim=-1)
            if not self.training:
                self.last_attn = attn.detach().cpu()
                
            attn = self.attn_dropout(attn)

            out = attn @ v  # (B,H,T,Hd)

            out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
            out = self.out_proj(out)
            out = self.resid_dropout(out)
            
            # No cache
            return out
        
        
        '''KV Cache Implementation'''
        position_offset = 0
        if past_kv is None:
                # allocate cache once
                max_len = self.max_seq_len
                k_cache = torch.empty(batch_size, self.num_heads, max_len, self.head_dim, device=x.device, dtype=k.dtype)
                v_cache = torch.empty_like(k_cache)
                pos = 0
        else:
                k_cache, v_cache, pos = past_kv
        q, k = self.rope(q, k, position_offset=pos)
            # write current k,v
        k_cache[:, :, pos:pos+seq_len, :] = k
        v_cache[:, :, pos:pos+seq_len, :] = v

        # use cached prefix
        k = k_cache[:, :, :pos+seq_len, :]
        v = v_cache[:, :, :pos+seq_len, :]        
        scores = (q @ k.transpose(-2,-1))*self.scale
        if seq_len>1:
            kv_len = pos+seq_len
            i = torch.arange(seq_len, device=x.device).unsqueeze(1)      # (T,1)
            j = torch.arange(kv_len, device=x.device).unsqueeze(0) # (1,kv_len)
            causal = j <= (pos + i)                                # (T,kv_len)
            scores = scores.masked_fill(~causal, float("-inf"))
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        out = attn_weights @ v
        
        out = out.transpose(1,2).contiguous().view(batch_size,seq_len,self.dim)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        
        present_kv = (k_cache, v_cache, pos + seq_len)
        
        # with cache
        return out, present_kv


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

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,past_kv=None,use_cache=False) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Optional attention mask
        Returns:
            Output tensor (batch, seq_len, dim)
        """
        if use_cache:
            attn_out,present_kv = self.attention(self.norm1(x),mask,past_kv=past_kv,use_cache=True)
        else:
            attn_out = self.attention(self.norm1(x),mask)
            present_kv = None
            
        # Pre-norm architecture (norm before attention/FF)
        x = x + attn_out
        x = x + self.feed_forward(self.norm2(x))
        
        if use_cache:
            return x,present_kv
        
        return x


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
        
        #sinusoidal
        self.pe = self.build_sinusoidal_pos_emb(max_seq_len,dim,device="cuda")
        self.register_buffer("pos_emb",self.pe,persistent=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights following standard Transformer initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def build_sinusoidal_pos_emb(self,max_len, dim, device):
        pe = torch.zeros(max_len, dim, device=device)
        position = torch.arange(0, max_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # (max_len, dim)

            
    @torch.no_grad()    
    def forward_with_cache(self, input_ids, past_kvs=None):
        """
        input_ids: (B, T) where T is usually 1 in generation
        past_kvs: list length num_layers of (k,v) or None
        Returns: logits, new_past_kvs
        """
        B, T = input_ids.shape
        x = self.token_embedding(input_ids)

        new_past_kvs = []
        for i, block in enumerate(self.blocks):
            past_kv = None if past_kvs is None else past_kvs[i]
            x, present_kv = block(x, past_kv=past_kv, use_cache=True)
            new_past_kvs.append(present_kv)

        x = self.norm_final(x)
        logits = self.lm_head(x)  # (B,T,V)
        return logits, new_past_kvs
    
    
    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: Input token indices (batch, seq_len)
            targets: Target token indices for training (batch, seq_len)
        Returns:
            logits: Output logits (batch, seq_len, vocab_size)
            loss: Cross-entropy loss if targets provided, else None
        """
        batch_size, seq_len = input_ids.shape

        # Embed tokens
        x = self.token_embedding(input_ids)  # (batch, seq_len, dim)
        
        #NOTE : Sinusoidal encoding kept for comparison
        # x= x+ self.pos_emb[:x.size(1),:].unsqueeze(0)

        # Create causal mask (lower triangular)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)

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

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        use_kv_cache = False
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
        self.eval()
        if use_kv_cache:
            # Process prompt and build initial cache
            logits, past_kvs = self.forward_with_cache(input_ids, past_kvs=None)
            
            # Generate new tokens one at a time with cache
            for _ in range(max_new_tokens):
                # Only need to process the last token
                logits = logits[:, -1, :] / temperature
                
                # Optional top-k sampling
                if top_k is not None:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < values[:, [-1]]] = -float('Inf')
                
                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                logits, past_kvs = self.forward_with_cache(next_token, past_kvs=past_kvs)
        
        else:
            for _ in range(max_new_tokens):
                # Crop context 
                input_context = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]

                # Forward pass
                logits, _ = self.forward(input_context)
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
) -> float:
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
    global LOSS_STEPS, LOSS_VALUES, GLOBAL_STEP

    for batch in dataloader:
        states,actions,target = batch
        states = states.to(device)
        actions = actions.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        logits,loss = model(actions,target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        LOSS_STEPS.append(GLOBAL_STEP)
        LOSS_VALUES.append(loss.item())
        GLOBAL_STEP += 1
        total_loss += loss.item()
        num_batches += 1
        
        if num_batches %100 ==0:
           print(f"Epoch {epoch} | Batch {num_batches}/{len(dataloader)} | Loss {loss.item():.4f}") 
    return total_loss/num_batches  

@torch.no_grad()
def eval_loss(model, dataloader, device):
    model.eval()
    total, n = 0.0, 0
    for states, actions, targets in dataloader:
        actions = actions.to(device)
        targets = targets.to(device)
        _, loss = model(actions, targets)
        total += loss.item()
        n += 1
    return total / max(n, 1)

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
    
    '''Dataset Creation'''
    file_path = "D:/University/OneDrive - UCB-O365/vla-foundations/data/trajectories.pkl"
    with open(file_path,'rb') as f:
        data = pickle.load(f)
    print("dataset keys",data.keys())
    states = data["states"].float()
    actions = data["actions"].long() 
    N = actions.size(0)
    perm = torch.randperm(N)

    split = int(0.9 * N)
    train_idx = perm[:split]
    val_idx   = perm[split:]

    x_actions = actions[:, :-1]
    y_actions = actions[:, 1:]
    x_states  = states[:, :-1, :]  

    train_ds = torch.utils.data.TensorDataset(x_states[train_idx], x_actions[train_idx], y_actions[train_idx])
    val_ds   = torch.utils.data.TensorDataset(x_states[val_idx],   x_actions[val_idx],   y_actions[val_idx])

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    
    '''Model Definition'''
    model = DecoderOnlyTransformer(
        vocab_size= vocab_size ,
        dim = dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_hidden_dim=ff_hidden_dim,
        max_seq_len=max_seq_len,
    )   
    model.to(device)

    #Optimizer 
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}")
        val = eval_loss(model, val_loader, device)
        print(f"Epoch {epoch+1}: train {train_loss:.4f} | val {val:.4f}")

    #  Save checkpoint
    torch.save(model.state_dict(), "checkpoints/best_model.pt")

    # Visualizing Training Loss Curve
    plt.figure()
    plt.plot(LOSS_STEPS, LOSS_VALUES)
    plt.xlabel("Iteration (global step)")
    plt.ylabel("Training Loss (CE)")
    plt.title("Training Loss Curve")
    plt.show()
    
    '''Attention Map Visualization'''
    
    # model.eval()
    # sample_actions = actions[0].unsqueeze(0).to(device)   # (1, 50)
    # input_ids = sample_actions[:, :-1]                   # (1, 49)

    # with torch.no_grad():
    #     _ = model(input_ids)  # forward pass populates last_attn

    # layer_idx = 0
    # head_idx = 0
    # att = model.blocks[layer_idx].attention.last_attn[0, head_idx]  # (T, T)

    # plt.figure()
    # plt.imshow(att.float().cpu().numpy(), aspect="auto")
    # plt.xlabel("Key position (attended-to token index)")
    # plt.ylabel("Query position (current token index)")
    # plt.title(f"Attention map — layer {layer_idx}, head {head_idx}")
    # plt.colorbar()
    # plt.show()
    
    '''KV Cache Comparison'''
    state_dict = torch.load("./checkpoints/best_model.pt",map_location="cuda")
    model.load_state_dict(state_dict)
    model.to(device="cuda")
    model.eval()
    prompt_len = 50
    max_new = 500
    prompt = actions[0,:prompt_len].unsqueeze(0).to("cuda")
    
    def run_no_cache_timed():
        with torch.no_grad():
            t0 = time.perf_counter()
            # Prompt phase
            result = model.generate(
                input_ids=prompt,
                max_new_tokens=max_new,
                temperature=1.0,
                top_k=None,
            )
            t1 = time.perf_counter()
        return result, t1 - t0

    def run_kv_cache_timed():
        with torch.no_grad():
            t0 = time.perf_counter()
            result = model.generate(
                input_ids=prompt,
                max_new_tokens=max_new,
                temperature=1.0,
                top_k=None,
                use_kv_cache=True
            )
            t1 = time.perf_counter()
        return result, t1 - t0

    
    _, time_no_cache = run_no_cache_timed()
    _, time_with_cache = run_kv_cache_timed()
    print(f"No cache total: {time_no_cache*1000:.2f} ms")
    print(f"With cache total: {time_with_cache*1000:.2f} ms")

if __name__ == "__main__":
    main()
