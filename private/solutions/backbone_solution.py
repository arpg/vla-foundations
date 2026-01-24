"""
SOLUTION: Scratch-1 Transformer Backbone (Complete Implementation)

This is the complete solution for the Scratch-1 assignment.
DO NOT share this file with students.

Includes:
- Complete RMSNorm implementation
- Complete CausalSelfAttention implementation
- Complete training loop
- DINOv2 vision backbone integration (optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (COMPLETE SOLUTION)"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # SOLUTION: Learnable scale parameter
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SOLUTION: Complete RMSNorm implementation

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
        Returns:
            Normalized tensor of same shape
        """
        # SOLUTION: Compute RMS efficiently using rsqrt
        # rms = sqrt(mean(x^2) + eps) = 1 / rsqrt(mean(x^2) + eps)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        # SOLUTION: Normalize and apply learnable scale
        return x * rms * self.scale


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) - PROVIDED, NO CHANGES NEEDED
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
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Helper function to rotate tensor"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key tensors"""
        seq_len = q.shape[2]

        # Get cached cos/sin
        cos = self.cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(0)

        # Apply rotation
        q_rotated = (q * cos) + (self.rotate_half(q) * sin)
        k_rotated = (k * cos) + (self.rotate_half(k) * sin)

        return q_rotated, k_rotated


class CausalSelfAttention(nn.Module):
    """Multi-Head Causal Self-Attention (COMPLETE SOLUTION)"""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout

        # SOLUTION: Linear projections for Q, K, V
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        # SOLUTION: Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # SOLUTION: Rotary position embeddings
        self.rope = RotaryPositionalEmbedding(dim=self.head_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        SOLUTION: Complete attention implementation

        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Optional causal mask (seq_len, seq_len)
        Returns:
            Output tensor (batch, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape

        # SOLUTION: Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # SOLUTION: Reshape for multi-head attention
        # (batch, seq_len, dim) -> (batch, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # SOLUTION: Apply RoPE
        q, k = self.rope(q, k)

        # SOLUTION: Scaled dot-product attention
        # scores = Q @ K^T / sqrt(head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # SOLUTION: Apply causal mask (prevent attending to future tokens)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # SOLUTION: Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # SOLUTION: Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # SOLUTION: Reshape back
        # (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.dim)

        # SOLUTION: Output projection
        output = self.o_proj(attn_output)
        output = self.resid_dropout(output)

        return output


class FeedForward(nn.Module):
    """Feed-Forward Network (PROVIDED, NO CHANGES NEEDED)"""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Single Transformer Block (PROVIDED, uses student implementations)"""

    def __init__(self, dim: int, num_heads: int, ff_hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = CausalSelfAttention(dim, num_heads, dropout)
        self.feed_forward = FeedForward(dim, ff_hidden_dim, dropout)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pre-norm Transformer block"""
        # Self-attention with residual
        x = x + self.attention(self.norm1(x), mask)

        # Feed-forward with residual
        x = x + self.feed_forward(self.norm2(x))

        return x


class DecoderOnlyTransformer(nn.Module):
    """
    Decoder-Only Transformer for Action Prediction

    SOLUTION: Complete implementation with optional DINOv2 integration
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
        use_vision_backbone: bool = False,
        vision_backbone_frozen: bool = True,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim = dim
        self.use_vision_backbone = use_vision_backbone

        # SOLUTION: Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, dim)

        # SOLUTION: Optional DINOv2 vision backbone
        if use_vision_backbone:
            try:
                # Load pretrained DINOv2
                self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

                # SOLUTION: Freeze DINOv2 parameters to prevent catastrophic forgetting
                if vision_backbone_frozen:
                    for param in self.dinov2.parameters():
                        param.requires_grad = False

                # SOLUTION: MLP projector to map DINOv2 features to transformer dim
                dinov2_dim = 384  # DINOv2-small output dimension
                self.projector = nn.Sequential(
                    nn.Linear(dinov2_dim, ff_hidden_dim),
                    nn.GELU(),
                    nn.Linear(ff_hidden_dim, dim),
                    nn.LayerNorm(dim)
                )
            except Exception as e:
                print(f"Warning: Failed to load DINOv2: {e}")
                self.use_vision_backbone = False

        # SOLUTION: Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # SOLUTION: Final layer norm
        self.norm_final = RMSNorm(dim)

        # SOLUTION: Language model head
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # SOLUTION: Create causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        )

        # SOLUTION: Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """SOLUTION: Initialize weights using scaled initialization"""
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
        vision_input: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        SOLUTION: Complete forward pass

        Args:
            input_ids: Token IDs (batch, seq_len)
            targets: Target token IDs for loss computation (batch, seq_len)
            vision_input: Optional vision input (batch, 3, H, W)

        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: Scalar loss if targets provided, else None
        """
        batch_size, seq_len = input_ids.shape

        # SOLUTION: Embed tokens
        x = self.token_embedding(input_ids)

        # SOLUTION: Optional vision features
        if self.use_vision_backbone and vision_input is not None:
            with torch.no_grad():
                vision_features = self.dinov2(vision_input)  # (batch, dinov2_dim)
            vision_features = self.projector(vision_features)  # (batch, dim)
            # Prepend vision features to sequence
            vision_features = vision_features.unsqueeze(1)  # (batch, 1, dim)
            x = torch.cat([vision_features, x], dim=1)  # (batch, seq_len+1, dim)
            seq_len += 1

        # SOLUTION: Get causal mask
        mask = self.causal_mask[:, :, :seq_len, :seq_len]

        # SOLUTION: Pass through transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # SOLUTION: Final normalization
        x = self.norm_final(x)

        # SOLUTION: Project to vocabulary
        logits = self.lm_head(x)

        # SOLUTION: Compute loss if targets provided
        loss = None
        if targets is not None:
            # Flatten for cross-entropy
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-100
            )

        return logits, loss


# SOLUTION: Complete training loop
def train_model(
    model: DecoderOnlyTransformer,
    train_loader,
    val_loader,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
):
    """
    SOLUTION: Complete training loop with best practices

    Args:
        model: The transformer model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0

        for batch_idx, (states, actions) in enumerate(train_loader):
            states = states.to(device)
            actions = actions.to(device)

            # Use actions as input (teacher forcing)
            input_ids = actions[:, :-1]
            targets = actions[:, 1:]

            # Forward pass
            logits, loss = model(input_ids, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for states, actions in val_loader:
                states = states.to(device)
                actions = actions.to(device)

                input_ids = actions[:, :-1]
                targets = actions[:, 1:]

                logits, loss = model(input_ids, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'checkpoints/best_model.pt')

        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

    return model


if __name__ == "__main__":
    # Example usage
    from generate_data import generate_dataset, create_dataloaders

    # Generate dataset
    print("Generating dataset...")
    dataset = generate_dataset(num_trajectories=1000, seq_length=50)
    train_loader, val_loader = create_dataloaders(dataset, batch_size=32)

    # Create model
    print("Creating model...")
    model = DecoderOnlyTransformer(
        vocab_size=256,
        dim=384,
        num_layers=4,
        num_heads=8,
        ff_hidden_dim=1024,
        dropout=0.1
    )

    # Train
    print("Training...")
    model = train_model(model, train_loader, val_loader, num_epochs=10)

    print("Training complete!")
