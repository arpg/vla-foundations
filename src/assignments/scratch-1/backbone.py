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
from pathlib import Path
import pickle
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Debug Info Flag
# Shows some steps along the way of understanding the data and printing additional information 
# during solution development.

debug_info = True

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
        # Initialize learnable scale parameter 'g' (gamma)
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
        Returns:
            Normalized tensor of same shape
        """
        # Implement RMSNorm
        # Step 1: Compute RMS (root mean square) along the last dimension
        # Step 2: Normalize by dividing x by RMS
        # Step 3: Apply learnable scale parameter

        # 1/RMS = sqrt(mean(x^2) + eps)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        # a_bar_i = (a_i / RMS(a)) * g_i
        return x * rms * self.scale


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

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        cos = self.cos_cached[:seq_len, ...]
        sin = self.sin_cached[:seq_len, ...]

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

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        global debug_info
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Optional attention mask (seq_len, seq_len)
        Returns:
            Output tensor (batch, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape

        # PART Implement Causal Self-Attention

        # Step 1: Project input to Q, K, V
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3*dim)

        # Split into Q, K, V and reshape for multi-head attention
        # Hint: Use .view() and .transpose() to get shape (batch, num_heads, seq_len, head_dim)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)

        # Print shapes for debugging
        if debug_info:
            print(f"qkv shape: {qkv.shape}")  # (batch, seq_len, 3, num_heads, head_dim)

        # Break out Q, K, V
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # Each is (batch, seq_len, num_heads, head_dim)

        # Query: What is position 'i' looking for?
        q = q.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)

        # Key: What does position 'j' advertise?
        k = k.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)

        # Value: "What does position 'j' contribute if selected?"
        v = v.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)

        # Step 2: Apply RoPE to Q and K
        q, k = self.rope(q, k)

        # Step 3: Compute attention scores
        # scores = (Q @ K^T) / sqrt(d_k)
        # Hint: Use torch.matmul or @ operator
        # Shape should be (batch, num_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch, num_heads, seq_len, seq_len)

        # Step 4: Apply causal mask
        # The mask should prevent position i from attending to positions > i
        # Hint: Create a lower-triangular matrix using torch.tril
        # Looks like something higher up will pass in the mask, so use that, otherwise redefine
        if mask is None:
            print("Had to create mask inside CausalSelfAttention because none was passed in.")
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        
        if debug_info:
            print(f"Mask shape: {mask.shape}")  # (seq_len, seq_len)
            # Print 5x5 top-left corner of the mask
            print("Mask (top-left 5x5):")
            print(mask[:5, :5])
            debug_info = False  # So we don't spam the output

        # Set masked positions to -inf BEFORE softmax
        # Example: scores = scores.masked_fill(mask == 0, float('-inf'))
        # Use mask to fill future information from current information
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # Step 5: Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        # attn_weights_dropout = self.attn_dropout(attn_weights)
        
        # This was for verifying dropout behavior
        # This memory was stored on cuda so I couldn't afford to keep both. Reverted back to single version.
        # if debug_info:
        #     # Compare non dropout and dropout attention weights
        #     print("Attention Weights (top-left 5x5) without dropout:")
        #     print(attn_weights[0, 0, :5, :5])
        #     print("Attention Weights (top-left 5x5) with dropout:")
        #     print(attn_weights_dropout[0, 0, :5, :5])
        #     attn_weights = attn_weights_dropout
        #     del attn_weights_dropout

        # Step 6: Apply attention to values
        out = torch.matmul(attn_weights, v)  # (batch, num_heads, seq_len, head_dim)

        # Step 7: Reshape and project back
        # Concatenate heads and apply output projection
        # Hint: Use .transpose() and .contiguous().view() to reshape
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)  # (batch, seq_len, dim)
        out = self.out_proj(out)

        # Step 8: Apply residual dropout
        out = self.resid_dropout(out)

        return out


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

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Optional attention mask
        Returns:
            Output tensor (batch, seq_len, dim)
        """
        # Pre-norm architecture (norm before attention/FF)
        x = x + self.attention(self.norm1(x), mask)
        x = x + self.feed_forward(self.norm2(x))
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
            # Bug: need contigous memory here for view to work
            loss = F.cross_entropy(
                logits.contiguous().view(-1, self.vocab_size),
                targets.contiguous().view(-1),
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
        for _ in range(max_new_tokens):
            # Crop context if too long
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

    # PART 4: Training loop
    # For each batch:
    for batch_idx, (states, actions) in enumerate(dataloader):
        # 1. Move data to device
        actions = actions.to(device)

        # Prepare inputs and targets for next-token prediction
        # Input: timestamps 0 to T-1
        # Target: timestamps 1 to T (to predict next token)
        inputs = actions[:, :-1]
        targets = actions[:, 1:]

        # 2. Forward pass (get logits and loss)
        logits, loss = model(inputs, targets=targets)
        
        # 3. Backward pass
        loss.backward()

        # 4. Gradient clipping (max_norm=1.0)
        # Hint: Use torch.nn.utils.clip_grad_norm_ for gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 5. Optimizer step
        optimizer.step()

        # 6. Zero gradients
        optimizer.zero_grad(set_to_none=True)

        # 7. Accumulate loss
        total_loss += loss.item()
        num_batches += 1

        # 8. Print progress every 100 batches
        # Hint: Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch}, Batch {batch_idx + 1}, Avg Loss: {avg_loss:.4f}")
    
    # Return average loss for the epoch
    return total_loss / num_batches

# Evaluation Function
def evaluate(
    model: DecoderOnlyTransformer,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """
    Evaluate the model on validation data

    Args:
        model: The transformer model
        dataloader: Validation data loader
        device: Device to evaluate on
    Returns:
        Average loss on validation set
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for states, actions in dataloader:
            actions = actions.to(device)

            inputs = actions[:, :-1]
            targets = actions[:, 1:]

            logits, loss = model(inputs, targets=targets)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches

def main():
    global debug_info
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

    # PART 1: Load dataset
    # Use the generate_data.py script to create synthetic trajectories
    # Load from data/trajectories.pkl from anywhere in file system
    local_data_path = Path("data/trajectories.pkl")
    data_path = local_data_path.resolve()
    print(f"Loading data from {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}. Please run generate_data.py to create it.")
    
    # Load generated trajectories pickle file
    with open(data_path, "rb") as f:
        trajectory_data = pickle.load(f)
    
    if len(trajectory_data["actions"]) == 0:
        raise ValueError("No trajectories found in the dataset.")
    
    if debug_info:
        # Print a bunch of information about the data
        print("Data loaded successfully. Trajectories: " + type(trajectory_data).__name__)
        for key in trajectory_data:
            print(f"Key: {key}, Type: {type(trajectory_data[key])}, Length: {len(trajectory_data[key])}")
        print(f"Loaded {len(trajectory_data['actions'])} trajectories")
        print(f"Example trajectory length: {len(trajectory_data['actions'][0])}")

    # Create Training and Validation DataSets with Torch
    number_of_trajectories = len(trajectory_data['actions'])
    train_size = int(0.9 * number_of_trajectories)

    # Randomly shuffle indices
    indices = torch.randperm(number_of_trajectories).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create TensorDatasets
    train_actions = TensorDataset(
        trajectory_data['states'][train_indices],
        trajectory_data['actions'][train_indices]
    )
    val_actions = TensorDataset(
        trajectory_data['states'][val_indices],
        trajectory_data['actions'][val_indices]
    )

    # Verify Dataset Size
    print(f"Training set size: {len(train_actions)}")
    print(f"Validation set size: {len(val_actions)}")

    # Create DataLoaders
    train_loader = DataLoader(train_actions, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_actions, batch_size=batch_size, shuffle=False)

    if debug_info:
        # Visualize one example data point
        example_state, example_action = train_actions[0]
        print(f"Example state shape: {example_state.shape}, Example action shape: {example_action.shape}")
        for traj_point in zip(example_action, example_state):
            print(
                f"Action token: {traj_point[0].item():>5} | "
                f"J0: {traj_point[1][0]: .2f}, J1: {traj_point[1][1]: .2f}, "
                f"J2: {traj_point[1][2]: .2f}, J3: {traj_point[1][3]: .2f}, "
                f"J4: {traj_point[1][4]: .2f}, J5: {traj_point[1][5]: .2f}, "
                f"J6: {traj_point[1][6]: .2f}, X: {traj_point[1][7]: .2f}, "
                f"Y: {traj_point[1][8]: .2f}, Z: {traj_point[1][9]: .2f}"
            )
        # Plot the End Effector Position over the trajectory as points in 3D space
        states = trajectory_data['states'][train_indices[0]].numpy()
        x = states[:, 7]
        y = states[:, 8]
        z = states[:, 9]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # plot as points from in rainbow color map
        ax.scatter(x, y, z, c=range(len(x)), cmap='rainbow', marker='o')
        # ax.plot(x, y, z, label='End Effector Trajectory')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('End Effector Position Over Trajectory')
        plt.show()


    # PART 2: Create model
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_hidden_dim=ff_hidden_dim,
        max_seq_len=max_seq_len,
        dropout=0.1,
    )
    model.to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    # PART 3: Create optimizer
    # Weight decay of 1e-2 for regularization on small models
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)

    # PART 4: Training loop
    best_model = None
    best_val_loss = float('inf')

    # Additional requirements for submission:
    loss_curve_values = []
    number_of_iterations = []
    for epoch in range(num_epochs):
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)

        # Evaluate on validation set
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        loss_curve_values.append((train_loss, val_loss))
        number_of_iterations.append((epoch + 1)*(len(train_loader)))
        # PART 5: Save checkpoint every 1000 steps
        # Small deviation: Only save the best model
        # This accounts for when a training produces a worse model than before
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            print(f"New best model found at epoch {epoch+1} with val loss {best_val_loss:.4f}")
            checkpoint_dir = Path("checkpoints")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"best_model.pt"
            torch.save(best_model, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")


    # Plot loss curves versus number of iterations
    train_losses, val_losses = zip(*loss_curve_values)
    plt.plot(number_of_iterations, train_losses, label='Train Loss')
    plt.plot(number_of_iterations, val_losses, label='Validation Loss')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid()
    plt.show()

    # Save the plots
    plots_dir = Path("images")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / "default_loss_curves.png"
    plt.savefig(plot_path)
    print(f"Saved loss curves plot to {plot_path}")

if __name__ == "__main__":
    main()
