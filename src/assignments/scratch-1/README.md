# Scratch-1: The Transformer Backbone

Implementation template for building a decoder-only Transformer from scratch.

## Quick Start

### 1. Generate the Dataset

```bash
python generate_data.py --num_trajectories 10000 --seq_length 50 --output data/trajectories.pkl
```

This creates a dataset of synthetic 7-DOF robot arm trajectories.

### 2. Complete the TODOs

Open `backbone.py` and complete the following sections:

- **RMSNorm** (lines 30-50): Implement Root Mean Square normalization
- **CausalSelfAttention** (lines 100-180): Implement multi-head causal attention with RoPE
- **train_epoch** (lines 380-420): Implement the training loop

### 3. Train the Model

```bash
python backbone.py
```

Expected behavior:
- Loss should decrease from ~5.5 to below 1.0
- Training takes ~10-15 minutes on GPU, ~1 hour on CPU
- Checkpoints saved to `checkpoints/`

## Files

- `backbone.py` - Main implementation template with TODOs
- `generate_data.py` - Dataset generation script
- `visualize.py` - Visualization tools (create this for your submission)
- `data/` - Generated datasets
- `checkpoints/` - Saved model checkpoints

## Dataset Details

**Format**: Dictionary with tensors
- `states`: (10000, 50, 10) - Joint angles + end-effector positions
- `actions`: (10000, 50) - Discretized actions (0-255)

**Specifications**:
- 10,000 trajectories
- 50 timesteps each
- 7-DOF joint angles + 3D end-effector position
- Actions discretized into 256 bins

## Model Architecture

```
Input: Tokenized actions (batch, seq_len)
  ↓
Token Embedding
  ↓
TransformerBlock × N:
  ├─ RMSNorm
  ├─ CausalSelfAttention (with RoPE)
  ├─ Residual Connection
  ├─ RMSNorm
  ├─ FeedForward (SwiGLU)
  └─ Residual Connection
  ↓
RMSNorm
  ↓
LM Head (Linear projection)
  ↓
Output: Logits (batch, seq_len, vocab_size)
```

**Hyperparameters**:
- Vocab size: 256
- Model dim: 256
- Layers: 4
- Attention heads: 8
- FF hidden dim: 1024
- Dropout: 0.1

## Key Concepts

### Causal Masking

Ensures that token at position `t` can only attend to positions `≤ t`:

```python
mask = torch.tril(torch.ones(seq_len, seq_len))
scores = scores.masked_fill(mask == 0, float('-inf'))
```

### RoPE (Rotary Position Embedding)

Rotates query/key representations based on position:
- More efficient than learned positional embeddings
- Better extrapolation to longer sequences
- Used in Llama-3, PaLM-E

### RMSNorm

Simpler than LayerNorm (no mean centering):
```
RMS(x) = sqrt(mean(x^2) + eps)
output = (x / RMS(x)) * scale
```

## Debugging Tips

### Loss not decreasing?
1. Check causal mask is applied correctly
2. Verify RoPE is applied to both Q and K
3. Enable gradient clipping
4. Reduce learning rate to 1e-5

### NaN loss?
1. Check for division by zero in RMSNorm
2. Ensure mask uses -inf, not large negative number
3. Enable gradient clipping

### Out of memory?
1. Reduce batch size to 16 or 8
2. Reduce sequence length
3. Use gradient accumulation

## Testing Your Implementation

### Quick Sanity Checks

```python
# Test RMSNorm
norm = RMSNorm(256)
x = torch.randn(2, 10, 256)
out = norm(x)
assert out.shape == x.shape
assert not torch.isnan(out).any()

# Test Attention
attn = CausalSelfAttention(256, 8)
x = torch.randn(2, 10, 256)
out = attn(x)
assert out.shape == x.shape

# Test causality
# Token at position 5 should NOT be affected by changes at position 7
```

## Submission Checklist

- [ ] RMSNorm implemented and tested
- [ ] CausalSelfAttention implemented with proper masking
- [ ] RoPE applied correctly to Q and K
- [ ] Training loop converges (loss < 1.0)
- [ ] Loss curve saved to `images/loss_curve.png`
- [ ] Attention maps visualized and saved
- [ ] Causal mask audit completed
- [ ] Code committed to your branch
- [ ] MDX report created in `content/course/submissions/scratch-1/`

## Resources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoFormer (RoPE)](https://arxiv.org/abs/2104.09864)
- [RMSNorm Paper](https://arxiv.org/abs/1910.07467)
- [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

## Getting Help

- **Office Hours**: Tuesday/Thursday 3-4 PM
- **Discussion Forum**: Post questions with `[scratch-1]` tag
- **Common Issues**: Check FAQ in assignment MDX file

Good luck! 🤖
