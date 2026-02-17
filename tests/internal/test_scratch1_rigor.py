"""
Rigorous Internal Tests for Scratch-1: The Transformer Backbone
CSCI 7000 - VLA Foundations

These tests are used for automated grading and are NOT visible to students.
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "assignments" / "scratch-1"))

try:
    from backbone import (
        RMSNorm,
        CausalSelfAttention,
        DecoderOnlyTransformer,
        RotaryPositionalEmbedding,
    )
    IMPORT_SUCCESS = True
except Exception as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def _make_causal_attention(dim, num_heads, max_seq_len, dropout=0.0):
    """Instantiate CausalSelfAttention with flexible API.

    The starter code defines CausalSelfAttention(dim, num_heads, dropout) without
    max_seq_len.  Some students add it, some don't.  Try both signatures so we
    don't penalise students for following the starter code exactly.
    """
    try:
        return CausalSelfAttention(dim=dim, num_heads=num_heads, max_seq_len=max_seq_len, dropout=dropout)
    except TypeError:
        return CausalSelfAttention(dim=dim, num_heads=num_heads, dropout=dropout)


@pytest.fixture
def device():
    """Get available device"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_config():
    """Small model config for fast testing"""
    return {
        "vocab_size": 256,
        "dim": 128,
        "num_layers": 2,
        "num_heads": 4,
        "ff_hidden_dim": 256,
        "max_seq_len": 32,
        "dropout": 0.0,  # Disable for testing
    }


# ============================================================================
# IMPORT TEST (Critical - must pass for any other test to run)
# ============================================================================

@pytest.mark.rigor
def test_import_success():
    """Test that student code can be imported"""
    assert IMPORT_SUCCESS, f"Failed to import student code: {IMPORT_ERROR if not IMPORT_SUCCESS else 'N/A'}"


# ============================================================================
# RMSNORM TESTS (10 points)
# ============================================================================

@pytest.mark.rigor
@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Import failed")
def test_rmsnorm_implementation():
    """Test RMSNorm produces correct outputs"""
    batch_size, seq_len, dim = 2, 10, 64

    # Initialize RMSNorm
    rmsnorm = RMSNorm(dim=dim)

    # Check that scale parameter exists and is learnable
    assert hasattr(rmsnorm, 'scale'), "RMSNorm must have 'scale' parameter"
    assert rmsnorm.scale is not None, "RMSNorm scale parameter cannot be None"
    assert isinstance(rmsnorm.scale, nn.Parameter), "scale must be nn.Parameter"
    assert rmsnorm.scale.shape == (dim,), f"scale shape should be ({dim},), got {rmsnorm.scale.shape}"

    # Test forward pass
    x = torch.randn(batch_size, seq_len, dim)
    output = rmsnorm(x)

    # Check output shape
    assert output.shape == x.shape, f"Output shape {output.shape} != input shape {x.shape}"

    # Check that output is not NaN or Inf
    assert torch.isfinite(output).all(), "RMSNorm output contains NaN or Inf"

    # Verify normalization: RMS should be approximately 1.0 (before scaling)
    # After scaling by learnable parameter, mean RMS should still be reasonable
    rms = torch.sqrt((output ** 2).mean(dim=-1))
    assert (rms > 0.1).all() and (rms < 10).all(), f"RMS values unreasonable: {rms.mean()}"


@pytest.mark.rigor
@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Import failed")
def test_rmsnorm_numerical_stability():
    """Test RMSNorm handles edge cases"""
    dim = 64
    rmsnorm = RMSNorm(dim=dim)

    # Test with very small values
    x_small = torch.randn(1, 10, dim) * 1e-6
    output_small = rmsnorm(x_small)
    assert torch.isfinite(output_small).all(), "RMSNorm fails with small inputs"

    # Test with very large values
    x_large = torch.randn(1, 10, dim) * 1e6
    output_large = rmsnorm(x_large)
    assert torch.isfinite(output_large).all(), "RMSNorm fails with large inputs"


# ============================================================================
# CAUSAL ATTENTION TESTS (15 points)
# ============================================================================

@pytest.mark.rigor
@pytest.mark.gradient
@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Import failed")
def test_causal_mask_leakage():
    """CRITICAL: Test that causal mask prevents future token leakage"""
    dim = 128
    num_heads = 4
    max_seq_len = 16

    attn = _make_causal_attention(dim=dim, num_heads=num_heads, max_seq_len=max_seq_len, dropout=0.0)
    attn.eval()  # Disable dropout

    batch_size = 2
    seq_len = 12

    # Create input where position information is critical
    # Each position has a unique value, so we can detect if future info leaks
    x = torch.arange(seq_len).float().view(1, seq_len, 1).expand(batch_size, seq_len, dim)
    x = x + torch.randn(batch_size, seq_len, dim) * 0.1  # Add small noise

    output = attn(x)

    # Check output shape
    assert output.shape == x.shape, f"Attention output shape mismatch: {output.shape} != {x.shape}"

    # Key test: Output at position i should NOT depend on positions > i
    # We'll test this by checking that changing future tokens doesn't affect past outputs

    # Run forward pass twice: once with original input, once with modified future
    x_modified = x.clone()
    x_modified[:, seq_len//2:, :] = torch.randn_like(x_modified[:, seq_len//2:, :]) * 1000  # Drastically change future

    with torch.no_grad():
        output_original = attn(x)
        output_modified = attn(x_modified)

    # Outputs for positions 0 to seq_len//2-1 should be identical (within numerical precision)
    early_positions = slice(0, seq_len//2)
    diff = (output_original[:, early_positions, :] - output_modified[:, early_positions, :]).abs().max()

    assert diff < 1e-4, f"Causal mask is leaking! Early positions changed by {diff} when future was modified. " \
                        f"Did you apply the mask BEFORE softmax?"


@pytest.mark.rigor
@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Import failed")
def test_causal_attention_shape_preservation():
    """Test that attention preserves tensor shapes"""
    dim = 128
    num_heads = 4
    batch_size, seq_len = 2, 16

    attn = _make_causal_attention(dim=dim, num_heads=num_heads, max_seq_len=32)
    x = torch.randn(batch_size, seq_len, dim)

    output = attn(x)
    assert output.shape == (batch_size, seq_len, dim), \
        f"Attention output shape {output.shape} != expected {(batch_size, seq_len, dim)}"


# ============================================================================
# ROPE TESTS (15 points)
# ============================================================================

@pytest.mark.rigor
@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Import failed")
def test_rope_embeddings():
    """Test RoPE is applied correctly"""
    dim = 64
    max_seq_len = 32
    rope = RotaryPositionalEmbedding(dim=dim, max_seq_len=max_seq_len)

    batch_size, num_heads, seq_len, head_dim = 2, 4, 16, dim // 4
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)

    q_rot, k_rot = rope(q, k)

    # Check shapes preserved
    assert q_rot.shape == q.shape, f"RoPE changed Q shape: {q_rot.shape} != {q.shape}"
    assert k_rot.shape == k.shape, f"RoPE changed K shape: {k_rot.shape} != {k.shape}"

    # Check that rotation actually happened (values changed)
    assert not torch.allclose(q, q_rot, atol=1e-6), "RoPE did not modify Q (are you applying it?)"
    assert not torch.allclose(k, k_rot, atol=1e-6), "RoPE did not modify K (are you applying it?)"

    # Check no NaNs
    assert torch.isfinite(q_rot).all(), "RoPE produced NaN/Inf in Q"
    assert torch.isfinite(k_rot).all(), "RoPE produced NaN/Inf in K"


# ============================================================================
# TRAINING CONVERGENCE TEST (10 points)
# ============================================================================

@pytest.mark.rigor
@pytest.mark.training
@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Import failed")
@pytest.mark.timeout(300)  # 5 minute timeout
def test_training_convergence(small_config, device):
    """Test that model can train and loss decreases"""

    # Create small dataset
    num_samples = 100
    seq_len = small_config["max_seq_len"]
    vocab_size = small_config["vocab_size"]

    # Generate random sequences (simulating robot trajectories)
    data = torch.randint(0, vocab_size, (num_samples, seq_len))

    # Create model
    model = DecoderOnlyTransformer(**small_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Train for a few steps
    model.train()
    initial_losses = []
    final_losses = []

    # Initial loss (first 5 batches)
    for i in range(5):
        batch = data[i:i+1].to(device)
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]

        logits, loss = model(input_ids, targets)
        initial_losses.append(loss.item())

    # Train for 50 steps
    for step in range(50):
        idx = step % num_samples
        batch = data[idx:idx+1].to(device)
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]

        optimizer.zero_grad()
        logits, loss = model(input_ids, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Final loss (last 5 batches)
    model.eval()
    with torch.no_grad():
        for i in range(5):
            batch = data[i:i+1].to(device)
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]

            logits, loss = model(input_ids, targets)
            final_losses.append(loss.item())

    initial_loss = sum(initial_losses) / len(initial_losses)
    final_loss = sum(final_losses) / len(final_losses)

    # Check that loss decreased
    improvement = initial_loss - final_loss
    assert improvement > 0.1, \
        f"Loss did not decrease significantly. Initial: {initial_loss:.3f}, Final: {final_loss:.3f}. " \
        f"Check your training loop implementation."

    # Check that final loss is reasonable
    # For random 256-way classification, loss ~= log(256) ~= 5.5
    # After training, should be noticeably lower
    assert final_loss < 5.0, \
        f"Final loss too high ({final_loss:.3f}). Model is not learning. " \
        f"Check gradient flow and loss computation."

    return initial_loss, final_loss  # Return for partial credit calculation


# ============================================================================
# MODEL ARCHITECTURE TESTS
# ============================================================================

@pytest.mark.rigor
@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Import failed")
def test_model_forward_pass(small_config, device):
    """Test that model forward pass works end-to-end"""
    model = DecoderOnlyTransformer(**small_config).to(device)

    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, small_config["vocab_size"], (batch_size, seq_len)).to(device)
    targets = torch.randint(0, small_config["vocab_size"], (batch_size, seq_len)).to(device)

    # Test forward with targets (training mode)
    logits, loss = model(input_ids, targets)

    assert logits.shape == (batch_size, seq_len, small_config["vocab_size"]), \
        f"Logits shape {logits.shape} incorrect"
    assert loss is not None, "Loss should not be None when targets provided"
    assert torch.isfinite(loss), "Loss is NaN or Inf"

    # Test forward without targets (inference mode)
    logits_only, loss_only = model(input_ids)
    assert logits_only.shape == (batch_size, seq_len, small_config["vocab_size"]), \
        f"Logits shape {logits_only.shape} incorrect"
    assert loss_only is None, "Loss should be None when no targets provided"


@pytest.mark.rigor
@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Import failed")
def test_model_has_trainable_parameters(small_config):
    """Test that model has trainable parameters"""
    model = DecoderOnlyTransformer(**small_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_params > 0, "Model has no trainable parameters!"

    # Should have at least a few hundred thousand parameters for this config
    assert trainable_params > 100_000, \
        f"Model has suspiciously few parameters ({trainable_params}). Something may be wrong."


# ============================================================================
# CODE QUALITY CHECKS
# ============================================================================

@pytest.mark.rigor
def test_no_syntax_errors():
    """Test that the code has no syntax errors"""
    # If we got here, import succeeded, so syntax is valid
    assert IMPORT_SUCCESS, "Code has syntax errors"


@pytest.mark.rigor
@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Import failed")
def test_no_todos_left():
    """Check that student didn't leave TODO comments with 'pass' statements"""
    import inspect

    # Check RMSNorm.forward
    source = inspect.getsource(RMSNorm.forward)
    assert "raise NotImplementedError" not in source, \
        "RMSNorm.forward still has NotImplementedError - did you implement it?"

    # Check CausalSelfAttention.forward
    source = inspect.getsource(CausalSelfAttention.forward)
    assert "raise NotImplementedError" not in source, \
        "CausalSelfAttention.forward still has NotImplementedError"


# ============================================================================
# HELPER FUNCTIONS FOR PARTIAL CREDIT
# ============================================================================

def calculate_partial_credit_for_training(initial_loss: float, final_loss: float) -> tuple[int, str]:
    """
    Calculate partial credit for training convergence
    Returns: (points_earned, feedback)
    """
    # Full credit: final_loss < 2.2
    # Partial credit: final_loss < 3.0
    # No credit: final_loss >= 3.0 or no improvement

    improvement = initial_loss - final_loss

    if final_loss < 2.2:
        return 10, f"✅ Excellent! Loss converged to {final_loss:.3f}"
    elif final_loss < 3.0 and improvement > 0.3:
        points = int(10 * (3.0 - final_loss) / 0.8)  # Linear scaling
        return points, f"⚠️ Partial credit: Loss is {final_loss:.3f} (target < 2.2). Shows learning but needs improvement."
    elif improvement > 0.1:
        return 3, f"⚠️ Minimal credit: Loss decreased slightly ({initial_loss:.3f} → {final_loss:.3f}) but not enough."
    else:
        return 0, f"❌ Loss did not improve ({initial_loss:.3f} → {final_loss:.3f}). Check your training loop."
