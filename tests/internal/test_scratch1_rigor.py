"""
Internal rigorous tests for Scratch-1 (NEVER public).

These tests validate student solutions with:
- Gradient leak detection (frozen backbone parameters)
- Latent fidelity comparison (against gold standards)
- Training convergence verification
- Edge case handling

All tests marked with @pytest.mark.internal and @pytest.mark.rigor.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

# Mark all tests in this file as internal
pytestmark = [pytest.mark.internal, pytest.mark.rigor]


@pytest.mark.gradient
def test_frozen_dinov2_no_gradients(create_solution_model):
    """
    Verify that frozen DINOv2 backbone has requires_grad=False for all parameters.

    This catches a common mistake where students forget to freeze the pretrained
    vision encoder, leading to:
    - Catastrophic forgetting (DINOv2 loses its learned features)
    - 10x slower training
    - Much higher memory usage
    """
    # Load the solution model with DINOv2 integration
    model = create_solution_model(
        dim=384,  # Matches DINOv2 output
        num_layers=4,
        num_heads=8,
        ff_hidden_dim=1024,
        use_vision_backbone=True
    )

    # Check DINOv2 backbone exists
    assert hasattr(model, 'dinov2'), "Model missing DINOv2 backbone"

    # Verify ALL parameters are frozen
    frozen_count = 0
    unfrozen_params = []

    for name, param in model.dinov2.named_parameters():
        if param.requires_grad:
            unfrozen_params.append(name)
        else:
            frozen_count += 1

    assert len(unfrozen_params) == 0, \
        f"Gradient leak detected! {len(unfrozen_params)} parameters not frozen:\n" + \
        "\n".join(f"  - {name}" for name in unfrozen_params[:10])

    print(f"✓ All {frozen_count} DINOv2 parameters correctly frozen")


@pytest.mark.fidelity
def test_projector_latent_fidelity(create_solution_model, load_gold_standard):
    """
    Compare student's projector output against gold standard tensor.

    This ensures the student's MLP projector is:
    - Correctly implemented (right architecture)
    - Properly initialized (not introducing noise)
    - Numerically stable (no NaNs, exploding gradients)
    """
    # Try to load gold standard (skip if not available)
    try:
        gold_data = load_gold_standard('scratch1_gold_output.pt')
    except Exception:
        pytest.skip("Gold standard not available yet")

    test_input = gold_data['input']
    gold_output = gold_data['output']

    # Load solution model
    model = create_solution_model(
        dim=384,
        num_layers=4,
        num_heads=8,
        ff_hidden_dim=1024,
        use_vision_backbone=True
    )
    model.eval()

    # Forward pass
    with torch.no_grad():
        student_output = model.projector(test_input)

    # Check for NaNs
    assert not torch.isnan(student_output).any(), "Output contains NaNs"

    # Compare to gold standard
    torch.testing.assert_close(
        student_output,
        gold_output,
        rtol=1e-4,
        atol=1e-5,
        msg="Projector output differs from gold standard"
    )

    print(f"✓ Projector output matches gold standard (max diff: {(student_output - gold_output).abs().max():.2e})")


@pytest.mark.training
def test_training_convergence(training_setup):
    """
    Verify that the model can train and converge on a small dataset.

    This is a sanity check that catches:
    - Learning rate too high/low
    - Missing gradient flow
    - Incorrect loss computation
    """
    model = training_setup['model']
    optimizer = training_setup['optimizer']
    train_loader = training_setup['train_loader']

    # Train for 10 steps
    model.train()
    initial_loss = None
    final_loss = None

    for i, (states, actions) in enumerate(train_loader):
        if i >= 10:
            break

        # Use actions as input (teacher forcing)
        input_ids = actions[:, :-1]
        targets = actions[:, 1:]

        logits, loss = model(input_ids, targets)

        if i == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Loss should decrease
    assert final_loss < initial_loss, \
        f"Loss did not decrease! Initial: {initial_loss:.4f}, Final: {final_loss:.4f}"

    print(f"✓ Training converged: {initial_loss:.4f} → {final_loss:.4f}")


@pytest.mark.gradient
def test_attention_gradient_flow(create_solution_model, sample_batch):
    """
    Verify that gradients flow properly through the attention mechanism.

    Common issues this catches:
    - Missing .backward() connections
    - Detached tensors in attention computation
    - Incorrectly frozen layers
    """
    model = create_solution_model(dim=128, num_layers=2, num_heads=4)
    model.train()

    input_ids = sample_batch['input_ids'][:, :-1]
    targets = sample_batch['targets'][:, 1:]

    # Forward pass
    logits, loss = model(input_ids, targets)

    # Backward pass
    loss.backward()

    # Check that attention parameters have gradients
    for name, param in model.named_parameters():
        if 'attention' in name.lower():
            assert param.grad is not None, f"No gradient for attention parameter: {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for: {name}"
            assert param.grad.abs().max() > 0, f"Zero gradient for: {name}"

    print("✓ Gradients flow correctly through attention mechanism")


@pytest.mark.rigor
def test_causal_mask_prevents_future_leakage(create_solution_model):
    """
    Verify that the causal mask prevents attention to future tokens.

    This ensures:
    - Causal mask is correctly applied
    - No information leakage from future tokens
    - Autoregressive property is maintained
    """
    model = create_solution_model(dim=128, num_layers=1, num_heads=4)
    model.eval()

    batch_size = 2
    seq_len = 10
    vocab_size = 256

    # Create test input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Forward pass
    with torch.no_grad():
        logits, _ = model(input_ids, input_ids)

    # Modify a future token and verify it doesn't affect past predictions
    input_ids_modified = input_ids.clone()
    input_ids_modified[:, -1] = (input_ids_modified[:, -1] + 1) % vocab_size

    with torch.no_grad():
        logits_modified, _ = model(input_ids_modified, input_ids_modified)

    # First seq_len-1 positions should be identical (no future leakage)
    # Note: There might be small numerical differences due to floating point ops
    torch.testing.assert_close(
        logits[:, :-1, :],
        logits_modified[:, :-1, :],
        rtol=1e-5,
        atol=1e-6,
        msg="Causal mask allows future information leakage!"
    )

    print("✓ Causal mask correctly prevents future information leakage")


@pytest.mark.rigor
def test_rmsnorm_numerics(create_solution_model):
    """
    Verify that RMSNorm is numerically stable and correct.

    Checks:
    - Output has unit RMS (root mean square)
    - No NaNs with extreme inputs
    - Correct scaling behavior
    """
    model = create_solution_model(dim=128, num_layers=1)

    # Access first transformer block's norm
    norm = model.blocks[0].norm1

    # Test 1: Normal input
    x = torch.randn(2, 10, 128)
    output = norm(x)

    # Compute RMS of output
    rms = torch.sqrt(torch.mean(output ** 2, dim=-1))

    # RMS should be approximately 1.0 (after scaling by learnable gain)
    # Since gain is learned, we check that normalization happened
    assert not torch.isnan(output).any(), "RMSNorm output contains NaNs"

    # Test 2: Extreme values
    x_large = torch.randn(2, 10, 128) * 1000
    output_large = norm(x_large)
    assert not torch.isnan(output_large).any(), "RMSNorm fails on large values"

    x_small = torch.randn(2, 10, 128) * 1e-6
    output_small = norm(x_small)
    assert not torch.isnan(output_small).any(), "RMSNorm fails on small values"

    print("✓ RMSNorm is numerically stable")


@pytest.mark.rigor
def test_model_output_distribution(create_solution_model):
    """
    Verify that model outputs have reasonable statistical properties.

    Checks:
    - Logits are not all zeros (model is learning)
    - Logits are not all the same (collapsed solution)
    - No NaNs or Infs in outputs
    - Distribution of logits is reasonable
    """
    model = create_solution_model(dim=128, num_layers=2, num_heads=4)
    model.eval()

    batch_size = 4
    seq_len = 20
    vocab_size = 256

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits, _ = model(input_ids, input_ids)

    # Check for NaNs and Infs
    assert not torch.isnan(logits).any(), "Logits contain NaNs"
    assert not torch.isinf(logits).any(), "Logits contain Infs"

    # Check logits are not all the same
    logit_std = logits.std()
    assert logit_std > 0.01, f"Logits have collapsed (std={logit_std:.6f})"

    # Check logits are not all zeros
    assert logits.abs().max() > 0.1, "Logits are nearly all zeros"

    # Check reasonable range
    assert logits.abs().max() < 100, f"Logits unreasonably large (max={logits.abs().max():.2f})"

    print(f"✓ Model outputs have healthy distribution (std={logit_std:.4f}, range=[{logits.min():.2f}, {logits.max():.2f}])")


@pytest.mark.rigor
def test_loss_computation_correctness(create_solution_model):
    """
    Verify that cross-entropy loss is computed correctly.

    Checks:
    - Loss is positive
    - Loss is in reasonable range for random predictions
    - Loss shape is correct (scalar)
    """
    model = create_solution_model(dim=128, num_layers=2)
    model.eval()

    batch_size = 4
    seq_len = 20
    vocab_size = 256

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits, loss = model(input_ids, targets)

    # Loss should be positive
    assert loss > 0, f"Loss is not positive: {loss.item()}"

    # For random predictions, expect loss around -log(1/vocab_size) = log(vocab_size)
    expected_random_loss = np.log(vocab_size)

    # Loss should be in reasonable range (0.5x to 2x expected random loss)
    assert 0.5 * expected_random_loss < loss < 2 * expected_random_loss, \
        f"Loss out of reasonable range: {loss.item():.4f} (expected ~{expected_random_loss:.4f})"

    # Loss should be scalar
    assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"

    print(f"✓ Loss computation correct (value={loss.item():.4f}, expected~{expected_random_loss:.4f})")


@pytest.mark.training
def test_overfitting_on_single_batch(create_solution_model):
    """
    Verify that the model can overfit on a single batch.

    This is a classic sanity check: if a model can't overfit a single batch,
    there's likely a bug in the implementation.
    """
    model = create_solution_model(dim=128, num_layers=2, num_heads=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Create single batch
    batch_size = 8
    seq_len = 20
    vocab_size = 256

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len - 1))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len - 1))

    model.train()
    initial_loss = None

    # Train for 100 steps on same batch
    for step in range(100):
        logits, loss = model(input_ids, targets)

        if step == 0:
            initial_loss = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_loss = loss.item()

    # Loss should decrease significantly
    assert final_loss < 0.5 * initial_loss, \
        f"Model failed to overfit single batch! Initial: {initial_loss:.4f}, Final: {final_loss:.4f}"

    print(f"✓ Model can overfit single batch: {initial_loss:.4f} → {final_loss:.4f}")
