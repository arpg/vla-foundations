"""
Public tests for Scratch-1 assignment (migrated from test_scratch1.py).

Tests the provided (non-TODO) components to ensure the assignment starter code works correctly.
Students can run these tests to validate their environment and provided components.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

# Add scratch-1 directory to path
scratch1_dir = Path(__file__).parent.parent.parent / "src" / "assignments" / "scratch-1"
sys.path.insert(0, str(scratch1_dir))

# Import modules from assignment
from backbone import (
    RotaryPositionalEmbedding,
    FeedForward,
    TransformerBlock,
    DecoderOnlyTransformer,
)
from generate_data import (
    forward_kinematics_7dof,
    generate_trajectory,
    generate_dataset,
    create_dataloaders,
)

# Mark all tests in this file as public
pytestmark = pytest.mark.public


class TestDataGeneration:
    """Test suite for data generation utilities."""

    def test_forward_kinematics(self):
        """Test that forward kinematics works correctly."""
        joint_angles = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        ee_pos = forward_kinematics_7dof(joint_angles)

        assert ee_pos.shape == (3,), f"Expected shape (3,), got {ee_pos.shape}"
        assert np.all(np.isfinite(ee_pos)), "FK output contains non-finite values"

    def test_trajectory_generation(self):
        """Test that trajectory generation produces valid outputs."""
        start_joints = np.random.uniform(-np.pi/2, np.pi/2, size=7)
        target_pos = np.array([1.0, 0.5, 0.3])
        states, actions = generate_trajectory(start_joints, target_pos, seq_length=50)

        assert states.shape == (50, 10), f"Expected states shape (50, 10), got {states.shape}"
        assert actions.shape == (50,), f"Expected actions shape (50,), got {actions.shape}"
        assert actions.dtype == np.int64, f"Expected actions dtype int64, got {actions.dtype}"
        assert np.all((actions >= 0) & (actions < 256)), "Actions out of range [0, 255]"

    def test_dataset_generation(self):
        """Test that dataset generation works correctly."""
        dataset = generate_dataset(num_trajectories=100, seq_length=50, seed=42)

        assert 'states' in dataset and 'actions' in dataset
        assert dataset['states'].shape == (100, 50, 10)
        assert dataset['actions'].shape == (100, 50)

    def test_dataloader_creation(self):
        """Test that dataloaders are created with correct batch sizes."""
        dataset = generate_dataset(num_trajectories=100, seq_length=50, seed=42)
        train_loader, val_loader = create_dataloaders(dataset, batch_size=16, train_split=0.8)

        states_batch, actions_batch = next(iter(train_loader))

        assert states_batch.shape[0] == 16, f"Expected batch size 16, got {states_batch.shape[0]}"
        assert states_batch.shape[1:] == (50, 10)
        assert actions_batch.shape == (16, 50)

        # Verify we have batches
        assert len(train_loader) > 0, "Train loader is empty"
        assert len(val_loader) > 0, "Val loader is empty"


class TestRotaryPositionalEmbedding:
    """Test suite for Rotary Position Embedding (provided component)."""

    def test_rope_preserves_shape(self):
        """Verify that RoPE preserves tensor shapes."""
        batch_size = 2
        num_heads = 4
        seq_len = 10
        head_dim = 32

        rope = RotaryPositionalEmbedding(dim=head_dim, max_seq_len=128)

        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)

        q_rot, k_rot = rope(q, k)

        assert q_rot.shape == q.shape, f"Q shape changed: {q.shape} -> {q_rot.shape}"
        assert k_rot.shape == k.shape, f"K shape changed: {k.shape} -> {k_rot.shape}"

    def test_rope_modifies_values(self):
        """Verify that RoPE actually rotates the embeddings."""
        batch_size = 2
        num_heads = 4
        seq_len = 10
        head_dim = 32

        rope = RotaryPositionalEmbedding(dim=head_dim, max_seq_len=128)

        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)

        q_rot, k_rot = rope(q, k)

        # Verify values changed (rotation applied)
        assert not torch.allclose(q, q_rot), "RoPE did not modify Q"
        assert not torch.allclose(k, k_rot), "RoPE did not modify K"


class TestFeedForward:
    """Test suite for FeedForward layer (provided component)."""

    def test_feedforward_shape_preservation(self):
        """Verify that FeedForward preserves input shape."""
        batch_size = 4
        seq_len = 20
        dim = 128
        hidden_dim = 512

        ff = FeedForward(dim=dim, hidden_dim=hidden_dim, dropout=0.0)

        x = torch.randn(batch_size, seq_len, dim)
        output = ff(x)

        assert output.shape == x.shape, f"Shape mismatch: {x.shape} -> {output.shape}"

    def test_feedforward_with_dropout(self):
        """Verify that FeedForward works with dropout enabled."""
        batch_size = 4
        seq_len = 20
        dim = 128
        hidden_dim = 512

        ff = FeedForward(dim=dim, hidden_dim=hidden_dim, dropout=0.1)
        ff.eval()

        x = torch.randn(batch_size, seq_len, dim)
        output = ff(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any(), "Output contains NaNs"

    def test_feedforward_parameter_count(self):
        """Verify that FeedForward has expected number of parameters."""
        dim = 128
        hidden_dim = 512

        ff = FeedForward(dim=dim, hidden_dim=hidden_dim, dropout=0.0)

        num_params = sum(p.numel() for p in ff.parameters())

        # Expected: (dim * hidden_dim) + hidden_dim + (hidden_dim * dim) + dim
        expected = (dim * hidden_dim) + hidden_dim + (hidden_dim * dim) + dim

        assert num_params == expected, f"Expected {expected} parameters, got {num_params}"


class TestTransformerBlock:
    """Test suite for TransformerBlock instantiation."""

    def test_transformer_block_instantiation(self):
        """Test that TransformerBlock can be instantiated."""
        dim = 128
        num_heads = 8
        ff_hidden_dim = 512

        block = TransformerBlock(
            dim=dim,
            num_heads=num_heads,
            ff_hidden_dim=ff_hidden_dim,
            dropout=0.0
        )

        # Verify components exist
        assert hasattr(block, 'attention'), "TransformerBlock missing attention"
        assert hasattr(block, 'feed_forward'), "TransformerBlock missing feed_forward"
        assert hasattr(block, 'norm1'), "TransformerBlock missing norm1"
        assert hasattr(block, 'norm2'), "TransformerBlock missing norm2"

        # Verify parameter count is reasonable
        num_params = sum(p.numel() for p in block.parameters())
        assert num_params > 0, "TransformerBlock has no parameters"

    def test_transformer_block_components(self):
        """Test that TransformerBlock has correct component types."""
        dim = 128
        num_heads = 8
        ff_hidden_dim = 512

        block = TransformerBlock(
            dim=dim,
            num_heads=num_heads,
            ff_hidden_dim=ff_hidden_dim,
            dropout=0.0
        )

        # Check types (basic smoke test)
        assert isinstance(block.feed_forward, FeedForward)
        # Note: Attention is student-implemented, so we can't check its type


class TestDecoderOnlyTransformer:
    """Test suite for full model instantiation."""

    def test_model_instantiation(self):
        """Test that the full model can be instantiated."""
        vocab_size = 256
        dim = 128
        num_layers = 2
        num_heads = 8
        ff_hidden_dim = 512
        max_seq_len = 50

        model = DecoderOnlyTransformer(
            vocab_size=vocab_size,
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_hidden_dim=ff_hidden_dim,
            max_seq_len=max_seq_len,
            dropout=0.0
        )

        # Verify components
        assert hasattr(model, 'token_embedding')
        assert hasattr(model, 'blocks')
        assert hasattr(model, 'norm_final')
        assert hasattr(model, 'lm_head')

        assert model.token_embedding.num_embeddings == vocab_size
        assert model.token_embedding.embedding_dim == dim
        assert len(model.blocks) == num_layers
        assert model.lm_head.out_features == vocab_size

    def test_model_parameter_count(self):
        """Test that the model has a reasonable number of parameters."""
        vocab_size = 256
        dim = 128
        num_layers = 2
        num_heads = 8
        ff_hidden_dim = 512

        model = DecoderOnlyTransformer(
            vocab_size=vocab_size,
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_hidden_dim=ff_hidden_dim,
            dropout=0.0
        )

        num_params = sum(p.numel() for p in model.parameters())

        # Model should have at least vocab_size * dim parameters (just for embedding)
        min_expected = vocab_size * dim
        assert num_params >= min_expected, f"Model has only {num_params} parameters, expected at least {min_expected}"


class TestExpectedShapes:
    """Test expected tensor shapes through the pipeline."""

    def test_input_shapes(self):
        """Test that input shapes are as expected."""
        batch_size = 4
        seq_len = 50
        vocab_size = 256

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))

        assert input_ids.shape == (batch_size, seq_len)
        assert targets.shape == (batch_size, seq_len)

    def test_expected_flow(self):
        """Document expected tensor flow through the model."""
        batch_size = 4
        seq_len = 50
        vocab_size = 256
        dim = 128

        # This is a documentation test - just verify our expectations
        assert True, "Shape flow: (B, L) -> (B, L, D) -> (B, L, D) -> (B, L, V)"
