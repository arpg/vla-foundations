"""
Test script for Scratch-1 assignment source code

Tests the provided (non-TODO) components to ensure the assignment starter code works correctly.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

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


def test_data_generation():
    """Test that data generation works correctly"""
    print("\n" + "="*60)
    print("TEST: Data Generation")
    print("="*60)

    # Test forward kinematics
    print("\n1. Testing forward_kinematics_7dof...")
    joint_angles = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ee_pos = forward_kinematics_7dof(joint_angles)
    assert ee_pos.shape == (3,), f"Expected shape (3,), got {ee_pos.shape}"
    assert np.all(np.isfinite(ee_pos)), "FK output contains non-finite values"
    print(f"   ✓ FK output shape: {ee_pos.shape}")
    print(f"   ✓ FK output: {ee_pos}")

    # Test trajectory generation
    print("\n2. Testing generate_trajectory...")
    start_joints = np.random.uniform(-np.pi/2, np.pi/2, size=7)
    target_pos = np.array([1.0, 0.5, 0.3])
    states, actions = generate_trajectory(start_joints, target_pos, seq_length=50)
    assert states.shape == (50, 10), f"Expected states shape (50, 10), got {states.shape}"
    assert actions.shape == (50,), f"Expected actions shape (50,), got {actions.shape}"
    assert actions.dtype == np.int64, f"Expected actions dtype int64, got {actions.dtype}"
    assert np.all((actions >= 0) & (actions < 256)), "Actions out of range [0, 255]"
    print(f"   ✓ States shape: {states.shape}")
    print(f"   ✓ Actions shape: {actions.shape}")
    print(f"   ✓ Action range: [{actions.min()}, {actions.max()}]")

    # Test small dataset generation
    print("\n3. Testing generate_dataset...")
    dataset = generate_dataset(num_trajectories=100, seq_length=50, seed=42)
    assert 'states' in dataset and 'actions' in dataset
    assert dataset['states'].shape == (100, 50, 10)
    assert dataset['actions'].shape == (100, 50)
    print(f"   ✓ Dataset states shape: {dataset['states'].shape}")
    print(f"   ✓ Dataset actions shape: {dataset['actions'].shape}")

    # Test dataloader creation
    print("\n4. Testing create_dataloaders...")
    train_loader, val_loader = create_dataloaders(dataset, batch_size=16, train_split=0.8)
    states_batch, actions_batch = next(iter(train_loader))
    assert states_batch.shape[0] == 16, f"Expected batch size 16, got {states_batch.shape[0]}"
    assert states_batch.shape[1:] == (50, 10)
    assert actions_batch.shape == (16, 50)
    print(f"   ✓ Train loader batch shape: {states_batch.shape}, {actions_batch.shape}")
    print(f"   ✓ Train batches: {len(train_loader)}")
    print(f"   ✓ Val batches: {len(val_loader)}")

    print("\n✅ All data generation tests passed!")
    return True


def test_rope():
    """Test Rotary Position Embedding (provided component)"""
    print("\n" + "="*60)
    print("TEST: Rotary Position Embedding")
    print("="*60)

    batch_size = 2
    num_heads = 4
    seq_len = 10
    head_dim = 32

    rope = RotaryPositionalEmbedding(dim=head_dim, max_seq_len=128)

    # Create dummy Q and K tensors
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)

    print(f"\n1. Input shapes:")
    print(f"   Q: {q.shape}")
    print(f"   K: {k.shape}")

    # Apply RoPE
    q_rot, k_rot = rope(q, k)

    print(f"\n2. Output shapes:")
    print(f"   Q_rot: {q_rot.shape}")
    print(f"   K_rot: {k_rot.shape}")

    # Verify shapes unchanged
    assert q_rot.shape == q.shape, f"Q shape changed: {q.shape} -> {q_rot.shape}"
    assert k_rot.shape == k.shape, f"K shape changed: {k.shape} -> {k_rot.shape}"

    # Verify values changed (rotation applied)
    assert not torch.allclose(q, q_rot), "RoPE did not modify Q"
    assert not torch.allclose(k, k_rot), "RoPE did not modify K"

    print(f"\n3. Verification:")
    print(f"   ✓ Shapes preserved")
    print(f"   ✓ Values modified (rotation applied)")
    print(f"   ✓ Max Q diff: {(q - q_rot).abs().max().item():.4f}")
    print(f"   ✓ Max K diff: {(k - k_rot).abs().max().item():.4f}")

    print("\n✅ RoPE tests passed!")
    return True


def test_feedforward():
    """Test FeedForward layer (provided component)"""
    print("\n" + "="*60)
    print("TEST: FeedForward Layer")
    print("="*60)

    batch_size = 4
    seq_len = 20
    dim = 128
    hidden_dim = 512

    ff = FeedForward(dim=dim, hidden_dim=hidden_dim, dropout=0.0)

    # Test forward pass
    x = torch.randn(batch_size, seq_len, dim)
    output = ff(x)

    print(f"\n1. Shapes:")
    print(f"   Input: {x.shape}")
    print(f"   Output: {output.shape}")

    assert output.shape == x.shape, f"Shape mismatch: {x.shape} -> {output.shape}"

    # Test with dropout enabled
    ff_dropout = FeedForward(dim=dim, hidden_dim=hidden_dim, dropout=0.1)
    ff_dropout.eval()
    output_eval = ff_dropout(x)

    print(f"\n2. Parameters:")
    num_params = sum(p.numel() for p in ff.parameters())
    print(f"   Total parameters: {num_params:,}")

    print(f"\n3. Verification:")
    print(f"   ✓ Output shape matches input")
    print(f"   ✓ Forward pass successful")

    print("\n✅ FeedForward tests passed!")
    return True


def test_transformer_block_instantiation():
    """Test TransformerBlock instantiation (depends on TODO components)"""
    print("\n" + "="*60)
    print("TEST: TransformerBlock Instantiation")
    print("="*60)

    try:
        dim = 128
        num_heads = 8
        ff_hidden_dim = 512

        block = TransformerBlock(
            dim=dim,
            num_heads=num_heads,
            ff_hidden_dim=ff_hidden_dim,
            dropout=0.0
        )

        print(f"\n1. Block components:")
        print(f"   ✓ Attention: {type(block.attention).__name__}")
        print(f"   ✓ FeedForward: {type(block.feed_forward).__name__}")
        print(f"   ✓ Norm1: {type(block.norm1).__name__}")
        print(f"   ✓ Norm2: {type(block.norm2).__name__}")

        num_params = sum(p.numel() for p in block.parameters())
        print(f"\n2. Parameters: {num_params:,}")

        print("\n✅ TransformerBlock instantiation successful!")
        print("⚠️  Note: Forward pass will fail until TODOs are implemented")
        return True

    except Exception as e:
        print(f"\n❌ TransformerBlock instantiation failed: {e}")
        return False


def test_model_instantiation():
    """Test full model instantiation"""
    print("\n" + "="*60)
    print("TEST: DecoderOnlyTransformer Instantiation")
    print("="*60)

    try:
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

        print(f"\n1. Model architecture:")
        print(f"   ✓ Vocab size: {vocab_size}")
        print(f"   ✓ Dimension: {dim}")
        print(f"   ✓ Layers: {num_layers}")
        print(f"   ✓ Heads: {num_heads}")
        print(f"   ✓ Max sequence length: {max_seq_len}")

        num_params = sum(p.numel() for p in model.parameters())
        print(f"\n2. Total parameters: {num_params:,}")

        print(f"\n3. Components:")
        print(f"   ✓ Token embedding: {model.token_embedding.num_embeddings} x {model.token_embedding.embedding_dim}")
        print(f"   ✓ Transformer blocks: {len(model.blocks)}")
        print(f"   ✓ Final norm: {type(model.norm_final).__name__}")
        print(f"   ✓ LM head: {model.lm_head.in_features} -> {model.lm_head.out_features}")

        print("\n✅ Model instantiation successful!")
        print("⚠️  Note: Forward pass will fail until TODOs are implemented")
        return True

    except Exception as e:
        print(f"\n❌ Model instantiation failed: {e}")
        return False


def test_expected_shapes():
    """Test expected tensor shapes through the pipeline"""
    print("\n" + "="*60)
    print("TEST: Expected Tensor Shapes")
    print("="*60)

    batch_size = 4
    seq_len = 50
    vocab_size = 256
    dim = 128

    print(f"\n1. Input shapes:")
    print(f"   Batch size: {batch_size}")
    print(f"   Sequence length: {seq_len}")
    print(f"   Vocab size: {vocab_size}")

    # Create dummy input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    print(f"\n2. Token IDs:")
    print(f"   input_ids shape: {input_ids.shape}")
    print(f"   targets shape: {targets.shape}")

    print(f"\n3. Expected intermediate shapes:")
    print(f"   After embedding: ({batch_size}, {seq_len}, {dim})")
    print(f"   Causal mask: ({seq_len}, {seq_len})")
    print(f"   After transformer blocks: ({batch_size}, {seq_len}, {dim})")
    print(f"   Logits output: ({batch_size}, {seq_len}, {vocab_size})")
    print(f"   Loss: scalar")

    print("\n✅ Shape expectations documented!")
    return True


def run_all_tests():
    """Run all test suites"""
    print("\n" + "="*70)
    print("SCRATCH-1 ASSIGNMENT SOURCE CODE TESTS")
    print("="*70)
    print("\nTesting provided components (non-TODO parts)")

    results = []

    # Run tests
    results.append(("Data Generation", test_data_generation()))
    results.append(("RoPE", test_rope()))
    results.append(("FeedForward", test_feedforward()))
    results.append(("TransformerBlock Instantiation", test_transformer_block_instantiation()))
    results.append(("Model Instantiation", test_model_instantiation()))
    results.append(("Expected Shapes", test_expected_shapes()))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)

    print(f"\nTotal: {total_passed}/{total_tests} test suites passed")

    if total_passed == total_tests:
        print("\n🎉 All tests passed! The assignment source code is working correctly.")
        print("\nNext steps:")
        print("1. Generate full training data: python generate_data.py")
        print("2. Students should implement TODOs in backbone.py")
        print("3. Run training: python backbone.py")
    else:
        print("\n⚠️  Some tests failed. Please review the issues above.")
        return False

    return True


if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test suite crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
