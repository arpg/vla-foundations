"""
Generate gold standard test fixtures for Scratch-1.

This script creates reference outputs for grading student solutions.
Run after verifying the solution works correctly.

Usage:
    python generate_gold_fixtures.py
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add paths
SCRIPT_DIR = Path(__file__).parent
SOLUTIONS_DIR = SCRIPT_DIR.parent / "solutions"
ASSIGNMENTS_DIR = SCRIPT_DIR.parent.parent / "src" / "assignments" / "scratch-1"

sys.path.insert(0, str(SOLUTIONS_DIR))
sys.path.insert(0, str(ASSIGNMENTS_DIR))


def generate_scratch1_gold_fixtures():
    """Generate gold standard fixtures for Scratch-1 tests."""

    from backbone_complete import DecoderOnlyTransformer, RMSNorm

    print("Generating Scratch-1 gold standard fixtures...")

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Create model with fixed configuration
    model = DecoderOnlyTransformer(
        vocab_size=256,
        dim=384,
        num_layers=4,
        num_heads=8,
        ff_hidden_dim=1024,
        max_seq_len=50,
        dropout=0.0,  # No dropout for deterministic output
    )
    model.eval()

    # Generate test input for projector (simulated DINOv2 output)
    # DINOv2-small outputs 384-dim features
    test_input = torch.randn(2, 384)  # batch_size=2, dinov2_dim=384

    # Create a simple projector to generate gold output
    # This simulates what a student's projector would produce
    projector = nn.Sequential(
        nn.Linear(384, 1024),
        nn.GELU(),
        nn.Linear(1024, 384),
        nn.LayerNorm(384)
    )
    projector.eval()

    with torch.no_grad():
        gold_output = projector(test_input)

    # Save projector gold standard
    gold_data = {
        'input': test_input,
        'output': gold_output,
        'projector_state_dict': projector.state_dict(),
    }

    output_path = SCRIPT_DIR / "scratch1_gold_output.pt"
    torch.save(gold_data, output_path)
    print(f"  Saved: {output_path}")

    # Generate attention pattern fixture
    batch_size = 2
    seq_len = 10

    # Sample input sequence
    sample_input_ids = torch.randint(0, 256, (batch_size, seq_len))

    with torch.no_grad():
        logits, _ = model(sample_input_ids, sample_input_ids)

    attention_fixture = {
        'input_ids': sample_input_ids,
        'expected_logits_shape': logits.shape,
        'logits_mean': logits.mean().item(),
        'logits_std': logits.std().item(),
    }

    attention_path = SCRIPT_DIR / "scratch1_attention_fixture.pt"
    torch.save(attention_fixture, attention_path)
    print(f"  Saved: {attention_path}")

    # Generate RMSNorm fixture
    rmsnorm = RMSNorm(dim=384)
    rmsnorm.eval()

    test_input_norm = torch.randn(2, 10, 384)
    with torch.no_grad():
        norm_output = rmsnorm(test_input_norm)

    rmsnorm_fixture = {
        'input': test_input_norm,
        'output': norm_output,
        'scale': rmsnorm.scale.clone(),
    }

    rmsnorm_path = SCRIPT_DIR / "scratch1_rmsnorm_fixture.pt"
    torch.save(rmsnorm_fixture, rmsnorm_path)
    print(f"  Saved: {rmsnorm_path}")

    print("\nGold standard fixtures generated successfully!")
    print(f"Files created in: {SCRIPT_DIR}")


if __name__ == "__main__":
    generate_scratch1_gold_fixtures()
