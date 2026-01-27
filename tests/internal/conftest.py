"""
Internal test fixtures and utilities (NEVER public).

Provides fixtures specific to internal grading tests:
- Gold standard data loading
- Solution model instantiation
- Training utilities
"""

import pytest
import torch
import sys
from pathlib import Path

# Path to gold standard fixtures
FIXTURES_DIR = Path(__file__).parent.parent.parent / "private" / "fixtures"


@pytest.fixture(scope="session", autouse=True)
def add_assignments_to_path():
    """
    Add src/assignments/scratch-1 to Python path for internal tests.

    This allows importing modules like:
        from backbone import DecoderOnlyTransformer
    """
    assignments_dir = Path(__file__).parent.parent.parent / "src" / "assignments" / "scratch-1"

    if str(assignments_dir) not in sys.path:
        sys.path.insert(0, str(assignments_dir))

    yield

    # Clean up
    if str(assignments_dir) in sys.path:
        sys.path.remove(str(assignments_dir))


@pytest.fixture
def create_solution_model():
    """
    Factory fixture for creating solution models with various configurations.

    Usage:
        def test_something(create_solution_model):
            model = create_solution_model(dim=384, num_layers=4)
    """
    def _create(**kwargs):
        from backbone import DecoderOnlyTransformer

        # Default configuration
        config = {
            'vocab_size': 256,
            'dim': 384,
            'num_layers': 4,
            'num_heads': 8,
            'ff_hidden_dim': 1024,
            'max_seq_len': 50,
            'dropout': 0.0
        }

        # Override with provided kwargs
        config.update(kwargs)

        return DecoderOnlyTransformer(**config)

    return _create


@pytest.fixture
def training_setup(create_solution_model):
    """
    Set up a complete training environment for convergence tests.

    Returns a dictionary with:
        - model: The transformer model
        - optimizer: AdamW optimizer
        - train_loader: DataLoader with small training set
    """
    from generate_data import generate_dataset, create_dataloaders

    # Generate small training dataset
    dataset = generate_dataset(num_trajectories=100, seq_length=50, seed=42)
    train_loader, _ = create_dataloaders(dataset, batch_size=16, train_split=0.9)

    # Create small model for fast training
    model = create_solution_model(dim=128, num_layers=2, num_heads=4, ff_hidden_dim=512)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    return {
        'model': model,
        'optimizer': optimizer,
        'train_loader': train_loader
    }


@pytest.fixture
def load_gold_standard():
    """
    Factory fixture for loading gold standard test fixtures.

    Usage:
        def test_something(load_gold_standard):
            gold_data = load_gold_standard('scratch1_gold_output.pt')
    """
    def _load(filename: str):
        filepath = FIXTURES_DIR / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Gold standard fixture not found: {filepath}")
        return torch.load(filepath, map_location='cpu', weights_only=False)

    return _load


@pytest.fixture
def sample_batch():
    """
    Create a sample batch for testing.

    Returns:
        Dictionary with 'input_ids' and 'targets' tensors.
    """
    batch_size = 4
    seq_len = 20
    vocab_size = 256

    # Create random input and target sequences
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    return {
        'input_ids': input_ids,
        'targets': targets,
    }
