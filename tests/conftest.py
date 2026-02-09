"""
Shared pytest configuration and fixtures.

This module provides common fixtures and utilities used across both
public and internal test suites.
"""

import pytest
import torch
import subprocess
import sys
from pathlib import Path


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def gold_standards_dir(project_root):
    """Return the directory containing gold standard fixtures."""
    return project_root / "tests" / "internal" / "fixtures"


@pytest.fixture
def load_gold_standard(gold_standards_dir):
    """
    Fixture factory for loading gold standard tensors.

    Usage:
        def test_something(load_gold_standard):
            data = load_gold_standard('scratch1_gold_output.pt')
    """
    def _load(filename: str):
        filepath = gold_standards_dir / filename
        if not filepath.exists():
            pytest.skip(f"Gold standard file not found: {filepath}")
        return torch.load(filepath)

    return _load


@pytest.fixture(scope="session", autouse=True)
def inject_solutions_for_internal_tests(request):
    """
    Automatically inject solutions before internal tests run.
    Reset after all tests complete.

    Only activates when running tests from tests/internal/ directory
    AND when dev_utils.py exists (i.e., in instructor environment).
    """
    # Check if we're running internal tests
    test_items = [item for item in request.session.items]
    internal_tests = [item for item in test_items if "tests/internal" in str(item.fspath)]

    if not internal_tests:
        # Not running internal tests, skip solution injection
        return

    # Check if dev_utils.py exists (instructor environment)
    dev_utils_path = Path(__file__).parent.parent / "scripts" / "dev_utils.py"
    if not dev_utils_path.exists():
        # Skip injection in student/grading environment
        print("\n=== Skipping solution injection (dev_utils.py not found) ===")
        return

    # Inject solutions
    print("\n=== Injecting solutions for internal tests ===")
    result = subprocess.run(
        [sys.executable, "scripts/dev_utils.py", "--inject", "scratch-1"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        pytest.exit(f"Failed to inject solutions: {result.stderr}")

    print(result.stdout)

    yield

    # Reset after tests
    print("\n=== Resetting to starter code ===")
    result = subprocess.run(
        [sys.executable, "scripts/dev_utils.py", "--reset", "scratch-1"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Warning: Failed to reset solutions: {result.stderr}")
    else:
        print(result.stdout)


@pytest.fixture
def small_transformer_config():
    """
    Configuration for a small transformer model (for fast tests).
    """
    return {
        'vocab_size': 256,
        'dim': 128,
        'num_layers': 2,
        'num_heads': 4,
        'ff_hidden_dim': 512,
        'max_seq_len': 50,
        'dropout': 0.0
    }


@pytest.fixture
def sample_batch():
    """
    Generate a small sample batch for testing.
    """
    batch_size = 4
    seq_len = 20
    vocab_size = 256

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    return {
        'input_ids': input_ids,
        'targets': targets,
        'batch_size': batch_size,
        'seq_len': seq_len,
        'vocab_size': vocab_size
    }
