# VLA Foundations Test Suite

This directory contains both **public** and **internal** tests for course assignments.

## Directory Structure

```
tests/
├── conftest.py                    # Shared pytest configuration
├── public/                        # Tests students can see and run
│   ├── __init__.py
│   └── test_scratch1_basic.py    # Basic validation tests
└── internal/                      # Internal grading tests (NEVER public)
    ├── __init__.py
    ├── conftest.py                # Internal test fixtures
    ├── fixtures/                  # Gold standard data
    │   └── scratch1_gold.pt       # Reference tensors
    └── test_scratch1_rigor.py     # Rigorous grading tests
```

## Test Categories

### Public Tests (`tests/public/`)

**Purpose**: Basic validation that students can run locally

**What they test**:
- Provided (non-TODO) components work correctly
- Data generation functions
- Model instantiation
- Basic shape validation

**Usage**:
```bash
# Run all public tests
pytest tests/public/ -v

# Run specific test file
pytest tests/public/test_scratch1_basic.py -v

# Run specific test
pytest tests/public/test_scratch1_basic.py::TestDataGeneration::test_forward_kinematics -v
```

Students can run these tests to verify:
1. Their environment is set up correctly
2. Provided starter code works as expected
3. Basic integration is correct

### Internal Tests (`tests/internal/`)

**Purpose**: Rigorous validation for grading (instructor only)

**What they test**:
- Gradient leak detection (frozen parameters)
- Latent fidelity (comparison against gold standards)
- Training convergence
- Edge cases and robustness
- Numerical stability

**Usage**:
```bash
# Inject solutions first (automatic via conftest.py)
pytest tests/internal/test_scratch1_rigor.py -v

# Run specific test category
pytest tests/internal/ -m gradient -v    # Only gradient tests
pytest tests/internal/ -m fidelity -v    # Only fidelity tests
pytest tests/internal/ -m training -v    # Only training tests
```

**Note**: Internal tests automatically inject solutions before running and reset after completion.

## Test Markers

Tests use pytest markers for categorization:

```bash
# Run only public tests
pytest -m public -v

# Run only internal tests
pytest -m internal -v

# Run gradient leak tests
pytest -m gradient -v

# Run fidelity tests
pytest -m fidelity -v

# Run training convergence tests
pytest -m training -v
```

Available markers (defined in `pytest.ini`):
- `public`: Tests students can see and run
- `internal`: Internal grading tests (never public)
- `rigor`: Rigorous validation tests
- `gradient`: Gradient flow validation
- `fidelity`: Output quality validation
- `training`: Training convergence tests

## Running Tests

### For Students (Public Tests Only)

```bash
# Install pytest
pip install pytest torch numpy

# Run public tests
cd /path/to/vla-foundations
pytest tests/public/ -v
```

### For Instructors (All Tests)

```bash
# Setup (one time)
pip install pytest torch numpy

# Run public tests
pytest tests/public/ -v

# Run internal tests (with solutions)
# Solutions are automatically injected by conftest.py
pytest tests/internal/ -v

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/assignments --cov-report=html
```

## Writing New Tests

### Public Test Template

```python
import pytest

# Mark as public
pytestmark = pytest.mark.public

class TestMyComponent:
    """Test suite for MyComponent."""

    def test_basic_functionality(self):
        """Test basic functionality."""
        # Test provided (non-TODO) components only
        assert True

    def test_shapes(self):
        """Test expected shapes."""
        # Validate tensor shapes
        assert True
```

### Internal Test Template

```python
import pytest

# Mark as internal
pytestmark = [pytest.mark.internal, pytest.mark.rigor]

@pytest.mark.gradient
def test_gradient_flow(create_solution_model):
    """Test gradient flow through the model."""
    model = create_solution_model()

    # Test with complete solution
    assert True

@pytest.mark.fidelity
def test_output_quality(create_solution_model, load_gold_standard):
    """Compare output against gold standard."""
    gold = load_gold_standard('my_gold.pt')

    # Compare student output to gold
    assert True
```

## Fixtures

### Shared Fixtures (`tests/conftest.py`)

- `project_root`: Path to project root
- `gold_standards_dir`: Path to gold standard fixtures
- `load_gold_standard(filename)`: Load gold standard tensor
- `small_transformer_config`: Small model config for fast tests
- `sample_batch`: Sample batch for testing

### Internal Fixtures (`tests/internal/conftest.py`)

- `create_solution_model(**kwargs)`: Factory for creating solution models
- `training_setup`: Complete training environment (model, optimizer, dataloader)

## Gold Standards

Gold standards are reference outputs used for rigorous testing:

```
tests/internal/fixtures/
└── scratch1_gold_output.pt  # Expected projector output
```

Gold standards contain:
```python
{
    'input': torch.Tensor,   # Test input
    'output': torch.Tensor,  # Expected output
    'metadata': {            # Additional context
        'created': '2024-01-23',
        'model_config': {...}
    }
}
```

## Continuous Integration

Tests run automatically on:
- PR creation (public tests only)
- PR merge to staging/main
- Manual trigger

See `.github/workflows/test.yml` for CI configuration.

## Troubleshooting

### Tests not discovered
```bash
# Verify pytest can find tests
pytest --collect-only

# Check pytest.ini configuration
cat pytest.ini
```

### Import errors
```bash
# Verify Python path
pytest tests/public/ -v --tb=short

# Check conftest.py adds correct paths
```

### Fixture not found
```bash
# List available fixtures
pytest --fixtures

# Check fixture scope and location
```

### Internal tests fail
```bash
# Verify solutions are injected
python scripts/manage_solutions.py --list

# Manually inject and test
python scripts/manage_solutions.py --inject scratch-1
pytest tests/internal/test_scratch1_rigor.py -v
python scripts/manage_solutions.py --reset scratch-1
```

## Best Practices

1. **Keep public tests simple**: Only test provided components
2. **Make internal tests comprehensive**: Cover edge cases and robustness
3. **Use descriptive test names**: `test_frozen_dinov2_no_gradients` not `test_1`
4. **Add docstrings**: Explain what each test validates and why
5. **Use markers**: Tag tests appropriately for easy filtering
6. **Keep tests fast**: Use small models and datasets where possible
7. **Isolate tests**: Each test should be independent
