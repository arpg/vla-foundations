---
name: new-assignment
description: "Scaffold new assignment with proper directory structure and templates"
user-invocable: true
---

# New Assignment: Assignment Scaffolding Generator

This skill creates a complete assignment structure with all necessary files, templates, and configurations.

## Execution Steps

### Step 1: Prompt for Assignment Details

Ask user for assignment information:

```
Use AskUserQuestion to ask:
- Question: "What is the assignment name?"
- Header: "Assignment Name"
- Options:
  - label: "Scratch-1", description: "Create Scratch-1 assignment"
  - label: "Scratch-2", description: "Create Scratch-2 assignment"
  - label: "Scratch-3", description: "Create Scratch-3 assignment"
  - label: "Custom", description: "Enter custom assignment name"
```

If "Custom":
```
Prompt: "Enter assignment name (e.g., scratch-4, project-1):"
```

Validate format:
```bash
ASSIGNMENT_NAME="scratch-X"

# Convert to lowercase with hyphens
ASSIGNMENT_NAME=$(echo "$ASSIGNMENT_NAME" | tr '[:upper:]' '[:lower:]' | tr '_' '-')

# Check if already exists
if [ -d "src/assignments/$ASSIGNMENT_NAME" ]; then
    echo "❌ ERROR: Assignment '$ASSIGNMENT_NAME' already exists!"
    exit 1
fi
```

### Step 2: Gather Assignment Metadata

Prompt for additional details:

```
Use AskUserQuestion to ask multiple questions:

1. Question: "What type of assignment is this?"
   Header: "Type"
   Options:
     - label: "Implementation", description: "Code implementation assignment"
     - label: "Analysis", description: "Data analysis and reporting"
     - label: "Project", description: "Multi-week project"

2. Question: "What is the main focus?"
   Header: "Focus Area"
   Options:
     - label: "Vision Transformers", description: "ViT, DINOv2, etc."
     - label: "Action Prediction", description: "Policy learning"
     - label: "Data Processing", description: "Dataset manipulation"
     - label: "Full Pipeline", description: "End-to-end VLA system"

3. Question: "Estimated difficulty level?"
   Header: "Difficulty"
   Options:
     - label: "Beginner", description: "Introductory concepts"
     - label: "Intermediate (Recommended)", description: "Core VLA concepts"
     - label: "Advanced", description: "Research-level implementation"
```

### Step 3: Create Directory Structure

Create the complete assignment structure:

```bash
ASSIGNMENT_NAME="scratch-X"

echo "Creating directory structure for $ASSIGNMENT_NAME..."

# Main assignment directory
mkdir -p "src/assignments/$ASSIGNMENT_NAME"

# Solution directory
mkdir -p "private/solutions/$ASSIGNMENT_NAME"

# Test directories
mkdir -p "tests/public"
mkdir -p "tests/internal"
mkdir -p "tests/internal/fixtures/$ASSIGNMENT_NAME"

# Content directory
mkdir -p "content/course/assignments"

echo "✓ Directory structure created"
```

### Step 4: Generate Starter Code Templates

Create Python starter files:

**File: `src/assignments/$ASSIGNMENT_NAME/__init__.py`**
```python
"""
$ASSIGNMENT_NAME: <Assignment Title>

<Brief description of the assignment>

Author: <Student Name>
Date: <Submission Date>
"""

__version__ = "1.0.0"
```

**File: `src/assignments/$ASSIGNMENT_NAME/model.py`**
```python
"""
Model implementation for $ASSIGNMENT_NAME.

TODO: Complete the implementation below.
"""

import torch
import torch.nn as nn


class AssignmentModel(nn.Module):
    """
    <Brief model description>

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension

    Example:
        >>> model = AssignmentModel(input_dim=384, hidden_dim=512, output_dim=256)
        >>> x = torch.randn(4, 10, 384)
        >>> output = model(x)
        >>> output.shape
        torch.Size([4, 10, 256])
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # TODO: Define your model architecture here
        # Example:
        # self.linear1 = nn.Linear(input_dim, hidden_dim)
        # self.activation = nn.ReLU()
        # self.linear2 = nn.Linear(hidden_dim, output_dim)

        raise NotImplementedError("Complete the model architecture")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Output tensor of shape (batch_size, seq_len, output_dim)
        """
        # TODO: Implement forward pass
        raise NotImplementedError("Complete the forward pass")


if __name__ == "__main__":
    # Quick test
    model = AssignmentModel(input_dim=384, hidden_dim=512, output_dim=256)
    x = torch.randn(4, 10, 384)

    try:
        output = model(x)
        print(f"✓ Forward pass successful: {output.shape}")
    except NotImplementedError:
        print("⚠️  Model not yet implemented")
```

**File: `src/assignments/$ASSIGNMENT_NAME/train.py`**
```python
"""
Training script for $ASSIGNMENT_NAME.

TODO: Complete the training loop.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from .model import AssignmentModel


def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train for one epoch.

    Args:
        model: The model to train
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # TODO: Implement training step
        # 1. Forward pass
        # 2. Compute loss
        # 3. Backward pass
        # 4. Optimizer step

        raise NotImplementedError("Complete the training loop")

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """
    Validate the model.

    Args:
        model: The model to validate
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on

    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # TODO: Implement validation step
            raise NotImplementedError("Complete the validation loop")

    return total_loss / len(dataloader)


def main():
    """Main training function."""
    # TODO: Set up training configuration
    config = {
        'input_dim': 384,
        'hidden_dim': 512,
        'output_dim': 256,
        'learning_rate': 1e-4,
        'batch_size': 32,
        'num_epochs': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print(f"Training on device: {config['device']}")

    # TODO: Initialize model, optimizer, loss
    # model = AssignmentModel(...)
    # optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    # criterion = nn.CrossEntropyLoss()

    # TODO: Load data
    # train_loader = ...
    # val_loader = ...

    # TODO: Training loop
    # for epoch in range(config['num_epochs']):
    #     train_loss = train_epoch(model, train_loader, optimizer, criterion, config['device'])
    #     val_loss = validate(model, val_loader, criterion, config['device'])
    #     print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

    raise NotImplementedError("Complete the main training function")


if __name__ == "__main__":
    main()
```

### Step 5: Generate Solution Templates

Create solution file templates in `private/solutions/$ASSIGNMENT_NAME/`:

**File: `private/solutions/$ASSIGNMENT_NAME/model_solution.py`**
```python
"""
SOLUTION for $ASSIGNMENT_NAME model.

This is the reference implementation.
NEVER sync this file to the public repository.
"""

import torch
import torch.nn as nn


class AssignmentModel(nn.Module):
    """Complete solution implementation."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # [SOLUTION] Complete implementation
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - complete solution."""
        # [SOLUTION] Complete implementation
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


if __name__ == "__main__":
    model = AssignmentModel(input_dim=384, hidden_dim=512, output_dim=256)
    x = torch.randn(4, 10, 384)
    output = model(x)
    print(f"✓ Solution forward pass: {output.shape}")
```

### Step 6: Generate Public Tests

**File: `tests/public/test_${ASSIGNMENT_NAME}_basic.py`**
```python
"""
Public tests for $ASSIGNMENT_NAME.

Students can run these tests to verify their implementation.
"""

import pytest
import torch

from src.assignments.$ASSIGNMENT_NAME.model import AssignmentModel


@pytest.mark.public
class TestModelStructure:
    """Test basic model structure and initialization."""

    def test_model_initialization(self):
        """Test that model can be initialized with correct parameters."""
        model = AssignmentModel(input_dim=384, hidden_dim=512, output_dim=256)

        assert model.input_dim == 384
        assert model.hidden_dim == 512
        assert model.output_dim == 256

    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shape."""
        model = AssignmentModel(input_dim=384, hidden_dim=512, output_dim=256)
        x = torch.randn(4, 10, 384)

        output = model(x)

        assert output.shape == (4, 10, 256), \
            f"Expected shape (4, 10, 256), got {output.shape}"

    def test_no_nans(self):
        """Test that model doesn't produce NaN values."""
        model = AssignmentModel(input_dim=384, hidden_dim=512, output_dim=256)
        model.eval()

        x = torch.randn(4, 10, 384)

        with torch.no_grad():
            output = model(x)

        assert not torch.isnan(output).any(), "Model produced NaN values"

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = AssignmentModel(input_dim=384, hidden_dim=512, output_dim=256)
        model.train()

        x = torch.randn(4, 10, 384, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check that input has gradients
        assert x.grad is not None, "No gradient flow to input"
        assert not torch.isnan(x.grad).any(), "Gradient contains NaN"
```

### Step 7: Generate Internal Tests

**File: `tests/internal/test_${ASSIGNMENT_NAME}_rigor.py`**
```python
"""
Internal grading tests for $ASSIGNMENT_NAME.

NEVER sync this file to the public repository.
"""

import pytest
import torch
from pathlib import Path

# Mark all tests as internal
pytestmark = [pytest.mark.internal, pytest.mark.rigor]

from src.assignments.$ASSIGNMENT_NAME.model import AssignmentModel


@pytest.mark.fidelity
def test_output_fidelity():
    """Compare student output against gold standard."""
    fixture_path = Path(f"tests/internal/fixtures/$ASSIGNMENT_NAME/gold_output.pt")

    if not fixture_path.exists():
        pytest.skip("Gold standard fixture not yet generated")

    gold_data = torch.load(fixture_path)
    test_input = gold_data['input']
    gold_output = gold_data['output']

    model = AssignmentModel(input_dim=384, hidden_dim=512, output_dim=256)
    model.eval()

    with torch.no_grad():
        student_output = model(test_input)

    # Check similarity
    torch.testing.assert_close(
        student_output,
        gold_output,
        rtol=1e-4,
        atol=1e-5,
        msg="Output differs significantly from gold standard"
    )


@pytest.mark.training
def test_training_convergence():
    """Verify model can train and converge."""
    model = AssignmentModel(input_dim=384, hidden_dim=512, output_dim=256)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Generate dummy data
    x = torch.randn(32, 10, 384)
    y = torch.randn(32, 10, 256)

    model.train()
    initial_loss = None

    for step in range(10):
        optimizer.zero_grad()
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)

        if step == 0:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()

    final_loss = loss.item()

    assert final_loss < initial_loss, \
        f"Model did not converge: {initial_loss:.4f} → {final_loss:.4f}"
```

### Step 8: Generate Assignment Spec (MDX)

**File: `content/course/assignments/${ASSIGNMENT_NAME}.mdx`**
```mdx
---
title: "$ASSIGNMENT_NAME: <Assignment Title>"
description: "<Brief description>"
due_date: "TBD"
points: 100
difficulty: "intermediate"
tags: ["vision", "transformers", "pytorch"]
---

# $ASSIGNMENT_NAME: <Assignment Title>

<div className="assignment-metadata">
  **Due Date**: TBD
  **Points**: 100
  **Difficulty**: Intermediate
</div>

## Overview

<Brief introduction to the assignment>

## Learning Objectives

By completing this assignment, you will:

- [ ] Understand <concept 1>
- [ ] Implement <component 1>
- [ ] Apply <technique 1>
- [ ] Evaluate <metric 1>

## Background

### Required Reading

- [Paper 1](https://arxiv.org/abs/...)
- [Tutorial 1](https://...)

### Prerequisites

- Assignment: <prerequisite-assignment>
- Concepts: <prerequisite-concepts>

## Assignment Tasks

### Task 1: <Task Title>

**Objective**: <What to implement>

**Files**: `src/assignments/$ASSIGNMENT_NAME/model.py`

**Instructions**:

1. Step 1
2. Step 2
3. Step 3

**Hints**:
- Hint 1
- Hint 2

### Task 2: <Task Title>

**Objective**: <What to implement>

**Files**: `src/assignments/$ASSIGNMENT_NAME/train.py`

**Instructions**:

1. Step 1
2. Step 2

## Testing

Run the public tests to verify your implementation:

\```bash
pytest tests/public/test_${ASSIGNMENT_NAME}_basic.py -v
\```

Expected output:
\```
tests/public/test_${ASSIGNMENT_NAME}_basic.py::TestModelStructure::test_model_initialization PASSED
tests/public/test_${ASSIGNMENT_NAME}_basic.py::TestModelStructure::test_forward_pass_shape PASSED
...
\```

## Submission

1. Create branch: `$ASSIGNMENT_NAME-<your-username>`
2. Commit your changes
3. Open PR to `staging` branch
4. Ensure all tests pass

## Grading Rubric

| Component | Points | Description |
|-----------|--------|-------------|
| Model Implementation | 40 | Correct architecture and forward pass |
| Training Loop | 30 | Proper training implementation |
| Code Quality | 15 | Clean, documented code |
| Tests Passing | 15 | All public tests pass |

## Resources

- [Documentation](https://pytorch.org/docs/)
- [Examples](https://...)

## FAQ

<details>
<summary>How do I handle X?</summary>

Answer...
</details>

---

**Questions?** Post in the course forum or attend office hours.
```

### Step 9: Update Navigation/Metadata

Add assignment to course navigation:

**File: `content/course/metadata.json`** (or wherever navigation is defined)
```json
{
  "assignments": [
    {
      "id": "$ASSIGNMENT_NAME",
      "title": "<Assignment Title>",
      "slug": "$ASSIGNMENT_NAME",
      "due_date": "TBD",
      "points": 100,
      "order": X
    }
  ]
}
```

### Step 10: Create README for Assignment

**File: `src/assignments/$ASSIGNMENT_NAME/README.md`**
```markdown
# $ASSIGNMENT_NAME: <Assignment Title>

## Quick Start

1. Review the assignment spec: [link to MDX]
2. Implement the TODOs in `model.py`
3. Complete the training loop in `train.py`
4. Run tests: `pytest tests/public/test_${ASSIGNMENT_NAME}_basic.py`
5. Submit via PR

## Files

- `model.py` - Model architecture (TODO)
- `train.py` - Training script (TODO)
- `__init__.py` - Package initialization

## Testing

\```bash
# Run public tests
pytest tests/public/test_${ASSIGNMENT_NAME}_basic.py -v

# Run specific test
pytest tests/public/test_${ASSIGNMENT_NAME}_basic.py::TestModelStructure::test_forward_pass_shape
\```

## Debugging

Common issues:

1. **Shape mismatch**: Check tensor dimensions
2. **NaN values**: Check for division by zero or log of negative numbers
3. **No gradient flow**: Ensure all operations are differentiable

## Resources

- PyTorch docs: https://pytorch.org/docs/
- Assignment spec: content/course/assignments/${ASSIGNMENT_NAME}.mdx
```

### Step 11: Generate .gitkeep Files

Ensure empty directories are tracked:

```bash
touch "tests/internal/fixtures/$ASSIGNMENT_NAME/.gitkeep"
```

### Step 12: Display Summary

```
╔══════════════════════════════════════════════════════════════╗
║         ASSIGNMENT SCAFFOLDING COMPLETE: $ASSIGNMENT_NAME    ║
╚══════════════════════════════════════════════════════════════╝

Assignment: $ASSIGNMENT_NAME
Type: <type>
Focus: <focus>
Difficulty: <difficulty>

════════════════════════════════════════════════════════════════

📁 Directory Structure Created:

Student Files (src/assignments/$ASSIGNMENT_NAME/):
  ✅ __init__.py
  ✅ model.py (with TODOs)
  ✅ train.py (with TODOs)
  ✅ README.md

Solution Files (private/solutions/$ASSIGNMENT_NAME/):
  ✅ model_solution.py
  ✅ train_solution.py (TODO: complete)

Test Files:
  ✅ tests/public/test_${ASSIGNMENT_NAME}_basic.py
  ✅ tests/internal/test_${ASSIGNMENT_NAME}_rigor.py
  ✅ tests/internal/fixtures/$ASSIGNMENT_NAME/

Content:
  ✅ content/course/assignments/${ASSIGNMENT_NAME}.mdx

════════════════════════════════════════════════════════════════

Next Steps:

1. Complete solution implementation:
   - Edit: private/solutions/$ASSIGNMENT_NAME/model_solution.py
   - Edit: private/solutions/$ASSIGNMENT_NAME/train_solution.py

2. Generate gold standard fixtures:
   - Run: /generate-fixtures
   - Select: $ASSIGNMENT_NAME

3. Update assignment spec:
   - Edit: content/course/assignments/${ASSIGNMENT_NAME}.mdx
   - Set due date, points, tasks

4. Test the assignment workflow:
   - Inject: python3 scripts/dev_utils.py --inject $ASSIGNMENT_NAME
   - Test: /test-rigor
   - Reset: python3 scripts/dev_utils.py --reset $ASSIGNMENT_NAME

5. Commit the scaffolding:
   - git add .
   - git commit -m "feat: scaffold $ASSIGNMENT_NAME assignment"
   - git push

6. When ready to release:
   - Run: /release
   - Select: $ASSIGNMENT_NAME

════════════════════════════════════════════════════════════════

Total files created: 12
```

## Notes

- All starter files contain TODO markers for students
- Solution files are marked with [SOLUTION] comments
- Public tests validate structure, internal tests validate correctness
- MDX spec provides complete assignment documentation
- Use `/generate-fixtures` after completing solution
- Use `/release` when ready to publish to students
