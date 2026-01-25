---
name: generate-fixtures
description: "Generate gold standard test fixtures from solution code"
user-invocable: true
---

# Generate Fixtures: Gold Standard Test Data Creation

This skill automates creation of reference data for internal fidelity tests by running solution code with fixed random seeds.

## Execution Steps

### Step 1: Prompt for Assignment

Ask the user which assignment to generate fixtures for:

```
Use AskUserQuestion to ask:
- Question: "Which assignment would you like to generate fixtures for?"
- Header: "Assignment"
- Options:
  - label: "Scratch-1", description: "Generate Scratch-1 gold standard data"
  - label: "Scratch-2", description: "Generate Scratch-2 gold standard data"
  - label: "Scratch-3", description: "Generate Scratch-3 gold standard data"
```

Parse the selection to get assignment name (e.g., "scratch-1").

### Step 2: Check Fixture Configuration

Read the assignment's fixture configuration if it exists:

```bash
# Check for fixture generation script
ls src/assignments/<assignment-name>/generate_fixtures.py
```

If no configuration exists, prompt user:

```
Use AskUserQuestion to ask:
- Question: "No fixture generator found. What type of fixtures should we create?"
- Options:
  - label: "Model outputs", description: "Forward pass outputs for fidelity comparison"
  - label: "Trained checkpoints", description: "Model weights after training"
  - label: "Both", description: "Generate both output fixtures and checkpoints"
```

### Step 3: Inject Solutions

Inject solution files:

```bash
python3 scripts/dev_utils.py --inject <assignment-name>
```

Verify injection succeeded.

### Step 4: Set Random Seeds

Ensure reproducibility by setting all random seeds:

```python
import torch
import numpy as np
import random

SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

# For deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

This should be embedded in the fixture generation script.

### Step 5: Generate Fixtures

If a `generate_fixtures.py` script exists:

```bash
cd src/assignments/<assignment-name>
python3 generate_fixtures.py --output ../../../tests/internal/fixtures/<assignment-name>/
```

Otherwise, manually create fixtures:

**For Model Output Fixtures:**
```python
# Load solution model
from src.assignments.<assignment_name> import backbone

model = backbone.DecoderOnlyTransformer(...)
model.eval()

# Create test inputs
test_input = torch.randn(4, 10, 384)  # Example

# Generate outputs
with torch.no_grad():
    gold_output = model(test_input)

# Save
torch.save({
    'input': test_input,
    'output': gold_output,
    'seed': SEED,
    'model_config': {...}
}, f'tests/internal/fixtures/<assignment-name>/gold_output.pt')
```

**For Trained Checkpoint Fixtures:**
```python
# Train solution model
model.train()
# ... training loop ...

# Save best checkpoint
torch.save({
    'model_state_dict': model.state_dict(),
    'epoch': best_epoch,
    'loss': best_loss,
    'seed': SEED,
    'training_config': {...}
}, f'tests/internal/fixtures/<assignment-name>/gold_checkpoint.pt')
```

### Step 6: Verify Fixtures

Load and verify the generated fixtures:

```bash
python3 -c "
import torch
data = torch.load('tests/internal/fixtures/<assignment-name>/gold_output.pt')
print('✓ Fixture loaded successfully')
print(f'  Input shape: {data[\"input\"].shape}')
print(f'  Output shape: {data[\"output\"].shape}')
print(f'  Seed: {data[\"seed\"]}')
print(f'  Contains NaNs: {torch.isnan(data[\"output\"]).any()}')
"
```

Check for:
- No NaN values
- Expected shapes
- Reasonable value ranges

### Step 7: Generate Fixture Documentation

Create a markdown file documenting the fixture:

```markdown
# <Assignment> Gold Standard Fixtures

Generated: $(date)
Seed: 42

## Files

### gold_output.pt
- **Purpose**: Reference output for fidelity comparison
- **Input shape**: [batch, seq_len, dim]
- **Output shape**: [batch, seq_len, vocab_size]
- **Model config**: {...}

### gold_checkpoint.pt
- **Purpose**: Trained model weights for convergence testing
- **Training epochs**: X
- **Final loss**: X.XXXX
- **Training config**: {...}

## Usage in Tests

```python
fixture = torch.load('tests/internal/fixtures/<assignment-name>/gold_output.pt')
test_input = fixture['input']
expected_output = fixture['output']

student_output = student_model(test_input)
torch.testing.assert_close(student_output, expected_output, rtol=1e-4, atol=1e-5)
```

## Regeneration

To regenerate these fixtures:
```bash
/generate-fixtures
# Select <assignment-name>
```
```

Save to: `tests/internal/fixtures/<assignment-name>/README.md`

### Step 8: Reset to Starter Code

Always reset after generating fixtures:

```bash
python3 scripts/dev_utils.py --reset <assignment-name>
```

### Step 9: Display Summary

```
╔══════════════════════════════════════════════════════════════╗
║          FIXTURE GENERATION COMPLETE: <ASSIGNMENT>           ║
╚══════════════════════════════════════════════════════════════╝

Fixtures Generated:
  ✅ tests/internal/fixtures/<assignment-name>/gold_output.pt
  ✅ tests/internal/fixtures/<assignment-name>/gold_checkpoint.pt
  ✅ tests/internal/fixtures/<assignment-name>/README.md

Verification:
  ✅ No NaN values detected
  ✅ Shapes verified
  ✅ Seed documented (42)

Reset to Starter: ✅

════════════════════════════════════════════════════════════════

Next Steps:
1. Review fixtures: cat tests/internal/fixtures/<assignment-name>/README.md
2. Update tests to use new fixtures
3. Run tests: /test-rigor
```

## Notes

- Always use seed=42 for reproducibility
- Fixtures should be committed to git (they're reference data)
- Update fixtures when solution implementation changes
- Include comprehensive metadata in saved fixtures
- Verify no NaNs or inf values before saving
