---
name: generate-fixtures
description: "Generate gold standard test fixtures"
---

# Generate Fixtures Command

Create reference data for fidelity testing.

## Usage

```
/generate-fixtures
```

## Execution

Use the Skill tool to invoke the generate-fixtures skill:

```
Skill: generate-fixtures
```

This will:
1. Prompt for assignment selection
2. Check for fixture generation configuration
3. Inject solution code
4. Set reproducible random seeds (seed=42)
5. Generate model output fixtures
6. Generate trained checkpoint fixtures (if applicable)
7. Verify fixtures (no NaNs, correct shapes)
8. Generate fixture documentation
9. Reset to starter code
10. Display summary

Fixtures are saved to: `tests/internal/fixtures/<assignment-name>/`

Use these fixtures in internal tests for fidelity comparison.
