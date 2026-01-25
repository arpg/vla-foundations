---
name: test-rigor
description: "Run internal grading tests with automatic solution injection/reset"
user-invocable: true
---

# Test Rigor: Internal Grading Test Runner

This skill automates the workflow for running rigorous internal tests against solutions.

## Execution Steps

### Step 1: Prompt for Assignment

Ask the user which assignment to test:

```
Use AskUserQuestion to ask:
- Question: "Which assignment would you like to test?"
- Header: "Assignment"
- Options:
  - label: "Scratch-1", description: "Test Scratch-1 internal grading suite"
  - label: "Scratch-2", description: "Test Scratch-2 internal grading suite"
  - label: "Scratch-3", description: "Test Scratch-3 internal grading suite"
  - label: "All", description: "Run all internal tests"
```

Parse the user's selection to get the assignment name (e.g., "scratch-1").

### Step 2: Inject Solutions

Before running tests, inject the solution files:

```bash
python3 scripts/dev_utils.py --inject <assignment-name>
```

Verify the injection succeeded by checking the output.

### Step 3: Run Internal Tests

Run pytest with internal markers:

```bash
pytest tests/internal/ -v -m rigor --tb=short
```

If user selected "All", run without filtering:

```bash
pytest tests/internal/ -v -m rigor --tb=short
```

Capture the full output including:
- Test results (passed/failed/skipped)
- Failure details with tracebacks
- Performance metrics if available

### Step 4: Generate Test Report

Save the test output to a report file:

```bash
pytest tests/internal/ -v -m rigor --tb=short --html=tests/internal/reports/test-report-$(date +%Y%m%d-%H%M%S).html --self-contained-html
```

Note: This requires pytest-html. If not installed, just save text output:

```bash
pytest tests/internal/ -v -m rigor --tb=short | tee tests/internal/reports/test-report-$(date +%Y%m%d-%H%M%S).txt
```

### Step 5: Reset to Starter Code

Always reset after testing, even if tests fail:

```bash
python3 scripts/dev_utils.py --reset <assignment-name>
```

Verify the reset succeeded.

### Step 6: Display Summary

Provide a clear summary:

```
╔══════════════════════════════════════════════════════════════╗
║                 INTERNAL TEST RESULTS: <ASSIGNMENT>          ║
╚══════════════════════════════════════════════════════════════╝

Solution Injection: ✅
Test Execution:     ✅ (or ❌)
Reset to Starter:   ✅

Test Summary:
  Total:   X tests
  Passed:  X
  Failed:  X
  Skipped: X

════════════════════════════════════════════════════════════════

Detailed Results:
[Show pass/fail breakdown by test category]

Report saved: tests/internal/reports/test-report-TIMESTAMP.txt
```

If tests failed, provide actionable insights:
- Which test categories failed (gradient, fidelity, training)
- Common failure patterns
- Recommendations for fixes

## Error Handling

- If injection fails: Stop and report error
- If tests fail: Continue to reset step, then report failures
- If reset fails: Warn user that manual cleanup needed

## Notes

- Always reset after tests to prevent accidental commits
- The `-m rigor` marker filters for internal grading tests only
- Test reports are saved to `tests/internal/reports/` (git-ignored)
- This skill is safe to run multiple times
