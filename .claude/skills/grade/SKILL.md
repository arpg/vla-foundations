---
name: grade
description: "Automated student submission review and grading workflow"
user-invocable: true
---

# Grade: Student PR Review & Grading Engine

This skill automates the complete grading workflow for student pull requests.

## Execution Steps

### Step 1: Get PR Information

Prompt user for PR number:

```
Use AskUserQuestion to ask:
- Question: "Which PR would you like to grade?"
- Header: "PR Number"
- Options:
  - label: "Auto-detect", description: "Find latest ungraded PR"
  - label: "Specify", description: "Enter PR number manually"
```

If "Auto-detect" selected:
```bash
# Get latest open PR to staging
gh pr list --base staging --state open --limit 1 --json number,title,author
```

If "Specify" selected, ask for the PR number.

### Step 2: Fetch PR Details

Get full PR information:

```bash
gh pr view <pr-number> --json number,title,author,headRefName,baseRefName,body,files,reviews
```

Parse the JSON to extract:
- Student branch name
- Assignment being submitted (from branch name pattern: `<assignment>-<username>`)
- Files changed
- Current review status

Display PR summary:
```
╔══════════════════════════════════════════════════════════════╗
║                    GRADING PR #<number>                      ║
╚══════════════════════════════════════════════════════════════╝

Student: <author>
Branch: <headRefName>
Assignment: <detected-assignment>
Base: <baseRefName>

Files Changed:
  - <list files>

Previous Reviews: <count>
```

### Step 3: Verify Assignment Detection

Confirm the detected assignment with user:

```
Use AskUserQuestion to ask:
- Question: "Detected assignment: <assignment>. Is this correct?"
- Header: "Confirmation"
- Options:
  - label: "Yes (Recommended)", description: "Grade <assignment>"
  - label: "Scratch-1", description: "Override to Scratch-1"
  - label: "Scratch-2", description: "Override to Scratch-2"
  - label: "Scratch-3", description: "Override to Scratch-3"
```

### Step 4: Fetch Student Code

Clone the student's branch to a temporary location:

```bash
# Create temp directory
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

# Fetch student branch
gh repo clone arpg/vla-foundations student-submission
cd student-submission
git checkout <student-branch>

# Copy student files to private repo for testing
cp -r src/assignments/<assignment>/* /Users/crh/projects/private-vla-foundations/src/assignments/<assignment>/
```

### Step 5: Run VLA Guard on Student Code

Check student code for accidental solution leaks:

```bash
cd /Users/crh/projects/private-vla-foundations

# Run verification (should pass - student shouldn't have solutions)
python3 scripts/dev_utils.py --verify-clean
```

If this detects high similarity (>80%), flag it:
```
⚠️  WARNING: Student code shows high similarity to solution!
This may indicate:
  - Academic integrity issue
  - Accidental solution leak in assignment
  - Coincidental implementation
```

### Step 6: Run Public Tests

First, run the public tests that students can see:

```bash
pytest tests/public/test_<assignment>_basic.py -v --tb=short
```

Record results:
- Number of tests passed/failed
- This shows what the student should have already verified

### Step 7: Inject Solutions for Comparison

Inject reference solution for internal tests:

```bash
# Backup student code
cp -r src/assignments/<assignment> /tmp/student-<assignment>-backup

# Inject solutions to create reference model
python3 scripts/dev_utils.py --inject <assignment>
```

### Step 8: Run Internal Grading Tests

Run rigorous internal tests:

```bash
pytest tests/internal/test_<assignment>_rigor.py -v --tb=short --json-report --json-report-file=tests/internal/reports/grade-pr<pr-number>.json
```

Capture detailed results:
- Gradient leak tests (frozen parameters)
- Fidelity tests (output comparison)
- Training convergence tests
- Edge case handling
- Performance benchmarks

### Step 9: Restore Student Code

```bash
# Remove injected solutions
python3 scripts/dev_utils.py --reset <assignment>

# Restore student code
cp -r /tmp/student-<assignment>-backup/* src/assignments/<assignment>/
```

### Step 10: Analyze Results

Parse test results and categorize:

**Critical Failures** (must fix):
- Gradient leaks (unfrozen parameters)
- Runtime errors (crashes)
- Missing required components

**Quality Issues** (should fix):
- Low fidelity scores (output differs from gold standard)
- Training doesn't converge
- Poor performance

**Minor Issues** (nice to have):
- Code style
- Missing docstrings
- Inefficient implementation

### Step 11: Generate Feedback Report

Create detailed feedback:

```markdown
# Grading Report: <Assignment> - PR #<number>

**Student**: <author>
**Submitted**: <date>
**Graded**: $(date)

---

## Summary

**Overall Status**: ✅ PASS / ⚠️ CONDITIONAL PASS / ❌ FAIL

### Test Results

**Public Tests** (Student-visible):
- Total: X
- Passed: X ✅
- Failed: X ❌

**Internal Grading Tests** (Rigorous):
- Gradient Tests: ✅ / ❌
- Fidelity Tests: ✅ / ❌ (Score: XX.X%)
- Training Tests: ✅ / ❌
- Edge Cases: ✅ / ❌

---

## Detailed Feedback

### ✅ What Worked Well
- [List strengths]
- [Correct implementations]

### ❌ Critical Issues (Must Fix)
- [List blocking issues]
- [Required changes]

### ⚠️ Quality Improvements (Recommended)
- [List suggestions]
- [Best practices]

### 💡 Optional Enhancements
- [Performance tips]
- [Code style suggestions]

---

## Test Details

<details>
<summary>Gradient Leak Test</summary>

```
[Show test output]
```

**Result**: ✅ PASS / ❌ FAIL
**Details**: [Explanation]
</details>

<details>
<summary>Fidelity Test</summary>

```
[Show comparison metrics]
```

**Result**: ✅ PASS / ❌ FAIL
**Similarity Score**: XX.X%
</details>

[Repeat for each test category]

---

## Next Steps

**If PASS**:
- [ ] Merge PR to staging
- [ ] Award points

**If CONDITIONAL PASS**:
- [ ] Address quality issues
- [ ] Re-submit for full credit

**If FAIL**:
- [ ] Fix critical issues listed above
- [ ] Re-run tests locally: `pytest tests/public/`
- [ ] Re-submit PR

---

**Graded by**: VLA Guard Grading Engine v1.0
**Report saved**: `tests/internal/reports/grade-pr<number>.md`
```

Save report to: `tests/internal/reports/grade-pr<pr-number>.md`

### Step 12: Post Feedback

Ask user if they want to post feedback:

```
Use AskUserQuestion to ask:
- Question: "How would you like to provide feedback?"
- Header: "Feedback Method"
- Options:
  - label: "Post as PR comment (Recommended)", description: "Comment on GitHub PR"
  - label: "Save to file only", description: "Just save report locally"
  - label: "Both", description: "Post comment AND save report"
```

If posting comment:
```bash
gh pr comment <pr-number> --body-file tests/internal/reports/grade-pr<pr-number>.md
```

### Step 13: Update PR Labels

Based on results, suggest label updates:

```bash
# If PASS
gh pr edit <pr-number> --add-label "ready-to-merge"

# If CONDITIONAL PASS
gh pr edit <pr-number> --add-label "needs-revision"

# If FAIL
gh pr edit <pr-number> --add-label "changes-requested"
```

### Step 14: Cleanup

```bash
# Remove temp directory
rm -rf "$TEMP_DIR"

# Ensure we're back on main branch
git checkout main

# Display summary
```

### Step 15: Display Grading Summary

```
╔══════════════════════════════════════════════════════════════╗
║               GRADING COMPLETE: PR #<number>                 ║
╚══════════════════════════════════════════════════════════════╝

Student: <author>
Assignment: <assignment>
Overall: ✅ PASS / ⚠️ CONDITIONAL / ❌ FAIL

Test Summary:
  Public Tests:   X/Y passed
  Gradient:       ✅ / ❌
  Fidelity:       ✅ / ❌ (XX.X%)
  Training:       ✅ / ❌
  Edge Cases:     ✅ / ❌

Feedback:
  Posted: ✅ GitHub PR comment
  Saved:  tests/internal/reports/grade-pr<number>.md

Labels Updated:
  Added: <label>

════════════════════════════════════════════════════════════════

Next Steps:
  - Review feedback: cat tests/internal/reports/grade-pr<number>.md
  - View PR: gh pr view <number> --web
  - Merge (if pass): gh pr merge <number> --squash
```

## Error Handling

- If PR not found: Report error and list available PRs
- If assignment detection fails: Prompt for manual selection
- If tests crash: Capture error, include in report, mark as FAIL
- If gh CLI not authenticated: Provide setup instructions

## Notes

- Always restore student code after injecting solutions
- Save all grading reports for record-keeping
- Internal test results are NEVER shared publicly
- Use Shadow CI results if available to cross-check
- This skill requires gh CLI: `brew install gh && gh auth login`
