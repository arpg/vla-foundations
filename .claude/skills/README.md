# VLA Guard Skills

This directory contains Claude Code skills for managing the VLA Foundations course repository.

## Available Skills

### 1. `/vla-guard` - Solution Leak Prevention Audit

**Purpose**: Comprehensive audit to prevent solution leaks before push/PR

**When to use**: Before any public sync or release

**What it does**:
- Scans for `[SOLUTION]` markers in code
- Verifies private directories not staged
- Checks git history for accidental commits
- Validates no sensitive files staged
- Runs similarity comparison with `dev_utils.py --verify-clean`

**Exit behavior**:
- ✅ PASS: Safe to proceed with sync
- ❌ FAIL: Blocks sync, provides remediation steps

---

### 2. `/pre-flight` - Combined Audit + Sanitization

**Purpose**: Complete pre-release check and sanitization pipeline

**When to use**: Before creating release tags

**What it does**:
1. Runs VLA Guard audit (fail-fast)
2. If audit passes, runs `scripts/sanitize.sh`
3. Verifies sanitization completed
4. Shows final verification status

**Fail-safe**: Only runs sanitization if audit passes

---

### 3. `/test-rigor` - Internal Grading Test Runner

**Purpose**: Run internal grading tests with automatic solution injection

**When to use**:
- Developing new assignments
- Updating solution code
- Verifying internal tests work

**What it does**:
1. Prompts for assignment selection
2. Injects solutions automatically
3. Runs `pytest tests/internal/ -v -m rigor`
4. Saves test report to `tests/internal/reports/`
5. Resets to starter code

**Safe to run**: Always resets after completion

---

### 4. `/generate-fixtures` - Gold Standard Fixture Generation

**Purpose**: Create reference data for fidelity tests

**When to use**:
- After completing solution implementation
- After updating solution code
- When creating new assignments

**What it does**:
1. Prompts for assignment
2. Injects solution code
3. Runs solution with fixed random seeds (seed=42)
4. Generates output fixtures and/or trained checkpoints
5. Verifies no NaNs in outputs
6. Creates fixture documentation
7. Resets to starter code

**Output**: `tests/internal/fixtures/<assignment>/gold_output.pt`

---

### 5. `/grade` - Automated Student PR Grading

**Purpose**: Complete grading workflow for student submissions

**When to use**: When reviewing student pull requests

**What it does**:
1. Prompts for PR number (or auto-detects latest)
2. Fetches student code from GitHub
3. Runs VLA Guard on student code (detect leaks)
4. Runs public tests
5. Injects reference solution
6. Runs internal grading tests (gradient, fidelity, training)
7. Restores student code
8. Generates detailed feedback report
9. Posts comment on PR (optional)
10. Updates PR labels

**Output**:
- Detailed markdown report
- GitHub PR comment (optional)
- Test results with pass/fail status

**Safe to run**: Automatically restores student code after grading

---

### 6. `/release` - Safe Assignment Publishing

**Purpose**: Orchestrate complete release workflow with safety checks

**When to use**: Ready to publish assignment to students

**What it does**:
1. Verifies on main branch
2. Checks for uncommitted changes
3. Runs VLA Guard pre-flight audit (fail-fast)
4. Prompts for release tag (e.g., `release-scratch-2`)
5. Shows changes since last release
6. Runs sanitization pipeline
7. Verifies sanitization (fail-safe)
8. Creates annotated git tag
9. Pushes tag to trigger GitHub Actions
10. Monitors workflow execution
11. Verifies public repository
12. Checks deployment status
13. Generates release summary

**Fail-safe**: Aborts at any failed check, provides remediation

**Output**:
- Release tag pushed to GitHub
- Public repo updated via orphan push
- Deployment to production
- Release summary: `.claude/releases/release-<tag>.md`

---

### 7. `/new-assignment` - Assignment Scaffolding

**Purpose**: Create complete assignment structure with templates

**When to use**: Starting a new assignment

**What it does**:
1. Prompts for assignment name and metadata
2. Creates directory structure:
   - `src/assignments/<name>/` (starter code with TODOs)
   - `private/solutions/<name>/` (solution templates)
   - `tests/public/` (student-visible tests)
   - `tests/internal/` (grading tests)
   - `content/course/assignments/` (MDX spec)
3. Generates Python templates with TODOs
4. Generates solution templates with [SOLUTION] markers
5. Generates test templates
6. Generates MDX assignment spec
7. Creates README files

**Output**: Complete assignment structure ready for development

**Next steps after scaffolding**:
1. Complete solution implementations
2. Run `/generate-fixtures`
3. Update assignment spec (MDX)
4. Test with `/test-rigor`
5. Release with `/release`

---

### 8. `/sync-check` - Post-Release Verification

**Purpose**: Verify public repository has no leaks after sync

**When to use**: After `/release` completes

**What it does**:
1. Prompts for release tag to verify
2. Clones public repo (read-only, temp location)
3. Scans for `[SOLUTION]` markers
4. Checks for private directories (`private/`, `tests/internal/`)
5. Checks for private scripts (`dev_utils.py`, `sanitize.sh`)
6. Checks for sensitive files (credentials, backups)
7. Verifies orphan push (no linked git history)
8. Compares file lists (private vs public)
9. Checks deployment status (HTTPS 200)
10. Runs sample fidelity check
11. Generates verification report
12. Cleans up temp files

**Output**: Comprehensive verification report

**Leak detection**: If any leaks found, provides urgent remediation steps

---

## Typical Workflows

### Creating a New Assignment

```bash
# 1. Scaffold assignment structure
/new-assignment
# Select: "Scratch-3", type, focus, difficulty

# 2. Complete solution implementations
# Edit: private/solutions/scratch-3/model_solution.py
# Edit: private/solutions/scratch-3/train_solution.py

# 3. Generate gold standard fixtures
/generate-fixtures
# Select: "Scratch-3"

# 4. Update assignment spec
# Edit: content/course/assignments/scratch-3.mdx
# Add: tasks, rubric, due date

# 5. Test internal grading
/test-rigor
# Select: "Scratch-3"
# Verify all tests pass

# 6. Commit changes
git add .
git commit -m "feat: add scratch-3 assignment"
git push

# 7. Release to students
/release
# Select: "Scratch-3"
# Tag: release-scratch-3

# 8. Verify public repo
/sync-check
# Select: "release-scratch-3"
```

### Grading Student Submissions

```bash
# 1. List open PRs
gh pr list --base staging --state open

# 2. Grade a PR
/grade
# Enter PR number or auto-detect

# 3. Review feedback report
cat tests/internal/reports/grade-pr123.md

# 4. If approved, merge
gh pr merge 123 --squash
```

### Before Every Release

```bash
# 1. Run pre-flight check
/pre-flight

# 2. If passes, create release
/release

# 3. Verify public repo
/sync-check
```

### Testing Changes to Solutions

```bash
# 1. Edit solution
# Edit: private/solutions/scratch-1/model_solution.py

# 2. Re-generate fixtures
/generate-fixtures
# Select: "Scratch-1"

# 3. Test internal grading
/test-rigor
# Select: "Scratch-1"

# 4. Commit if all passes
git add .
git commit -m "fix: update scratch-1 solution"
```

## Error Handling

All skills include comprehensive error handling:

- **VLA Guard fails**: Aborts operation, shows remediation steps
- **Sanitization fails**: Restores original state with `git reset --hard`
- **Tests fail**: Continues to cleanup, shows failure details
- **GitHub API fails**: Provides alternative manual steps
- **Deployment check fails**: May be expected during build time

## Safety Features

1. **Fail-Safe Defaults**: Skills abort on any failed check
2. **Automatic Cleanup**: Always reset/restore after operations
3. **Read-Only Operations**: Where possible (sync-check)
4. **Confirmation Prompts**: Before destructive operations
5. **Detailed Reports**: All operations generate audit trails

## Directory Structure

```
.claude/
├── skills/                  # Skill definitions
│   ├── vla-guard/
│   │   └── SKILL.md
│   ├── test-rigor/
│   │   └── SKILL.md
│   ├── generate-fixtures/
│   │   └── SKILL.md
│   ├── grade/
│   │   └── SKILL.md
│   ├── release/
│   │   └── SKILL.md
│   ├── new-assignment/
│   │   └── SKILL.md
│   ├── sync-check/
│   │   └── SKILL.md
│   └── README.md           # This file
├── commands/               # Command shortcuts
│   ├── vla-guard.md
│   ├── pre-flight.md
│   ├── test-rigor.md
│   ├── generate-fixtures.md
│   ├── grade.md
│   ├── release.md
│   ├── new-assignment.md
│   └── sync-check.md
├── releases/               # Release summaries
│   └── .gitkeep
└── sync-reports/           # Sync verification reports
    └── .gitkeep

tests/internal/reports/     # Test and grading reports
└── .gitkeep
```

## Requirements

These skills require:

- `gh` CLI: `brew install gh && gh auth login`
- `pytest`: `pip install pytest pytest-html`
- Python 3.11+
- Git configured with user.name and user.email

## Notes

- All skills are safe to run multiple times
- Reports are git-ignored but directories are tracked
- Skills automatically handle solution injection/reset
- Use `/vla-guard` before any public-facing operation
- Always run `/sync-check` after `/release`

## Support

For issues or questions about skills:
1. Check skill SKILL.md for detailed documentation
2. Review error messages and remediation steps
3. Check `.claude/releases/` and `.claude/sync-reports/` for audit trails
