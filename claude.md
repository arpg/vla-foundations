# VLA Foundations Development Guide for AI SWE Agents (Private Repo)

This is the **private instructor repository** for VLA Foundations, containing complete assignment solutions, internal grading tests, and instructor operations. The public student-facing repository is at `arpg/vla-foundations`. This repo uses **Next.js (App Router)** for the textbook, **Tailwind CSS** for styling, **MDX** for content, and **pnpm** for package management.

Read more about the dual-repository architecture in [INSTRUCTOR.md](INSTRUCTOR.md).

---

## Repository Architecture

This is a **two-repository system**:

```
Private Repo (crheckman/private-vla-foundations)
├── private/                    # Complete assignment solutions (NEVER PUBLIC)
│   └── solutions/
├── tests/internal/             # Internal grading tests (NEVER PUBLIC)
│   ├── fixtures/              # Gold standard test data
│   └── reports/               # Grading reports (git-ignored)
├── scripts/
│   ├── dev_utils.py           # Solution management (inject/reset/verify-clean)
│   ├── sanitize.sh            # Automated sanitization pipeline
│   └── _sanitize_todos.py     # TODO comment sanitizer
├── .claude/
│   ├── skills/                # Claude Code skills for automation
│   └── commands/              # Slash command shortcuts
└── src/assignments/           # Starter code with [SOLUTION] hints

                    ↓ (Orphan push on release tag)

Public Repo (arpg/vla-foundations)
├── src/assignments/           # Starter code (TODOs only)
├── tests/public/              # Student-visible tests
├── content/                   # Textbook and assignment specs
└── [NO private/ or tests/internal/]
```

**Critical**: Never commit `private/` or `tests/internal/` to public branches.

---

## Initial Setup

### Prerequisites
```bash
# Install dependencies
pnpm install

# Install uv (Python package manager) - REQUIRED
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python dependencies via uv
uv sync

# Install GitHub CLI (required for skills)
brew install gh
gh auth login
```

### Python Environment (uv)
**All Python commands MUST use `uv run`** to ensure correct dependencies:
```bash
# Run Python scripts
uv run python scripts/dev_utils.py --list

# Run pytest
uv run pytest tests/internal/ -v -m rigor

# Run any Python file
uv run python src/assignments/scratch-1/generate_data.py
```

### Development
```bash
# Run development server
pnpm dev

# Build production (static export in out/)
pnpm build

# Lint Next.js
pnpm lint
```

---

## Claude Code Skills (Automation)

This repository has **7 Claude Code skills** for workflow automation. See [.claude/skills/README.md](.claude/skills/README.md) for complete documentation.

### Core Skills

#### `/vla-guard` - Solution Leak Audit
**Purpose**: Prevent solution leaks before any public operation

**Usage**:
```bash
/vla-guard
```

**What it does**:
- Scans for `[SOLUTION]` markers in `src/` and `content/`
- Verifies `private/` and `tests/internal/` not staged
- Checks git history for accidental commits
- Runs `dev_utils.py --verify-clean` (similarity detection)
- **Blocks** sync if any check fails

**When to use**: Before every push, PR, or release

---

#### `/test-rigor` - Internal Grading Tests
**Purpose**: Run internal grading tests with automatic solution injection/reset

**Usage**:
```bash
/test-rigor
# Select: "Scratch-1" / "Scratch-2" / "All"
```

**What it does**:
1. Injects solutions: `python3 scripts/dev_utils.py --inject <assignment>`
2. Runs pytest: `pytest tests/internal/ -v -m rigor`
3. Generates report: `tests/internal/reports/test-report-<timestamp>.txt`
4. Resets to starter code: `python3 scripts/dev_utils.py --reset <assignment>`

**Safe to run multiple times** - always resets after completion.

---

#### `/generate-fixtures` - Gold Standard Fixtures
**Purpose**: Generate reference data for fidelity tests from solution code

**Usage**:
```bash
/generate-fixtures
# Select assignment
```

**What it does**:
1. Injects solutions
2. Sets fixed random seeds (seed=42)
3. Runs solution code to generate outputs
4. Saves to `tests/internal/fixtures/<assignment>/gold_output.pt`
5. Verifies no NaNs
6. Generates fixture documentation
7. Resets to starter code

**When to use**: After completing solution implementation or updating solution code

---

#### `/grade` - Automated PR Grading
**Purpose**: Complete grading workflow for student pull requests

**Usage**:
```bash
/grade
# Enter PR number or auto-detect latest
```

**What it does**:
1. Fetches student code from GitHub PR
2. Runs VLA Guard on student code (detect plagiarism/leaks)
3. Runs `tests/public/` (student-visible tests)
4. Injects reference solution
5. Runs `tests/internal/` (gradient leak, fidelity, training tests)
6. Restores student code
7. Generates detailed markdown feedback report
8. Posts comment on PR (optional)
9. Updates PR labels (ready-to-merge / needs-revision / changes-requested)

**Output**: `tests/internal/reports/grade-pr<number>.md`

**When to use**: When reviewing student submissions

---

#### `/release` - Safe Assignment Publishing
**Purpose**: Orchestrate complete release workflow with comprehensive safety checks

**Usage**:
```bash
/release
# Select: "Scratch-1" / "Scratch-2" / etc.
```

**What it does**:
1. Verifies on main branch, no uncommitted changes
2. Runs `/vla-guard` pre-flight audit (fail-fast)
3. Prompts for release tag (e.g., `release-scratch-2`)
4. Shows changes since last release
5. Runs `scripts/sanitize.sh` (removes private/, [SOLUTION] markers, etc.)
6. Verifies sanitization (fail-safe)
7. Creates annotated git tag
8. Pushes tag → triggers `.github/workflows/sync-to-public.yml`
9. Monitors GitHub Actions workflow execution
10. Verifies public repository (no leaks)
11. Checks deployment status (https://www.vlm-robotics.dev)
12. Generates release summary: `.claude/releases/release-<tag>.md`

**Fail-safe**: Aborts at ANY failed check, provides remediation instructions

**When to use**: When ready to publish assignment to students

---

#### `/new-assignment` - Assignment Scaffolding
**Purpose**: Create complete assignment structure with templates

**Usage**:
```bash
/new-assignment
# Enter name, type, focus, difficulty
```

**What it does**:
1. Creates directory structure:
   - `src/assignments/<name>/` (starter code with TODOs)
   - `private/solutions/<name>/` (solution templates)
   - `tests/public/test_<name>_basic.py` (student-visible tests)
   - `tests/internal/test_<name>_rigor.py` (grading tests)
   - `content/course/assignments/<name>.mdx` (assignment spec)
2. Generates Python templates
3. Generates test templates
4. Creates README files

**Next steps after scaffolding**:
1. Complete solution implementations
2. Run `/generate-fixtures`
3. Update MDX spec
4. Run `/test-rigor`
5. Commit changes
6. Run `/release`

---

#### `/sync-check` - Post-Release Verification
**Purpose**: Verify public repository has no leaks after release sync

**Usage**:
```bash
/sync-check
# Select: "Latest" or specify release tag
```

**What it does**:
1. Clones public repo to temp directory (read-only)
2. Scans for `[SOLUTION]` markers
3. Checks for private directories (`private/`, `tests/internal/`)
4. Checks for private scripts (`dev_utils.py`, `sanitize.sh`)
5. Checks for sensitive files (credentials, `*_solution.py`)
6. Verifies orphan push strategy (no linked git history)
7. Compares file lists (private vs public)
8. Checks deployment status (HTTPS 200)
9. Runs sample fidelity check
10. Generates verification report: `.claude/sync-reports/sync-check-<timestamp>.md`
11. Cleans up temp files

**When to use**: Always run after `/release` completes

**Critical**: If leaks detected, report provides urgent remediation steps

---

## Commands Useful in Development

### Solution Management
```bash
# List all available solutions
uv run python scripts/dev_utils.py --list

# Inject solutions for testing/grading
uv run python scripts/dev_utils.py --inject scratch-1

# Reset to starter code
uv run python scripts/dev_utils.py --reset scratch-1

# Verify no solution leaks (similarity check)
uv run python scripts/dev_utils.py --verify-clean
```

### Testing
```bash
# Run public tests (students can see these)
uv run pytest tests/public/ -v

# Run internal grading tests (after injecting solutions)
uv run pytest tests/internal/ -v -m rigor

# Run specific test file
uv run pytest tests/internal/test_scratch1_rigor.py -v

# Generate HTML report
uv run pytest tests/internal/ --html=tests/internal/reports/report.html --self-contained-html
```

### Pre-Release Checks
```bash
# Complete pre-flight check
/pre-flight

# Or manually:
uv run python scripts/dev_utils.py --verify-clean
bash scripts/sanitize.sh  # (Only in orphan branch workflow)
```

### GitHub Operations
```bash
# List open student PRs
gh pr list --base staging --state open

# View PR details
gh pr view 123

# Comment on PR
gh pr comment 123 --body "Feedback here"

# Merge PR
gh pr merge 123 --squash
```

---

## Linting and Formatting

### Semantic Line Breaks
**All MDX files MUST use one sentence per line.** This is mandatory to allow granular, line-by-line feedback in Pull Requests.

**Bad:**
```markdown
This is a very long sentence with multiple ideas. It continues on the same line. This makes PR review difficult.
```

**Good:**
```markdown
This is a sentence on its own line.
Each idea gets its own line.
This makes PR review much easier.
```

### LaTeX
Use formal LaTeX for all mathematical derivations:
```markdown
The loss function is:
$$
\mathcal{L} = -\sum_{t=1}^T \log p(a_t | s_t, I_t)
$$
```

Do not use code blocks for math.

### Next.js Linting
```bash
pnpm lint
```

---

## Testing Philosophy

### Public Tests (`tests/public/`)
**Purpose**: Student-visible validation tests

**What they test**:
- Basic model structure (initialization, shapes)
- Forward pass correctness (no NaNs, correct dimensions)
- Gradient flow (backpropagation works)

**Students can run these**: `pytest tests/public/test_scratch1_basic.py -v`

### Internal Tests (`tests/internal/`)
**Purpose**: Rigorous grading tests (NEVER synced to public)

**What they test**:
- **Gradient Leak Test**: Verify frozen parameters (e.g., DINOv2 backbone)
- **Latent Fidelity Test**: Compare output against gold standard fixtures
- **Training Convergence Test**: Verify model can train and loss decreases
- **Edge Case Tests**: Boundary conditions, error handling

**Markers**:
- `@pytest.mark.internal` - All internal tests
- `@pytest.mark.rigor` - Strict grading tests
- `@pytest.mark.gradient` - Gradient flow tests
- `@pytest.mark.fidelity` - Output comparison tests
- `@pytest.mark.training` - Training convergence tests

**Run with**: `pytest tests/internal/ -v -m rigor`

---

## Interacting with the App

### Local Development
```bash
pnpm dev
# Access at http://localhost:3000
```

### Staging Previews
Every Pull Request to `staging` branch triggers deployment to:
```
https://vlm-robotics.dev/staging/pulls/[PR_NUMBER]/
```

**Review Protocol**:
1. Read the rendered audit on the staging site
2. Comment on the **source MDX** in GitHub "Files Changed" tab
3. Use the **Rich Diff** view in GitHub to verify LaTeX rendering

### Production
Production site deployed at:
```
https://www.vlm-robotics.dev
```

Deployment triggered by:
- Push to `main` branch (after staging → main merge)
- GitHub Action: `.github/workflows/deploy.yml`
- Deploys to ristoffer.ch via SSH

---

## Patterns & Standards

### Amazon Principle
We do not write "summaries." We write rigorous, durable **Audits**. A high-fidelity audit IS the textbook chapter.

### Textbook Audit Sidebars
Every audit MUST contain these three technical sidebars:

1. **The Lineage of Failure**: Why previous approaches died
2. **Intuitive Derivation**: The geometric/mathematical intuition of the loss function
3. **Implementation Gotchas**: Practitioners' notes on coordinate frames, normalization, or hyperparameters

### The Interface Focus
When auditing VLA models, focus on the **Interface**:
- **Input Projection**: Pixels → Tokens
- **Action Head**: Tokens → Trajectories
- **The Loss/Objective Function**

### Git Hygiene
We are a **rebase-only** lab. Use `git rebase main`. PRs containing "Merge branch 'main'" commits will be closed.

**Correct workflow**:
```bash
git fetch origin
git rebase origin/main
git push --force-with-lease
```

### Sanitization
All private solutions are marked with `[SOLUTION]` tags:
```python
# TODO: Implement RMSNorm forward pass
# [SOLUTION] Use torch.rsqrt for efficiency
result = torch.rsqrt(variance + self.eps)
```

The sanitization pipeline:
1. `scripts/_sanitize_todos.py` - Removes `[SOLUTION]` markers
2. `scripts/sanitize.sh` - Orchestrates full cleanup (private dirs, scripts, README)
3. Triggered automatically by `.github/workflows/sync-to-public.yml` on release tags

**Load-bearing wall**: `scripts/sanitize.sh` is the primary defense against solution leaks.

### Orphan Push Strategy
When syncing to public repo, we use **orphan branches** to break all git history links:

```bash
git checkout --orphan temp-public-branch
git add -A
git commit -m "Public Release: $(date)"
git push public temp-public-branch:main --force
```

**Benefits**:
- No commit history from private repo exposed
- Public repo has completely independent git history
- Maximum security against accidental leaks via `git log`

---

## File Map of Interest

### GitHub Actions
- [.github/workflows/sync-to-public.yml](.github/workflows/sync-to-public.yml) - Automated sync to public repo (orphan push)
- [.github/workflows/shadow-tester.yml](.github/workflows/shadow-tester.yml) - Shadow CI for student PRs
- [.github/workflows/deploy.yml](.github/workflows/deploy.yml) - Production deployment to ristoffer.ch

### Configuration
- [next.config.ts](next.config.ts) - Next.js config with dynamic routing for staging
- [pytest.ini](pytest.ini) - pytest markers configuration
- [tailwind.config.ts](tailwind.config.ts) - Tailwind CSS configuration

### Scripts
- [scripts/dev_utils.py](scripts/dev_utils.py) - Solution management (inject/reset/verify-clean)
- [scripts/sanitize.sh](scripts/sanitize.sh) - Complete sanitization pipeline
- [scripts/_sanitize_todos.py](scripts/_sanitize_todos.py) - TODO comment sanitizer

### Claude Code Skills
- [.claude/skills/](/.claude/skills/) - All skill definitions
- [.claude/skills/README.md](.claude/skills/README.md) - Comprehensive skills documentation
- [.claude/commands/](.claude/commands/) - Command shortcuts

### Components
- [components/audit/AuditLayout.tsx](components/audit/AuditLayout.tsx) - Primary wrapper for rendered textbook chapters

### Testing
- [tests/conftest.py](tests/conftest.py) - pytest fixtures (auto-inject for internal tests)
- [tests/public/](tests/public/) - Student-visible tests
- [tests/internal/](tests/internal/) - Internal grading tests

### Documentation
- [INSTRUCTOR.md](INSTRUCTOR.md) - Complete instructor guide (consolidated)
- [SKILLS_COMPLETE.md](SKILLS_COMPLETE.md) - Skills implementation summary
- [REFACTOR_COMPLETE.md](REFACTOR_COMPLETE.md) - Repository hardening summary

---

## Typical Workflows

### Creating a New Assignment
```bash
# 1. Scaffold structure
/new-assignment

# 2. Implement solutions
# Edit: private/solutions/scratch-3/model_solution.py

# 3. Generate fixtures
/generate-fixtures

# 4. Update spec
# Edit: content/course/assignments/scratch-3.mdx

# 5. Test grading
/test-rigor

# 6. Commit
git add . && git commit -m "feat: add scratch-3 assignment"

# 7. Release
/release

# 8. Verify
/sync-check
```

### Grading Student Work
```bash
# 1. List PRs
gh pr list --base staging

# 2. Grade PR
/grade

# 3. Review report
cat tests/internal/reports/grade-pr123.md

# 4. Merge if approved
gh pr merge 123 --squash
```

### Pre-Release Checklist
```bash
# 1. Audit
/vla-guard

# 2. Pre-flight (audit + sanitize)
/pre-flight

# 3. Release
/release

# 4. Verify
/sync-check
```

---

## Shadow CI

Student PRs to the public repo trigger **Shadow CI** - hidden testing with internal grading suite:

1. Student opens PR to `arpg/vla-foundations` (public)
2. Public `.github/workflows/vla-audit.yml` triggers `repository_dispatch` to private repo
3. Private `.github/workflows/shadow-tester.yml` runs:
   - Fetches student code
   - Injects solutions
   - Runs internal tests
   - Posts Pass/Fail comment on public PR (no details)
4. Instructor uses `/grade` for detailed feedback

**Purpose**: Catch critical failures early without exposing grading logic.

---

## Security Boundaries

### NEVER Sync to Public
- `private/` directory (complete solutions)
- `tests/internal/` directory (grading tests)
- `scripts/dev_utils.py` (solution management)
- `scripts/sanitize.sh` (sanitization script)
- `scripts/_sanitize_todos.py` (helper script)
- `.claude/` directory (instructor automation)
- Files with `[SOLUTION]` markers

### Multi-Layer Protection
1. **Pre-commit hook** - Blocks commits with `[SOLUTION]` in public files
2. **VLA Guard skill** - Scans for leaks before operations
3. **Sanitization pipeline** - Removes private content automatically
4. **Post-sanitization validation** - Fail-safe check in GitHub Actions
5. **Orphan push** - Breaks git history links
6. **Sync-check skill** - Verifies public repo after release

---

## Requirements

- **Node.js** 18+
- **pnpm** 8+
- **Python** 3.11+
- **uv** (Python package manager): `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **gh CLI** (for skills): `brew install gh && gh auth login`

Python dependencies (managed by uv via `pyproject.toml`):
- pytest, pytest-html
- torch
- numpy

---

## Support

- **Instructor Guide**: [INSTRUCTOR.md](INSTRUCTOR.md)
- **Skills Documentation**: [.claude/skills/README.md](.claude/skills/README.md)
- **Public Repo**: https://github.com/arpg/vla-foundations
- **Course Website**: https://www.vlm-robotics.dev

---

**Remember**: This is the private instructor repository. Always run `/vla-guard` before any public-facing operation.
