# VLA Private Source of Truth (PST)

**CONFIDENTIAL - INTERNAL USE ONLY**

This repository contains solutions, hidden tests, and deployment keys for the VLA Foundations course.

**Live Site**: https://www.vlm-robotics.dev

---

## Repository Map

| Repository | Purpose | Push Changes Here | Visibility |
|------------|---------|-------------------|------------|
| **`crheckman/private-vla-foundations`** | Instructor development | ✅ YES - Work here | 🔒 Private |
| **`arpg/vla-foundations`** | Student access | ❌ NO - Auto-synced | 🌐 Public |

### ⚠️ DO NOT push directly to `arpg/vla-foundations`

The public repository is **automatically synced** from this private repository using GitHub Actions when you create a release tag.

**To release changes to public:**
```bash
git tag release-scratch-1
git push origin release-scratch-1
# GitHub Actions automatically sanitizes and syncs to public repo
```

---

## Key Scripts

### Solution Management

```bash
# Swap stubs for full solutions (for internal testing)
python scripts/manage_solutions.py --inject scratch-1

# Restore stubs before pushing to public
python scripts/manage_solutions.py --reset scratch-1

# List available solutions
python scripts/manage_solutions.py --list
```

### Sanitization

```bash
# The core fail-safe for removing solution leaks
bash scripts/sanitize.sh
```

This script:
1. Deletes `private/` and `tests/internal/` directories
2. Deletes solution management scripts
3. Sanitizes TODO comments (removes `[SOLUTION]` hints)
4. Sanitizes MDX files (removes instructor notes)
5. Removes draft assignment blocks
6. Commits sanitized state

---

## Shadow CI

Rigorous internal tests are automatically run against student PRs via the **Shadow Tester** workflow.

When a student opens a PR to the public `staging` branch:
1. Public repo triggers a `repository_dispatch` event to this private repo
2. Shadow Tester workflow runs internal tests against student code
3. Results are posted back to the public PR as comments

This allows you to run hidden tests without exposing test logic or solutions.

---

## PST System Setup

### GitHub Secrets

**In the Public Repo** (`arpg/vla-foundations`):
- `PRIVATE_DISPATCH_TOKEN`: Fine-grained PAT with access to the private repo (triggers Shadow CI)

**In the Private Repo** (`crheckman/private-vla-foundations`):
- `PUBLIC_REPO_TOKEN`: PAT with push access to the public repo (for sanitization sync)

### Branch Protection

**Public repo:**
- `main` and `staging` should require PRs and passing status checks
- Enable "Require conversation resolution before merging"

**Private repo:**
- No restrictions (instructor has full control)

---

## Assignment Lifecycle

### Creating a New Assignment

1. **Create stub and solution files** in the private repo:
   ```bash
   # Create stub with TODOs
   touch src/assignments/scratch-2/model.py

   # Create full solution
   touch private/solutions/model_solution.py

   # Mark solution blocks with # [SOLUTION] tags
   ```

2. **Create assignment specification**:
   ```bash
   touch content/course/assignments/scratch-2.mdx
   ```

3. **Wrap in DRAFT block** (prevents public release):
   ```mdx
   <div className="draft-warning">
   ⚠️ DRAFT: NOT YET ASSIGNED

   This assignment is under development and not yet released to students.
   </div>

   # Assignment content here...
   ```

### Updating an Assignment

1. **Apply changes** to the solution/stub in the private repo
2. **Run inject** to verify updates against internal rigorous tests:
   ```bash
   python scripts/manage_solutions.py --inject scratch-1
   pytest tests/internal/test_scratch1_rigor.py
   ```

### Promoting from Draft to Released

1. **Remove the "DRAFT" warning block** from the MDX file in the private repo
2. **Create a release tag**:
   ```bash
   git tag release-scratch-1
   git push origin release-scratch-1
   ```
3. **GitHub Actions automatically**:
   - Runs `scripts/sanitize.sh`
   - Removes draft blocks
   - Deletes solution markers
   - Pushes to `public-release` branch in public repo
4. **Review and merge** the release branch into public `main` or `staging`

---

## Directory Structure

```
private-vla-foundations/
├── private/                           # NEVER synced to public
│   ├── solutions/                     # Complete assignment solutions
│   │   ├── backbone_solution.py       # Scratch-1 full implementation
│   │   ├── generate_data_solution.py  # Enhanced data generation
│   │   └── checkpoints/               # Trained model weights
│   │       └── scratch1_gold.pt       # Gold standard for testing
│   └── README.md                      # Solution usage guide
│
├── tests/                             # Testing infrastructure
│   ├── public/                        # Tests students can see/run
│   │   └── test_scratch1_basic.py     # Basic validation
│   └── internal/                      # NEVER synced to public
│       ├── test_scratch1_rigor.py     # Rigorous grading tests
│       ├── fixtures/                  # Gold standard data
│       │   └── scratch1_gold.pt       # Reference tensor
│       └── conftest.py                # Internal test fixtures
│
├── scripts/                           # NEVER synced to public
│   ├── manage_solutions.py            # Inject/reset utility
│   ├── sanitize.sh                    # Main sanitization script
│   ├── _sanitize_todos.py             # Helper for TODO cleanup
│   └── setup_private_repo.sh          # Initial setup script
│
├── src/assignments/                   # Assignment stubs (synced with hints)
│   └── scratch-1/
│       ├── backbone.py                # TODO: [SOLUTION] hints
│       └── generate_data.py
│
├── content/                           # MDX content
│   ├── textbook/                      # 8-chapter textbook
│   └── course/                        # Course materials
│       └── assignments/               # Assignment specs
│           └── scratch-1.mdx          # May contain <!-- INSTRUCTOR NOTE -->
│
└── .github/workflows/
    ├── sync-to-public.yml             # Auto-sync on release tags
    └── shadow-tester.yml              # Run internal tests on student PRs
```

---

## Testing Framework

### Public Tests (`tests/public/`)
- Basic validation that students can run
- Checks provided (non-TODO) components
- Visible in public repo

### Internal Tests (`tests/internal/`)
- Rigorous grading tests (NEVER public)
- Gradient leak detection
- Latent fidelity comparison against gold standard
- Training convergence validation
- Edge case testing

### Running Tests

```bash
# Run public tests (students can run these)
pytest tests/public/

# Run internal tests (auto-injects solutions)
pytest tests/internal/

# Run specific test file
pytest tests/internal/test_scratch1_rigor.py -v
```

---

## Operational Guides

For detailed operational procedures, see:
- **PRIVATE_OPERATIONS.md**: Comprehensive guide to review workflows, server configuration, deployment, and troubleshooting

---

## Quick Reference

### Review a Student PR
```bash
# Public repo PR workflow
gh pr checkout {PR_NUMBER}
gh pr view {PR_NUMBER} --web

# Trigger Shadow CI manually (if needed)
gh workflow run shadow-tester.yml -f pr_number={PR_NUMBER}
```

### Create a New Release
```bash
# Tag and push from private repo
git tag release-scratch-1
git push origin release-scratch-1

# Monitor the sync workflow
gh run list --workflow=sync-to-public.yml
```

### Test Sanitization Locally
```bash
# Dry run (doesn't commit)
bash scripts/sanitize.sh --dry-run

# Check for solution leaks
grep -r "TODO: \[SOLUTION\]" src/ content/
grep -r "\[SOLUTION\]" src/ content/
```

---

## Contact

- **Instructor**: Christoffer Heckman
- **Email**: christoffer.heckman@colorado.edu
- **Course**: CSCI 7000, Spring 2026

---

## License

**CONFIDENTIAL** - All rights reserved.

This repository and its contents are private and confidential.
Unauthorized access or distribution is prohibited.
