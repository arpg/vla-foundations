# Private Solution Repository Setup Guide

This document describes the complete architecture for managing private solutions and internal grading tests for VLA Foundations assignments.

## Overview

The VLA Foundations repository uses a **private mirror architecture** that:
1. Stores complete assignment solutions (never public)
2. Provides internal "Grading Engine" tests with rigorous validation
3. Syncs sanitized code to public repository automatically
4. Minimizes unnecessary rebuilds during sync

## Repository Architecture

```
┌────────────────────────────────────────────────────┐
│   PRIVATE REPO (instructor only)                   │
│   github.com/crheckman/private-vla-foundations     │
│   ├── private/solutions/     (NEVER PUBLIC)        │
│   ├── tests/internal/        (NEVER PUBLIC)        │
│   ├── scripts/manage_solutions.py (inject/reset)   │
│   ├── scripts/sanitize.sh   (automated cleanup)    │
│   └── src/assignments/      (with solution hints)  │
└────────────────────┬───────────────────────────────┘
                     │ GitHub Action on tag: release-scratch-*
                     │ Runs scripts/sanitize.sh
                     │ Deletes private/, sanitizes comments
                     ▼
┌────────────────────────────────────────────────────┐
│   PUBLIC REPO (arpg/vla-foundations)               │
│   ├── src/assignments/      (starter code only)    │
│   ├── tests/public/         (basic validation)     │
│   ├── content/              (assignment specs)     │
│   └── .github/workflows/    (deploy on content/)   │
└────────────────────────────────────────────────────┘
```

## Directory Structure

### Private Repository

```
vla-foundations/                    # Private repo root
├── private/                        # NEVER synced to public
│   ├── solutions/
│   │   ├── backbone_solution.py   # Complete implementations
│   │   └── checkpoints/
│   │       └── scratch1_gold.pt   # Trained models
│   └── README.md                   # Solution usage guide
│
├── tests/
│   ├── conftest.py                 # Shared pytest config
│   ├── public/                     # Tests students can run
│   │   └── test_scratch1_basic.py
│   └── internal/                   # NEVER synced to public
│       ├── conftest.py
│       ├── fixtures/               # Gold standards
│       │   └── scratch1_gold.pt
│       └── test_scratch1_rigor.py  # Rigorous grading tests
│
├── scripts/
│   ├── manage_solutions.py         # Inject/reset solutions
│   ├── sanitize.sh                 # Main sanitization script
│   └── _sanitize_todos.py          # TODO comment cleanup
│
├── src/assignments/scratch-1/
│   ├── backbone.py                 # Starter code with TODO: [SOLUTION] hints
│   └── generate_data.py
│
└── .github/workflows/
    └── sync-to-public.yml          # Automated sync workflow
```

### Public Repository

```
vla-foundations/                    # Public repo (after sanitization)
├── tests/
│   └── public/                     # Only public tests
│       └── test_scratch1_basic.py
│
├── src/assignments/scratch-1/
│   ├── backbone.py                 # Starter code (sanitized, no hints)
│   └── generate_data.py
│
└── content/                        # Assignment specifications
    └── assignments/
        └── scratch-1.mdx
```

## Setup Instructions

### 1. Create Private Repository

```bash
# On GitHub: Create new private repository
# Name: private-vla-foundations
# Organization: crheckman (or your personal account)

# Clone the current repository
git clone https://github.com/arpg/vla-foundations.git private-vla-foundations
cd private-vla-foundations

# Update remote to point to new private repo
git remote remove origin
git remote add origin https://github.com/crheckman/private-vla-foundations.git
git push -u origin main

# Add public repo as remote
git remote add public https://github.com/arpg/vla-foundations.git
```

### 2. Initialize Private Structure

The structure has already been created by this implementation. Verify:

```bash
# Check directory structure
ls -la private/
ls -la tests/internal/
ls -la scripts/

# Verify .gitignore excludes private content
cat .gitignore | grep private
```

### 3. Configure GitHub Secrets

In your private repository settings (Settings → Secrets and variables → Actions):

1. **PUBLIC_REPO_TOKEN**:
   - Create a Personal Access Token (PAT) with `repo` scope
   - Go to GitHub Settings → Developer settings → Personal access tokens
   - Generate new token (classic)
   - Select scope: `repo` (full control)
   - Copy token and add as secret `PUBLIC_REPO_TOKEN`

### 4. Create First Solution

```bash
# Create solution file
touch private/solutions/backbone_solution.py

# Edit and implement complete solution
# (See private/solutions/backbone_solution.py for example)

# Test solution injection
python scripts/manage_solutions.py --inject scratch-1

# Verify files were copied
ls -la src/assignments/scratch-1/

# Reset to starter code
python scripts/manage_solutions.py --reset scratch-1
```

### 5. Write Internal Tests

```bash
# Edit tests/internal/test_scratch1_rigor.py
# Add rigorous validation tests

# Run internal tests (auto-injects solutions)
pytest tests/internal/test_scratch1_rigor.py -v

# Solutions are automatically reset after tests complete
```

### 6. Test Sanitization

```bash
# Create a test branch
git checkout -b test-sanitization

# Run sanitization script
bash scripts/sanitize.sh

# Verify private content was removed
ls private/  # Should fail - directory deleted
ls tests/internal/  # Should fail - directory deleted

# Check for solution leaks
grep -r "SOLUTION" src/ content/  # Should find nothing

# Review changes
git status
git diff

# Reset to main
git checkout main
git branch -D test-sanitization
```

## Workflow for Instructors

### Developing a New Assignment

1. **Create assignment in private repo**:
   ```bash
   # Work in private-vla-foundations
   cd src/assignments/scratch-2
   # Create starter code with TODO: [SOLUTION] hints
   ```

2. **Implement complete solution**:
   ```bash
   # Create solution file
   vim private/solutions/scratch2_solution.py
   # Implement complete, working solution
   ```

3. **Write internal tests**:
   ```bash
   # Create test file
   vim tests/internal/test_scratch2_rigor.py
   # Add gradient, fidelity, convergence tests
   ```

4. **Generate gold standards**:
   ```bash
   # Inject solution
   python scripts/manage_solutions.py --inject scratch-2

   # Train model
   cd src/assignments/scratch-2
   python main.py

   # Save gold standard
   cp checkpoints/best.pt ../../../tests/internal/fixtures/scratch2_gold.pt
   ```

5. **Test everything**:
   ```bash
   # Test internal grading
   pytest tests/internal/test_scratch2_rigor.py -v

   # Test public validation
   python scripts/manage_solutions.py --reset scratch-2
   pytest tests/public/test_scratch2_basic.py -v
   ```

### Releasing to Students

1. **Commit all changes to private repo**:
   ```bash
   git add .
   git commit -m "Add scratch-2 assignment with solution"
   git push origin main
   ```

2. **Create release tag**:
   ```bash
   # Tag triggers automatic sync to public repo
   git tag release-scratch-2
   git push origin release-scratch-2
   ```

3. **GitHub Actions automatically**:
   - Runs `scripts/sanitize.sh`
   - Removes `private/` directory
   - Removes `tests/internal/` directory
   - Sanitizes `TODO: [SOLUTION]` comments
   - Removes instructor notes from MDX files
   - Pushes to public repo branch `public-release-scratch-2`

4. **Review and merge**:
   ```bash
   # Go to public repo: https://github.com/arpg/vla-foundations
   # Review the `public-release-scratch-2` branch
   # Create PR to merge into main
   # Review changes carefully
   # Merge PR
   ```

### Updating an Assignment

1. **Make changes in private repo**:
   ```bash
   # Edit starter code
   vim src/assignments/scratch-1/backbone.py

   # Update solution if needed
   vim private/solutions/backbone_solution.py

   # Commit changes
   git add .
   git commit -m "Update scratch-1: clarify RMSNorm TODO"
   git push origin main
   ```

2. **Create new release tag**:
   ```bash
   git tag release-scratch-1-v2
   git push origin release-scratch-1-v2
   ```

3. **Review and merge to public** (same as above)

## Testing Infrastructure

### Public Tests (`tests/public/`)

**Purpose**: Basic validation that students can run

```bash
# Students run these tests
pytest tests/public/test_scratch1_basic.py -v
```

Tests validate:
- Provided (non-TODO) components work
- Environment is set up correctly
- Basic shapes and types are correct

### Internal Tests (`tests/internal/`)

**Purpose**: Rigorous grading validation (instructor only)

```bash
# Instructors run these tests
pytest tests/internal/test_scratch1_rigor.py -v
```

Tests validate:
- Gradient leak detection (frozen parameters)
- Latent fidelity (match gold standards)
- Training convergence
- Edge cases and robustness

### Test Markers

```bash
# Run only public tests
pytest -m public -v

# Run only internal tests
pytest -m internal -v

# Run gradient leak tests
pytest -m gradient -v

# Run fidelity tests
pytest -m fidelity -v
```

## Solution Management

### Commands

```bash
# List available solutions
python scripts/manage_solutions.py --list

# Inject solutions for testing
python scripts/manage_solutions.py --inject scratch-1

# Reset to starter code
python scripts/manage_solutions.py --reset scratch-1
```

### Auto-injection for Internal Tests

Internal tests automatically inject solutions:

```bash
# Just run the tests - solutions injected automatically
pytest tests/internal/test_scratch1_rigor.py -v

# Solutions are reset after tests complete
```

This is handled by the `inject_solutions_for_internal_tests` fixture in `tests/conftest.py`.

## Build Isolation

Deployment workflows ignore changes to:
- `src/assignments/` (assignment code)
- `scripts/` (solution management)
- `tests/` (test files)
- `private/` (solutions)
- `*.md` (documentation)

This prevents unnecessary rebuilds when:
- Adding new assignments
- Updating tests
- Updating solutions
- Writing documentation

Only changes to `app/`, `content/`, `components/`, etc. trigger deployment.

## Security Checklist

- [ ] Private repo is actually private (not public)
- [ ] `.gitignore` includes `private/` and `tests/internal/`
- [ ] `PUBLIC_REPO_TOKEN` secret is configured
- [ ] Sanitization script tested locally
- [ ] No `[SOLUTION]` markers in public repo
- [ ] Gold standards are in `tests/internal/fixtures/`
- [ ] Solution files use `*_solution.py` naming

## Troubleshooting

### Solutions not injecting

**Problem**: `manage_solutions.py --inject` fails

**Solution**:
1. Check solution files exist: `ls private/solutions/*_solution.py`
2. Verify naming convention: must end with `_solution.py`
3. Check target directory exists: `ls src/assignments/scratch-1/`

### Tests can't import modules

**Problem**: `ImportError` in tests

**Solution**:
1. Check `conftest.py` adds correct paths to `sys.path`
2. Verify assignment directory structure
3. Try running from project root: `pytest tests/internal/ -v`

### Sanitization not working

**Problem**: Private content appears in public repo

**Solution**:
1. Run sanitization locally: `bash scripts/sanitize.sh`
2. Check for errors in output
3. Verify `.gitignore` is correct
4. Review `sync-to-public.yml` workflow logs

### Deployment triggered on code changes

**Problem**: Website rebuilds when changing assignment code

**Solution**:
1. Check `deploy.yml` and `deploy-staging.yml`
2. Verify `paths` and `paths-ignore` filters
3. Test with: `git push origin main` (should not trigger)

## FAQ

**Q: Can students see the private repo?**
A: No, it's a completely separate private repository.

**Q: What if I accidentally commit a solution to public?**
A: Immediately revert the commit, run sanitization, force push to clean history.

**Q: How do I add a new assignment?**
A: Create starter code, write solution in `private/solutions/`, write tests, tag with `release-<assignment-name>`.

**Q: Can I test sanitization without pushing?**
A: Yes! Run `bash scripts/sanitize.sh` locally, review changes, then `git reset --hard HEAD`.

**Q: Do I need to manually sync to public?**
A: No, tagging with `release-*` automatically triggers sync via GitHub Actions.

## Next Steps

1. ✅ Private repo structure created
2. ✅ Solution management scripts working
3. ✅ Test infrastructure set up
4. ✅ Sanitization pipeline tested
5. ⏭️ Create private GitHub repository
6. ⏭️ Configure secrets (PUBLIC_REPO_TOKEN)
7. ⏭️ Test complete workflow end-to-end
8. ⏭️ Document for other instructors

## Support

For questions or issues with this infrastructure:
- Review this document
- Check `private/README.md` for solution-specific help
- Check `tests/README.md` for testing help
- Review GitHub Actions logs for sync issues
