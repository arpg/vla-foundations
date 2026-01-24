# Quick Reference: Private Solution Architecture

## Common Commands

### Solution Management

```bash
# List available solutions
python3 scripts/manage_solutions.py --list

# Inject solutions for testing
python3 scripts/manage_solutions.py --inject scratch-1

# Reset to starter code
python3 scripts/manage_solutions.py --reset scratch-1
```

### Running Tests

```bash
# Install pytest (one time)
pip install pytest torch numpy

# Run public tests (what students run)
pytest tests/public/ -v

# Run internal tests (with auto solution injection)
pytest tests/internal/test_scratch1_rigor.py -v

# Run specific test categories
pytest -m gradient -v      # Gradient leak tests
pytest -m fidelity -v      # Fidelity comparison tests
pytest -m training -v      # Training convergence tests
```

### Releasing to Public

```bash
# 1. Commit changes to private repo
git add .
git commit -m "Add/update assignment"
git push origin main

# 2. Create release tag (triggers automatic sync)
git tag release-scratch-1
git push origin release-scratch-1

# 3. GitHub Actions runs automatically:
#    - Sanitizes code
#    - Removes private/ and tests/internal/
#    - Pushes to public repo branch

# 4. Review and merge in public repo
#    - Go to https://github.com/arpg/vla-foundations
#    - Review public-release-* branch
#    - Create PR and merge
```

### Testing Sanitization

```bash
# Test locally before releasing
git checkout -b test-sanitize
bash scripts/sanitize.sh
git status  # Review what was removed
git diff    # Review what was changed
git checkout main
git branch -D test-sanitize
```

## File Locations

### Solutions
- `private/solutions/backbone_solution.py` - Complete implementation
- `private/solutions/checkpoints/` - Trained model weights

### Tests
- `tests/public/` - Tests students can run
- `tests/internal/` - Internal grading tests (NEVER public)
- `tests/internal/fixtures/` - Gold standard reference data

### Scripts
- `scripts/manage_solutions.py` - Inject/reset solutions
- `scripts/sanitize.sh` - Sanitize for public release
- `scripts/_sanitize_todos.py` - Remove solution hints

### Workflows
- `.github/workflows/sync-to-public.yml` - Auto-sync on release tags
- `.github/workflows/deploy.yml` - Deploy production (path-filtered)
- `.github/workflows/deploy-staging.yml` - Deploy staging (path-filtered)

## Solution File Naming

Solutions must follow this pattern:
- `<filename>_solution.py` → maps to `src/assignments/<assignment>/<filename>.py`

Examples:
- `backbone_solution.py` → `src/assignments/scratch-1/backbone.py`
- `generate_data_solution.py` → `src/assignments/scratch-1/generate_data.py`

## TODO Comment Conventions

Use `[SOLUTION]` markers for hints that should be removed:

```python
# In starter code (private repo):
# TODO: [SOLUTION] Use torch.rsqrt for efficiency
result = None  # Student implements this

# After sanitization (public repo):
# TODO: Complete implementation
result = None
```

Inline solution comments:
```python
# In starter code (private repo):
result = torch.rsqrt(x)  # [SOLUTION]: More efficient than 1/sqrt

# After sanitization (public repo):
result = torch.rsqrt(x)
```

## Test Markers

```python
# Public test
@pytest.mark.public
def test_basic_functionality():
    pass

# Internal test
@pytest.mark.internal
@pytest.mark.gradient
def test_gradient_leak():
    pass
```

Available markers:
- `public` - Tests students can run
- `internal` - Internal grading tests
- `gradient` - Gradient flow validation
- `fidelity` - Output quality vs gold standard
- `training` - Training convergence
- `rigor` - Rigorous validation

## Directory Structure

```
vla-foundations/
├── private/                    # NEVER public
│   ├── solutions/             # Complete implementations
│   └── README.md
├── tests/
│   ├── public/                # Students can run
│   └── internal/              # NEVER public
│       ├── fixtures/          # Gold standards
│       └── test_*_rigor.py
├── scripts/
│   ├── manage_solutions.py    # Inject/reset
│   └── sanitize.sh            # Sanitize for public
├── src/assignments/
│   └── scratch-1/
│       ├── backbone.py        # Starter with TODOs
│       └── generate_data.py
└── .github/workflows/
    └── sync-to-public.yml     # Auto-sync workflow
```

## Workflow

1. **Develop** in private repo with solutions
2. **Test** with internal rigorous tests
3. **Tag** with `release-*` pattern
4. **Auto-sync** removes private content
5. **Review** public release branch
6. **Merge** to publish to students

## Security

**NEVER commit to public:**
- `private/` directory
- `tests/internal/` directory
- Files matching `*_solution.py`
- Gold standard fixtures
- `TODO: [SOLUTION]` comments

**Always verify before public release:**
```bash
grep -r "\[SOLUTION\]" src/ content/  # Should find nothing
ls private/  # Should fail (directory deleted)
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Solutions not injecting | Check naming: `*_solution.py` |
| Tests can't import | Check `conftest.py` adds paths |
| Private content in public | Run `sanitize.sh`, verify output |
| Deployment on code change | Check `paths-ignore` in workflows |
| Test fixture not found | Generate gold standard first |

## Setup with GitHub CLI

**Quick setup (5 commands):**
```bash
# 1. Create private repo
gh repo create crheckman/private-vla-foundations --private --confirm

# 2. Create PAT token (opens browser)
gh browse https://github.com/settings/tokens/new
# Select "repo" scope, copy token

# 3. Add secret (paste token when prompted)
gh secret set PUBLIC_REPO_TOKEN --repo crheckman/private-vla-foundations

# 4. Update remotes
git remote rename origin public
git remote add origin https://github.com/crheckman/private-vla-foundations.git

# 5. Push
git push -u origin main staging
```

**Or use automated script:**
```bash
./scripts/setup_private_repo.sh
```

See **SETUP_WITH_GH_CLI.md** for detailed guide.

## Next Steps

1. ✅ Use GitHub CLI setup (see above)
2. Test solution management
3. Test complete workflow
4. Add more assignments

## Full Documentation

- `PRIVATE_REPO_SETUP.md` - Complete setup guide
- `private/README.md` - Solution management
- `tests/README.md` - Testing infrastructure
- `pytest.ini` - Test configuration
