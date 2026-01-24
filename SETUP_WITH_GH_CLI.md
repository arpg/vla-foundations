# Setup Using GitHub CLI (`gh`)

This guide shows you how to set up the private repository using only the GitHub CLI.

## Prerequisites

```bash
# Check gh is installed
gh --version

# Login if needed
gh auth login
```

## Quick Setup (Copy & Paste)

### 1. Create Private Repository

```bash
# Set your repository name and owner
REPO_NAME="private-vla-foundations"
REPO_OWNER="crheckman"  # or your GitHub username/org

# Create private repo
gh repo create "$REPO_OWNER/$REPO_NAME" \
    --private \
    --description "Private solutions and internal tests for VLA Foundations" \
    --confirm

echo "✓ Created https://github.com/$REPO_OWNER/$REPO_NAME"
```

### 2. Create Personal Access Token

```bash
# Open token creation page in browser
gh browse https://github.com/settings/tokens/new

# Instructions:
# 1. Note: "VLA Sync Token $(date +%Y-%m-%d)"
# 2. Expiration: 90 days (or your preference)
# 3. Scopes: Check ☑ "repo" (Full control)
# 4. Click "Generate token"
# 5. Copy the token (starts with ghp_ or github_pat_)
```

### 3. Add Secret to Repository

```bash
# Method A: Use helper script
./scripts/add_github_secret.sh "$REPO_OWNER/$REPO_NAME"
# Follow prompts to paste your PAT token

# Method B: Manual command
gh secret set PUBLIC_REPO_TOKEN --repo "$REPO_OWNER/$REPO_NAME"
# Paste your PAT when prompted

# Verify secret was added
gh secret list --repo "$REPO_OWNER/$REPO_NAME"
```

### 4. Update Git Remotes

```bash
# Rename current origin to public (if pointing to arpg/vla-foundations)
git remote rename origin public

# Add private repo as new origin
git remote add origin "https://github.com/$REPO_OWNER/$REPO_NAME.git"

# Verify remotes
git remote -v
# Should show:
#   origin → crheckman/private-vla-foundations (private)
#   public → arpg/vla-foundations (public)
```

### 5. Push to Private Repository

```bash
# Push main branch
git push -u origin main

# Push staging branch (if it exists)
git push -u origin staging

# View your repo
gh repo view "$REPO_OWNER/$REPO_NAME" --web
```

## Complete Automated Setup

Or use the automated script:

```bash
# Run automated setup script
./scripts/setup_private_repo.sh

# This will:
# - Create private repository
# - Guide you through token creation
# - Add secret to repository
# - Update git remotes
# - Push branches
```

## Verification

```bash
# 1. Check repository exists
gh repo view "$REPO_OWNER/$REPO_NAME"

# 2. Check secret exists
gh secret list --repo "$REPO_OWNER/$REPO_NAME"
# Should show: PUBLIC_REPO_TOKEN

# 3. Check remotes
git remote -v
# Should show both origin (private) and public

# 4. View repo in browser
gh repo view "$REPO_OWNER/$REPO_NAME" --web
```

## Test the Workflow

### Test Solution Management

```bash
# List solutions
python3 scripts/manage_solutions.py --list

# Inject solutions
python3 scripts/manage_solutions.py --inject scratch-1

# Check injection worked
ls -la src/assignments/scratch-1/*.backup.py

# Reset to starter code
python3 scripts/manage_solutions.py --reset scratch-1
```

### Test Sanitization (Safe)

```bash
# Create test branch
git checkout -b test-sanitize

# Run sanitization
bash scripts/sanitize.sh

# Review what was removed
git status
git diff

# Check for leaks
grep -r "\[SOLUTION\]" src/ content/  # Should find nothing
ls private/  # Should fail - directory deleted

# Reset
git checkout main
git branch -D test-sanitize
```

### Test Release Sync

```bash
# Create test release tag
git tag release-scratch-1-test
git push origin release-scratch-1-test

# Watch GitHub Actions run
gh run watch --repo "$REPO_OWNER/$REPO_NAME"

# View the workflow in browser
gh repo view "$REPO_OWNER/$REPO_NAME" --web
# Click "Actions" tab

# Check public repo for new branch
gh repo view arpg/vla-foundations --web
# Look for branch: public-release-scratch-1-test

# Clean up test tag
git tag -d release-scratch-1-test
git push origin :refs/tags/release-scratch-1-test
```

## Install Testing Dependencies

```bash
# Install pytest and dependencies
pip install pytest torch numpy

# Run public tests
pytest tests/public/ -v

# Run internal tests (auto-injects solutions)
pytest tests/internal/ -v

# Run specific test categories
pytest -m gradient -v
pytest -m fidelity -v
```

## Managing Secrets

```bash
# List secrets
gh secret list --repo "$REPO_OWNER/$REPO_NAME"

# Add/update a secret
gh secret set SECRET_NAME --repo "$REPO_OWNER/$REPO_NAME"

# Delete a secret
gh secret delete SECRET_NAME --repo "$REPO_OWNER/$REPO_NAME"
```

## Troubleshooting

### Token Creation

If you need to create a token with specific scopes via CLI:

```bash
# This requires gh extension
gh extension install github/gh-token

# Create token (requires confirmation)
gh token create --scopes repo --note "VLA Sync $(date +%Y-%m-%d)"
```

### Secret Not Working

```bash
# Check secret exists
gh secret list --repo "$REPO_OWNER/$REPO_NAME"

# Re-add the secret
gh secret set PUBLIC_REPO_TOKEN --repo "$REPO_OWNER/$REPO_NAME"
# Paste token when prompted

# Test with a dummy workflow run
gh workflow run sync-to-public.yml --repo "$REPO_OWNER/$REPO_NAME"
```

### Remote Issues

```bash
# List remotes
git remote -v

# Fix public remote
git remote set-url public https://github.com/arpg/vla-foundations.git

# Fix origin remote
git remote set-url origin "https://github.com/$REPO_OWNER/$REPO_NAME.git"
```

## Useful gh Commands

```bash
# View repo info
gh repo view

# View repo in browser
gh repo view --web

# List workflows
gh workflow list

# View workflow runs
gh run list

# Watch a workflow run
gh run watch

# View secrets
gh secret list

# View repo settings in browser
gh browse --settings

# Clone your private repo (from elsewhere)
gh repo clone "$REPO_OWNER/$REPO_NAME"
```

## Environment Variables

For convenience, add to your `~/.bashrc` or `~/.zshrc`:

```bash
export VLA_PRIVATE_REPO="crheckman/private-vla-foundations"
export VLA_PUBLIC_REPO="arpg/vla-foundations"

# Then use:
gh secret list --repo "$VLA_PRIVATE_REPO"
gh repo view "$VLA_PRIVATE_REPO" --web
```

## Next Steps

Once setup is complete:

1. **Read documentation**:
   - `PRIVATE_REPO_SETUP.md` - Complete guide
   - `QUICK_REFERENCE.md` - Command cheat sheet

2. **Test workflow end-to-end**:
   - Inject solution
   - Run internal tests
   - Create release tag
   - Watch sync to public
   - Review public release branch

3. **Add more assignments**:
   - Create solution in `private/solutions/`
   - Write internal tests in `tests/internal/`
   - Tag with `release-<assignment>`

## Support

For issues with GitHub CLI:
- Documentation: https://cli.github.com/manual/
- Issues: https://github.com/cli/cli/issues

For issues with this setup:
- See `PRIVATE_REPO_SETUP.md` troubleshooting section
- Check GitHub Actions logs: `gh run list --repo "$REPO_OWNER/$REPO_NAME"`
