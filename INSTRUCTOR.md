# VLA Foundations: Instructor Guide

**CONFIDENTIAL - INTERNAL USE ONLY**

This is the comprehensive instructor guide for managing the VLA Foundations course infrastructure, including the private solution repository, student review workflows, and deployment procedures.

---

## Table of Contents

1. [Repository Architecture](#repository-architecture)
2. [Setup & Configuration](#setup--configuration)
3. [Solution Management](#solution-management)
4. [Student Review Workflow](#student-review-workflow)
5. [Shadow CI & Testing](#shadow-ci--testing)
6. [Deployment Procedures](#deployment-procedures)
7. [Server Configuration](#server-configuration)
8. [Troubleshooting](#troubleshooting)

---

## Repository Architecture

### Overview

The VLA Foundations course uses a **dual-repository architecture**:

```
┌────────────────────────────────────────────────────┐
│   PRIVATE REPO (instructor only)                   │
│   github.com/crheckman/private-vla-foundations     │
│   ├── private/solutions/     (NEVER PUBLIC)        │
│   ├── tests/internal/        (NEVER PUBLIC)        │
│   ├── scripts/dev_utils.py   (solution mgmt)       │
│   ├── scripts/sanitize.sh    (auto cleanup)        │
│   └── src/assignments/       (with [SOLUTION] hints)│
└────────────────────┬───────────────────────────────┘
                     │ GitHub Action on release tag
                     │ Runs sanitize.sh → orphan push
                     ▼
┌────────────────────────────────────────────────────┐
│   PUBLIC REPO (arpg/vla-foundations)               │
│   ├── src/assignments/      (starter code only)    │
│   ├── tests/public/         (basic validation)     │
│   ├── content/              (assignment specs)     │
│   └── .github/workflows/    (deploy + Shadow CI)   │
└────────────────────────────────────────────────────┘
```

### Directory Structure (Private Repo)

```
private-vla-foundations/
├── private/                        # NEVER synced to public
│   ├── solutions/
│   │   ├── backbone_solution.py   # Complete implementations
│   │   └── checkpoints/
│   │       └── scratch1_gold.pt   # Trained model weights
│   └── README.md
│
├── tests/
│   ├── conftest.py                 # Shared pytest config
│   ├── public/                     # Tests students can run
│   │   └── test_scratch1_basic.py
│   └── internal/                   # NEVER synced to public
│       ├── conftest.py
│       ├── fixtures/               # Gold standard tensors
│       │   └── scratch1_gold.pt
│       └── test_scratch1_rigor.py  # Gradient leak, fidelity tests
│
├── scripts/
│   ├── dev_utils.py                # Solution injection & verification
│   ├── sanitize.sh                 # Main sanitization script
│   └── _sanitize_todos.py          # TODO comment cleanup
│
├── src/assignments/scratch-1/
│   ├── backbone.py                 # Starter with TODO: [SOLUTION] hints
│   └── generate_data.py
│
└── .github/workflows/
    ├── sync-to-public.yml          # Orphan push workflow
    └── shadow-tester.yml           # Internal test runner
```

---

## Setup & Configuration

### Initial Setup

1. **Create Private Repository** (if not already done):
   ```bash
   # On GitHub: Create new private repository
   # Name: private-vla-foundations
   # Organization: crheckman

   # Clone and configure
   git clone https://github.com/arpg/vla-foundations.git private-vla-foundations
   cd private-vla-foundations
   git remote remove origin
   git remote add origin https://github.com/crheckman/private-vla-foundations.git
   git push -u origin main

   # Add public repo as remote
   git remote add public https://github.com/arpg/vla-foundations.git
   ```

2. **Configure GitHub Secrets**:

   **Private Repo** (`crheckman/private-vla-foundations`):
   ```bash
   # Create PAT with 'repo' scope for public repo
   gh secret set PUBLIC_REPO_TOKEN --repo crheckman/private-vla-foundations
   ```

   **Public Repo** (`arpg/vla-foundations`):
   ```bash
   # Create PAT with 'contents:write' + 'repository_dispatch' for private repo
   gh secret set PRIVATE_DISPATCH_TOKEN --repo arpg/vla-foundations
   ```

3. **Verify Structure**:
   ```bash
   cd private-vla-foundations
   ls -la private/
   ls -la tests/internal/
   ls -la scripts/
   ```

---

## Solution Management

### Using `scripts/dev_utils.py`

**Inject Solutions** (for internal testing):
```bash
python scripts/dev_utils.py --inject scratch-1
pytest tests/internal/test_scratch1_rigor.py -v
```

**Reset to Starter Code**:
```bash
python scripts/dev_utils.py --reset scratch-1
```

**Verify No Solution Leaks**:
```bash
python scripts/dev_utils.py --verify-clean
# Exits with error if any solution code found in src/assignments/
```

**List Available Solutions**:
```bash
python scripts/dev_utils.py --list
```

### Creating New Solutions

1. **Create starter code** in `src/assignments/new-assignment/`:
   ```python
   # TODO: [SOLUTION] Use torch.rsqrt for efficiency
   def rmsnorm(x):
       # TODO: Complete implementation
       pass
   ```

2. **Create complete solution** in `private/solutions/`:
   ```bash
   touch private/solutions/new_assignment_solution.py
   # Implement complete solution
   ```

3. **Create internal tests** in `tests/internal/`:
   ```bash
   touch tests/internal/test_new_assignment_rigor.py
   # Write gradient leak, fidelity, convergence tests
   ```

4. **Test solution**:
   ```bash
   python scripts/dev_utils.py --inject new-assignment
   pytest tests/internal/test_new_assignment_rigor.py -v
   python scripts/dev_utils.py --reset new-assignment
   ```

5. **Create assignment spec** in `content/course/assignments/`:
   ```mdx
   ---
   title: "New Assignment"
   ---

   <div className="draft-warning">
   ⚠️ DRAFT: NOT YET ASSIGNED
   </div>

   # Assignment content...
   ```

---

## Student Review Workflow

### Paper Audit Review Process

Students submit paper audits via Pull Requests to the `staging` branch. The system automatically:
1. Runs standards linter (semantic line breaks, clean git history)
2. Deploys preview to GitHub Pages
3. Posts preview link in PR comments
4. Triggers Shadow CI for internal testing

#### Student Submission

Students follow these steps:
```bash
git checkout -b audit/username-topic
# Add audit file to: content/textbook/audits/staging/username.mdx
git add content/textbook/audits/staging/username.mdx
git commit -m "audit: add [Paper] deep-dive by [Name]"
git push origin audit/username-topic
```

Then create PR with:
- **Base branch:** `staging`
- **Title:** `Audit: [Paper Topic] - [Name]`

#### Automated Checks

The `vla-audit.yml` workflow runs:
- ✅ Semantic line breaks check
- ✅ Clean git history check (no merge commits)
- ✅ Full site build
- ✅ Deploy preview to GitHub Pages
- ✅ Trigger Shadow CI

**If linter fails:**
- Bot posts detailed fix instructions
- Build blocked until standards met

**If linter passes:**
- Preview URL: `https://arpg.github.io/vla-foundations/staging/pulls/{PR}/textbook/audits/staging/{username}/`
- Bot posts preview link in PR comments

#### Instructor Review

1. **Navigate to preview URL** (posted in PR comments)
2. **Review rendered audit** for:
   - Mathematical rigor and LaTeX correctness
   - Architectural depth and comparative analysis
   - Citations and references
   - Physical grounding critique ("load-bearing" analysis)
3. **Add feedback** using GitHub PR review features
4. **Request changes** or approve

#### Promoting to Production

When audit reaches "Level 3 (A - Mastery)" quality:

```bash
# Checkout PR branch
gh pr checkout {PR_NUMBER}

# Move file to production location
git mv content/textbook/audits/staging/{username}.mdx \
       content/textbook/audits/{canonical-name}.mdx

# Commit and push
git commit -m "Promote audit to production: ready for merge"
git push

# Merge PR
gh pr merge {PR_NUMBER} --merge

# Merge staging → main
git checkout main
git merge staging
git push origin main
```

### Grading Levels

**Level 1 (C): Basic Summary**
- Correct frontmatter and basic formatting
- Summary of paper's main contributions
- Basic LaTeX rendering

**Level 2 (B): Technical Deep-Dive**
- High technical depth with architectural details
- Correct mathematical formulations
- Sound critique of the model
- Proper citations

**Level 3 (A - Mastery): Merge-Ready**
- Exhaustive technical analysis
- Comparative deep-dive across multiple papers
- Original insights on scaling laws and bottlenecks
- Physical grounding critique (semantic-motor gap, information decay)
- Professional "Senior Staff" engineering quality
- **Ready to merge to main as canonical reference**

---

## Shadow CI & Testing

### How Shadow CI Works

When a student opens a PR to the public `staging` branch:

1. **Public repo** (`vla-audit.yml`) triggers `repository_dispatch` to private repo
2. **Private repo** (`shadow-tester.yml`) receives event:
   - Checks out private test suite
   - Fetches student code from public PR
   - Runs internal rigorous tests
3. **Results posted** back to public PR as comments

### Internal Test Structure

**Tests students can see** (`tests/public/`):
- Basic validation tests
- Check provided (non-TODO) components

**Tests students cannot see** (`tests/internal/`):
- **Gradient leak tests**: Verify frozen backbone parameters
- **Latent fidelity tests**: Compare against gold standard tensors
- **Training convergence tests**: Validate model can learn
- **Edge case tests**: Boundary conditions and error handling

### Running Tests Manually

```bash
# Run public tests (students can also run these)
pytest tests/public/ -v

# Run internal tests (auto-injects solutions)
pytest tests/internal/ -v

# Run specific test file
pytest tests/internal/test_scratch1_rigor.py -v
```

---

## Deployment Procedures

### Syncing to Public Repository

**Method: Orphan Push** (breaks git history link)

When you create a release tag in the private repo:

```bash
# Tag release
git tag release-scratch-1
git push origin release-scratch-1
```

GitHub Actions automatically:
1. Checks out private repo
2. Runs `scripts/sanitize.sh`:
   - Deletes `private/` and `tests/internal/`
   - Removes `[SOLUTION]` markers
   - Removes draft MDX blocks
   - Overwrites README.md with public version
3. Creates orphan branch (no history)
4. Force pushes to public repo `main` branch

**Workflow** (`.github/workflows/sync-to-public.yml`):
```yaml
- git checkout --orphan temp-public-branch
- git add -A
- git commit -m "Public Release: $(date)"
- git push public temp-public-branch:main --force
```

### Manual Deployment

If needed, deploy manually:

```bash
# SSH to production server
ssh -i ~/.ssh/id_ed25519_automation crh@ristoffer.ch

# Navigate to project
cd /var/www/vlm-robotics.dev

# Pull latest
git pull origin main

# Build
npm run build

# Deploy
cp -r out/* public_html/
```

### Deployment Script

```bash
# From local machine
./scripts/deploy.sh
```

This script:
1. Builds Next.js site locally
2. Syncs to remote server via rsync
3. Deploys to `https://www.vlm-robotics.dev`

---

## Server Configuration

### Apache Setup

**Configuration**: `/etc/apache2/sites-available/vlm-robotics.dev.conf`

```apache
<VirtualHost *:80>
    ServerName vlm-robotics.dev
    DocumentRoot /var/www/vlm-robotics.dev/public_html

    # Proxy staging requests to Next.js server
    ProxyPass /staging http://localhost:3002
    ProxyPassReverse /staging http://localhost:3002
    ProxyPreserveHost On

    <Directory /var/www/vlm-robotics.dev/public_html>
        AllowOverride All
        Require all granted
    </Directory>
</VirtualHost>
```

**Enable and restart:**
```bash
sudo a2ensite vlm-robotics.dev.conf
sudo a2enmod proxy proxy_http rewrite
sudo systemctl restart apache2
```

### Process Management with systemd

**Service file**: `/etc/systemd/system/vla-staging.service`

```ini
[Unit]
Description=VLA Foundations Staging (Next.js)
After=network.target

[Service]
Type=simple
User=crh
WorkingDirectory=/home/crh/vla-staging
ExecStart=/usr/bin/npm start -- -p 3002
Restart=always
RestartSec=10
Environment=NODE_ENV=production

[Install]
WantedBy=multi-user.target
```

**Manage service:**
```bash
sudo systemctl enable vla-staging
sudo systemctl start vla-staging
sudo systemctl status vla-staging
sudo journalctl -u vla-staging -f  # View logs
```

---

## Troubleshooting

### Linter Issues

**Semantic line breaks failing:**
```bash
# Bad (sentence on same line):
Multi-modality is the framework that enables a model to process and generate information across disparate data types. In robotics, this represents the shift from a robot that sees its environment versus one that can understand it.

# Good (one sentence per line):
Multi-modality is the framework that enables a model to process and generate information across disparate data types.
In robotics, this represents the shift from a robot that sees its environment versus one that can understand it.
```

**Git history check failing:**
```bash
# Student has merge commits
git log --oneline
# Should not see: "Merge branch 'main'" or "Merge branch 'staging'"

# Fix: Student needs to rebase
git rebase staging
git push --force-with-lease
```

### Preview Deployment Issues

**Preview not deploying:**
1. Check GitHub Actions: https://github.com/arpg/vla-foundations/actions
2. Verify linter passed (green checkmark)
3. Check PR comments for deployment link

**Student can't find preview:**
- Preview URL in `github-actions` bot comment
- Format: `https://arpg.github.io/vla-foundations/staging/pulls/{PR}/textbook/audits/staging/{username}/`

### Server Issues

**Check server status:**
```bash
ssh crh@ristoffer.ch "ps aux | grep next-server | grep -v grep"
```

**View logs:**
```bash
ssh crh@ristoffer.ch "tail -f ~/vla-staging-server.log"
```

**Restart server:**
```bash
ssh crh@ristoffer.ch "cd ~/vla-staging && pkill -f 'next start' && nohup npm start -- -p 3002 > ~/vla-staging-server.log 2>&1 &"
```

**Test endpoints:**
```bash
curl http://localhost:3002/
curl http://vlm-robotics.dev/staging/api/comments/23
```

### Solution Leak Detection

**Pre-commit hook blocked commit:**
```
🚫 COMMIT BLOCKED - SOLUTION LEAK DETECTED

The following files contain [SOLUTION] markers:
  - src/assignments/scratch-1/backbone.py

To fix:
  1. Run: python3 scripts/_sanitize_todos.py
  2. Review the changes
  3. Re-stage: git add <files>
  4. Commit again
```

**Verify clean before sync:**
```bash
python scripts/dev_utils.py --verify-clean
grep -r "\[SOLUTION\]" src/ content/
```

---

## Quick Reference

### Common Commands

**Review student PR:**
```bash
gh pr checkout {PR_NUMBER}
gh pr view {PR_NUMBER} --web
```

**Check linter locally:**
```bash
python3 scripts/audit_linter.py
```

**Promote audit to production:**
```bash
git mv content/textbook/audits/staging/{user}.mdx content/textbook/audits/{name}.mdx
git commit -m "Promote audit to production"
git push
```

**Merge to main:**
```bash
gh pr merge {PR_NUMBER} --merge
git checkout main && git merge staging && git push origin main
```

**Create release:**
```bash
git tag release-scratch-1
git push origin release-scratch-1
# Monitor: gh run list --workflow=sync-to-public.yml
```

**Check server:**
```bash
ssh crh@ristoffer.ch "ps aux | grep 'next start' | grep -v grep"
ssh crh@ristoffer.ch "tail -f ~/vla-staging-server.log"
```

---

## Assignment Lifecycle

### Creating a New Assignment

1. Create stub and solution in private repo
2. Mark solution blocks with `# [SOLUTION]`
3. Create assignment spec in `content/course/assignments/`
4. Wrap spec in draft block
5. Test with internal rigorous tests

### Releasing an Assignment

1. Remove draft block from MDX
2. Create release tag: `git tag release-scratch-2`
3. Push tag: `git push origin release-scratch-2`
4. GitHub Actions auto-sanitizes and syncs via orphan push
5. Verify public repo received update

### Grading Student Submissions

1. Student opens PR to `staging`
2. Shadow CI runs automatically
3. Review Shadow CI results in PR comments
4. Review code and report manually
5. Approve and merge when ready

---

**END OF INSTRUCTOR GUIDE**

For detailed technical implementation, see:
- `README.md` - Private repo overview
- `scripts/dev_utils.py` - Solution management code
- `scripts/sanitize.sh` - Sanitization script
- `.github/workflows/` - CI/CD workflows
