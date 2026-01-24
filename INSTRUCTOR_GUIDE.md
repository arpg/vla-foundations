# Instructor Guide: Paper Audit Review Workflow

## Overview

Students submit paper audits via Pull Requests to the `staging` branch. The system automatically:
1. Runs a standards linter
2. Deploys preview to GitHub Pages
3. Posts preview link in PR comments

Once approved, you promote the audit from `staging/` to production and merge to `main`.

---

## Student Submission Workflow

### 1. Student Creates PR

Students follow these steps (from `content/course/assignments/paper-audit.mdx`):

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

### 2. Automated Linter Check

The `vla-audit.yml` workflow automatically runs:

**Standards Checked:**
- ✅ Semantic line breaks (one sentence per line)
- ✅ Clean git history (no "Merge branch" commits)

**If linter fails:**
- Bot posts detailed instructions in PR comments
- Build is blocked until standards are met

**If linter passes:**
- Build deploys to: `https://arpg.github.io/vla-foundations/staging/pulls/{PR_NUMBER}/textbook/audits/staging/{username}/`
- Bot posts preview link in PR comments

### 3. Instructor Review

Navigate to the preview URL (posted in PR comments) to review the rendered audit.

**Add feedback directly in GitHub PR:**
- Use GitHub's native PR review features
- Comment on specific lines
- Request changes or approve

**Common feedback areas:**
- Mathematical rigor and LaTeX correctness
- Architectural depth and comparative analysis
- Citations and references
- Physical grounding critique (the "load-bearing" analysis)

---

## Promoting Audit to Production (Merge-Ready)

When an audit reaches "Level 3 (A - Mastery)" quality:

### Step 1: Move File from Staging to Production

```bash
# Checkout the PR branch
gh pr checkout {PR_NUMBER}

# Move the audit file to production location
git mv content/textbook/audits/staging/{username}.mdx content/textbook/audits/{canonical-name}.mdx

# Example:
git mv content/textbook/audits/staging/gyanigkali.mdx content/textbook/audits/multimodality_audit.mdx

# Commit the promotion
git commit -m "Promote audit to production: ready for merge"

# Push the change
git push
```

### Step 2: Merge PR to Staging

```bash
gh pr merge {PR_NUMBER} --merge --repo arpg/vla-foundations
```

This merges the PR into `staging` branch.

### Step 3: Merge Staging to Main

```bash
git checkout main
git pull origin main
git merge staging
git push origin main
```

### Step 4: Verify Production Deployment

The `Deploy to Production` workflow will automatically run and:
- Pull latest `main` branch
- Run `npm run build`
- Sync to production server with `rsync --delete`
- Deploy to: `https://vlm-robotics.dev/textbook/audits/{canonical-name}/`

Check the deployment: https://github.com/arpg/vla-foundations/actions/workflows/deploy.yml

---

## URL Structure

| Environment | Location | Audience |
|------------|----------|----------|
| **PR Preview** | `arpg.github.io/vla-foundations/staging/pulls/{PR}/...` | Student + Instructor |
| **Staging** | `vlm-robotics.dev/staging/...` | Testing only |
| **Production** | `vlm-robotics.dev/textbook/audits/{canonical-name}` | Public |

---

## File Structure

```
content/textbook/audits/
├── staging/                    # Student drafts under review
│   ├── gyanigkali.mdx         # PR #23 - in review
│   ├── johndoe.mdx            # PR #24 - in review
│   └── .gitkeep
├── multimodality_audit.mdx    # Published audit (merged from staging/)
├── siglip_audit.mdx           # Published audit
└── _template.mdx              # Template for students
```

**Important:** Files in `staging/` are **only visible in preview deployments**, not in production. Once approved, they **must be moved** to the root `audits/` directory to appear at `vlm-robotics.dev`.

---

## Grading Levels

### Level 1 (C): Basic Summary
- Correct frontmatter and basic formatting
- Summary of paper's main contributions
- Basic LaTeX rendering

### Level 2 (B): Technical Deep-Dive
- High technical depth with architectural details
- Correct mathematical formulations
- Sound critique of the model
- Proper citations

### Level 3 (A - Mastery): Merge-Ready
- Exhaustive technical analysis
- Comparative deep-dive across multiple papers
- Original insights on scaling laws and bottlenecks
- Physical grounding critique (semantic-motor gap, information decay)
- Professional "Senior Staff" engineering quality
- **Ready to merge to main as canonical reference**

---

## Troubleshooting

### Linter Keeps Failing

**Check line breaks:**
```bash
# The student's file should have semantic line breaks
# Bad:
Multi-modality is the framework that enables a model to process and generate information across disparate data types. In robotics, this represents the shift from a robot that sees its environment versus one that can understand it.

# Good:
Multi-modality is the framework that enables a model to process and generate information across disparate data types.
In robotics, this represents the shift from a robot that sees its environment versus one that can understand it.
```

**Check git history:**
```bash
# If student has merge commits:
git log --oneline
# Should not see: "Merge branch 'main'" or "Merge branch 'staging'"

# Student needs to rebase:
git rebase staging
git push --force-with-lease
```

### Preview Isn't Deploying

1. Check GitHub Actions: https://github.com/arpg/vla-foundations/actions
2. Verify linter passed (green checkmark)
3. Check PR comments for deployment link

### Production Page Still Shows Old Audit

This happens if the deploy script doesn't delete old files.

**Fix:** The deploy.yml now uses `rsync --delete` to properly sync files. Force a redeploy:
```bash
git commit --allow-empty -m "Trigger redeploy"
git push origin main
```

### Student Can't Find Their Preview

The preview URL is posted automatically in PR comments by the bot:
- Look for the comment from `github-actions` bot
- URL format: `https://arpg.github.io/vla-foundations/staging/pulls/{PR}/textbook/audits/staging/{username}/`

---

## Tips for Effective Reviews

1. **Use GitHub's PR review features** - Line-specific comments are best
2. **Require semantic line breaks** - Makes commenting much easier
3. **Check LaTeX rendering** - View the preview, not just the markdown
4. **Verify references** - Click through to cited papers
5. **Test on mobile** - Ensure equations and diagrams are responsive
6. **Look for the "load-bearing" analysis** - This is what distinguishes A-level work

---

## Quick Reference Commands

```bash
# Review a PR
gh pr checkout {PR_NUMBER}
gh pr view {PR_NUMBER} --web

# Check linter locally
python3 scripts/audit_linter.py

# Promote to production
git mv content/textbook/audits/staging/{user}.mdx content/textbook/audits/{name}.mdx
git commit -m "Promote audit to production"
git push

# Merge to staging and main
gh pr merge {PR_NUMBER} --merge
git checkout main && git merge staging && git push origin main
```
