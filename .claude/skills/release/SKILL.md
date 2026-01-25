---
name: release
description: "Safely create and publish assignment releases to public repo"
user-invocable: true
---

# Release: Safe Assignment Publishing Workflow

This skill orchestrates the complete release workflow with comprehensive safety checks.

## Execution Steps

### Step 1: Verify Current Branch

Ensure we're on the main branch:

```bash
CURRENT_BRANCH=$(git branch --show-current)

if [ "$CURRENT_BRANCH" != "main" ]; then
    echo "⚠️  Warning: Currently on branch '$CURRENT_BRANCH'"
    echo "   Releases should be created from 'main'"
fi
```

Prompt user:
```
Use AskUserQuestion to ask:
- Question: "Current branch: $CURRENT_BRANCH. Continue with release?"
- Header: "Branch Check"
- Options:
  - label: "Yes, continue", description: "Create release from current branch"
  - label: "Switch to main (Recommended)", description: "Switch to main branch first"
  - label: "Cancel", description: "Abort release"
```

If "Switch to main":
```bash
git checkout main
git pull origin main
```

### Step 2: Check for Uncommitted Changes

```bash
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "❌ ERROR: You have uncommitted changes"
    git status --short
fi
```

If changes detected:
```
Use AskUserQuestion to ask:
- Question: "Uncommitted changes detected. What would you like to do?"
- Header: "Uncommitted Changes"
- Options:
  - label: "Commit changes", description: "Commit before releasing"
  - label: "Stash changes", description: "Stash and continue"
  - label: "Cancel", description: "Abort release"
```

### Step 3: Run Pre-Flight Check

Invoke the VLA Guard skill:

```
Use the Skill tool to invoke: vla-guard
```

Wait for VLA Guard to complete.

**If VLA Guard FAILS**:
- Display failure report
- List all detected issues
- ABORT the release
- Provide remediation instructions

**If VLA Guard PASSES**:
- Proceed to next step

### Step 4: Prompt for Release Tag

Ask user for release information:

```
Use AskUserQuestion to ask:
- Question: "Which assignment are you releasing?"
- Header: "Assignment"
- Options:
  - label: "Scratch-1", description: "Release Scratch-1 assignment"
  - label: "Scratch-2", description: "Release Scratch-2 assignment"
  - label: "Scratch-3", description: "Release Scratch-3 assignment"
  - label: "Custom", description: "Enter custom tag name"
```

If custom tag:
```
Prompt user: "Enter release tag name (format: release-scratch-X):"
```

Validate tag format:
```bash
TAG_NAME="release-scratch-X"

# Check format
if [[ ! "$TAG_NAME" =~ ^release-scratch-[0-9]+$ ]]; then
    echo "⚠️  Warning: Tag doesn't match expected format 'release-scratch-X'"
fi

# Check if tag already exists
if git rev-parse "$TAG_NAME" >/dev/null 2>&1; then
    echo "❌ ERROR: Tag '$TAG_NAME' already exists!"
    echo "Existing tags:"
    git tag -l "release-scratch-*"
fi
```

### Step 5: Review Changes Since Last Release

Show what's new in this release:

```bash
# Find previous release tag
PREV_TAG=$(git tag -l "release-scratch-*" | sort -V | tail -n 1)

if [ -n "$PREV_TAG" ]; then
    echo "Changes since $PREV_TAG:"
    git log "$PREV_TAG..HEAD" --oneline --decorate

    echo ""
    echo "Files changed:"
    git diff "$PREV_TAG..HEAD" --name-status
else
    echo "This is the first release tag"
    git log --oneline --decorate | head -10
fi
```

Prompt for confirmation:
```
Use AskUserQuestion to ask:
- Question: "Review the changes above. Proceed with release?"
- Header: "Confirm Release"
- Options:
  - label: "Yes, create release (Recommended)", description: "Create tag and trigger sync"
  - label: "Show detailed diff", description: "Review full code changes"
  - label: "Cancel", description: "Abort release"
```

If "Show detailed diff":
```bash
git diff "$PREV_TAG..HEAD"
# Then re-prompt for confirmation
```

### Step 6: Run Sanitization Pipeline

Run the sanitization script:

```bash
echo "=== Running Sanitization Pipeline ==="
bash scripts/sanitize.sh
```

Capture the output and verify:
- ✅ `private/` removed
- ✅ `tests/internal/` removed
- ✅ `scripts/dev_utils.py` removed
- ✅ TODO comments sanitized
- ✅ MDX instructor notes removed
- ✅ README.md overwritten

If sanitization fails:
- Display error
- ABORT release
- Restore original state with `git reset --hard HEAD`

### Step 7: Verify Sanitization

Run post-sanitization checks:

```bash
echo "=== FAIL-SAFE: Verifying Sanitization ==="

LEAK_FOUND=0

# Check 1: [SOLUTION] markers
if grep -r '\[SOLUTION\]' src/ content/ 2>/dev/null; then
    echo "❌ LEAK: [SOLUTION] markers found!"
    LEAK_FOUND=1
fi

# Check 2: Private directories
if [ -d "private/" ] || [ -d "tests/internal/" ]; then
    echo "❌ LEAK: Private directories still exist!"
    LEAK_FOUND=1
fi

# Check 3: dev_utils.py
if [ -f "scripts/dev_utils.py" ]; then
    echo "❌ LEAK: dev_utils.py not removed!"
    LEAK_FOUND=1
fi

if [ $LEAK_FOUND -eq 1 ]; then
    echo ""
    echo "🚫 ABORTING RELEASE - Sanitization verification failed!"
    git reset --hard HEAD  # Restore original state
    exit 1
fi

echo "✅ Sanitization verified - no leaks detected"
```

### Step 8: Create Release Tag

Create and push the release tag:

```bash
# Create annotated tag
git tag -a "$TAG_NAME" -m "Release: $TAG_NAME

Assignment: <assignment-name>
Date: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
Commit: $(git rev-parse --short HEAD)

This tag triggers automated sync to public repository.
Sanitization pipeline will run via GitHub Actions.
"

# Show tag info
git show "$TAG_NAME" --no-patch
```

Confirm before pushing:
```
Use AskUserQuestion to ask:
- Question: "Tag created locally. Push to trigger public sync?"
- Header: "Push Tag"
- Options:
  - label: "Yes, push tag (Recommended)", description: "Push tag and trigger GitHub Actions"
  - label: "Preview workflow", description: "Show what will happen"
  - label: "Cancel", description: "Delete tag and abort"
```

If "Preview workflow":
```
Display:
╔══════════════════════════════════════════════════════════════╗
║                    RELEASE WORKFLOW PREVIEW                  ║
╚══════════════════════════════════════════════════════════════╝

1. Push tag to private repo
   → git push origin $TAG_NAME

2. GitHub Actions triggered (.github/workflows/sync-to-public.yml)
   → Checkout private repo
   → Run sanitization
   → Verify no leaks (fail-safe)
   → Create orphan branch
   → Force push to public repo main

3. Public repo updated
   → URL: https://github.com/arpg/vla-foundations
   → Branch: main
   → Commit: "Public Release: <timestamp>"

4. Deployment triggered (public repo)
   → Build Next.js site
   → Deploy to production
   → URL: https://www.vlm-robotics.dev

Estimated time: 3-5 minutes
```

If "Yes, push tag":
```bash
git push origin "$TAG_NAME"
```

### Step 9: Monitor GitHub Actions

Watch the workflow execution:

```bash
echo "=== Monitoring GitHub Actions ==="
echo "Tag pushed: $TAG_NAME"
echo ""

# Get the workflow run URL
echo "Opening GitHub Actions in browser..."
gh run watch --repo crheckman/private-vla-foundations

# Alternative: poll for status
sleep 10
gh run list --workflow=sync-to-public.yml --limit 1 --json status,conclusion,url
```

Display real-time status:
```
╔══════════════════════════════════════════════════════════════╗
║              MONITORING RELEASE: $TAG_NAME                   ║
╚══════════════════════════════════════════════════════════════╝

GitHub Actions Status:
  Workflow: sync-to-public.yml
  Status:   in_progress / queued / completed

Steps:
  ⏳ Checkout private repo
  ⏳ Run sanitization
  ⏳ Verify no leaks
  ⏳ Create orphan branch
  ⏳ Push to public repo

Watch live: <workflow-url>
```

Update status every 15 seconds until completion.

### Step 10: Verify Public Repo

Once workflow completes, verify the public repo:

```bash
echo "=== Verifying Public Repository ==="

# Clone public repo to temp location
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"
git clone https://github.com/arpg/vla-foundations.git
cd vla-foundations

# Check latest commit
echo "Latest commit on public main:"
git log -1 --oneline

# Verify sanitization
echo ""
echo "Checking for leaks in public repo:"

LEAK_FOUND=0

if grep -r '\[SOLUTION\]' . 2>/dev/null; then
    echo "❌ LEAK DETECTED: [SOLUTION] markers found in public repo!"
    LEAK_FOUND=1
fi

if [ -d "private/" ]; then
    echo "❌ LEAK DETECTED: private/ directory in public repo!"
    LEAK_FOUND=1
fi

if [ -f "scripts/dev_utils.py" ]; then
    echo "❌ LEAK DETECTED: dev_utils.py in public repo!"
    LEAK_FOUND=1
fi

cd /Users/crh/projects/private-vla-foundations
rm -rf "$TEMP_DIR"

if [ $LEAK_FOUND -eq 0 ]; then
    echo "✅ Public repo verification passed"
else
    echo "❌ Public repo verification FAILED!"
    echo "   URGENT: Solution leak detected in public repo!"
fi
```

### Step 11: Check Deployment Status

Verify the website deployed correctly:

```bash
echo "=== Checking Deployment Status ==="

# Check if site is reachable
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" https://www.vlm-robotics.dev)

if [ "$HTTP_STATUS" = "200" ]; then
    echo "✅ Production site is live"
    echo "   URL: https://www.vlm-robotics.dev"
else
    echo "⚠️  Warning: Site returned HTTP $HTTP_STATUS"
fi

# Check assignment page if applicable
ASSIGNMENT_URL="https://www.vlm-robotics.dev/textbook/assignments/<assignment-name>"
ASSIGNMENT_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$ASSIGNMENT_URL")

if [ "$ASSIGNMENT_STATUS" = "200" ]; then
    echo "✅ Assignment page accessible"
    echo "   URL: $ASSIGNMENT_URL"
fi
```

### Step 12: Generate Release Summary

Create a release summary document:

```markdown
# Release Summary: $TAG_NAME

**Released**: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
**Private Commit**: $(git rev-parse HEAD)
**Released By**: $(git config user.name)

---

## Pre-Flight Checks

- ✅ VLA Guard audit passed
- ✅ No uncommitted changes
- ✅ On main branch
- ✅ Sanitization successful

## Release Details

**Tag**: $TAG_NAME
**Assignment**: <assignment-name>
**Previous Release**: $PREV_TAG

### Changes Included

```
[Git log output from PREV_TAG..HEAD]
```

### Files Changed

```
[Git diff --name-status output]
```

---

## Verification Results

### Sanitization Pipeline
- ✅ private/ removed
- ✅ tests/internal/ removed
- ✅ dev_utils.py removed
- ✅ [SOLUTION] markers sanitized
- ✅ Instructor notes removed
- ✅ README.md sanitized

### Public Repository
- ✅ Orphan push successful
- ✅ No solution leaks detected
- ✅ Git history isolated

### Deployment
- ✅ Production site live
- ✅ Assignment page accessible
- URL: https://www.vlm-robotics.dev/textbook/assignments/<assignment-name>

---

## GitHub Actions

**Workflow**: sync-to-public.yml
**Status**: ✅ Success
**Duration**: X minutes
**URL**: <workflow-url>

---

## Next Steps

- [ ] Announce assignment to students
- [ ] Monitor student PRs
- [ ] Use `/grade` to review submissions

---

**Release completed successfully** ✅
```

Save to: `.claude/releases/release-$TAG_NAME.md`

### Step 13: Display Final Summary

```
╔══════════════════════════════════════════════════════════════╗
║            RELEASE COMPLETE: $TAG_NAME ✅                    ║
╚══════════════════════════════════════════════════════════════╝

Pre-Flight:          ✅ Passed
Sanitization:        ✅ Verified
Tag Created:         ✅ $TAG_NAME
GitHub Actions:      ✅ Success
Public Repo:         ✅ Verified clean
Deployment:          ✅ Live

════════════════════════════════════════════════════════════════

📍 Public Repository
   URL: https://github.com/arpg/vla-foundations
   Commit: <public-commit-hash>

🌐 Production Site
   URL: https://www.vlm-robotics.dev
   Status: ✅ Online

📄 Assignment Page
   URL: https://www.vlm-robotics.dev/textbook/assignments/<assignment>
   Status: ✅ Accessible

════════════════════════════════════════════════════════════════

Next Steps:
  1. View release summary: cat .claude/releases/release-$TAG_NAME.md
  2. Announce to students: <assignment-name> is now live
  3. Monitor submissions: gh pr list --base staging
  4. Grade PRs: /grade

Release time: X minutes
```

## Error Handling

- If VLA Guard fails: ABORT, display issues, provide fix instructions
- If sanitization fails: ABORT, restore state with `git reset --hard`
- If tag exists: ABORT, suggest incrementing version
- If GitHub Actions fails: Display logs, provide debugging steps
- If public verification fails: URGENT alert, manual intervention required

## Notes

- This skill requires `gh` CLI authenticated
- Always verify public repo after release
- Never skip VLA Guard - it's the first line of defense
- Orphan push strategy ensures no history leaks
- Release summaries are saved for record-keeping
