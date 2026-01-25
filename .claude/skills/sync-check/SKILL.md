---
name: sync-check
description: "Verify public repository has no leaks after synchronization"
user-invocable: true
---

# Sync Check: Post-Release Verification

This skill performs comprehensive verification of the public repository after a release sync to ensure no solution leaks occurred.

## Execution Steps

### Step 1: Check Recent Release Tags

List recent release tags to identify what was just synced:

```bash
echo "Recent release tags:"
git tag -l "release-scratch-*" | sort -V | tail -5
```

Prompt user:
```
Use AskUserQuestion to ask:
- Question: "Which release would you like to verify?"
- Header: "Release Tag"
- Options:
  - label: "Latest (Recommended)", description: "Verify most recent release"
  - label: "Specify tag", description: "Enter specific release tag"
```

Get the release tag to verify (e.g., `release-scratch-2`).

### Step 2: Clone Public Repository

Clone the public repo to a temporary location:

```bash
echo "=== Cloning Public Repository ==="

TEMP_DIR=$(mktemp -d)
echo "Temp directory: $TEMP_DIR"

cd "$TEMP_DIR"

# Clone public repo
gh repo clone arpg/vla-foundations public-repo
cd public-repo

# Get latest commit info
LATEST_COMMIT=$(git rev-parse HEAD)
COMMIT_MESSAGE=$(git log -1 --pretty=format:"%s")
COMMIT_DATE=$(git log -1 --pretty=format:"%ai")

echo ""
echo "Public Repository Status:"
echo "  Latest commit: $LATEST_COMMIT"
echo "  Message: $COMMIT_MESSAGE"
echo "  Date: $COMMIT_DATE"
echo ""
```

### Step 3: Scan for Solution Markers

Check for any [SOLUTION] markers that should have been sanitized:

```bash
echo "=== Scanning for Solution Markers ==="

LEAK_COUNT=0

# Check for [SOLUTION] markers
echo "Checking for [SOLUTION] markers..."
SOLUTION_MARKERS=$(grep -r '\[SOLUTION\]' src/ content/ 2>/dev/null || true)

if [ -n "$SOLUTION_MARKERS" ]; then
    echo "❌ CRITICAL: Found [SOLUTION] markers in public repo!"
    echo "$SOLUTION_MARKERS"
    LEAK_COUNT=$((LEAK_COUNT + 1))
else
    echo "✅ No [SOLUTION] markers found"
fi

# Check for TODO: [SOLUTION] patterns
echo ""
echo "Checking for TODO: [SOLUTION] patterns..."
TODO_SOLUTION=$(grep -r 'TODO:.*\[SOLUTION\]' src/ content/ 2>/dev/null || true)

if [ -n "$TODO_SOLUTION" ]; then
    echo "❌ CRITICAL: Found TODO: [SOLUTION] in public repo!"
    echo "$TODO_SOLUTION"
    LEAK_COUNT=$((LEAK_COUNT + 1))
else
    echo "✅ No TODO: [SOLUTION] patterns found"
fi
```

### Step 4: Check for Private Directories

Verify private directories were removed:

```bash
echo ""
echo "=== Checking for Private Directories ==="

# Check for private/
if [ -d "private/" ]; then
    echo "❌ CRITICAL: private/ directory exists in public repo!"
    ls -la private/
    LEAK_COUNT=$((LEAK_COUNT + 1))
else
    echo "✅ private/ directory not found (correct)"
fi

# Check for tests/internal/
if [ -d "tests/internal/" ]; then
    echo "❌ CRITICAL: tests/internal/ directory exists in public repo!"
    ls -la tests/internal/
    LEAK_COUNT=$((LEAK_COUNT + 1))
else
    echo "✅ tests/internal/ directory not found (correct)"
fi

# Check for scripts/dev/
if [ -d "scripts/dev/" ]; then
    echo "❌ CRITICAL: scripts/dev/ directory exists in public repo!"
    ls -la scripts/dev/
    LEAK_COUNT=$((LEAK_COUNT + 1))
else
    echo "✅ scripts/dev/ directory not found (correct)"
fi
```

### Step 5: Check for Private Scripts

Verify development utilities were removed:

```bash
echo ""
echo "=== Checking for Private Scripts ==="

# Check for dev_utils.py
if [ -f "scripts/dev_utils.py" ]; then
    echo "❌ CRITICAL: dev_utils.py exists in public repo!"
    LEAK_COUNT=$((LEAK_COUNT + 1))
else
    echo "✅ dev_utils.py not found (correct)"
fi

# Check for manage_solutions.py (old name)
if [ -f "scripts/manage_solutions.py" ]; then
    echo "❌ CRITICAL: manage_solutions.py exists in public repo!"
    LEAK_COUNT=$((LEAK_COUNT + 1))
else
    echo "✅ manage_solutions.py not found (correct)"
fi

# Check for sanitization scripts
if [ -f "scripts/sanitize.sh" ]; then
    echo "❌ WARNING: sanitize.sh exists in public repo (should be removed)"
    LEAK_COUNT=$((LEAK_COUNT + 1))
else
    echo "✅ sanitize.sh not found (correct)"
fi

if [ -f "scripts/_sanitize_todos.py" ]; then
    echo "❌ WARNING: _sanitize_todos.py exists in public repo"
    LEAK_COUNT=$((LEAK_COUNT + 1))
else
    echo "✅ _sanitize_todos.py not found (correct)"
fi
```

### Step 6: Check for Sensitive Files

Look for common sensitive file patterns:

```bash
echo ""
echo "=== Checking for Sensitive Files ==="

# Check for solution files (should all be in private repo)
SOLUTION_FILES=$(find . -name "*_solution.py" 2>/dev/null || true)

if [ -n "$SOLUTION_FILES" ]; then
    echo "❌ CRITICAL: Found solution files in public repo!"
    echo "$SOLUTION_FILES"
    LEAK_COUNT=$((LEAK_COUNT + 1))
else
    echo "✅ No *_solution.py files found"
fi

# Check for backup files
BACKUP_FILES=$(find . -name "*.backup.py" -o -name "*.bak" 2>/dev/null || true)

if [ -n "$BACKUP_FILES" ]; then
    echo "⚠️  WARNING: Found backup files in public repo"
    echo "$BACKUP_FILES"
fi

# Check for credential files
CRED_FILES=$(find . -name ".env" -o -name "credentials.json" -o -name "*.key" 2>/dev/null || true)

if [ -n "$CRED_FILES" ]; then
    echo "❌ CRITICAL: Found credential files in public repo!"
    echo "$CRED_FILES"
    LEAK_COUNT=$((LEAK_COUNT + 1))
else
    echo "✅ No credential files found"
fi
```

### Step 7: Check Git History

Verify orphan push strategy worked (no linked history):

```bash
echo ""
echo "=== Checking Git History ==="

# Count total commits (should be small if orphan push works)
COMMIT_COUNT=$(git rev-list --count HEAD)
echo "Total commits in public repo: $COMMIT_COUNT"

if [ $COMMIT_COUNT -gt 100 ]; then
    echo "⚠️  WARNING: Large commit count ($COMMIT_COUNT)"
    echo "   Orphan push may not be working correctly"
fi

# Check for commits with "solution" in message
SOLUTION_COMMITS=$(git log --all --oneline | grep -i "solution" || true)

if [ -n "$SOLUTION_COMMITS" ]; then
    echo "⚠️  WARNING: Found commits mentioning 'solution':"
    echo "$SOLUTION_COMMITS"
fi

# Get commit history summary
echo ""
echo "Recent commits:"
git log --oneline -10

# Check if history looks like orphan (no parents for root commit)
PARENT_COUNT=$(git rev-list --parents HEAD | tail -1 | wc -w)

if [ $PARENT_COUNT -eq 1 ]; then
    echo "✅ Orphan push confirmed (no parent commits)"
else
    echo "⚠️  WARNING: Root commit has $((PARENT_COUNT - 1)) parent(s)"
    echo "   Expected orphan commit with no parents"
fi
```

### Step 8: Compare File Lists

Compare files between private and public repos:

```bash
echo ""
echo "=== Comparing File Lists ==="

# Get file list from public repo
PUBLIC_FILES=$(find . -type f -not -path "./.git/*" | sort)

# Get file list from private repo
cd /Users/crh/projects/private-vla-foundations
PRIVATE_FILES=$(find . -type f -not -path "./.git/*" -not -path "./private/*" -not -path "./tests/internal/*" | sort)

# Save lists for comparison
echo "$PUBLIC_FILES" > /tmp/public-files.txt
echo "$PRIVATE_FILES" > /tmp/private-files.txt

# Files in public but not in private (suspicious)
PUBLIC_ONLY=$(comm -23 /tmp/public-files.txt /tmp/private-files.txt)

if [ -n "$PUBLIC_ONLY" ]; then
    echo "⚠️  Files in public repo but not in private:"
    echo "$PUBLIC_ONLY" | head -20
fi

# Files in private but not in public (expected - solutions, internal tests)
PRIVATE_ONLY=$(comm -13 /tmp/public-files.txt /tmp/private-files.txt)

echo ""
echo "Files in private repo but not in public (should include private/, tests/internal/):"
echo "$PRIVATE_ONLY" | grep -E "(private/|tests/internal/|scripts/dev_utils)" | head -10
```

### Step 9: Check Deployment Status

Verify the website is accessible and reflects the release:

```bash
echo ""
echo "=== Checking Deployment Status ==="

# Check production site
PROD_URL="https://www.vlm-robotics.dev"
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$PROD_URL")

if [ "$HTTP_STATUS" = "200" ]; then
    echo "✅ Production site is live"
    echo "   URL: $PROD_URL"
    echo "   HTTP: $HTTP_STATUS"
else
    echo "❌ Production site returned HTTP $HTTP_STATUS"
fi

# If we know the assignment, check its page
ASSIGNMENT_SLUG=$(echo "$RELEASE_TAG" | sed 's/release-//')

if [ -n "$ASSIGNMENT_SLUG" ]; then
    ASSIGNMENT_URL="$PROD_URL/textbook/assignments/$ASSIGNMENT_SLUG"
    ASSIGNMENT_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$ASSIGNMENT_URL")

    if [ "$ASSIGNMENT_STATUS" = "200" ]; then
        echo "✅ Assignment page is live"
        echo "   URL: $ASSIGNMENT_URL"
    else
        echo "⚠️  Assignment page returned HTTP $ASSIGNMENT_STATUS"
        echo "   URL: $ASSIGNMENT_URL"
    fi
fi

# Check build timestamp (if available in HTML)
BUILD_TIME=$(curl -s "$PROD_URL" | grep -o 'build-time">[^<]*' | sed 's/build-time">//' || echo "Unknown")
echo "   Build time: $BUILD_TIME"
```

### Step 10: Run Sample Fidelity Test

If possible, clone a student starter and verify it matches public:

```bash
echo ""
echo "=== Sample Fidelity Check ==="

# This is a spot check that starter code matches expectations
# Pick a known file and check its content

SAMPLE_FILE="src/assignments/scratch-1/backbone.py"

if [ -f "$SAMPLE_FILE" ]; then
    # Count TODOs (students should have TODOs to complete)
    TODO_COUNT=$(grep -c "TODO" "$SAMPLE_FILE" || echo "0")
    echo "Sample file: $SAMPLE_FILE"
    echo "  TODO count: $TODO_COUNT"

    if [ $TODO_COUNT -gt 0 ]; then
        echo "  ✅ Contains TODOs for students"
    else
        echo "  ⚠️  WARNING: No TODOs found (might be complete solution)"
    fi

    # Check for solution markers
    if grep -q '\[SOLUTION\]' "$SAMPLE_FILE"; then
        echo "  ❌ CRITICAL: Contains [SOLUTION] markers!"
        LEAK_COUNT=$((LEAK_COUNT + 1))
    else
        echo "  ✅ No solution markers"
    fi
fi
```

### Step 11: Generate Verification Report

Create detailed verification report:

```markdown
# Sync Verification Report: $RELEASE_TAG

**Verified**: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
**Public Repo**: https://github.com/arpg/vla-foundations
**Public Commit**: $LATEST_COMMIT
**Commit Message**: $COMMIT_MESSAGE

---

## Summary

**Overall Status**: ✅ PASS / ❌ FAIL

**Leak Count**: $LEAK_COUNT critical issues detected

---

## Verification Checks

### Solution Markers
- ✅ / ❌ No [SOLUTION] markers found
- ✅ / ❌ No TODO: [SOLUTION] patterns found

### Private Directories
- ✅ / ❌ private/ removed
- ✅ / ❌ tests/internal/ removed
- ✅ / ❌ scripts/dev/ removed

### Private Scripts
- ✅ / ❌ dev_utils.py removed
- ✅ / ❌ sanitize.sh removed
- ✅ / ❌ _sanitize_todos.py removed

### Sensitive Files
- ✅ / ❌ No *_solution.py files
- ✅ / ❌ No credential files
- ✅ / ❌ No backup files

### Git History
- ✅ / ❌ Orphan push confirmed
- ✅ / ❌ No solution-related commits
- Total commits: $COMMIT_COUNT

### Deployment
- ✅ / ❌ Production site live
- ✅ / ❌ Assignment page accessible
- Build time: $BUILD_TIME

---

## Detailed Findings

[If LEAK_COUNT > 0, list all issues found]

### Critical Issues
- Issue 1
- Issue 2

### Warnings
- Warning 1
- Warning 2

---

## File Comparison

**Files in public repo**: <count>
**Files in private repo (excluding private/)**: <count>
**Difference**: <count>

### Unexpected Public Files
[List files in public but not in private]

### Expected Private-Only Files
[List important files in private/ and tests/internal/]

---

## Git History Analysis

**Total commits**: $COMMIT_COUNT
**Orphan root**: ✅ / ❌
**Last 5 commits**:
```
[Git log output]
```

---

## Deployment Status

**Production URL**: https://www.vlm-robotics.dev
**HTTP Status**: $HTTP_STATUS
**Assignment URL**: https://www.vlm-robotics.dev/textbook/assignments/$ASSIGNMENT_SLUG
**Assignment Status**: $ASSIGNMENT_STATUS

---

## Recommendations

**If PASS**:
- ✅ Public repository is clean
- ✅ Safe to announce release to students
- ✅ Monitor student PRs with `/grade`

**If FAIL**:
- ❌ URGENT: Solution leak detected in public repo!
- ❌ Required Actions:
  1. [List required fixes]
  2. Re-run sanitization
  3. Force push corrected version
  4. Re-verify with `/sync-check`

---

**Verified by**: VLA Guard Sync Check v1.0
**Report saved**: `.claude/sync-reports/sync-check-$RELEASE_TAG.md`
```

Save to: `/Users/crh/projects/private-vla-foundations/.claude/sync-reports/sync-check-$(date +%Y%m%d-%H%M%S).md`

### Step 12: Cleanup

```bash
# Return to private repo
cd /Users/crh/projects/private-vla-foundations

# Remove temp directory
rm -rf "$TEMP_DIR"
rm -f /tmp/public-files.txt /tmp/private-files.txt

echo ""
echo "✓ Cleanup complete"
```

### Step 13: Display Summary

```
╔══════════════════════════════════════════════════════════════╗
║           SYNC VERIFICATION COMPLETE: $RELEASE_TAG           ║
╚══════════════════════════════════════════════════════════════╝

Release Tag:     $RELEASE_TAG
Public Commit:   $LATEST_COMMIT
Verified:        $(date)

════════════════════════════════════════════════════════════════

Verification Results:

  Solution Markers:   ✅ / ❌
  Private Dirs:       ✅ / ❌
  Private Scripts:    ✅ / ❌
  Sensitive Files:    ✅ / ❌
  Git History:        ✅ / ❌
  Deployment:         ✅ / ❌

Overall Status: ✅ PASS / ❌ FAIL
Leak Count: $LEAK_COUNT

════════════════════════════════════════════════════════════════

Public Repository:
  URL: https://github.com/arpg/vla-foundations
  Commit: $LATEST_COMMIT
  Files: <count>

Production Site:
  URL: https://www.vlm-robotics.dev
  Status: HTTP $HTTP_STATUS
  Assignment: $PROD_URL/textbook/assignments/$ASSIGNMENT_SLUG

════════════════════════════════════════════════════════════════

Report saved: .claude/sync-reports/sync-check-<timestamp>.md

[If PASS]
✅ Public repository verified clean - safe to announce to students

[If FAIL]
❌ CRITICAL: $LEAK_COUNT issues detected!
   Review report: cat .claude/sync-reports/sync-check-<timestamp>.md
   Fix issues and re-verify
```

## Error Handling

- If public repo clone fails: Check network, check repo URL
- If temp directory creation fails: Use fallback location
- If gh CLI not authenticated: Provide setup instructions
- If deployment check fails: May be expected during build time

## Notes

- This skill is critical after every release
- Always run after `/release` completes
- Save all reports for audit trail
- If leaks detected, immediately remediate
- Consider running weekly as preventive check
- Public repo clone is read-only (safe operation)
