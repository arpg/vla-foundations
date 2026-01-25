---
name: pre-flight
description: "Run VLA Guard audit and sanitization pipeline before push/PR"
---

# Pre-Flight Check

This command runs a comprehensive pre-flight check before pushing to the repository or creating a pull request.

## Execution Steps

### Step 1: Run VLA Guard Audit

First, invoke the VLA Guard skill to perform a comprehensive security audit:

```
Use the Skill tool to invoke the vla-guard skill.
```

Wait for the VLA Guard audit to complete.

### Step 2: Evaluate Audit Results

- If the audit **PASSES** (all checks green):
  - Proceed to Step 3

- If the audit **FAILS** (any check red):
  - **STOP** execution
  - Display the failure report
  - Provide remediation instructions
  - Exit without running sanitization

### Step 3: Run Sanitization Pipeline (Only if Audit Passed)

If and only if the VLA Guard audit passes, run the sanitization script:

```bash
bash scripts/sanitize.sh
```

This will:
1. Delete `private/` directory
2. Delete `tests/internal/` directory
3. Delete `scripts/dev_utils.py` and other dev scripts
4. Sanitize TODO comments (remove `[SOLUTION]` markers)
5. Sanitize MDX files (remove instructor notes and draft blocks)
6. Overwrite README.md with public version
7. Clean .gitignore
8. Stage all changes
9. Commit sanitized state

### Step 4: Final Verification

After sanitization, run a final check:

```bash
# Verify no solution markers remain
grep -r "\[SOLUTION\]" src/ content/ 2>/dev/null || echo "✓ Clean"

# Verify private directories are gone
[ ! -d "private/" ] && [ ! -d "tests/internal/" ] && echo "✓ Private directories removed"

# Show git status
git status
```

### Step 5: Summary Report

Provide a clear summary:

```
╔══════════════════════════════════════════════════════════════╗
║                    PRE-FLIGHT COMPLETE                       ║
╚══════════════════════════════════════════════════════════════╝

Phase 1: VLA Guard Audit
  ✅ No solution markers found
  ✅ No private files staged
  ✅ No leaks in git history
  ✅ Similarity check passed

Phase 2: Sanitization
  ✅ Private directories removed
  ✅ Solution markers sanitized
  ✅ Draft blocks removed
  ✅ README.md sanitized
  ✅ Changes committed

════════════════════════════════════════════════════════════════
🟢 READY FOR PUBLIC SYNC
════════════════════════════════════════════════════════════════

Next steps:
  1. Review changes: git show HEAD
  2. Tag release: git tag release-scratch-X
  3. Push tag: git push origin release-scratch-X
  4. GitHub Actions will orphan-push to public repo
```

## Usage

Run this command before:
- Creating a release tag
- Pushing to remote
- Creating a PR (if manual sync)
- Any operation that exposes code to public

```bash
# In Claude Code CLI
/pre-flight
```

## Notes

- This command combines VLA Guard audit with sanitization
- Sanitization only runs if audit passes (fail-safe)
- The orphan push to public repo is still handled by GitHub Actions on release tags
- This is a **local pre-flight check** before triggering the remote workflow
