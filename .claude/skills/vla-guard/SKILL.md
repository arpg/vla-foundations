---
name: vla-guard
description: "Final audit to prevent solution/internal test leaks before PR or Push."
user-invocable: true
---

# VLA Guard: Solution Leak Prevention Audit

This skill performs a comprehensive audit to prevent accidental leaking of solutions or internal tests to the public repository.

## Execution Steps

### Step 1: Identify All Solutions

Run the solution listing command to identify all valid solution files:

```bash
python3 scripts/dev_utils.py --list
```

Parse the output to extract all solution files and their corresponding target files.

### Step 2: Scan for Solution Content Leaks

Scan `src/assignments/` and `content/course/` for any content that matches solutions or contains solution markers:

1. **Check for [SOLUTION] markers**:
   ```bash
   grep -r "\[SOLUTION\]" src/assignments/ content/course/
   ```

   If any matches are found, flag them as CRITICAL leaks.

2. **Check for TODO: [SOLUTION] patterns**:
   ```bash
   grep -r "TODO:.*\[SOLUTION\]" src/assignments/ content/course/
   ```

   If any matches are found, flag them as CRITICAL leaks.

3. **Run automated similarity check**:
   ```bash
   python3 scripts/dev_utils.py --verify-clean
   ```

   This compares assignment files against solution files and detects high similarity (>80%) that indicates solution code is present.

### Step 3: Verify Private Directories Not Staged

Check that `private/` and `tests/internal/` directories are NOT staged in git:

```bash
git diff --cached --name-only | grep -E "^(private/|tests/internal/)"
```

If any files from these directories are staged:
- **CRITICAL**: Flag as leak attempt
- List all staged private files
- Recommend: `git reset HEAD private/ tests/internal/`

### Step 4: Check Git History for Accidental Commits

Examine the last 3 commits to ensure no solution files were accidentally committed then deleted:

```bash
# Get list of all files changed in last 3 commits
git log -3 --name-only --pretty=format:"" | sort -u

# Check if any solution files appear in history
git log -3 --name-only --pretty=format:"" | grep -E "(private/|tests/internal/|_solution\.py)"
```

If solution files are found in recent history:
- **CRITICAL**: Solution files were committed (even if later deleted)
- Git history preserves these files
- Recommend: Interactive rebase to remove from history OR squash commits

### Step 5: Check for Sensitive Configuration Leaks

Verify no sensitive files are staged:

```bash
git diff --cached --name-only | grep -E "(\.env|secrets|credentials|\.key|id_rsa)"
```

If sensitive files found:
- **WARNING**: Potential credential leak
- List files
- Recommend unstaging

## Output Format

Provide a clear summary:

```
╔══════════════════════════════════════════════════════════════╗
║                   VLA GUARD: AUDIT REPORT                    ║
╚══════════════════════════════════════════════════════════════╝

✅ PASSED: No [SOLUTION] markers found
✅ PASSED: Similarity check (no solution code in assignments)
✅ PASSED: No private directories staged
✅ PASSED: No solution files in recent git history
✅ PASSED: No sensitive files staged

════════════════════════════════════════════════════════════════
🟢 AUDIT PASSED - Safe to push/PR
════════════════════════════════════════════════════════════════
```

OR if issues found:

```
╔══════════════════════════════════════════════════════════════╗
║                   VLA GUARD: AUDIT REPORT                    ║
╚══════════════════════════════════════════════════════════════╝

❌ FAILED: [SOLUTION] markers found in:
   - src/assignments/scratch-1/backbone.py:42
   - src/assignments/scratch-1/backbone.py:89

⚠️  WARNING: Private files staged:
   - private/solutions/backbone_solution.py
   - tests/internal/test_scratch1_rigor.py

✅ PASSED: No solution files in recent git history

════════════════════════════════════════════════════════════════
🔴 AUDIT FAILED - DO NOT push/PR
════════════════════════════════════════════════════════════════

Required Actions:
1. Run: python3 scripts/_sanitize_todos.py
2. Run: git reset HEAD private/ tests/internal/
3. Re-run: /vla-guard
```

## Exit Behavior

- If ALL checks pass: Exit with success, safe to proceed
- If ANY check fails: Exit with error, block push/PR

## Integration with Sanitization

This skill should be run BEFORE `scripts/sanitize.sh` to catch issues early.
The custom `/pre-flight` command will invoke this skill and only run sanitization if the guard passes.

## Notes

- This skill uses the renamed `dev_utils.py` (formerly `manage_solutions.py`)
- Solution markers use the format `[SOLUTION]`, not `### SOLUTION`
- The similarity threshold for solution detection is 80% (defined in `dev_utils.py`)
