---
name: sync-check
description: "Verify public repo after release"
---

# Sync Check Command

Comprehensive post-release verification.

## Usage

```
/sync-check
```

## Execution

Use the Skill tool to invoke the sync-check skill:

```
Skill: sync-check
```

This will:
1. Prompt for release tag to verify
2. Clone public repository (read-only)
3. Scan for solution markers
4. Check for private directories
5. Check for private scripts
6. Check for sensitive files
7. Verify git history (orphan push)
8. Compare file lists (private vs public)
9. Check deployment status
10. Run sample fidelity check
11. Generate verification report
12. Display summary

**IMPORTANT**: Always run this after `/release` to verify no leaks.

If any leaks detected, the report will provide remediation steps.
