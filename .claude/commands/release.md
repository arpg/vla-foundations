---
name: release
description: "Create and publish assignment release"
---

# Release Command

Safe assignment publishing workflow with comprehensive checks.

## Usage

```
/release
```

## Execution

Use the Skill tool to invoke the release skill:

```
Skill: release
```

This will:
1. Verify current branch (main)
2. Check for uncommitted changes
3. Run VLA Guard pre-flight audit
4. Prompt for release tag
5. Review changes since last release
6. Run sanitization pipeline
7. Verify sanitization (fail-safe)
8. Create and push release tag
9. Monitor GitHub Actions workflow
10. Verify public repository
11. Check deployment status
12. Generate release summary

**IMPORTANT**: Only run this when ready to publish to students.
