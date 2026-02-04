# CI/CD Scripts

**Critical infrastructure scripts** used in GitHub Actions workflows.

## Contents

### Production Scripts

- **`manage_solutions.py`** - Inject/reset assignment solutions (used in testing)
- **`sanitize.sh`** - Main sanitization pipeline for public sync
- **`_sanitize_todos.py`** - Remove solution hints from code
- **`audit_linter.py`** - Validate paper audit MDX files

### Usage in CI/CD

| Script | Workflow | Purpose |
|--------|----------|---------|
| `audit_linter.py` | `vla-audit.yml` | Validate audit frontmatter |
| `sanitize.sh` | `sync-to-public.yml` | Remove private content |
| `_sanitize_todos.py` | `sync-to-public.yml` | Strip solution hints |
| `manage_solutions.py` | (local testing) | Inject/reset solutions |

### Critical Requirements

1. **Fail-Safe**: All scripts must return non-zero exit codes on failure
2. **Idempotent**: Can be run multiple times safely
3. **Validated**: Must pass linting before sync
4. **Documented**: Clear error messages and usage

## Development Scripts

Local development helpers are in `scripts/dev/`. These are **not** used in CI/CD.

## Modification Guidelines

Changes to scripts in this directory affect production workflows. Always:

1. Test locally first
2. Verify exit codes
3. Check GitHub Actions logs
4. Update documentation

## Security

These scripts handle sensitive operations:
- `sanitize.sh` - Removes private content before public sync
- `manage_solutions.py` - Manages private solutions

Never commit secrets or tokens to these scripts.
