#!/bin/bash
# Sanitize repository for public release
#
# This script removes all private content and solution hints before syncing to the public repo.
# It should be run automatically by GitHub Actions on release tags.
#
# Usage:
#   bash scripts/sanitize.sh
#
# What it does:
#   1. Deletes private/ and tests/internal/ directories
#   2. Deletes solution management scripts
#   3. Sanitizes TODO comments (removes [SOLUTION] hints)
#   4. Sanitizes MDX files (removes instructor notes)
#   5. Commits sanitized state
#

set -euo pipefail

echo "=== Sanitizing repository for public release ==="
echo ""

# Step 1: Delete private directories and scripts
echo "Step 1: Removing private directories and scripts..."
if [ -d "private/" ]; then
    rm -rf private/
    echo "  ✓ Removed: private/"
else
    echo "  ⊘ Not found: private/"
fi

if [ -d "tests/internal/" ]; then
    rm -rf tests/internal/
    echo "  ✓ Removed: tests/internal/"
else
    echo "  ⊘ Not found: tests/internal/"
fi

if [ -d "scripts/dev/" ]; then
    rm -rf scripts/dev/
    echo "  ✓ Removed: scripts/dev/"
else
    echo "  ⊘ Not found: scripts/dev/"
fi

if [ -f "scripts/dev_utils.py" ]; then
    rm -f scripts/dev_utils.py
    echo "  ✓ Removed: scripts/dev_utils.py"
else
    echo "  ⊘ Not found: scripts/dev_utils.py"
fi

# Remove instructor documentation files
for file in INSTRUCTOR.md INSTRUCTOR_GUIDE.md API_SETUP.md SETUP_WITH_GH_CLI.md QUICK_START_SSH.md; do
    if [ -f "$file" ]; then
        rm -f "$file"
        echo "  ✓ Removed: $file"
    fi
done

# Remove Claude Code skills and commands (instructor-only workflow automation)
if [ -d ".claude/" ]; then
    rm -rf .claude/
    echo "  ✓ Removed: .claude/"
fi

# Remove sync workflow itself (shouldn't be in public repo)
if [ -f ".github/workflows/sync-to-public.yml" ]; then
    rm -f .github/workflows/sync-to-public.yml
    echo "  ✓ Removed: .github/workflows/sync-to-public.yml"
fi

echo ""

# Step 2: Sanitize TODO comments in Python files
echo "Step 2: Sanitizing TODO comments..."
if [ -f "scripts/_sanitize_todos.py" ]; then
    python3 scripts/_sanitize_todos.py
else
    echo "  ⚠️  Warning: _sanitize_todos.py not found, skipping"
fi
echo ""

# Step 3: Sanitize MDX files (remove instructor notes and draft blocks)
echo "Step 3: Sanitizing MDX instructor notes and draft blocks..."
if [ -d "content/" ]; then
    # Find and remove instructor notes (comments between <!-- INSTRUCTOR NOTE and -->)
    find content/ -name "*.mdx" -type f -exec sed -i.bak '/<!-- INSTRUCTOR NOTE/,/-->/d' {} \;

    # Remove draft warning blocks in assignment files
    find content/course/assignments/ -name "*.mdx" -type f -exec sed -i.bak '/<div className="draft-warning">/,/<\/div>/d' {} \; 2>/dev/null || true

    # Alternative: Also remove blocks with explicit DRAFT marker
    find content/course/assignments/ -name "*.mdx" -type f -exec sed -i.bak '/⚠️ DRAFT: NOT YET ASSIGNED/,/^$/d' {} \; 2>/dev/null || true

    # Remove backup files
    find content/ -name "*.bak" -delete

    echo "  ✓ Instructor notes and draft blocks removed from MDX files"
else
    echo "  ⊘ Not found: content/"
fi
echo ""

# Step 4: Remove sanitization helper scripts (no longer needed in public repo)
echo "Step 4: Removing sanitization scripts..."
if [ -f "scripts/_sanitize_todos.py" ]; then
    rm -f scripts/_sanitize_todos.py
    echo "  ✓ Removed: scripts/_sanitize_todos.py"
fi

if [ -f "scripts/sanitize.sh" ]; then
    # Note: This script deletes itself!
    rm -f scripts/sanitize.sh
    echo "  ✓ Removed: scripts/sanitize.sh"
fi
echo ""

# Step 5: Overwrite README.md with public version
echo "Step 5: Overwriting README.md with public version..."
if [ -f "README.md" ]; then
    # Create a clean student-facing README
    cat > README.md << 'README_EOF'
# VLA Foundations: Course Repository

Vision-Language-Action Foundations for Robotics - CSCI 7000, Spring 2026

**Live Site**: https://www.vlm-robotics.dev

This repository contains the framework for CSCI 7000: VLA Foundations for Robotics.

---

## Workflow for Students

### Setup
Follow the Scratch-0 assignment to configure your environment.

### Branching
All work must be done on a branch named `[assignment]-[username]`.

**Example**: `scratch-1-heckman`

```bash
git checkout -b scratch-1-johndoe
```

### Implementation
- Code stubs are in `src/assignments/`
- Documentation and reports belong in `content/course/submissions/`

### Submission
Open a Pull Request to the `staging` branch. **Do not target `main`**.

1. Go to https://github.com/arpg/vla-foundations
2. Click "Pull requests" → "New pull request"
3. **Base branch**: `staging` (NOT `main`)
4. **Compare branch**: your branch name
5. Title: `Assignment X: Your Name`
6. Add a description of your work

### Review Process
1. Wait for CI checks to pass (GitHub Actions will validate your submission)
2. Wait for instructor review
3. Address any requested changes
4. **ONLY the instructor can merge pull requests**
5. Once approved, the instructor will merge to `staging`, then to `main`

**You do NOT have permission to merge your own PRs. All merges are done by the instructor.**

---

## Engineering Standards

### Semantic Line Breaks
Use semantic line breaks (one sentence per line) in all MDX files.

**Bad:**
```markdown
This is a very long sentence with multiple ideas. It continues on the same line. This makes PR review difficult.
```

**Good:**
```markdown
This is a sentence on its own line.
Each idea gets its own line.
This makes PR review much easier.
```

### Linear Git History
Maintain a linear git history.
Use `git rebase staging` instead of `git merge staging`.

**Example:**
```bash
# Update your branch with latest staging changes
git fetch origin
git rebase origin/staging

# If conflicts occur, resolve them and continue
git rebase --continue

# Force push your rebased branch
git push --force-with-lease
```

---

## Resources

### Documentation
- [MDX Syntax](https://mdxjs.com/)
- [Next.js App Router](https://nextjs.org/docs/app)
- [Tailwind CSS](https://tailwindcss.com/docs)
- [LaTeX Math Symbols](https://www.overleaf.com/learn/latex/List_of_Greek_letters_and_math_symbols)

### Papers & Datasets
- [RT-1 Paper](https://arxiv.org/abs/2212.06817)
- [RT-2 Paper](https://arxiv.org/abs/2307.15818)
- [Open-X Embodiment](https://robotics-transformer-x.github.io/)
- [DROID Dataset](https://droid-dataset.github.io/)

---

## Contact

- **Instructor**: Christoffer Heckman
- **Email**: christoffer.heckman@colorado.edu
- **Course**: CSCI 7000, Spring 2026
- **GitHub**: https://github.com/arpg/vla-foundations

---

## License

Copyright © 2026 Christoffer Heckman. All rights reserved.

Course materials are for educational use by enrolled students only.
README_EOF

    echo "  ✓ README.md overwritten with public version"
else
    echo "  ⚠️  Warning: README.md not found"
fi
echo ""

# Step 6: Update .gitignore (remove private entries since they're already deleted)
echo "Step 6: Cleaning .gitignore..."
if [ -f ".gitignore" ]; then
    # Remove lines related to private content
    sed -i.bak '/^private\//d' .gitignore
    sed -i.bak '/^tests\/internal\//d' .gitignore
    rm -f .gitignore.bak
    echo "  ✓ Cleaned .gitignore"
fi
echo ""

# Step 7: Stage all changes
echo "Step 7: Staging changes..."
git add -A
echo "  ✓ All changes staged"
echo ""

# Step 8: Commit sanitized state
echo "Step 8: Committing sanitized state..."
if git diff --cached --quiet; then
    echo "  ⊘ No changes to commit (already clean)"
else
    git commit -m "Sanitized release: $(date -u +%Y-%m-%d)" || {
        echo "  ⚠️  Commit failed (possibly no changes)"
    }
    echo "  ✓ Committed sanitized state"
fi
echo ""

echo "=== Sanitization complete ==="
echo ""
echo "Next steps:"
echo "  1. Review changes: git show HEAD"
echo "  2. Push to public repo: git push public HEAD:main --force"
echo ""
