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

# Step 1: Delete private directories
echo "Step 1: Removing private directories..."
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

if [ -f "scripts/manage_solutions.py" ]; then
    rm -f scripts/manage_solutions.py
    echo "  ✓ Removed: scripts/manage_solutions.py"
else
    echo "  ⊘ Not found: scripts/manage_solutions.py"
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

# Step 3: Sanitize MDX files (remove instructor notes)
echo "Step 3: Sanitizing MDX instructor notes..."
if [ -d "content/" ]; then
    # Find and remove instructor notes (comments between <!-- INSTRUCTOR NOTE and -->)
    find content/ -name "*.mdx" -type f -exec sed -i.bak '/<!-- INSTRUCTOR NOTE/,/-->/d' {} \;

    # Remove backup files
    find content/ -name "*.bak" -delete

    echo "  ✓ Instructor notes removed from MDX files"
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

# Step 5: Update .gitignore (remove private entries since they're already deleted)
echo "Step 5: Cleaning .gitignore..."
if [ -f ".gitignore" ]; then
    # Remove lines related to private content
    sed -i.bak '/^private\//d' .gitignore
    sed -i.bak '/^tests\/internal\//d' .gitignore
    rm -f .gitignore.bak
    echo "  ✓ Cleaned .gitignore"
fi
echo ""

# Step 6: Stage all changes
echo "Step 6: Staging changes..."
git add -A
echo "  ✓ All changes staged"
echo ""

# Step 7: Commit sanitized state
echo "Step 7: Committing sanitized state..."
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
