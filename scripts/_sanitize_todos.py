#!/usr/bin/env python3
"""
Internal helper: Sanitize TODO comments by removing solution hints.

Replaces:
    # TODO: [SOLUTION] Use torch.rsqrt for efficiency
With:
    # TODO: Complete implementation

Also removes inline solution comments:
    result = torch.rsqrt(x)  # [SOLUTION]: More efficient than 1/sqrt
With:
    result = torch.rsqrt(x)

This script is run as part of the sanitization process before syncing to the public repo.
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple


# Pattern for TODO lines with [SOLUTION]
TODO_PATTERN = re.compile(r'#\s*TODO:\s*\[SOLUTION\].*$', re.MULTILINE)
TODO_REPLACEMENT = "# TODO: Complete implementation"

# Pattern for inline [SOLUTION] comments
INLINE_PATTERN = re.compile(r'\s*#\s*\[SOLUTION\]:?.*$', re.MULTILINE)


def sanitize_file(file_path: Path) -> Tuple[bool, int]:
    """
    Sanitize a single Python file by removing solution hints.

    Args:
        file_path: Path to the Python file to sanitize

    Returns:
        Tuple of (changed, num_changes) where changed is True if file was modified
    """
    if not file_path.exists():
        return False, 0

    content = file_path.read_text()
    original = content
    num_changes = 0

    # Remove TODO: [SOLUTION] lines
    content, todo_count = TODO_PATTERN.subn(TODO_REPLACEMENT, content)
    num_changes += todo_count

    # Remove inline [SOLUTION] comments
    content, inline_count = INLINE_PATTERN.subn('', content)
    num_changes += inline_count

    if content != original:
        file_path.write_text(content)
        return True, num_changes

    return False, 0


def sanitize_directory(directory: Path, exclude_patterns: List[str] = None) -> Tuple[int, int]:
    """
    Recursively sanitize all Python files in a directory.

    Args:
        directory: Directory to sanitize
        exclude_patterns: List of glob patterns to exclude (e.g., ['*_solution.py'])

    Returns:
        Tuple of (files_changed, total_changes)
    """
    if exclude_patterns is None:
        exclude_patterns = ['*_solution.py', '*backup*.py']

    files_changed = 0
    total_changes = 0

    for py_file in directory.rglob("*.py"):
        # Skip if file matches any exclude pattern
        if any(py_file.match(pattern) for pattern in exclude_patterns):
            continue

        changed, num_changes = sanitize_file(py_file)
        if changed:
            files_changed += 1
            total_changes += num_changes
            print(f"  ✓ Sanitized: {py_file.relative_to(directory.parent)} ({num_changes} changes)")

    return files_changed, total_changes


def main():
    """Main entry point for sanitization script."""
    # Assume script is in scripts/ directory
    project_root = Path(__file__).parent.parent

    # Sanitize src/assignments directory
    assignments_dir = project_root / "src" / "assignments"

    if not assignments_dir.exists():
        print(f"❌ Assignments directory not found: {assignments_dir}")
        sys.exit(1)

    print("Sanitizing TODO comments in assignments...")

    files_changed, total_changes = sanitize_directory(
        assignments_dir,
        exclude_patterns=['*_solution.py', '*backup*.py', '__pycache__/*']
    )

    if files_changed > 0:
        print(f"\n✅ Sanitized {files_changed} files ({total_changes} total changes)")
    else:
        print("\n✓ No solution hints found (already clean)")

    # Also sanitize content/ directory (MDX files might have Python code blocks)
    content_dir = project_root / "content"
    if content_dir.exists():
        print("\nChecking content/ directory...")

        # For content, just report if any [SOLUTION] markers found
        solution_markers = []
        for mdx_file in content_dir.rglob("*.mdx"):
            content = mdx_file.read_text()
            if '[SOLUTION]' in content:
                solution_markers.append(mdx_file)

        if solution_markers:
            print(f"⚠️  Found [SOLUTION] markers in {len(solution_markers)} MDX files:")
            for f in solution_markers:
                print(f"    - {f.relative_to(project_root)}")
            print("    (MDX sanitization handled by sanitize.sh)")


if __name__ == "__main__":
    main()
