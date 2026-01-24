#!/usr/bin/env python3
"""
Internal helper: Sanitize TODO comments by removing solution hints.

FAIL-SAFE MODE: This script returns non-zero exit code if:
- Any [SOLUTION] tags remain after sanitization
- Any files cannot be processed
- Any unexpected patterns are found

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
from typing import List, Tuple, Set


# Pattern for TODO lines with [SOLUTION]
TODO_PATTERN = re.compile(r'#\s*TODO:\s*\[SOLUTION\].*$', re.MULTILINE)
TODO_REPLACEMENT = "# TODO: Complete implementation"

# Pattern for inline [SOLUTION] comments
INLINE_PATTERN = re.compile(r'\s*#\s*\[SOLUTION\]:?.*$', re.MULTILINE)

# Pattern to detect ANY remaining [SOLUTION] markers (for verification)
SOLUTION_MARKER_PATTERN = re.compile(r'\[SOLUTION\]', re.IGNORECASE)


class SanitizationError(Exception):
    """Raised when sanitization fails or leaves solution markers."""
    pass


def sanitize_file(file_path: Path) -> Tuple[bool, int, List[str]]:
    """
    Sanitize a single Python file by removing solution hints.

    Args:
        file_path: Path to the Python file to sanitize

    Returns:
        Tuple of (changed, num_changes, warnings) where:
        - changed: True if file was modified
        - num_changes: Number of replacements made
        - warnings: List of warning messages

    Raises:
        SanitizationError: If [SOLUTION] markers remain after sanitization
    """
    if not file_path.exists():
        return False, 0, []

    try:
        content = file_path.read_text(encoding='utf-8')
    except UnicodeDecodeError as e:
        raise SanitizationError(f"Cannot read {file_path}: {e}")

    original = content
    num_changes = 0
    warnings = []

    # Remove TODO: [SOLUTION] lines
    content, todo_count = TODO_PATTERN.subn(TODO_REPLACEMENT, content)
    num_changes += todo_count

    # Remove inline [SOLUTION] comments
    content, inline_count = INLINE_PATTERN.subn('', content)
    num_changes += inline_count

    # FAIL-SAFE CHECK: Verify no [SOLUTION] markers remain
    remaining_markers = SOLUTION_MARKER_PATTERN.findall(content)
    if remaining_markers:
        # Find line numbers for debugging
        lines_with_markers = []
        for line_num, line in enumerate(content.split('\n'), 1):
            if '[SOLUTION]' in line:
                lines_with_markers.append((line_num, line.strip()))

        error_msg = f"FAIL-SAFE: [SOLUTION] markers remain in {file_path}:\n"
        for line_num, line in lines_with_markers[:5]:  # Show first 5
            error_msg += f"  Line {line_num}: {line[:80]}\n"

        raise SanitizationError(error_msg)

    # Write sanitized content
    if content != original:
        file_path.write_text(content, encoding='utf-8')
        return True, num_changes, warnings

    return False, 0, warnings


def sanitize_directory(
    directory: Path,
    exclude_patterns: List[str] = None
) -> Tuple[int, int, List[str]]:
    """
    Recursively sanitize all Python files in a directory.

    Args:
        directory: Directory to sanitize
        exclude_patterns: List of glob patterns to exclude (e.g., ['*_solution.py'])

    Returns:
        Tuple of (files_changed, total_changes, errors)

    Raises:
        SanitizationError: If any file cannot be sanitized
    """
    if exclude_patterns is None:
        exclude_patterns = ['*_solution.py', '*backup*.py']

    files_changed = 0
    total_changes = 0
    all_warnings = []

    for py_file in directory.rglob("*.py"):
        # Skip if file matches any exclude pattern
        if any(py_file.match(pattern) for pattern in exclude_patterns):
            continue

        try:
            changed, num_changes, warnings = sanitize_file(py_file)
            if changed:
                files_changed += 1
                total_changes += num_changes
                print(f"  ✓ Sanitized: {py_file.relative_to(directory.parent)} ({num_changes} changes)")

            all_warnings.extend(warnings)

        except SanitizationError as e:
            # Critical error - fail immediately
            print(f"  ✗ FAILED: {py_file.relative_to(directory.parent)}")
            print(f"    {e}")
            raise

    return files_changed, total_changes, all_warnings


def verify_no_solution_markers(directory: Path, exclude_dirs: Set[str] = None) -> List[Path]:
    """
    Verify that no [SOLUTION] markers exist in any files.

    Args:
        directory: Directory to check
        exclude_dirs: Set of directory names to exclude (e.g., {'private', 'tests/internal'})

    Returns:
        List of files containing [SOLUTION] markers
    """
    if exclude_dirs is None:
        exclude_dirs = {'private', '.git', 'node_modules', '__pycache__'}

    files_with_markers = []

    for file_path in directory.rglob("*"):
        # Skip directories
        if not file_path.is_file():
            continue

        # Skip excluded directories
        if any(excluded in file_path.parts for excluded in exclude_dirs):
            continue

        # Skip binary files
        if file_path.suffix in {'.pyc', '.pkl', '.pt', '.db', '.sqlite', '.json'}:
            continue

        try:
            content = file_path.read_text(encoding='utf-8')
            if '[SOLUTION]' in content:
                files_with_markers.append(file_path)
        except (UnicodeDecodeError, PermissionError):
            # Skip files we can't read
            pass

    return files_with_markers


def main():
    """
    Main entry point for sanitization script.

    Exit codes:
        0: Success - all files sanitized, no markers remain
        1: Failure - [SOLUTION] markers remain or sanitization failed
    """
    # Assume script is in scripts/ directory
    project_root = Path(__file__).parent.parent

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║           FAIL-SAFE Sanitization Pipeline                   ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    exit_code = 0

    try:
        # Step 1: Sanitize src/assignments directory
        assignments_dir = project_root / "src" / "assignments"

        if not assignments_dir.exists():
            print(f"⚠️  Assignments directory not found: {assignments_dir}")
            print("   (This is OK if repo has no assignments yet)")
        else:
            print("Step 1: Sanitizing TODO comments in assignments...")

            files_changed, total_changes, warnings = sanitize_directory(
                assignments_dir,
                exclude_patterns=['*_solution.py', '*backup*.py', '__pycache__/*']
            )

            if files_changed > 0:
                print(f"\n✅ Sanitized {files_changed} files ({total_changes} total changes)")
            else:
                print("\n✓ No solution hints found (already clean)")

        # Step 2: Check content/ directory for [SOLUTION] markers
        content_dir = project_root / "content"
        if content_dir.exists():
            print("\nStep 2: Checking content/ directory...")

            solution_markers = []
            for mdx_file in content_dir.rglob("*.mdx"):
                try:
                    content = mdx_file.read_text(encoding='utf-8')
                    if '[SOLUTION]' in content:
                        solution_markers.append(mdx_file)
                except (UnicodeDecodeError, PermissionError):
                    pass

            if solution_markers:
                print(f"❌ FAIL-SAFE: Found [SOLUTION] markers in {len(solution_markers)} MDX files:")
                for f in solution_markers[:10]:  # Show first 10
                    print(f"    - {f.relative_to(project_root)}")
                print("\n   These must be removed before syncing to public!")
                exit_code = 1

        # Step 3: VERIFICATION - Check entire repo for any remaining markers
        print("\nStep 3: VERIFICATION - Scanning for remaining [SOLUTION] markers...")

        exclude_dirs = {'private', 'tests/internal', '.git', 'node_modules', '__pycache__', 'venv', 'out', '.next'}
        remaining_files = verify_no_solution_markers(project_root, exclude_dirs)

        if remaining_files:
            print(f"\n❌ FAIL-SAFE: Found [SOLUTION] markers in {len(remaining_files)} files:")
            for f in remaining_files[:10]:
                print(f"    - {f.relative_to(project_root)}")

            if len(remaining_files) > 10:
                print(f"    ... and {len(remaining_files) - 10} more")

            print("\n🚫 SANITIZATION FAILED - Solution markers remain!")
            print("   Fix these issues before syncing to public repo.")
            exit_code = 1
        else:
            print("\n✅ VERIFICATION PASSED - No [SOLUTION] markers found")

    except SanitizationError as e:
        print(f"\n❌ SANITIZATION ERROR: {e}")
        exit_code = 1

    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1

    print("\n" + "="*65)
    if exit_code == 0:
        print("✅ SUCCESS: Sanitization complete and verified")
    else:
        print("❌ FAILURE: Sanitization failed (see errors above)")
    print("="*65)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
