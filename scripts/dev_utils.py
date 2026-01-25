#!/usr/bin/env python3
"""
Development utilities for managing assignment solutions and verifying repository cleanliness.

This script provides tools for instructors to:
- Inject complete solutions for internal testing
- Reset to starter code
- Verify no solution leaks before public sync

Commands:
    --inject ASSIGNMENT: Copy solution files from private/solutions/ to src/assignments/
    --reset ASSIGNMENT: Restore starter code using git checkout
    --list: List all available solution files
    --verify-clean: Scan for solution leaks in public-facing code

Usage:
    python scripts/dev_utils.py --inject scratch-1
    python scripts/dev_utils.py --reset scratch-1
    python scripts/dev_utils.py --list
    python scripts/dev_utils.py --verify-clean
"""

import argparse
import shutil
import subprocess
from pathlib import Path
import sys
import difflib
from typing import List, Tuple, Set


class SolutionManager:
    """Manages solution files and verifies repository cleanliness."""

    def __init__(self, project_root: Path = None):
        if project_root is None:
            # Assume script is in scripts/ directory
            self.project_root = Path(__file__).parent.parent
        else:
            self.project_root = Path(project_root)

        self.solutions_dir = self.project_root / "private" / "solutions"
        self.assignments_dir = self.project_root / "src" / "assignments"

    def inject_solutions(self, assignment: str) -> None:
        """
        Copy solution files into src/assignments/, overwriting stubs.

        Args:
            assignment: Assignment name (e.g., 'scratch-1')
        """
        target_dir = self.assignments_dir / assignment

        if not target_dir.exists():
            raise ValueError(f"Assignment directory not found: {target_dir}")

        if not self.solutions_dir.exists():
            raise ValueError(f"Solutions directory not found: {self.solutions_dir}")

        # Find solution files
        solution_files = list(self.solutions_dir.glob("*_solution.py"))

        if not solution_files:
            print(f"⚠️  No solution files found in {self.solutions_dir}")
            return

        print(f"Injecting solutions for {assignment}...")

        # Map solution files to target files
        for solution_file in solution_files:
            # Remove '_solution' from filename
            target_name = solution_file.name.replace("_solution", "")
            target_path = target_dir / target_name

            # Backup original if it exists
            if target_path.exists():
                backup = target_path.with_suffix(f".backup{target_path.suffix}")
                shutil.copy2(target_path, backup)
                print(f"  📦 Backed up: {target_path.name} → {backup.name}")

            # Copy solution
            shutil.copy2(solution_file, target_path)
            print(f"  ✓ Injected: {solution_file.name} → {target_path.relative_to(self.project_root)}")

        print(f"\n✅ Solutions injected successfully!")

    def reset_solutions(self, assignment: str) -> None:
        """
        Restore starter code using git checkout.

        Args:
            assignment: Assignment name (e.g., 'scratch-1')
        """
        target_dir = self.assignments_dir / assignment

        if not target_dir.exists():
            raise ValueError(f"Assignment directory not found: {target_dir}")

        print(f"Resetting {assignment} to starter code...")

        # Use git checkout to restore original files
        result = subprocess.run(
            ["git", "checkout", "HEAD", "--", str(target_dir)],
            capture_output=True,
            text=True,
            cwd=self.project_root
        )

        if result.returncode == 0:
            print(f"✓ Reset: Restored starter code for {assignment}")

            # Clean up backup files
            backup_files = list(target_dir.glob("*.backup.*"))
            for backup in backup_files:
                backup.unlink()
                print(f"  🗑️  Removed backup: {backup.name}")

            print(f"\n✅ Reset complete!")
        else:
            print(f"✗ Error: {result.stderr}")
            raise RuntimeError("Git checkout failed")

    def list_solutions(self) -> None:
        """List all available solution files."""
        if not self.solutions_dir.exists():
            print(f"⚠️  Solutions directory not found: {self.solutions_dir}")
            print("     Create it with: mkdir -p private/solutions")
            return

        solutions = list(self.solutions_dir.glob("*_solution.py"))

        if not solutions:
            print(f"No solution files found in {self.solutions_dir}")
            print("\nExpected naming convention: <filename>_solution.py")
            print("Example: backbone_solution.py → backbone.py")
            return

        print(f"Available solutions in {self.solutions_dir.relative_to(self.project_root)}:")
        for sol in sorted(solutions):
            target_name = sol.name.replace("_solution", "")
            print(f"  • {sol.name} → {target_name}")

    def verify_clean(self) -> bool:
        """
        Verify that src/assignments/ contains no solution code.

        Scans all Python files in src/assignments/ and compares them against
        files in private/solutions/. If any significant matches are found,
        this indicates a potential solution leak.

        Returns:
            True if clean (no leaks detected), False otherwise

        Raises:
            SystemExit: Exits with code 1 if leaks detected
        """
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║           Verifying Clean Repository (No Solution Leaks)    ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        print()

        if not self.solutions_dir.exists():
            print("✓ No solutions directory found - repository is clean")
            return True

        if not self.assignments_dir.exists():
            print("✓ No assignments directory found - repository is clean")
            return True

        # Get all solution files
        solution_files = list(self.solutions_dir.glob("*_solution.py"))

        if not solution_files:
            print("✓ No solution files found - repository is clean")
            return True

        print(f"Scanning {self.assignments_dir.relative_to(self.project_root)} for solution leaks...")
        print()

        leaks_detected = False
        leak_report = []

        # For each solution file, check if corresponding file in assignments matches
        for solution_file in solution_files:
            # Get corresponding assignment file
            target_name = solution_file.name.replace("_solution", "")

            # Find all matching files in assignments directory
            assignment_files = list(self.assignments_dir.rglob(target_name))

            for assignment_file in assignment_files:
                similarity = self._compare_files(solution_file, assignment_file)

                # If similarity > 80%, likely contains solution code
                if similarity > 0.8:
                    leaks_detected = True
                    leak_report.append({
                        'solution': solution_file.relative_to(self.project_root),
                        'assignment': assignment_file.relative_to(self.project_root),
                        'similarity': similarity
                    })
                    print(f"  ❌ LEAK DETECTED: {assignment_file.relative_to(self.project_root)}")
                    print(f"     Matches {solution_file.name} at {similarity:.1%} similarity")
                elif similarity > 0.5:
                    print(f"  ⚠️  Warning: {assignment_file.relative_to(self.project_root)}")
                    print(f"     Partial match with {solution_file.name} ({similarity:.1%})")
                else:
                    print(f"  ✓ Clean: {assignment_file.relative_to(self.project_root)}")

        print()
        print("="*65)

        if leaks_detected:
            print("❌ VERIFICATION FAILED - Solution leaks detected!")
            print()
            print("The following files contain solution code:")
            for leak in leak_report:
                print(f"  • {leak['assignment']} ({leak['similarity']:.1%} match)")
            print()
            print("Actions required:")
            print("  1. Run: python scripts/dev_utils.py --reset <assignment>")
            print("  2. Or manually remove solution code from the files above")
            print("  3. Re-run: python scripts/dev_utils.py --verify-clean")
            print()
            sys.exit(1)
        else:
            print("✅ VERIFICATION PASSED - No solution leaks detected")
            print()
            return True

    def _compare_files(self, file1: Path, file2: Path) -> float:
        """
        Compare two Python files and return similarity ratio.

        Uses difflib.SequenceMatcher to compare file contents after
        normalizing whitespace and removing comments.

        Args:
            file1: Path to first file
            file2: Path to second file

        Returns:
            Similarity ratio between 0.0 and 1.0
        """
        try:
            content1 = self._normalize_code(file1.read_text())
            content2 = self._normalize_code(file2.read_text())

            # Use SequenceMatcher to compare
            matcher = difflib.SequenceMatcher(None, content1, content2)
            return matcher.ratio()

        except Exception as e:
            print(f"  ⚠️  Error comparing files: {e}")
            return 0.0

    def _normalize_code(self, code: str) -> str:
        """
        Normalize Python code for comparison.

        Removes:
        - Comments (including [SOLUTION] markers)
        - Docstrings
        - Extra whitespace

        Args:
            code: Raw Python code

        Returns:
            Normalized code string
        """
        lines = []
        in_docstring = False
        docstring_char = None

        for line in code.split('\n'):
            stripped = line.strip()

            # Skip empty lines
            if not stripped:
                continue

            # Handle docstrings
            if stripped.startswith('"""') or stripped.startswith("'''"):
                if not in_docstring:
                    in_docstring = True
                    docstring_char = stripped[:3]
                    if stripped.endswith(docstring_char) and len(stripped) > 3:
                        in_docstring = False
                else:
                    if stripped.endswith(docstring_char):
                        in_docstring = False
                continue

            if in_docstring:
                continue

            # Skip comment-only lines
            if stripped.startswith('#'):
                continue

            # Remove inline comments
            if '#' in stripped:
                code_part = stripped.split('#')[0].rstrip()
                if code_part:
                    lines.append(code_part)
            else:
                lines.append(stripped)

        return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Development utilities for managing solutions and verifying cleanliness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inject solutions for testing
  python scripts/dev_utils.py --inject scratch-1
  pytest tests/internal/test_scratch1_rigor.py -v

  # Reset to starter code
  python scripts/dev_utils.py --reset scratch-1

  # List available solutions
  python scripts/dev_utils.py --list

  # Verify no solution leaks (before public sync)
  python scripts/dev_utils.py --verify-clean
        """
    )

    parser.add_argument(
        "--inject",
        metavar="ASSIGNMENT",
        help="Inject solutions for assignment (e.g., scratch-1)"
    )
    parser.add_argument(
        "--reset",
        metavar="ASSIGNMENT",
        help="Reset to starter code"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available solutions"
    )
    parser.add_argument(
        "--verify-clean",
        action="store_true",
        help="Verify no solution leaks in src/assignments/"
    )

    args = parser.parse_args()

    # Check if any command was provided
    if not any([args.inject, args.reset, args.list, args.verify_clean]):
        parser.print_help()
        sys.exit(0)

    # Create manager
    manager = SolutionManager()

    try:
        if args.inject:
            manager.inject_solutions(args.inject)
        elif args.reset:
            manager.reset_solutions(args.reset)
        elif args.list:
            manager.list_solutions()
        elif args.verify_clean:
            manager.verify_clean()

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
