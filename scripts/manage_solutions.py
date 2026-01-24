#!/usr/bin/env python3
"""
Manage assignment solutions for grading and internal testing.

This script allows instructors to inject complete solutions into the assignment
directories for testing, and then reset back to starter code.

Commands:
    --inject ASSIGNMENT: Copy solution files from private/solutions/ to src/assignments/
    --reset ASSIGNMENT: Restore starter code using git checkout
    --list: List all available solution files

Usage:
    python scripts/manage_solutions.py --inject scratch-1
    python scripts/manage_solutions.py --reset scratch-1
    python scripts/manage_solutions.py --list
"""

import argparse
import shutil
import subprocess
from pathlib import Path
import sys


class SolutionManager:
    """Manages solution files for assignments."""

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


def main():
    parser = argparse.ArgumentParser(
        description="Manage assignment solutions for internal testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inject solutions for scratch-1
  python scripts/manage_solutions.py --inject scratch-1

  # Run internal tests with solutions
  pytest tests/internal/test_scratch1_rigor.py -v

  # Reset to starter code
  python scripts/manage_solutions.py --reset scratch-1

  # List available solutions
  python scripts/manage_solutions.py --list
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

    args = parser.parse_args()

    # Check if any command was provided
    if not any([args.inject, args.reset, args.list]):
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

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
