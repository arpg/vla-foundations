#!/usr/bin/env python3
"""
VLA Audit Linter - Enforces "Senior Staff" engineering standards for student paper audits.

Checks:
1. Semantic Line Breaks: Sentences should be on separate lines for easier PR commenting
2. Clean History: No "Merge branch" commits allowed in PR branch
"""

import sys
import os
import subprocess

def check_semantic_breaks(file_path):
    """Check for potential wall-of-text issues in MDX files."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    errors = []
    for i, line in enumerate(lines, start=1):
        # Skip frontmatter and code blocks
        stripped = line.strip()
        if stripped.startswith('---') or stripped.startswith('```') or stripped.startswith('#'):
            continue

        # Flag lines that are very long and contain multiple sentences
        if len(line) > 150 and (line.count('. ') > 1 or line.count('? ') > 1):
            errors.append(
                f"{file_path}:Line {i}: Potential wall-of-text detected. "
                "Use semantic line breaks (one sentence per line)."
            )
    return errors

def check_git_history():
    """Check if PR branch contains merge commits from main."""
    try:
        # Check if 'Merge branch' exists in the current branch's unique commits
        cmd = "git log main..HEAD --oneline"
        result = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode()
        if "Merge branch 'main'" in result or "Merge branch 'staging'" in result:
            return ["❌ Dirty history detected: Remove 'Merge branch' commits by rebasing."]
    except subprocess.CalledProcessError:
        # If git command fails (e.g., not in a git repo), skip this check
        pass
    return []

def find_audit_files():
    """Find all MDX audit files in the staging directory."""
    audit_dir = "content/textbook/audits/staging"
    if not os.path.exists(audit_dir):
        return []

    files = []
    for filename in os.listdir(audit_dir):
        if filename.endswith(".mdx") and filename != "README.mdx":
            files.append(os.path.join(audit_dir, filename))
    return files

if __name__ == "__main__":
    audit_files = find_audit_files()

    if not audit_files:
        print("ℹ️  No audit files found in content/textbook/audits/staging/")
        sys.exit(0)

    print(f"🔍 Linting {len(audit_files)} audit file(s)...")

    all_errors = []
    for f in audit_files:
        all_errors.extend(check_semantic_breaks(f))
    all_errors.extend(check_git_history())

    if all_errors:
        print("\n❌ Audit Linter Failed\n")
        print("\n".join(all_errors))
        print("\n📋 Fix these issues before requesting instructor review.")
        sys.exit(1)

    print("✅ Audit Linter Passed - All standards met!")
    sys.exit(0)
