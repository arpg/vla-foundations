#!/usr/bin/env python3
"""
VLA Audit Linter - Enforces "Senior Staff" engineering standards for student paper audits.

Checks:
1. Semantic Line Breaks: Sentences should be on separate lines for easier PR commenting
2. Clean History: No "Merge branch" commits allowed in PR branch
3. MDX Syntax: Proper frontmatter, no HTML comments, escaped angle brackets
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

def validate_frontmatter(file_path, content, lines):
    """Validate YAML frontmatter contains required fields."""
    errors = []

    # Extract frontmatter
    if not content.startswith('---'):
        errors.append(
            f"{file_path}: Missing YAML frontmatter. File must start with '---' followed by "
            "title, author, paper, and topic fields."
        )
        return errors

    # Find the end of frontmatter
    frontmatter_end = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == '---':
            frontmatter_end = i
            break

    if frontmatter_end is None:
        errors.append(
            f"{file_path}: Malformed YAML frontmatter. Missing closing '---'."
        )
        return errors

    frontmatter_lines = lines[1:frontmatter_end]
    frontmatter_text = '\n'.join(frontmatter_lines)

    # Required fields for audit MDX files
    required_fields = ['title', 'author', 'topic', 'paper']

    for field in required_fields:
        # Check if field exists (case-insensitive)
        if not any(line.strip().lower().startswith(f'{field}:') for line in frontmatter_lines):
            errors.append(
                f"{file_path}: Missing required frontmatter field: '{field}'"
            )

    # Validate field values are not empty
    for line in frontmatter_lines:
        stripped = line.strip()
        if ':' in stripped:
            field_name, field_value = stripped.split(':', 1)
            field_name = field_name.strip().lower()
            field_value = field_value.strip()

            if field_name in required_fields:
                # Check for empty values or placeholder values
                if not field_value or field_value in ['""', "''", 'null', 'TBD', 'TODO']:
                    errors.append(
                        f"{file_path}: Empty or placeholder value for required field: '{field_name}'"
                    )

    return errors


def check_mdx_syntax(file_path):
    """Check for MDX-specific syntax issues."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.splitlines()

    errors = []

    # Check 1: Validate frontmatter fields
    errors.extend(validate_frontmatter(file_path, content, lines))

    # Check 2: No HTML comments (should use JSX-style {/* */})
    if '<!--' in content:
        for i, line in enumerate(lines, start=1):
            if '<!--' in line:
                errors.append(
                    f"{file_path}:Line {i}: HTML comment detected. "
                    "MDX requires JSX-style comments: {{/* comment */}}"
                )

    # Check 3: Unescaped angle brackets outside code blocks
    in_code_block = False
    in_frontmatter = False
    frontmatter_count = 0

    for i, line in enumerate(lines, start=1):
        # Track code blocks
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            continue

        # Track frontmatter
        if line.strip() == '---':
            frontmatter_count += 1
            if frontmatter_count <= 2:
                in_frontmatter = not in_frontmatter
            continue

        # Skip if in code block or frontmatter
        if in_code_block or in_frontmatter:
            continue

        # Check for problematic angle brackets (not in LaTeX $...$ or proper HTML tags)
        # Look for < followed by a number or special char (like <! or <?)
        if '<' in line:
            # Remove LaTeX expressions
            cleaned = line
            import re
            # Remove inline math
            cleaned = re.sub(r'\$[^\$]+\$', '', cleaned)
            # Remove display math
            cleaned = re.sub(r'\$\$[^\$]+\$\$', '', cleaned)
            # Remove code spans
            cleaned = re.sub(r'`[^`]+`', '', cleaned)

            # Now check for problematic patterns
            if re.search(r'<[0-9!?]', cleaned):
                errors.append(
                    f"{file_path}:Line {i}: Unescaped angle bracket detected. "
                    "Use HTML entities (&lt;) or wrap in code backticks."
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
        all_errors.extend(check_mdx_syntax(f))
        all_errors.extend(check_semantic_breaks(f))
    all_errors.extend(check_git_history())

    if all_errors:
        print("\n❌ Audit Linter Failed\n")
        print("\n".join(all_errors))
        print("\n📋 Fix these issues before requesting instructor review.")
        sys.exit(1)

    print("✅ Audit Linter Passed - All standards met!")
    sys.exit(0)
