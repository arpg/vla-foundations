#!/usr/bin/env python3
"""
Autograder for Scratch-1: The Transformer Backbone
CSCI 7000 - VLA Foundations

Usage:
    python scripts/grade_scratch1.py --pr 53
    python scripts/grade_scratch1.py --batch 34,35,36,37
    python scripts/grade_scratch1.py --pr 53 --dry-run

Author: Chris's Grading Assistant (Chris-Bot)
"""

import subprocess
import argparse
import json
import re
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from datetime import datetime


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TestResult:
    """Single test result"""
    name: str
    passed: bool
    points_possible: int
    points_earned: int
    feedback: str
    traceback: Optional[str] = None
    category: str = "other"


@dataclass
class GradingReport:
    """Complete grading report"""
    pr_number: int
    author: str
    branch_name: str
    total_points: int
    max_points: int = 100
    test_results: List[TestResult] = field(default_factory=list)
    has_report: bool = False
    report_files: List[str] = field(default_factory=list)
    has_mastery: bool = False
    mastery_features: List[str] = field(default_factory=list)
    critical_failure: Optional[str] = None
    execution_time: float = 0.0


# ============================================================================
# GIT OPERATIONS
# ============================================================================

class GitManager:
    """Handles all git operations safely"""

    BOT_NAME = "crh-bot"
    BOT_EMAIL = "260777175+crh-bot@users.noreply.github.com"

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.original_branch = None
        self.had_stash = False
        self.original_git_name = None
        self.original_git_email = None

    def set_bot_identity(self):
        """Set git config to Chris Bot for commits on student branches"""
        # Save original config
        for field, attr in [("user.name", "original_git_name"), ("user.email", "original_git_email")]:
            result = subprocess.run(
                ["git", "config", field], cwd=self.repo_path,
                capture_output=True, text=True
            )
            setattr(self, attr, result.stdout.strip() if result.returncode == 0 else None)

        subprocess.run(["git", "config", "user.name", self.BOT_NAME], cwd=self.repo_path, check=True)
        subprocess.run(["git", "config", "user.email", self.BOT_EMAIL], cwd=self.repo_path, check=True)

    def restore_identity(self):
        """Restore original git config after bot operations"""
        for field, val in [("user.name", self.original_git_name), ("user.email", self.original_git_email)]:
            if val:
                subprocess.run(["git", "config", field, val], cwd=self.repo_path)

    def bot_commit(self, message: str, files: List[str] = None):
        """Make a commit as Chris Bot on the current branch"""
        self.set_bot_identity()
        try:
            if files:
                subprocess.run(["git", "add"] + files, cwd=self.repo_path, check=True)
            else:
                subprocess.run(["git", "add", "-A"], cwd=self.repo_path, check=True)
            subprocess.run(
                ["git", "commit", "-m", message],
                cwd=self.repo_path, capture_output=True, text=True, check=True
            )
            print(f"  🤖 Chris Bot committed: {message}")
        finally:
            self.restore_identity()

    def get_current_branch(self) -> str:
        """Get current git branch"""
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()

    def fetch_pr_info(self, pr_number: int) -> Dict[str, str]:
        """Fetch PR information from GitHub"""
        try:
            result = subprocess.run(
                ["gh", "pr", "view", str(pr_number), "--json", "number,title,author,headRefName,state"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to fetch PR #{pr_number}: {e.stderr}")

    def checkout_pr_branch(self, pr_number: int) -> Tuple[str, str]:
        """
        Safely checkout PR branch
        Returns: (branch_name, author)
        """
        # Save current state
        self.original_branch = self.get_current_branch()

        # Check for uncommitted changes
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )

        if result.stdout.strip():
            print("  📦 Stashing uncommitted changes...")
            subprocess.run(["git", "stash"], cwd=self.repo_path, check=True)
            self.had_stash = True

        # Fetch PR info
        pr_info = self.fetch_pr_info(pr_number)
        branch_name = pr_info["headRefName"]
        author = pr_info["author"]["login"]

        # Fetch and checkout PR branch
        print(f"  🔄 Fetching PR branch: {branch_name}")
        subprocess.run(
            ["git", "fetch", "origin", f"pull/{pr_number}/head:{branch_name}"],
            cwd=self.repo_path,
            check=True,
            capture_output=True
        )

        print(f"  ✓ Checking out branch: {branch_name}")
        subprocess.run(
            ["git", "checkout", branch_name],
            cwd=self.repo_path,
            check=True,
            capture_output=True
        )

        return branch_name, author

    def restore_original_state(self):
        """Restore to original branch and unstash if needed"""
        if self.original_branch:
            print(f"  ↩ Returning to branch: {self.original_branch}")
            # Use -f to force checkout, discarding any injected changes
            subprocess.run(
                ["git", "checkout", "-f", self.original_branch],
                cwd=self.repo_path,
                check=True,
                capture_output=True
            )

        if self.had_stash:
            print("  📦 Restoring stashed changes...")
            result = subprocess.run(
                ["git", "stash", "pop"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"  ⚠️  Warning: Stash pop had conflicts. Resolve manually if needed.")


# ============================================================================
# TEST EXECUTION
# ============================================================================

class TestRunner:
    """Runs pytest remotely on arpg-4090 (RTX 4090 GPU) and captures results"""

    REMOTE_HOST = "arpg-4090"
    REMOTE_REPO = "~/vla-foundations"

    def __init__(self, repo_path: Path, git_manager: 'GitManager' = None):
        self.repo_path = repo_path
        self.test_file = repo_path / "tests" / "internal" / "test_scratch1_rigor.py"
        self.git_manager = git_manager
        self._temp_branch = None

    def inject_internal_tests(self):
        """Copy internal tests and dependencies from main branch to current branch"""
        try:
            # Inject test files
            (self.repo_path / "tests" / "internal").mkdir(parents=True, exist_ok=True)

            for filename in ["test_scratch1_rigor.py", "__init__.py"]:
                result = subprocess.run(
                    ["git", "show", f"main:tests/internal/{filename}"],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True,
                    check=True
                )
                (self.repo_path / "tests" / "internal" / filename).write_text(result.stdout)

            # Inject conftest.py
            result = subprocess.run(
                ["git", "show", f"main:tests/conftest.py"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            (self.repo_path / "tests" / "conftest.py").write_text(result.stdout)

            # Inject pyproject.toml for platform-specific deps
            result = subprocess.run(
                ["git", "show", f"main:pyproject.toml"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            (self.repo_path / "pyproject.toml").write_text(result.stdout)

            # Remove old lock file (remote will regenerate)
            lock_file = self.repo_path / "uv.lock"
            if lock_file.exists():
                lock_file.unlink()

            print(f"  ✓ Internal tests and dependencies injected")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to inject tests/dependencies: {e.stderr}")

    def run_tests(self) -> Dict[str, any]:
        """
        Push current state to a temp branch, run pytest on arpg-4090 via SSH,
        parse and return results, then clean up the temp branch.
        """
        self._temp_branch = f"_grading-temp-{int(time.time())}"

        try:
            # Stage and commit all injected files with bot identity
            print(f"  📤 Pushing to temp branch: {self._temp_branch}")
            if self.git_manager:
                self.git_manager.bot_commit(
                    f"[grading] temp commit for remote test run",
                )
            else:
                # Fallback: commit without bot identity
                subprocess.run(["git", "add", "-A"], cwd=self.repo_path, check=True)
                subprocess.run(
                    ["git", "commit", "--allow-empty", "-m", "[grading] temp commit for remote test run"],
                    cwd=self.repo_path, capture_output=True, text=True
                )

            # Push to temp branch
            subprocess.run(
                ["git", "push", "origin", f"HEAD:{self._temp_branch}", "--force"],
                cwd=self.repo_path, capture_output=True, text=True, check=True,
            )

            # Run tests remotely via SSH
            ssh_cmd = (
                f"source ~/.local/bin/env 2>/dev/null; "
                f"cd {self.REMOTE_REPO} && "
                f"git fetch origin {self._temp_branch} && "
                f"git checkout -f FETCH_HEAD && "
                f"rm -f uv.lock && uv sync && "
                f"uv run pytest tests/internal/test_scratch1_rigor.py -v -m rigor --tb=short"
            )

            print(f"  🧪 Running tests on {self.REMOTE_HOST}...")
            result = subprocess.run(
                ["ssh", self.REMOTE_HOST, ssh_cmd],
                capture_output=True, text=True, timeout=300,
            )

            return self._parse_text_output(result)

        finally:
            # Clean up temp branch from remote
            self._cleanup_temp_branch()

    def _cleanup_temp_branch(self):
        """Delete the temporary branch from the remote"""
        if self._temp_branch:
            print(f"  🧹 Cleaning up temp branch: {self._temp_branch}")
            subprocess.run(
                ["git", "push", "origin", "--delete", self._temp_branch],
                cwd=self.repo_path, capture_output=True, text=True,
            )
            self._temp_branch = None

    def _parse_text_output(self, result: subprocess.CompletedProcess) -> dict:
        """Parse pytest text output"""
        output = result.stdout + result.stderr

        # Extract test results using regex
        # Pattern matches: test_scratch1_rigor.py::test_name PASSED/FAILED/SKIPPED
        test_pattern = r"::(test_\w+)\s+(PASSED|FAILED|ERROR|SKIPPED)"
        matches = re.findall(test_pattern, output, re.MULTILINE)

        tests = []
        seen = set()
        for test_name, outcome in matches:
            if test_name not in seen:
                tests.append({
                    "name": test_name,
                    "outcome": outcome.lower(),
                    "duration": 0,
                    "call": {}
                })
                seen.add(test_name)

        # If no tests found, check if there was an import error
        if not tests and ("ImportError" in output or "ModuleNotFoundError" in output):
            tests.append({
                "name": "test_import_success",
                "outcome": "failed",
                "duration": 0,
                "call": {"longrepr": output[:1000]}
            })

        return {
            "tests": tests,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
        }


# ============================================================================
# SCORING LOGIC
# ============================================================================

class Scorer:
    """Maps test results to rubric scores"""

    RUBRIC = {
        "causal_attention": {"points": 15, "tests": ["test_causal_mask_leakage", "test_causal_attention_shape"]},
        "rmsnorm":          {"points": 10, "tests": ["test_rmsnorm_implementation", "test_rmsnorm_numerical"]},
        "training":         {"points": 10, "tests": ["test_training_convergence"]},
        "rope":             {"points": 15, "tests": ["test_rope_embeddings"]},
        "code_quality":     {"points": 10, "tests": ["test_import_success", "test_no_syntax_errors", "test_no_todos_left"]},
        "model":            {"points": 10, "tests": ["test_model_forward_pass", "test_model_has_trainable"]},
        "report":           {"points": 30, "tests": []},  # Manual review
        "mastery":          {"points": 10, "tests": []},  # Manual review
    }

    # Explicit per-test point allocation (avoids integer division rounding)
    # Total: 8+7+5+5+10+15+4+3+3+5+5 = 70
    TEST_POINTS = {
        "test_causal_mask_leakage": 8,
        "test_causal_attention_shape_preservation": 7,
        "test_rmsnorm_implementation": 5,
        "test_rmsnorm_numerical_stability": 5,
        "test_training_convergence": 10,
        "test_rope_embeddings": 15,
        "test_import_success": 4,
        "test_no_syntax_errors": 3,
        "test_no_todos_left": 3,
        "test_model_forward_pass": 5,
        "test_model_has_trainable_parameters": 5,
    }

    FEEDBACK_TEMPLATES = {
        "causal_mask_leakage": {
            "pass": "✅ Perfect! Your causal mask correctly prevents future token leakage.",
            "fail": "❌ Your mask is leaking future information. Did you apply it **BEFORE** softmax? The mask should be added to logits, not attention weights.",
        },
        "rmsnorm_implementation": {
            "pass": "✅ RMSNorm implemented correctly with proper normalization and learnable scale.",
            "fail": "❌ RMSNorm implementation has issues. Check: (1) RMS calculation, (2) normalization, (3) learnable scale parameter.",
        },
        "training_convergence": {
            "pass": "✅ Excellent! Your model trains successfully and loss converges.",
            "fail_no_improve": "❌ Loss did not decrease during training. Check your training loop, gradient flow, and optimizer setup.",
            "fail_too_high": "⚠️ Model trains but final loss is too high ({:.3f}). Target is < 2.2. Check learning rate and training iterations.",
        },
        "rope_embeddings": {
            "pass": "✅ RoPE correctly applied to Q and K tensors.",
            "fail": "❌ RoPE implementation has issues. Verify: (1) rotation is applied, (2) shapes are preserved, (3) no NaN/Inf values.",
        },
        "import_success": {
            "pass": "✅ Code imports successfully.",
            "fail": "❌ CRITICAL: Code failed to import. See error traceback below.",
        },
        "model_forward_pass": {
            "pass": "✅ Model forward pass works end-to-end with correct output shapes.",
            "fail": "❌ Model forward pass failed. Check that DecoderOnlyTransformer returns (logits, loss) with correct shapes.",
        },
        "model_has_trainable_parameters": {
            "pass": "✅ Model has the expected number of trainable parameters.",
            "fail": "❌ Model has too few or no trainable parameters. Check your layer initialization.",
        },
    }

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path

    def score_tests(self, test_results: dict) -> List[TestResult]:
        """Convert pytest results to scored TestResult objects"""
        scored_results = []

        # Process each test
        for test in test_results.get("tests", []):
            test_name = test["name"].split("::")[-1] if "::" in test["name"] else test["name"]
            outcome = test["outcome"]

            # Find matching rubric category via substring match
            category = None
            for cat, info in self.RUBRIC.items():
                for rubric_test in info["tests"]:
                    if rubric_test in test_name:
                        category = cat
                        break
                if category:
                    break

            # Look up explicit point value for this test
            points_possible = self.TEST_POINTS.get(test_name, 0)

            if category:
                passed = outcome in ["passed", "PASSED"]
                skipped = outcome in ["skipped", "SKIPPED"]
                points_earned = points_possible if passed else 0

                # Get feedback
                feedback_key = test_name.replace("test_", "")
                if skipped:
                    feedback = "⏭️ Test skipped (likely due to import failure or missing dependencies)"
                else:
                    feedback = self._get_feedback(feedback_key, passed, test.get("call", {}))

                scored_results.append(TestResult(
                    name=test_name,
                    passed=passed,
                    points_possible=points_possible,
                    points_earned=points_earned,
                    feedback=feedback,
                    traceback=self._extract_traceback(test),
                    category=category,
                ))

        return scored_results

    def _get_feedback(self, test_key: str, passed: bool, call_info: dict) -> str:
        """Get appropriate feedback for test result"""
        templates = self.FEEDBACK_TEMPLATES.get(test_key, {})
        if passed:
            return templates.get("pass", "✅ Test passed.")
        else:
            return templates.get("fail", "❌ Test failed.")

    def _extract_traceback(self, test: dict) -> Optional[str]:
        """Extract traceback from test result"""
        call = test.get("call", {})
        if "longrepr" in call:
            return call["longrepr"]
        return None

    def check_for_report(self) -> Tuple[bool, List[str]]:
        """Check if student submitted a report"""
        report_patterns = [
            "content/course/submissions/scratch-1/*.mdx",
            "content/course/submissions/scratch-1/*.md",
            "report.md",
            "README.md",
        ]

        found_files = []
        for pattern in report_patterns:
            files = list(self.repo_path.glob(pattern))
            found_files.extend([str(f.relative_to(self.repo_path)) for f in files])

        return len(found_files) > 0, found_files

    def check_for_mastery(self) -> Tuple[bool, List[str]]:
        """Check for mastery-level features"""
        features = []

        backbone_file = self.repo_path / "src" / "assignments" / "scratch-1" / "backbone.py"
        if backbone_file.exists():
            content = backbone_file.read_text()

            # Check for KV-caching
            if "kv_cache" in content.lower() or "cache_k" in content:
                features.append("KV-Caching implementation")

            # Check for ablation study references
            if "ablation" in content.lower() or "sinusoidal" in content.lower():
                features.append("RoPE vs Sinusoidal ablation study")

        return len(features) > 0, features


# ============================================================================
# REPORT GENERATION
# ============================================================================

class ReportGenerator:
    """Generates Chris-Bot style markdown reports"""

    def __init__(self, chris_bot_image: str = "~/chris_robot.png"):
        self.chris_bot_image = chris_bot_image

    def generate_report(self, report: GradingReport) -> str:
        """Generate complete markdown report (DEPRECATED: use generate_public_report or generate_private_report)"""
        if report.critical_failure:
            return self._generate_failure_report(report)
        else:
            return self._generate_standard_report(report)

    def generate_public_report(self, report: GradingReport) -> str:
        """Generate public-facing report for GitHub PR (feedback-focused, no numeric scores)"""
        if report.critical_failure:
            return self._generate_failure_report(report)

        # Group test results by category
        results_by_category = {}
        for result in report.test_results:
            if result.category not in results_by_category:
                results_by_category[result.category] = []
            results_by_category[result.category].append(result)

        md = f"""![Chris-Bot]({self.chris_bot_image})
### 🤖 Chris's Grading Assistant - Feedback Report

**Student:** @{report.author}
**PR:** #{report.pr_number}
**Branch:** `{report.branch_name}`

Hi! I've reviewed your submission. Here's what I found:

---

## 📊 Component Feedback

"""
        # Causal Attention
        causal_results = results_by_category.get("causal_attention", [])
        causal_passed = all(r.passed for r in causal_results)
        md += f"### {'✅' if causal_passed else '❌'} Causal Self-Attention\n\n"
        for result in causal_results:
            md += f"{result.feedback}\n\n"

        # RMSNorm
        rmsnorm_results = results_by_category.get("rmsnorm", [])
        rmsnorm_passed = all(r.passed for r in rmsnorm_results)
        md += f"### {'✅' if rmsnorm_passed else '❌'} RMSNorm\n\n"
        for result in rmsnorm_results:
            md += f"{result.feedback}\n\n"

        # Training
        training_results = results_by_category.get("training", [])
        training_passed = all(r.passed for r in training_results)
        md += f"### {'✅' if training_passed else '⚠️'} Training Loop\n\n"
        for result in training_results:
            md += f"{result.feedback}\n\n"

        # RoPE
        rope_results = results_by_category.get("rope", [])
        rope_passed = all(r.passed for r in rope_results)
        md += f"### {'✅' if rope_passed else '❌'} RoPE Embeddings\n\n"
        for result in rope_results:
            md += f"{result.feedback}\n\n"

        # Model Architecture
        model_results = results_by_category.get("model", [])
        model_passed = all(r.passed for r in model_results) if model_results else True
        md += f"### {'✅' if model_passed else '❌'} Model Architecture\n\n"
        if model_results:
            for result in model_results:
                md += f"{result.feedback}\n\n"
        else:
            md += "No model architecture tests ran.\n\n"

        # Code Quality
        quality_results = results_by_category.get("code_quality", [])
        quality_passed = all(r.passed for r in quality_results)
        md += f"### {'✅' if quality_passed else '❌'} Code Quality\n\n"
        if quality_passed:
            md += "Your code imports and runs cleanly. Nice! ✨\n\n"
        else:
            for result in quality_results:
                md += f"{result.feedback}\n\n"

        # Report
        md += "---\n\n## 📝 Documentation & Analysis\n\n"
        if report.has_report:
            md += "✅ Report submitted! I found:\n"
            for f in report.report_files:
                md += f"- `{f}`\n"
            md += "\nYour instructor will review the quality of your analysis.\n\n"
        else:
            md += "⚠️ **No report found!**\n\n"

        # Mastery
        if report.has_mastery:
            md += "---\n\n## 🎯 Mastery Features Detected\n\n"
            md += "I noticed you implemented:\n"
            for feature in report.mastery_features:
                md += f"- {feature}\n"
            md += "\nGreat work going beyond the requirements! Your instructor will verify implementation quality.\n\n"

        md += "---\n\n"
        md += "> *Grading is automated but reviewed by an instructor. If you have questions, reach out on Slack!*\n"

        return md

    def generate_private_report(self, report: GradingReport) -> str:
        """Generate private report for gradebook (score-focused)"""
        # Group test results by category
        results_by_category = {}
        for result in report.test_results:
            if result.category not in results_by_category:
                results_by_category[result.category] = []
            results_by_category[result.category].append(result)

        # Calculate category scores
        causal_results = results_by_category.get("causal_attention", [])
        causal_points = sum(r.points_earned for r in causal_results)
        causal_max = sum(r.points_possible for r in causal_results)

        rmsnorm_results = results_by_category.get("rmsnorm", [])
        rmsnorm_points = sum(r.points_earned for r in rmsnorm_results)
        rmsnorm_max = sum(r.points_possible for r in rmsnorm_results)

        training_results = results_by_category.get("training", [])
        training_points = sum(r.points_earned for r in training_results)
        training_max = sum(r.points_possible for r in training_results)

        rope_results = results_by_category.get("rope", [])
        rope_points = sum(r.points_earned for r in rope_results)
        rope_max = sum(r.points_possible for r in rope_results)

        quality_results = results_by_category.get("code_quality", [])
        quality_points = sum(r.points_earned for r in quality_results)
        quality_max = sum(r.points_possible for r in quality_results)

        model_results = results_by_category.get("model", [])
        model_points = sum(r.points_earned for r in model_results)
        model_max = sum(r.points_possible for r in model_results)

        automated_total = causal_points + rmsnorm_points + training_points + rope_points + quality_points + model_points
        automated_max = causal_max + rmsnorm_max + training_max + rope_max + quality_max + model_max

        percentage = (report.total_points / report.max_points) * 100 if report.max_points > 0 else 0

        md = f"""# PRIVATE GRADING REPORT - PR #{report.pr_number}

**Student:** {report.author}
**Branch:** {report.branch_name}
**Graded:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## AUTOMATED SCORES ({automated_total}/{automated_max} pts)

| Component | Score | Max | Details |
|-----------|-------|-----|---------|
| Causal Attention | {causal_points} | {causal_max} | {"✅ Passed" if causal_points == causal_max else "❌ Failed"} |
| RMSNorm | {rmsnorm_points} | {rmsnorm_max} | {"✅ Passed" if rmsnorm_points == rmsnorm_max else "❌ Failed"} |
| Training | {training_points} | {training_max} | {"✅ Passed" if training_points == training_max else "❌ Failed"} |
| RoPE | {rope_points} | {rope_max} | {"✅ Passed" if rope_points == rope_max else "❌ Failed"} |
| Model Architecture | {model_points} | {model_max} | {"✅ Passed" if model_points == model_max else "❌ Failed"} |
| Code Quality | {quality_points} | {quality_max} | {"✅ Passed" if quality_points == quality_max else "❌ Failed"} |

---

## MANUAL REVIEW REQUIRED

### Documentation (0-30 pts): _____ / 30

Report files found:
"""
        if report.has_report:
            for f in report.report_files:
                md += f"- {f}\n"
        else:
            md += "- ⚠️ NO REPORT FOUND\n"

        md += """
Check for:
- [ ] Loss curve visualization (clear, labeled)
- [ ] Attention map visualization (interpretable)
- [ ] The Audit: causal mask removal analysis

### Mastery Components (0-10 pts): _____ / 10

"""
        if report.has_mastery:
            md += "Features detected:\n"
            for feature in report.mastery_features:
                md += f"- {feature}\n"
        else:
            md += "- No mastery features detected\n"

        md += f"""
---

## FINAL SCORE

**Automated:** {automated_total}/{automated_max}
**Documentation:** _____ / 30
**Mastery:** _____ / 10
**Adjustments:** _____ (e.g., -1 for programmatic fixes like missing dependencies)

**TOTAL:** _____ / 100

---

## TEST DETAILS

"""
        # Detailed test results
        for category, results in results_by_category.items():
            md += f"\n### {category.replace('_', ' ').title()}\n\n"
            for result in results:
                status = "✅ PASS" if result.passed else "❌ FAIL"
                md += f"- **{result.name}**: {status} ({result.points_earned}/{result.points_possible} pts)\n"
                md += f"  - {result.feedback}\n"
                if result.traceback:
                    md += f"  - Error: `{result.traceback[:200]}...`\n"

        return md

    def _generate_standard_report(self, report: GradingReport) -> str:
        """Generate standard grading report"""
        # Calculate grade letter
        percentage = (report.total_points / report.max_points) * 100
        if percentage >= 90:
            grade_letter = "A"
            grade_emoji = "🎉"
        elif percentage >= 80:
            grade_letter = "B+"
            grade_emoji = "👍"
        elif percentage >= 70:
            grade_letter = "B"
            grade_emoji = "✅"
        elif percentage >= 60:
            grade_letter = "C"
            grade_emoji = "⚠️"
        else:
            grade_letter = "F"
            grade_emoji = "❌"

        # Group test results by category
        results_by_category = {}
        for result in report.test_results:
            if result.category not in results_by_category:
                results_by_category[result.category] = []
            results_by_category[result.category].append(result)

        # Build report sections
        md = f"""![Chris-Bot]({self.chris_bot_image})
### 🤖 Chris's Grading Assistant Report

**Student:** @{report.author}
**PR:** #{report.pr_number}
**Branch:** `{report.branch_name}`
**Score:** {report.total_points}/{report.max_points} ({percentage:.1f}%)
**Execution Time:** {report.execution_time:.1f}s

---

## 📊 Test Results

"""

        # Causal Attention
        causal_results = results_by_category.get("causal_attention", [])
        causal_points = sum(r.points_earned for r in causal_results)
        causal_max = sum(r.points_possible for r in causal_results)
        md += f"### {'✅' if causal_points == causal_max else '❌'} Causal Self-Attention ({causal_points}/{causal_max} pts)\n\n"
        for result in causal_results:
            md += f"{result.feedback}\n\n"
            if result.traceback and not result.passed:
                md += f"<details>\n<summary>🔍 Error Details</summary>\n\n```\n{result.traceback[:500]}\n```\n</details>\n\n"

        # RMSNorm
        rmsnorm_results = results_by_category.get("rmsnorm", [])
        rmsnorm_points = sum(r.points_earned for r in rmsnorm_results)
        rmsnorm_max = sum(r.points_possible for r in rmsnorm_results)
        md += f"### {'✅' if rmsnorm_points == rmsnorm_max else '❌'} RMSNorm ({rmsnorm_points}/{rmsnorm_max} pts)\n\n"
        for result in rmsnorm_results:
            md += f"{result.feedback}\n\n"

        # Training
        training_results = results_by_category.get("training", [])
        training_points = sum(r.points_earned for r in training_results)
        training_max = sum(r.points_possible for r in training_results)
        md += f"### {'✅' if training_points == training_max else '⚠️'} Training Loop ({training_points}/{training_max} pts)\n\n"
        for result in training_results:
            md += f"{result.feedback}\n\n"

        # RoPE
        rope_results = results_by_category.get("rope", [])
        rope_points = sum(r.points_earned for r in rope_results)
        rope_max = sum(r.points_possible for r in rope_results)
        md += f"### {'✅' if rope_points == rope_max else '❌'} RoPE Embeddings ({rope_points}/{rope_max} pts)\n\n"
        for result in rope_results:
            md += f"{result.feedback}\n\n"

        # Model Architecture
        model_results = results_by_category.get("model", [])
        model_points = sum(r.points_earned for r in model_results)
        model_max = sum(r.points_possible for r in model_results)
        md += f"### {'✅' if model_points == model_max else '❌'} Model Architecture ({model_points}/{model_max} pts)\n\n"
        for result in model_results:
            md += f"{result.feedback}\n\n"
            if result.traceback and not result.passed:
                md += f"<details>\n<summary>🔍 Error Details</summary>\n\n```\n{result.traceback[:500]}\n```\n</details>\n\n"

        # Code Quality
        quality_results = results_by_category.get("code_quality", [])
        quality_points = sum(r.points_earned for r in quality_results)
        quality_max = sum(r.points_possible for r in quality_results)
        md += f"### {'✅' if quality_points == quality_max else '❌'} Code Quality ({quality_points}/{quality_max} pts)\n\n"
        if quality_points == quality_max:
            md += "Your code compiled and ran without critical errors. Nice! ✨\n\n"
        else:
            for result in quality_results:
                md += f"{result.feedback}\n\n"
                if result.traceback and not result.passed:
                    md += f"<details>\n<summary>🔍 Error Details</summary>\n\n```\n{result.traceback[:500]}\n```\n</details>\n\n"

        # Report (Manual Review)
        md += "---\n\n## 📝 Report & Documentation (30 pts): **NEEDS MANUAL REVIEW**\n\n"
        if report.has_report:
            md += "I found the following report files:\n"
            for f in report.report_files:
                md += f"- `{f}`\n"
            md += "\n**Instructor:** Please manually review the report for:\n"
            md += "- [ ] Loss curve visualization\n"
            md += "- [ ] Attention map visualization\n"
            md += "- [ ] The Audit (causal mask removal explanation)\n\n"
        else:
            md += "⚠️ **No report found!** Expected file at `content/course/submissions/scratch-1/[your-name].mdx`\n\n"
            md += "You must submit a report with:\n"
            md += "- Loss curve showing convergence\n"
            md += "- Attention visualization\n"
            md += "- The Audit: What happens when you remove the causal mask?\n\n"

        # Mastery
        md += "---\n\n## 🎯 Mastery Components (+10 pts)\n\n"
        if report.has_mastery:
            md += f"Found mastery features: {', '.join(report.mastery_features)}\n\n"
            md += "**Instructor:** Please verify implementation quality.\n\n"
        else:
            md += "No mastery components detected.\n\n"
            md += "For full credit (A grade), implement:\n"
            md += "- KV-Caching for efficient inference\n"
            md += "- RoPE vs Sinusoidal embeddings ablation study\n\n"

        # Final Grade
        md += f"""---

## 🏆 Final Grade: {grade_emoji} {report.total_points}/{report.max_points} ({grade_letter})

"""

        if percentage >= 90:
            md += "Outstanding work! You've mastered the core Transformer concepts. 🎉\n\n"
        elif percentage >= 70:
            md += "Good work! You've demonstrated understanding of the key concepts. Keep pushing! 💪\n\n"
        else:
            md += "Keep working on this. Review the feedback above and test locally before resubmitting. You've got this! 🚀\n\n"

        md += """---

> *If you disagree with this automated assessment, please slack me directly. I am a robot, but my owner is reasonable.*
"""

        return md

    def _generate_failure_report(self, report: GradingReport) -> str:
        """Generate report for critical failures"""
        return f"""![Chris-Bot]({self.chris_bot_image})
### 🤖 Chris's Grading Assistant Report

**Student:** @{report.author}
**PR:** #{report.pr_number}
**Score:** 0/100

---

## 🚨 CRITICAL FAILURE

Your code could not be imported or executed. Here's what went wrong:

```
{report.critical_failure}
```

**Common fixes:**
- Make sure `backbone.py` is in the correct location: `src/assignments/scratch-1/backbone.py`
- Check for syntax errors (missing colons, parentheses, etc.)
- Ensure all TODOs are implemented (not left as `pass` or `NotImplementedError`)
- Test locally with: `python src/assignments/scratch-1/test_scratch1.py`

**Next steps:**
1. Fix the import error shown above
2. Test your code locally
3. Push your changes
4. Request a re-grade by commenting on this PR

---

> *If you disagree with this automated assessment, please slack me directly. I am a robot, but my owner is reasonable.*
"""

    def save_report(self, report: GradingReport, output_dir: Path):
        """Save both public and private reports"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save public report (for GitHub PR comment)
        public_file = output_dir / f"PR_{report.pr_number}_public.md"
        public_content = self.generate_public_report(report)
        public_file.write_text(public_content)

        # Save private report (for gradebook)
        private_file = output_dir / f"PR_{report.pr_number}_private.md"
        private_content = self.generate_private_report(report)
        private_file.write_text(private_content)

        print(f"  📄 Public report: {public_file}")
        print(f"  📄 Private report: {private_file}")
        return public_file, private_file


# ============================================================================
# MAIN GRADER
# ============================================================================

class Scratch1Grader:
    """Main grading orchestrator"""

    def __init__(self, repo_path: Path, dry_run: bool = False):
        self.repo_path = repo_path
        self.dry_run = dry_run
        self.git_manager = GitManager(repo_path)
        self.test_runner = TestRunner(repo_path, git_manager=self.git_manager)
        self.scorer = Scorer(repo_path)
        self.report_generator = ReportGenerator()

    def grade_pr(self, pr_number: int) -> GradingReport:
        """Grade a single PR"""
        print(f"\n{'='*60}")
        print(f"Grading PR #{pr_number}")
        print(f"{'='*60}")

        start_time = datetime.now()

        try:
            # Checkout PR branch
            branch_name, author = self.git_manager.checkout_pr_branch(pr_number)

            # Inject internal tests into student's branch
            self.test_runner.inject_internal_tests()

            # Run tests
            test_results = self.test_runner.run_tests()

            # Score tests
            scored_results = self.scorer.score_tests(test_results)

            # Check for report and mastery
            has_report, report_files = self.scorer.check_for_report()
            has_mastery, mastery_features = self.scorer.check_for_mastery()

            # Calculate total score
            total_points = sum(r.points_earned for r in scored_results)

            # Create report
            report = GradingReport(
                pr_number=pr_number,
                author=author,
                branch_name=branch_name,
                total_points=total_points,
                test_results=scored_results,
                has_report=has_report,
                report_files=report_files,
                has_mastery=has_mastery,
                mastery_features=mastery_features,
                execution_time=(datetime.now() - start_time).total_seconds(),
            )

            print(f"  ✓ Score: {total_points}/100")
            return report

        except Exception as e:
            # Critical failure
            print(f"  ✗ CRITICAL FAILURE: {str(e)}")
            return GradingReport(
                pr_number=pr_number,
                author="unknown",
                branch_name="unknown",
                total_points=0,
                critical_failure=str(e),
                execution_time=(datetime.now() - start_time).total_seconds(),
            )

        finally:
            # Always restore git state
            if not self.dry_run:
                self.git_manager.restore_original_state()

    def grade_batch(self, pr_numbers: List[int]) -> List[GradingReport]:
        """Grade multiple PRs"""
        reports = []
        for pr_num in pr_numbers:
            try:
                report = self.grade_pr(pr_num)
                reports.append(report)
            except Exception as e:
                print(f"  ✗ Failed to grade PR #{pr_num}: {e}")

        return reports


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Autograder for Scratch-1: The Transformer Backbone"
    )
    parser.add_argument("--pr", type=int, help="Single PR number to grade")
    parser.add_argument("--batch", type=str, help="Comma-separated PR numbers")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (no git operations)")
    parser.add_argument("--output", type=str, default="grading_reports", help="Output directory")

    args = parser.parse_args()

    if not args.pr and not args.batch:
        parser.error("Must specify either --pr or --batch")

    # Setup
    repo_path = Path(__file__).parent.parent
    output_dir = repo_path / args.output
    grader = Scratch1Grader(repo_path, dry_run=args.dry_run)

    # Grade
    if args.pr:
        report = grader.grade_pr(args.pr)
        grader.report_generator.save_report(report, output_dir)
        print(f"\n✅ Grading complete! Report saved to {output_dir}")

    elif args.batch:
        pr_numbers = [int(x.strip()) for x in args.batch.split(",")]
        print(f"\n📋 Batch grading {len(pr_numbers)} PRs...")
        reports = grader.grade_batch(pr_numbers)

        # Save all reports
        for report in reports:
            grader.report_generator.save_report(report, output_dir)

        # Summary
        print(f"\n{'='*60}")
        print(f"Batch Grading Summary")
        print(f"{'='*60}")
        for report in reports:
            status = "✓" if report.total_points > 0 else "✗"
            print(f"{status} PR #{report.pr_number}: {report.total_points}/100")

        print(f"\n✅ All reports saved to {output_dir}")


if __name__ == "__main__":
    main()
