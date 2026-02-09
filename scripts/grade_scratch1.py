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

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.original_branch = None
        self.had_stash = False

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
            subprocess.run(
                ["git", "checkout", self.original_branch],
                cwd=self.repo_path,
                check=True,
                capture_output=True
            )

        if self.had_stash:
            print("  📦 Restoring stashed changes...")
            subprocess.run(["git", "stash", "pop"], cwd=self.repo_path, check=True)


# ============================================================================
# TEST EXECUTION
# ============================================================================

class TestRunner:
    """Runs pytest and captures results"""

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.test_file = repo_path / "tests" / "internal" / "test_scratch1_rigor.py"

    def run_tests(self) -> Dict[str, any]:
        """
        Run pytest on internal tests
        Returns: dict with test results
        """
        if not self.test_file.exists():
            raise FileNotFoundError(f"Test file not found: {self.test_file}")

        # Run pytest with verbose output
        cmd = [
            "uv", "run", "pytest",
            str(self.test_file),
            "-v",
            "--tb=short",
            "-m", "rigor",
        ]

        print(f"  🧪 Running tests...")
        result = subprocess.run(
            cmd,
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )

        # Parse text output
        return self._parse_text_output(result)

    def _parse_text_output(self, result: subprocess.CompletedProcess) -> dict:
        """Parse pytest text output"""
        output = result.stdout + result.stderr

        # Extract test results using regex
        # Pattern matches: test_scratch1_rigor.py::test_name PASSED/FAILED
        test_pattern = r"(test_\w+)\s+(PASSED|FAILED|ERROR)"
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
        "rmsnorm": {"points": 10, "tests": ["test_rmsnorm_implementation", "test_rmsnorm_numerical"]},
        "training": {"points": 10, "tests": ["test_training_convergence"]},
        "rope": {"points": 15, "tests": ["test_rope_embeddings"]},
        "code_quality": {"points": 20, "tests": ["test_import_success", "test_no_syntax_errors", "test_no_todos_left"]},
        "report": {"points": 30, "tests": []},  # Manual review
        "mastery": {"points": 10, "tests": []},  # Manual review
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
    }

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path

    def score_tests(self, test_results: dict) -> List[TestResult]:
        """Convert pytest results to scored TestResult objects"""
        scored_results = []

        # Map test names to rubric categories
        test_mapping = {}
        for category, info in self.RUBRIC.items():
            for test_name in info["tests"]:
                test_mapping[test_name] = (category, info["points"])

        # Process each test
        for test in test_results.get("tests", []):
            test_name = test["name"].split("::")[-1] if "::" in test["name"] else test["name"]
            outcome = test["outcome"]

            # Find matching rubric category
            category = None
            points_possible = 0
            for cat, info in self.RUBRIC.items():
                for rubric_test in info["tests"]:
                    if rubric_test in test_name:
                        category = cat
                        points_possible = info["points"] // len(info["tests"]) if info["tests"] else 0
                        break

            if category:
                passed = outcome in ["passed", "PASSED"]
                points_earned = points_possible if passed else 0

                # Get feedback
                feedback_key = test_name.replace("test_", "")
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
        """Generate complete markdown report"""
        if report.critical_failure:
            return self._generate_failure_report(report)
        else:
            return self._generate_standard_report(report)

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

        # Code Quality
        quality_results = results_by_category.get("code_quality", [])
        quality_points = sum(r.points_earned for r in quality_results)
        quality_max = sum(r.points_possible for r in quality_results)
        md += f"### {'✅' if quality_points == quality_max else '❌'} Code Quality ({quality_points}/{quality_max} pts)\n\n"
        md += "Your code compiled and ran without critical errors. Nice! ✨\n\n"

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
        """Save report to file"""
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f"PR_{report.pr_number}_report.md"

        content = self.generate_report(report)
        filename.write_text(content)

        print(f"  📄 Report saved: {filename}")
        return filename


# ============================================================================
# MAIN GRADER
# ============================================================================

class Scratch1Grader:
    """Main grading orchestrator"""

    def __init__(self, repo_path: Path, dry_run: bool = False):
        self.repo_path = repo_path
        self.dry_run = dry_run
        self.git_manager = GitManager(repo_path)
        self.test_runner = TestRunner(repo_path)
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
