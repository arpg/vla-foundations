#!/usr/bin/env python3
"""Collect peer review data from GitHub audit PRs and generate reviewer stats.

Usage:
    python scripts/collect_reviewer_data.py

Outputs data/reviewer-stats.json with per-reviewer comment counts,
quality scores, and category breakdowns for the top-reviewers dashboard.
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Audit PRs to collect data from
AUDIT_PRS = [23, 27, 32, 48, 54, 55, 57, 61]

# Exclude these users from the reviewer leaderboard
EXCLUDED_LOGINS = {"crh-bot"}

# Show these users with "Instructor" tier instead of computed tier
INSTRUCTOR_LOGINS = {"crheckman"}

# Quality scoring weights
SCORE_WEIGHTS = {
    "technical_depth": 3,
    "constructive_suggestion": 2,
    "clarification_request": 2,
    "substantive": 1,
    "brief_low_effort": 0,
}

# Heuristic keywords
CONSTRUCTIVE_KEYWORDS = [
    "consider", "could", "might want", "suggestion", "suggest",
    "maybe", "perhaps", "would recommend", "alternatively", "instead",
    "how about", "what if", "one option",
]

TECHNICAL_KEYWORDS = [
    "line ", "L", "equation", "formula", "loss", "gradient",
    "attention", "transformer", "embedding", "softmax", "relu",
    "normalization", "dropout", "weight", "bias", "dimension",
    "tensor", "matrix", "vector", "convergence", "backprop",
    "implementation", "complexity", "O(", "runtime", "latency",
    "parameter", "hyperparameter", "learning rate", "batch size",
    "architecture", "layer", "block", "head", "token",
]


def gh_api(endpoint: str) -> list[dict]:
    """Call GitHub API via gh CLI and return parsed JSON."""
    result = subprocess.run(
        ["gh", "api", endpoint, "--paginate"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  Warning: gh api {endpoint} failed: {result.stderr.strip()}", file=sys.stderr)
        return []
    try:
        data = json.loads(result.stdout)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        # --paginate may concatenate multiple JSON arrays
        # Try parsing as concatenated arrays
        text = result.stdout.strip()
        if text.startswith("[") and "][" in text:
            parts = text.replace("][", "]\n[").split("\n")
            combined = []
            for part in parts:
                combined.extend(json.loads(part))
            return combined
        return []


def categorize_comment(body: str) -> dict[str, bool]:
    """Categorize a comment using heuristic rules."""
    body_lower = body.lower().strip()
    length = len(body.strip())

    categories = {
        "technical_depth": False,
        "constructive_suggestion": False,
        "clarification_request": False,
        "praise": False,
        "substantive": False,
        "brief_low_effort": False,
    }

    # Brief / low-effort
    if length < 30 or body_lower in ("lgtm", "nice", "looks good", "good", "+1", "approved"):
        categories["brief_low_effort"] = True
        return categories

    # Substantive (>100 chars, not just praise)
    if length > 100:
        categories["substantive"] = True

    # Technical depth: references code, formulas, specific implementation details
    tech_hits = sum(1 for kw in TECHNICAL_KEYWORDS if kw.lower() in body_lower)
    if tech_hits >= 2 or any(c in body for c in ["`", "```", "$$"]):
        categories["technical_depth"] = True

    # Constructive suggestion
    if any(kw in body_lower for kw in CONSTRUCTIVE_KEYWORDS):
        categories["constructive_suggestion"] = True

    # Clarification / question
    if "?" in body and length > 30:
        categories["clarification_request"] = True

    # Praise (but not ONLY praise if also technical)
    praise_words = ["great", "nice", "good job", "well done", "excellent", "awesome", "love"]
    if any(pw in body_lower for pw in praise_words):
        categories["praise"] = True

    return categories


def compute_raw_points(categories_totals: dict[str, int]) -> int:
    """Compute total weighted contribution points (not averaged)."""
    return (
        categories_totals.get("technical_depth", 0) * SCORE_WEIGHTS["technical_depth"]
        + categories_totals.get("constructive_suggestion", 0) * SCORE_WEIGHTS["constructive_suggestion"]
        + categories_totals.get("clarification_request", 0) * SCORE_WEIGHTS["clarification_request"]
        + categories_totals.get("substantive", 0) * SCORE_WEIGHTS["substantive"]
        + categories_totals.get("brief_low_effort", 0) * SCORE_WEIGHTS["brief_low_effort"]
    )


def normalize_scores(raw_points_list: list[int], benchmark_raw: int) -> list[float]:
    """Normalize raw points to 0-10 using sqrt scaling against instructor benchmark.

    The instructor's total contribution sets the 10.0 ceiling. Students
    are scored relative to that. sqrt scaling rewards both quality and
    volume without making it a pure comment-count contest.
    """
    import math
    if benchmark_raw == 0:
        return [0.0] * len(raw_points_list)
    sqrt_bench = math.sqrt(benchmark_raw)
    return [min(10.0, round(math.sqrt(r) / sqrt_bench * 10, 1)) for r in raw_points_list]


def quality_tier(score: float) -> str:
    """Tier thresholds calibrated to instructor benchmark (10.0 = instructor)."""
    if score >= 5.0:
        return "Exemplary"
    elif score >= 3.5:
        return "Strong"
    elif score >= 2.0:
        return "Solid"
    else:
        return "Developing"


def main():
    project_root = Path(__file__).resolve().parent.parent
    output_path = project_root / "data" / "reviewer-stats.json"
    output_path.parent.mkdir(exist_ok=True)

    # Accumulate per-reviewer data
    reviewers: dict[str, dict] = {}
    audit_prs: list[dict] = []

    for pr_num in AUDIT_PRS:
        print(f"Fetching PR #{pr_num}...")

        # Get PR metadata
        pr_info = gh_api(f"repos/arpg/vla-foundations/pulls/{pr_num}")
        pr_title = pr_info[0].get("title", f"PR #{pr_num}") if pr_info else f"PR #{pr_num}"
        pr_authors = []
        if pr_info and pr_info[0].get("user"):
            pr_authors.append(pr_info[0]["user"]["login"])

        # Collect all comments from three sources
        inline_comments = gh_api(f"repos/arpg/vla-foundations/pulls/{pr_num}/comments")
        reviews = gh_api(f"repos/arpg/vla-foundations/pulls/{pr_num}/reviews")
        issue_comments = gh_api(f"repos/arpg/vla-foundations/issues/{pr_num}/comments")

        pr_total = 0

        # Process inline code review comments
        for comment in inline_comments:
            user = comment.get("user", {})
            login = user.get("login", "unknown")
            if login in EXCLUDED_LOGINS or user.get("type") == "Bot":
                continue

            body = comment.get("body", "")
            if not body.strip():
                continue

            if login not in reviewers:
                reviewers[login] = {
                    "login": login,
                    "avatar_url": user.get("avatar_url", ""),
                    "name": user.get("login", login),  # GitHub API doesn't always return name
                    "total_comments": 0,
                    "inline_comments": 0,
                    "discussion_comments": 0,
                    "prs_reviewed": set(),
                    "comments_bodies": [],
                    "category_totals": {
                        "technical_depth": 0,
                        "constructive_suggestion": 0,
                        "clarification_request": 0,
                        "praise": 0,
                    },
                }

            reviewers[login]["total_comments"] += 1
            reviewers[login]["inline_comments"] += 1
            reviewers[login]["prs_reviewed"].add(pr_num)
            reviewers[login]["comments_bodies"].append(body)

            cats = categorize_comment(body)
            for cat, hit in cats.items():
                if hit and cat in reviewers[login]["category_totals"]:
                    reviewers[login]["category_totals"][cat] += 1

            pr_total += 1

        # Process review submissions (body text, not just approve/reject)
        for review in reviews:
            user = review.get("user", {})
            login = user.get("login", "unknown")
            if login in EXCLUDED_LOGINS or user.get("type") == "Bot":
                continue

            body = review.get("body", "")
            if not body or not body.strip():
                continue

            if login not in reviewers:
                reviewers[login] = {
                    "login": login,
                    "avatar_url": user.get("avatar_url", ""),
                    "name": user.get("login", login),
                    "total_comments": 0,
                    "inline_comments": 0,
                    "discussion_comments": 0,
                    "prs_reviewed": set(),
                    "comments_bodies": [],
                    "category_totals": {
                        "technical_depth": 0,
                        "constructive_suggestion": 0,
                        "clarification_request": 0,
                        "praise": 0,
                    },
                }

            reviewers[login]["total_comments"] += 1
            reviewers[login]["discussion_comments"] += 1
            reviewers[login]["prs_reviewed"].add(pr_num)
            reviewers[login]["comments_bodies"].append(body)

            cats = categorize_comment(body)
            for cat, hit in cats.items():
                if hit and cat in reviewers[login]["category_totals"]:
                    reviewers[login]["category_totals"][cat] += 1

            pr_total += 1

        # Process issue-level discussion comments
        for comment in issue_comments:
            user = comment.get("user", {})
            login = user.get("login", "unknown")
            if login in EXCLUDED_LOGINS or user.get("type") == "Bot":
                continue

            body = comment.get("body", "")
            if not body.strip():
                continue

            if login not in reviewers:
                reviewers[login] = {
                    "login": login,
                    "avatar_url": user.get("avatar_url", ""),
                    "name": user.get("login", login),
                    "total_comments": 0,
                    "inline_comments": 0,
                    "discussion_comments": 0,
                    "prs_reviewed": set(),
                    "comments_bodies": [],
                    "category_totals": {
                        "technical_depth": 0,
                        "constructive_suggestion": 0,
                        "clarification_request": 0,
                        "praise": 0,
                    },
                }

            reviewers[login]["total_comments"] += 1
            reviewers[login]["discussion_comments"] += 1
            reviewers[login]["prs_reviewed"].add(pr_num)
            reviewers[login]["comments_bodies"].append(body)

            cats = categorize_comment(body)
            for cat, hit in cats.items():
                if hit and cat in reviewers[login]["category_totals"]:
                    reviewers[login]["category_totals"][cat] += 1

            pr_total += 1

        audit_prs.append({
            "number": pr_num,
            "title": pr_title,
            "authors": pr_authors,
            "total_comments": pr_total,
        })

        print(f"  → {pr_total} reviewer comments (excluding bots)")

    # Pass 1: collect raw data per reviewer
    pre_list = []
    for login, data in reviewers.items():
        total = data["total_comments"]
        cats = data["category_totals"]
        raw = compute_raw_points(cats)

        bodies = sorted(data["comments_bodies"], key=len, reverse=True)
        samples = []
        for b in bodies[:3]:
            preview = b[:200].strip()
            if len(b) > 200:
                preview += "..."
            samples.append(preview)

        pre_list.append({
            "login": data["login"],
            "avatar_url": data["avatar_url"],
            "name": data["name"],
            "total_comments": total,
            "inline_comments": data["inline_comments"],
            "discussion_comments": data["discussion_comments"],
            "prs_reviewed": sorted(data["prs_reviewed"]),
            "is_instructor": data["login"] in INSTRUCTOR_LOGINS,
            "sample_comments": samples,
            "comment_categories": cats,
            "_raw_points": raw,
        })

    # Pass 2: normalize student scores against instructor benchmark
    instructor_raw = next(
        (p["_raw_points"] for p in pre_list if p["is_instructor"]), 1
    )
    student_raws = [p["_raw_points"] for p in pre_list if not p["is_instructor"]]
    student_scores = normalize_scores(student_raws, benchmark_raw=instructor_raw)

    # Build final list with normalized scores
    reviewer_list = []
    student_idx = 0
    for p in pre_list:
        if p["is_instructor"]:
            score = 10.0  # benchmark
            tier = "Instructor"
        else:
            score = student_scores[student_idx]
            tier = quality_tier(score)
            student_idx += 1

        entry = dict(p)
        del entry["_raw_points"]
        entry["quality_score"] = score
        entry["quality_tier"] = tier
        reviewer_list.append(entry)

    # Sort by quality score descending, then by total comments descending
    reviewer_list.sort(key=lambda r: (-r["quality_score"], -r["total_comments"]))

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reviewers": reviewer_list,
        "audit_prs": audit_prs,
    }

    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nWrote {len(reviewer_list)} reviewers to {output_path}")
    print(f"Total comments across all PRs: {sum(r['total_comments'] for r in reviewer_list)}")


if __name__ == "__main__":
    main()
