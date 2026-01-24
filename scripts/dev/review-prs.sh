#!/bin/bash
# Script to review and merge Scratch-0 contributor profile PRs

set -e

PR_NUMBERS=(21 20 19 18 17 16 15 14 13 12 7 6 5 4)

echo "========================================"
echo "Scratch-0 PR Review Tool"
echo "========================================"
echo ""

for PR in "${PR_NUMBERS[@]}"; do
    echo "----------------------------------------"
    echo "Reviewing PR #$PR"
    echo "----------------------------------------"

    # Fetch PR details
    gh pr view $PR --json title,author,files,state

    echo ""
    echo "Files changed:"
    gh pr diff $PR --name-only

    echo ""
    read -p "View full diff? (y/n): " VIEW_DIFF
    if [[ $VIEW_DIFF == "y" ]]; then
        gh pr diff $PR
    fi

    echo ""
    read -p "Merge this PR? (y/n/s to skip): " MERGE

    if [[ $MERGE == "y" ]]; then
        echo "Merging PR #$PR..."
        gh pr merge $PR --squash --delete-branch
        echo "✓ Merged PR #$PR"
    elif [[ $MERGE == "s" ]]; then
        echo "Skipped PR #$PR"
    else
        echo "Not merging PR #$PR"
    fi

    echo ""
done

echo "========================================"
echo "Review complete!"
echo "========================================"
