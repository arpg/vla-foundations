#!/bin/bash
# Quick script to add PUBLIC_REPO_TOKEN secret using gh CLI
# Usage: ./scripts/add_github_secret.sh [repo-owner/repo-name]

set -euo pipefail

REPO="${1:-}"

if [ -z "$REPO" ]; then
    echo "Usage: $0 <owner/repo-name>"
    echo "Example: $0 crheckman/private-vla-foundations"
    exit 1
fi

echo "Adding PUBLIC_REPO_TOKEN secret to $REPO"
echo ""
echo "This token needs 'repo' scope to push to the public repository."
echo ""
echo "Creating a new Personal Access Token..."
echo ""

# Option 1: Use an existing token if the user has one
if gh auth token &> /dev/null; then
    CURRENT_TOKEN=$(gh auth token)
    echo "You're currently authenticated with a token."
    echo ""
    read -p "Use your current token as PUBLIC_REPO_TOKEN? [y/N]: " USE_CURRENT

    if [[ "$USE_CURRENT" =~ ^[Yy]$ ]]; then
        echo "$CURRENT_TOKEN" | gh secret set PUBLIC_REPO_TOKEN --repo "$REPO"
        echo "✓ Secret added successfully!"
        exit 0
    fi
fi

echo ""
echo "Please create a new Personal Access Token (PAT):"
echo "  1. Go to: https://github.com/settings/tokens/new"
echo "  2. Note: 'VLA Sync Token $(date +%Y-%m-%d)'"
echo "  3. Expiration: 90 days (or your preference)"
echo "  4. Scopes: Check ☑ 'repo' (Full control of private repositories)"
echo "  5. Click 'Generate token'"
echo "  6. Copy the token (starts with 'ghp_' or 'github_pat_')"
echo ""
echo "Or run this to open the page:"
echo "  gh browse https://github.com/settings/tokens/new"
echo ""

read -sp "Paste your new PAT token here: " NEW_TOKEN
echo ""

if [ -z "$NEW_TOKEN" ]; then
    echo "❌ No token provided. Aborted."
    exit 1
fi

# Validate token format
if [[ ! "$NEW_TOKEN" =~ ^(ghp_|github_pat_) ]]; then
    echo "⚠️  Warning: Token doesn't start with 'ghp_' or 'github_pat_'"
    read -p "Continue anyway? [y/N]: " CONTINUE
    if [[ ! "$CONTINUE" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "Adding secret to repository..."
echo "$NEW_TOKEN" | gh secret set PUBLIC_REPO_TOKEN --repo "$REPO"

echo "✓ Secret PUBLIC_REPO_TOKEN added to $REPO"
echo ""
echo "Verify with:"
echo "  gh secret list --repo $REPO"
