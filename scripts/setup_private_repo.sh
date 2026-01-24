#!/bin/bash
# Automated setup script for private solution repository
# Uses GitHub CLI to create repo, configure secrets, and set up remotes

set -euo pipefail

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║         Private VLA Foundations Repository Setup                    ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if gh is installed and authenticated
if ! command -v gh &> /dev/null; then
    echo "❌ Error: GitHub CLI (gh) is not installed"
    echo "   Install from: https://cli.github.com/"
    exit 1
fi

echo "1. Checking GitHub authentication..."
if ! gh auth status &> /dev/null; then
    echo "❌ Not authenticated with GitHub CLI"
    echo "   Run: gh auth login"
    exit 1
fi
echo "   ✓ Authenticated as $(gh api user --jq .login)"
echo ""

# Prompt for repository details
echo "2. Repository Configuration"
read -p "   Private repository name [private-vla-foundations]: " REPO_NAME
REPO_NAME=${REPO_NAME:-private-vla-foundations}

read -p "   Organization/user (press Enter for personal account): " ORG
if [ -z "$ORG" ]; then
    ORG=$(gh api user --jq .login)
fi

FULL_REPO="$ORG/$REPO_NAME"
echo "   Will create: $FULL_REPO (private)"
echo ""

# Confirm before proceeding
read -p "3. Continue with setup? [y/N]: " CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "   Aborted."
    exit 0
fi
echo ""

# Create private repository
echo "4. Creating private repository..."
if gh repo view "$FULL_REPO" &> /dev/null; then
    echo "   ⚠️  Repository $FULL_REPO already exists"
    read -p "   Use existing repository? [y/N]: " USE_EXISTING
    if [[ ! "$USE_EXISTING" =~ ^[Yy]$ ]]; then
        echo "   Aborted."
        exit 0
    fi
else
    gh repo create "$FULL_REPO" \
        --private \
        --description "Private solutions and internal tests for VLA Foundations course" \
        --disable-wiki \
        --confirm
    echo "   ✓ Created private repository: $FULL_REPO"
fi
echo ""

# Create Personal Access Token for syncing to public repo
echo "5. Creating Personal Access Token (PAT) for public repo sync..."
echo "   This token will be used by GitHub Actions to push to the public repo."
echo ""

# Generate token name with timestamp
TOKEN_NAME="vla-sync-$(date +%Y%m%d)"

# Check if token with similar name exists
EXISTING_TOKEN=$(gh api user --jq .login 2>/dev/null || echo "")
echo "   Creating token: $TOKEN_NAME"
echo "   Scopes: repo (full control of private repositories)"
echo ""

# Create the token
# Note: gh CLI doesn't support creating PATs directly via API for security reasons
# We need to guide the user to create it manually or use the web interface

echo "   ⚠️  GitHub CLI requires you to create the PAT manually for security."
echo ""
echo "   Option 1: Create via web (opens browser):"
echo "     gh auth token  # Shows your current token (limited)"
echo "     gh browse --repo $FULL_REPO --settings  # Opens repo settings"
echo ""
echo "   Option 2: Create PAT manually:"
echo "     1. Go to: https://github.com/settings/tokens/new"
echo "     2. Name: $TOKEN_NAME"
echo "     3. Expiration: 90 days (or longer)"
echo "     4. Scopes: Check 'repo' (Full control of private repositories)"
echo "     5. Click 'Generate token'"
echo "     6. Copy the token (starts with 'ghp_')"
echo ""

read -p "   Have you created the PAT? [y/N]: " HAS_TOKEN
if [[ ! "$HAS_TOKEN" =~ ^[Yy]$ ]]; then
    echo "   Please create the PAT first, then re-run this script."
    echo "   Or manually add the secret later with:"
    echo "     gh secret set PUBLIC_REPO_TOKEN --repo $FULL_REPO"
    exit 0
fi

echo ""
read -sp "   Paste your PAT token: " PAT_TOKEN
echo ""
echo ""

# Set the secret in the private repository
echo "6. Adding secret to private repository..."
echo "$PAT_TOKEN" | gh secret set PUBLIC_REPO_TOKEN --repo "$FULL_REPO"
echo "   ✓ Secret PUBLIC_REPO_TOKEN added to $FULL_REPO"
echo ""

# Update git remotes
echo "7. Updating git remotes..."

# Check if 'origin' points to public repo
CURRENT_ORIGIN=$(git remote get-url origin 2>/dev/null || echo "")
if [[ "$CURRENT_ORIGIN" == *"arpg/vla-foundations"* ]]; then
    echo "   Renaming 'origin' to 'public'..."
    git remote rename origin public 2>/dev/null || echo "   (public remote already exists)"
fi

# Add new private repo as origin
if git remote get-url origin &> /dev/null; then
    echo "   Updating 'origin' to point to private repo..."
    git remote set-url origin "https://github.com/$FULL_REPO.git"
else
    echo "   Adding 'origin' for private repo..."
    git remote add origin "https://github.com/$FULL_REPO.git"
fi

# Ensure public remote exists
if ! git remote get-url public &> /dev/null; then
    echo "   Adding 'public' remote for public repo..."
    git remote add public https://github.com/arpg/vla-foundations.git
fi

echo "   ✓ Git remotes configured:"
git remote -v | grep -E "^(origin|public)" | sed 's/^/     /'
echo ""

# Push to private repository
echo "8. Pushing to private repository..."
read -p "   Push main and staging branches? [y/N]: " PUSH_NOW
if [[ "$PUSH_NOW" =~ ^[Yy]$ ]]; then
    echo "   Pushing main branch..."
    git push -u origin main

    if git rev-parse --verify staging &> /dev/null; then
        echo "   Pushing staging branch..."
        git push -u origin staging
    fi

    echo "   ✓ Pushed to private repository"
else
    echo "   Skipped. Push manually with:"
    echo "     git push -u origin main"
    echo "     git push -u origin staging"
fi
echo ""

# Summary
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                        Setup Complete! ✅                            ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Private Repository: https://github.com/$FULL_REPO"
echo "Public Repository:  https://github.com/arpg/vla-foundations"
echo ""
echo "Git Remotes:"
echo "  origin → https://github.com/$FULL_REPO.git (private)"
echo "  public → https://github.com/arpg/vla-foundations.git (public)"
echo ""
echo "Secrets Configured:"
echo "  ✓ PUBLIC_REPO_TOKEN (for syncing to public repo)"
echo ""
echo "Next Steps:"
echo "  1. Test solution management:"
echo "       python3 scripts/manage_solutions.py --list"
echo ""
echo "  2. Test sanitization (safe - uses temp branch):"
echo "       git checkout -b test-sanitize"
echo "       bash scripts/sanitize.sh"
echo "       git status"
echo "       git checkout main && git branch -D test-sanitize"
echo ""
echo "  3. Create a test release:"
echo "       git tag release-scratch-1-test"
echo "       git push origin release-scratch-1-test"
echo "       # Watch GitHub Actions at: https://github.com/$FULL_REPO/actions"
echo ""
echo "  4. Install pytest and run tests:"
echo "       pip install pytest torch numpy"
echo "       pytest tests/public/ -v"
echo "       pytest tests/internal/ -v"
echo ""
echo "📚 Documentation:"
echo "  • PRIVATE_REPO_SETUP.md - Complete setup guide"
echo "  • QUICK_REFERENCE.md - Command cheat sheet"
echo ""
