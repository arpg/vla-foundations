#!/usr/bin/env bash

# VLA Foundations Deployment Script
# Deploys the Next.js static site to the remote server at vlm-robotics.dev

set -e  # Exit on error

# Configuration
REMOTE_USER="crh"
REMOTE_HOST="ristoffer.ch"
REMOTE_PATH="/var/www/vlm-robotics.dev"
SSH_KEY="$HOME/.ssh/id_ed25519_automation"
LOCAL_BUILD_DIR="./out"

echo "======================================="
echo "VLA Foundations Deployment"
echo "======================================="

# Step 1: Build the site locally
echo ""
echo "[1/4] Building Next.js site..."
pnpm install
pnpm build

# Check if build succeeded
if [ ! -d "$LOCAL_BUILD_DIR" ]; then
  echo "Error: Build failed. The 'out' directory does not exist."
  exit 1
fi

echo "✓ Build successful"

# Step 2: Rsync the built site to remote server
echo ""
echo "[2/4] Syncing files to remote server..."
rsync -avz --delete \
  -e "ssh -i $SSH_KEY" \
  "$LOCAL_BUILD_DIR/" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/public_html/"

echo "✓ Files synced"

# Step 3: Verify deployment
echo ""
echo "[3/4] Verifying deployment..."
ssh -i "$SSH_KEY" "${REMOTE_USER}@${REMOTE_HOST}" \
  "ls -lh ${REMOTE_PATH}/public_html/ | head -10"

echo "✓ Deployment verified"

# Step 4: Show live URL
echo ""
echo "[4/4] Deployment complete!"
echo "======================================="
echo "Live URL: https://www.vlm-robotics.dev"
echo "======================================="
