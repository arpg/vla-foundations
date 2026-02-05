#!/bin/bash
# Deploy script for staging with dynamic Next.js (runs as server, not static export)
# Run this on ristoffer.ch

set -e

echo "=== VLA Foundations Staging Deployment (Dynamic) ==="

# Navigate to staging directory
cd ~/vla-staging || { mkdir -p ~/vla-staging && cd ~/vla-staging; }

# Clone or pull staging branch
if [ ! -d ".git" ]; then
  echo "Cloning repository..."
  git clone -b staging https://github.com/arpg/vla-foundations.git .
else
  echo "Pulling latest changes..."
  git fetch origin staging
  git reset --hard origin/staging
fi

# Create dynamic config (no static export)
echo "Creating dynamic Next.js config..."
cat > next.config.ts <<'NEXTCONFIG'
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // No static export - run as dynamic server
  trailingSlash: true,
  images: {
    unoptimized: true,
  },
  turbopack: {},
};

export default nextConfig;
NEXTCONFIG

# Install dependencies
echo "Installing dependencies..."
npm install

# Build Next.js
echo "Building Next.js..."
npm run build

# Kill existing Next.js server
echo "Stopping existing server..."
pkill -f "next start" || true
sleep 2

# Start Next.js server on port 3002 in background
echo "Starting Next.js server on port 3002..."
nohup npm start -- -p 3002 > ~/vla-staging-server.log 2>&1 &
NEXT_PID=$!
echo "Next.js server started with PID $NEXT_PID"

# Wait for server to start
echo "Waiting for server to start..."
sleep 5

# Test if server is running
if curl -s http://localhost:3002 > /dev/null; then
  echo "✓ Next.js server is running"
else
  echo "✗ Next.js server failed to start"
  echo "Check logs: tail ~/vla-staging-server.log"
  exit 1
fi

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Next.js server: http://localhost:3002"
echo "Logs: tail -f ~/vla-staging-server.log"
echo ""
echo "Now configure Apache to proxy /staging to localhost:3002"
echo "Add to your Apache config:"
echo ""
echo "  ProxyPass /staging http://localhost:3002/staging"
echo "  ProxyPassReverse /staging http://localhost:3002/staging"
echo ""
echo "Or if you want staging on a subdomain:"
echo ""
echo "  <VirtualHost *:80>"
echo "    ServerName staging.vlm-robotics.dev"
echo "    ProxyPass / http://localhost:3002/"
echo "    ProxyPassReverse / http://localhost:3002/"
echo "  </VirtualHost>"
echo ""
