# Quick Start: Manual SSH Setup

If you want to get the review system running immediately without waiting for automated deployment:

## SSH Commands

```bash
# 1. SSH to your server
ssh crh@ristoffer.ch

# 2. Navigate to staging directory
cd ~/vla-staging

# 3. Pull latest code
git fetch origin staging
git reset --hard origin/staging

# 4. Build Next.js site
npm install
npm run build

# 5. Deploy static files
cp -r out/* /var/www/vlm-robotics.dev/public_html/staging/

# 6. Set up API server
cd ~/vla-staging/api-server
npm install

# Kill any existing API server
pkill -f "node server.js" || true

# Start API server in background
nohup node server.js > ~/vla-staging-api.log 2>&1 &

# Note the PID
echo "API server started with PID $!"

# 7. Configure Apache proxy (one-time setup)
# Edit your Apache config or create .htaccess:
sudo nano /etc/apache2/sites-available/vlm-robotics.dev.conf

# Add these lines inside <VirtualHost>:
ProxyPass /api http://localhost:3001/api
ProxyPassReverse /api http://localhost:3001/api

# Restart Apache
sudo systemctl restart apache2

# 8. Test API
curl http://localhost:3001/health
# Should return: {"status":"ok","timestamp":"..."}

# 9. Test via Apache proxy
curl https://vlm-robotics.dev/api/comments/23
# Should return comment data

# 10. View the review interface
# Visit: https://vlm-robotics.dev/staging/textbook/audits/multimodality_audit?review=true
```

## Verify Everything Works

```bash
# Check if API server is running
ps aux | grep "node server.js"

# View API logs
tail -f ~/vla-staging-api.log

# Test API endpoints
curl http://localhost:3001/health
curl http://localhost:3001/api/comments/23

# Check Apache proxy
curl https://vlm-robotics.dev/api/comments/23
```

## Troubleshooting

**API not responding:**
```bash
cd ~/vla-staging/api-server
pkill -f "node server.js"
nohup node server.js > ~/vla-staging-api.log 2>&1 &
```

**Check what's using port 3001:**
```bash
lsof -i :3001
```

**View API logs:**
```bash
tail -50 ~/vla-staging-api.log
```

Once you've done this manual setup, future deployments will be automatic via GitHub Actions.
