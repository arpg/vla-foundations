# VLA Foundations: Private Operations Guide

**CONFIDENTIAL - INTERNAL USE ONLY**

This document consolidates all instructor-facing operational documentation for the VLA Foundations course infrastructure.

---

## Table of Contents

1. [Paper Audit Review Workflow](#paper-audit-review-workflow)
2. [Server Configuration](#server-configuration)
3. [Deployment Procedures](#deployment-procedures)
4. [Troubleshooting](#troubleshooting)

---

## Paper Audit Review Workflow

### Overview

Students submit paper audits via Pull Requests to the `staging` branch. The system automatically:
1. Runs a standards linter
2. Deploys preview to GitHub Pages
3. Posts preview link in PR comments

Once approved, you promote the audit from `staging/` to production and merge to `main`.

### Student Submission Workflow

#### 1. Student Creates PR

Students follow these steps (from `content/course/assignments/paper-audit.mdx`):

```bash
git checkout -b audit/username-topic
# Add audit file to: content/textbook/audits/staging/username.mdx
git add content/textbook/audits/staging/username.mdx
git commit -m "audit: add [Paper] deep-dive by [Name]"
git push origin audit/username-topic
```

Then create PR with:
- **Base branch:** `staging`
- **Title:** `Audit: [Paper Topic] - [Name]`

#### 2. Automated Linter Check

The `vla-audit.yml` workflow automatically runs:

**Standards Checked:**
- ✅ Semantic line breaks (one sentence per line)
- ✅ Clean git history (no "Merge branch" commits)

**If linter fails:**
- Bot posts detailed instructions in PR comments
- Build is blocked until standards are met

**If linter passes:**
- Build deploys to: `https://arpg.github.io/vla-foundations/staging/pulls/{PR_NUMBER}/textbook/audits/staging/{username}/`
- Bot posts preview link in PR comments

#### 3. Instructor Review

Navigate to the preview URL (posted in PR comments) to review the rendered audit.

**Add feedback directly in GitHub PR:**
- Use GitHub's native PR review features
- Comment on specific lines
- Request changes or approve

**Common feedback areas:**
- Mathematical rigor and LaTeX correctness
- Architectural depth and comparative analysis
- Citations and references
- Physical grounding critique (the "load-bearing" analysis)

### Promoting Audit to Production (Merge-Ready)

When an audit reaches "Level 3 (A - Mastery)" quality:

#### Step 1: Move File from Staging to Production

```bash
# Checkout the PR branch
gh pr checkout {PR_NUMBER}

# Move the audit file to production location
git mv content/textbook/audits/staging/{username}.mdx content/textbook/audits/{canonical-name}.mdx

# Example:
git mv content/textbook/audits/staging/gyanigkali.mdx content/textbook/audits/multimodality_audit.mdx

# Commit the promotion
git commit -m "Promote audit to production: ready for merge"

# Push the change
git push
```

#### Step 2: Merge PR to Staging

```bash
gh pr merge {PR_NUMBER} --merge --repo arpg/vla-foundations
```

This merges the PR into `staging` branch.

#### Step 3: Merge Staging to Main

```bash
git checkout main
git pull origin main
git merge staging
git push origin main
```

#### Step 4: Verify Production Deployment

The `Deploy to Production` workflow will automatically run and:
- Pull latest `main` branch
- Run `npm run build`
- Sync to production server with `rsync --delete`
- Deploy to: `https://vlm-robotics.dev/textbook/audits/{canonical-name}/`

Check the deployment: https://github.com/arpg/vla-foundations/actions/workflows/deploy.yml

### URL Structure

| Environment | Location | Audience |
|------------|----------|----------|
| **PR Preview** | `arpg.github.io/vla-foundations/staging/pulls/{PR}/...` | Student + Instructor |
| **Staging** | `vlm-robotics.dev/staging/...` | Testing only |
| **Production** | `vlm-robotics.dev/textbook/audits/{canonical-name}` | Public |

### File Structure

```
content/textbook/audits/
├── staging/                    # Student drafts under review
│   ├── gyanigkali.mdx         # PR #23 - in review
│   ├── johndoe.mdx            # PR #24 - in review
│   └── .gitkeep
├── multimodality_audit.mdx    # Published audit (merged from staging/)
├── siglip_audit.mdx           # Published audit
└── _template.mdx              # Template for students
```

**Important:** Files in `staging/` are **only visible in preview deployments**, not in production. Once approved, they **must be moved** to the root `audits/` directory to appear at `vlm-robotics.dev`.

### Grading Levels

#### Level 1 (C): Basic Summary
- Correct frontmatter and basic formatting
- Summary of paper's main contributions
- Basic LaTeX rendering

#### Level 2 (B): Technical Deep-Dive
- High technical depth with architectural details
- Correct mathematical formulations
- Sound critique of the model
- Proper citations

#### Level 3 (A - Mastery): Merge-Ready
- Exhaustive technical analysis
- Comparative deep-dive across multiple papers
- Original insights on scaling laws and bottlenecks
- Physical grounding critique (semantic-motor gap, information decay)
- Professional "Senior Staff" engineering quality
- **Ready to merge to main as canonical reference**

### Tips for Effective Reviews

1. **Use GitHub's PR review features** - Line-specific comments are best
2. **Require semantic line breaks** - Makes commenting much easier
3. **Check LaTeX rendering** - View the preview, not just the markdown
4. **Verify references** - Click through to cited papers
5. **Test on mobile** - Ensure equations and diagrams are responsive
6. **Look for the "load-bearing" analysis** - This is what distinguishes A-level work

---

## Server Configuration

### Apache Setup

#### Option 1: Subdirectory (/staging)

Add to `/etc/apache2/sites-available/vlm-robotics.dev.conf`:

```apache
<VirtualHost *:80>
    ServerName vlm-robotics.dev

    # Static production site
    DocumentRoot /var/www/vlm-robotics.dev/public_html

    # Proxy staging requests to Next.js server
    ProxyPass /staging http://localhost:3002
    ProxyPassReverse /staging http://localhost:3002
    ProxyPreserveHost On

    <Directory /var/www/vlm-robotics.dev/public_html>
        AllowOverride All
        Require all granted
    </Directory>
</VirtualHost>
```

#### Option 2: Subdomain (staging.vlm-robotics.dev) [RECOMMENDED]

Create `/etc/apache2/sites-available/staging.vlm-robotics.dev.conf`:

```apache
<VirtualHost *:80>
    ServerName staging.vlm-robotics.dev

    # Proxy everything to Next.js server
    ProxyPass / http://localhost:3002/
    ProxyPassReverse / http://localhost:3002/
    ProxyPreserveHost On

    # WebSocket support (if needed)
    RewriteEngine On
    RewriteCond %{HTTP:Upgrade} websocket [NC]
    RewriteCond %{HTTP:Connection} upgrade [NC]
    RewriteRule ^/?(.*) "ws://localhost:3002/$1" [P,L]
</VirtualHost>
```

Enable and restart:

```bash
sudo a2ensite staging.vlm-robotics.dev.conf
sudo a2enmod proxy proxy_http rewrite
sudo systemctl restart apache2
```

#### API Proxy Configuration

Add to Apache config or create `/var/www/vlm-robotics.dev/public_html/api/.htaccess`:

```apache
RewriteEngine On
RewriteRule ^(.*)$ http://localhost:3001/api/$1 [P,L]
```

Or in your Apache virtual host config:

```apache
<VirtualHost *:80>
    ServerName vlm-robotics.dev

    # Proxy API requests to Node.js
    ProxyPass /api http://localhost:3001/api
    ProxyPassReverse /api http://localhost:3001/api

    # Static files
    DocumentRoot /var/www/vlm-robotics.dev/public_html
</VirtualHost>
```

### Environment Variables

Create `/home/crh/vla-staging/.env.local`:

```bash
# Set base path if using subdirectory
# Leave empty if using subdomain
BASE_PATH=

# Port for Next.js server
PORT=3002
```

### Process Management with systemd

Instead of nohup, use systemd for proper process management:

Create `/etc/systemd/system/vla-staging.service`:

```ini
[Unit]
Description=VLA Foundations Staging (Next.js)
After=network.target

[Service]
Type=simple
User=crh
WorkingDirectory=/home/crh/vla-staging
ExecStart=/usr/bin/npm start -- -p 3002
Restart=always
RestartSec=10
Environment=NODE_ENV=production

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable vla-staging
sudo systemctl start vla-staging
sudo systemctl status vla-staging
```

View logs:

```bash
sudo journalctl -u vla-staging -f
```

### API Server Setup

The commenting system requires a Node.js API server running alongside the static site.

#### Manual API Server Start

SSH to ristoffer.ch:

```bash
cd ~/vla-staging/api-server
npm install
npm start &
```

Or with PM2 (process manager):

```bash
cd ~/vla-staging/api-server
npm install -g pm2
pm2 start server.js --name vla-comments-api
pm2 save
pm2 startup  # Enable auto-start on boot
```

#### Verify API is Running

```bash
# Check PM2 status
pm2 status

# Test health endpoint
curl http://localhost:3001/health

# Test from outside (after proxy setup)
curl https://vlm-robotics.dev/api/comments/23
```

---

## Deployment Procedures

### Quick Start: Manual SSH Deployment

If you want to get the review system running immediately without waiting for automated deployment:

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
```

### Verify Everything Works

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

### Automatic Deployment

The GitHub Actions workflow automatically:
1. Builds and deploys the static site
2. Installs API server dependencies
3. Restarts the API server with PM2

Update `.github/workflows/deploy-staging.yml` to restart the service:

```yaml
- name: Restart staging service
  run: |
    sudo systemctl restart vla-staging
```

### Server Management Commands

**Check Status:**
```bash
ssh crh@ristoffer.ch "ps aux | grep next-server | grep -v grep"
```

**View Logs:**
```bash
ssh crh@ristoffer.ch "tail -f ~/vla-staging-server.log"
```

**Restart Server:**
```bash
ssh crh@ristoffer.ch "cd ~/vla-staging && kill -9 \$(ps aux | grep 'next-server' | grep -v grep | awk '{print \$2}') && nohup npm start -- -p 3002 > ~/vla-staging-server.log 2>&1 &"
```

**Test API:**
```bash
curl http://vlm-robotics.dev/staging/api/comments/23/
```

---

## Troubleshooting

### Linter Issues

#### Linter Keeps Failing

**Check line breaks:**
```bash
# The student's file should have semantic line breaks
# Bad:
Multi-modality is the framework that enables a model to process and generate information across disparate data types. In robotics, this represents the shift from a robot that sees its environment versus one that can understand it.

# Good:
Multi-modality is the framework that enables a model to process and generate information across disparate data types.
In robotics, this represents the shift from a robot that sees its environment versus one that can understand it.
```

**Check git history:**
```bash
# If student has merge commits:
git log --oneline
# Should not see: "Merge branch 'main'" or "Merge branch 'staging'"

# Student needs to rebase:
git rebase staging
git push --force-with-lease
```

### Preview Deployment Issues

#### Preview Isn't Deploying

1. Check GitHub Actions: https://github.com/arpg/vla-foundations/actions
2. Verify linter passed (green checkmark)
3. Check PR comments for deployment link

#### Production Page Still Shows Old Audit

This happens if the deploy script doesn't delete old files.

**Fix:** The deploy.yml now uses `rsync --delete` to properly sync files. Force a redeploy:
```bash
git commit --allow-empty -m "Trigger redeploy"
git push origin main
```

#### Student Can't Find Their Preview

The preview URL is posted automatically in PR comments by the bot:
- Look for the comment from `github-actions` bot
- URL format: `https://arpg.github.io/vla-foundations/staging/pulls/{PR}/textbook/audits/staging/{username}/`

### API Server Issues

#### API not responding:
```bash
cd ~/vla-staging/api-server
pkill -f "node server.js"
nohup node server.js > ~/vla-staging-api.log 2>&1 &
```

#### Check what's using port 3001:
```bash
lsof -i :3001
```

#### View API logs:
```bash
tail -50 ~/vla-staging-api.log
```

#### PM2 Troubleshooting

**API not responding:**
```bash
pm2 logs vla-comments-api
```

**Restart API server:**
```bash
pm2 restart vla-comments-api
```

**Check if port 3001 is in use:**
```bash
lsof -i :3001
```

### Testing Commands

```bash
# Test Next.js server directly
curl http://localhost:3002/

# Test via Apache
curl http://staging.vlm-robotics.dev/

# Test API endpoint
curl http://staging.vlm-robotics.dev/api/comments/23
```

---

## Quick Reference Commands

### Review a PR
```bash
gh pr checkout {PR_NUMBER}
gh pr view {PR_NUMBER} --web
```

### Check linter locally
```bash
python3 scripts/audit_linter.py
```

### Promote to production
```bash
git mv content/textbook/audits/staging/{user}.mdx content/textbook/audits/{name}.mdx
git commit -m "Promote audit to production"
git push
```

### Merge to staging and main
```bash
gh pr merge {PR_NUMBER} --merge
git checkout main && git merge staging && git push origin main
```

### Check server status
```bash
ssh crh@ristoffer.ch "ps aux | grep 'next start' | grep -v grep"
```

### View server logs
```bash
ssh crh@ristoffer.ch "tail -f ~/vla-staging-server.log"
```

### Restart server
```bash
ssh crh@ristoffer.ch "cd ~/vla-staging && pkill -f 'next start' && nohup npm start -- -p 3002 > ~/vla-staging-server.log 2>&1 &"
```

---

## System Architecture

### Components

1. **Audit Pages** (`/textbook/audits/[slug]`)
   - Renders student paper audits from MDX files
   - Located in `content/textbook/audits/`
   - Supports both public view and instructor review mode

2. **Review Mode** (`/textbook/audits/[slug]?review=true`)
   - Adds commenting sidebar for instructor feedback
   - Comments tied to PR numbers for version tracking
   - Visual interface for adding/resolving comments

3. **Comments API** (`/api/comments/`)
   - RESTful API for managing comments
   - Comments stored in `database/comments/{pr-number}.json`
   - Persists in git repository for audit trail

4. **Staging Deployment**
   - Staging branch deploys to `vlm-robotics.dev/staging/`
   - Automatic deployment on push to `staging` branch
   - Allows review before merging to production

### Comment Data Structure

Comments are stored in JSON files:

```json
{
  "prNumber": 23,
  "auditSlug": "multimodality_audit",
  "comments": [
    {
      "id": "uuid",
      "sectionId": "introduction",
      "text": "Great start! Consider expanding on the architectural implications.",
      "author": "instructor",
      "timestamp": "2026-01-21T...",
      "resolved": false
    }
  ]
}
```

### API Endpoints

- `GET /api/comments/{prNumber}` - Fetch all comments for a PR
- `POST /api/comments/{prNumber}` - Add new comment
- `PUT /api/comments/{prNumber}/{commentId}` - Update comment (mark resolved)
- `DELETE /api/comments/{prNumber}/{commentId}` - Delete comment

---

## Local Development

To run locally:

```bash
# Terminal 1: Start API server
cd api-server
npm install
npm start

# Terminal 2: Start Next.js dev server
npm run dev
```

The frontend will proxy API requests to `http://localhost:3001`.

---

**END OF DOCUMENT**
