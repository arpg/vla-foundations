# API Server Setup for Comments

The commenting system requires a Node.js API server running alongside the static site.

## Architecture

- **Static Site**: Next.js static export (no server-side rendering)
- **API Server**: Express.js server on port 3001
- **Proxy**: Apache/Nginx proxies `/api/*` requests to `localhost:3001`

## Server Setup

### 1. Apache Proxy Configuration

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

### 2. Manual API Server Start

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

### 3. Verify API is Running

```bash
# Check PM2 status
pm2 status

# Test health endpoint
curl http://localhost:3001/health

# Test from outside (after proxy setup)
curl https://vlm-robotics.dev/api/comments/23
```

## Deployment

The GitHub Actions workflow automatically:
1. Builds and deploys the static site
2. Installs API server dependencies
3. Restarts the API server with PM2

## Troubleshooting

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
