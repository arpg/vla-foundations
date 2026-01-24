# Apache Configuration for Dynamic Staging

## Option 1: Subdirectory (/staging)

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

## Option 2: Subdomain (staging.vlm-robotics.dev) [RECOMMENDED]

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

## Environment Variables

Create `/home/crh/vla-staging/.env.local`:

```bash
# Set base path if using subdirectory
# Leave empty if using subdomain
BASE_PATH=

# Port for Next.js server
PORT=3002
```

## Process Management

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

## Testing

```bash
# Test Next.js server directly
curl http://localhost:3002/

# Test via Apache
curl http://staging.vlm-robotics.dev/

# Test API endpoint
curl http://staging.vlm-robotics.dev/api/comments/23
```

## Automatic Deployment

Update `.github/workflows/deploy-staging.yml` to restart the service:

```yaml
- name: Restart staging service
  run: |
    sudo systemctl restart vla-staging
```
