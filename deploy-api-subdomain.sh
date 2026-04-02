#!/bin/bash
# Run this on the VPS to set up api.adelphostech.com nginx config
# Usage: bash deploy-api-subdomain.sh

# 1. Write nginx config for api.adelphostech.com
sudo tee /etc/nginx/sites-available/api.adelphostech.com > /dev/null << 'NGINX_EOF'
server {
    listen 80;
    listen [::]:80;
    server_name api.adelphostech.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    listen [::]:443 ssl;
    server_name api.adelphostech.com;

    ssl_certificate     /etc/letsencrypt/live/adelphostech.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/adelphostech.com/privkey.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         HIGH:!aNULL:!MD5;

    client_max_body_size 50m;

    # WebSocket endpoint
    location /ws/ {
        proxy_pass http://127.0.0.1:8030;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
        proxy_connect_timeout 60s;
        proxy_buffering off;
        proxy_cache off;
    }

    # All other API endpoints
    location / {
        proxy_pass http://127.0.0.1:8030;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
        proxy_send_timeout 120s;
        proxy_buffering off;
    }
}
NGINX_EOF

# 2. Enable the site
sudo ln -sf /etc/nginx/sites-available/api.adelphostech.com /etc/nginx/sites-enabled/api.adelphostech.com

# 3. Test and reload nginx
sudo nginx -t && sudo systemctl reload nginx

# 4. Pull latest code and restart backend
cd /home/amlak/adelphos-property && git pull origin main
sudo systemctl restart adelphos-property
sleep 5
sudo systemctl is-active adelphos-property

echo "Done! api.adelphostech.com is now configured."
echo "Next: Add DNS A record -> api.adelphostech.com -> 83.111.242.6"
