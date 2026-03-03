#!/bin/bash
# ============================================================
# Adelphos Tech — Hostinger VPS Setup Script
# Run this on your server as root:
#   bash setup_server.sh
# ============================================================
set -e

APP_DIR="/var/www/adelphos"
REPO="https://github.com/Adelphos-tech/adelphos-new-website.git"

echo "==> Updating system packages..."
apt-get update -y && apt-get upgrade -y

echo "==> Installing dependencies..."
apt-get install -y python3 python3-pip python3-venv nginx certbot python3-certbot-nginx git curl

echo "==> Cloning / updating repo..."
if [ -d "$APP_DIR/.git" ]; then
    cd "$APP_DIR" && git pull origin main
else
    git clone "$REPO" "$APP_DIR"
fi

echo "==> Creating Python virtual environment..."
cd "$APP_DIR"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "==> Setting up .env (if not already present)..."
if [ ! -f "$APP_DIR/.env" ]; then
    cp "$APP_DIR/.env.example" "$APP_DIR/.env"
    echo ""
    echo "  *** ACTION REQUIRED ***"
    echo "  Edit $APP_DIR/.env and fill in your real API keys before starting the service."
    echo "  Run:  nano $APP_DIR/.env"
    echo ""
fi

echo "==> Installing systemd service..."
cp "$APP_DIR/deploy/adelphos.service" /etc/systemd/system/adelphos.service
systemctl daemon-reload
systemctl enable adelphos
systemctl restart adelphos
systemctl status adelphos --no-pager

echo "==> Configuring nginx..."
cp "$APP_DIR/deploy/nginx.conf" /etc/nginx/sites-available/adelphos
ln -sf /etc/nginx/sites-available/adelphos /etc/nginx/sites-enabled/adelphos
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl restart nginx

echo ""
echo "============================================================"
echo "  DONE! App is running at http://$(curl -s ifconfig.me)"
echo ""
echo "  Next steps:"
echo "  1. Edit /var/www/adelphos/.env with your real keys"
echo "  2. Replace YOUR_DOMAIN.com in /etc/nginx/sites-available/adelphos"
echo "  3. Run: certbot --nginx -d yourdomain.com -d www.yourdomain.com"
echo "  4. Point GoDaddy DNS A record to this server's IP"
echo "============================================================"
