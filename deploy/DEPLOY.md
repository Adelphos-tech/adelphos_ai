# Adelphos Tech — Deployment Guide
## Hostinger VPS + GoDaddy Domain

---

## Step 1 — SSH into your Hostinger VPS

Find your server IP in: **Hostinger hPanel → VPS → Manage → Overview**

```bash
ssh root@YOUR_SERVER_IP
```

---

## Step 2 — Run the setup script (one command)

```bash
curl -s https://raw.githubusercontent.com/Adelphos-tech/adelphos-new-website/main/deploy/setup_server.sh | bash
```

This will:
- Install Python, nginx, certbot, git
- Clone the repo to `/var/www/adelphos`
- Set up a Python virtualenv and install dependencies
- Register a systemd service (auto-restarts on crash/reboot)
- Configure nginx as reverse proxy

---

## Step 3 — Fill in your .env on the server

```bash
nano /var/www/adelphos/.env
```

Add your real keys (copy from your local `.env`):
```
DEEPGRAM_API_KEY=your_key_here
VLLM_BASE_URL=http://your_vllm_server:8005/v1
VLLM_API_KEY=EMPTY
VLLM_MODEL=Qwen/Qwen2.5-14B-Instruct-AWQ
TTS_API_URL=http://your_tts_server:8004/tts
TTS_VOICE=rizwan.wav
HOST=127.0.0.1
PORT=8080
```

Then restart the app:
```bash
systemctl restart adelphos
```

---

## Step 4 — Set your domain in nginx

```bash
nano /etc/nginx/sites-available/adelphos
```

Replace **both** occurrences of `YOUR_DOMAIN.com` with your real domain e.g. `adelphostech.in`

Then test and reload:
```bash
nginx -t && systemctl reload nginx
```

---

## Step 5 — Point GoDaddy DNS to your server

1. Log in to [GoDaddy → My Products → DNS](https://dcc.godaddy.com)
2. Select your domain → **DNS Management**
3. **Delete** any existing `A` record for `@`
4. **Add** a new record:
   - Type: `A`
   - Name: `@`
   - Value: `YOUR_SERVER_IP`
   - TTL: 600
5. **Add** another record:
   - Type: `CNAME`
   - Name: `www`
   - Value: `@`
   - TTL: 600
6. Wait ~5–10 minutes for DNS to propagate

---

## Step 6 — Enable HTTPS (free SSL via Let's Encrypt)

Once DNS is pointing to your server:
```bash
certbot --nginx -d yourdomain.com -d www.yourdomain.com
```

Follow the prompts. Certbot will auto-renew every 90 days.

---

## Useful Commands (on the server)

| Action | Command |
|---|---|
| Check app status | `systemctl status adelphos` |
| View live logs | `journalctl -u adelphos -f` |
| Restart app | `systemctl restart adelphos` |
| Pull latest code | `cd /var/www/adelphos && git pull && systemctl restart adelphos` |
| Check nginx | `nginx -t && systemctl reload nginx` |

---

## Updating the site later

From your local machine, push to GitHub as usual:
```bash
git add -A
git commit -m "your message"
git push origin main
```

Then on the server, pull and restart:
```bash
cd /var/www/adelphos && git pull origin main && systemctl restart adelphos
```
