# Deploy Adelphos Voice Agent to Render

## Quick Deploy

1. Push this code to GitHub
2. Go to https://dashboard.render.com
3. Click "New +" → "Web Service"
4. Connect your GitHub repository
5. Render will auto-detect the `render.yaml` file
6. Add environment variables (see below)
7. Deploy!

## Environment Variables Required

In Render dashboard, add these environment variables:

- `DEEPGRAM_API_KEY` - Your Deepgram API key
- `GROQ_API_KEY` - Your Groq API key

## After Deployment

1. Once deployed, Render gives you a URL like:
   `https://adelphos-voice-agent.onrender.com`

2. Update the frontend HTML file with your backend URL:
   ```javascript
   const WS_URL = 'wss://adelphos-voice-agent.onrender.com/ws/voice';
   ```

3. Share the HTML file - anyone can open it and it will connect to your deployed backend

## Free Tier Limits

- Render free tier: spins down after 15 min inactivity, wakes up on next request (takes ~30s)
- For always-on, upgrade to paid tier or use a uptime ping service

## Alternative: Self-Host

Run locally with ngrok:
```bash
# Terminal 1: Start backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Start ngrok
ngrok http 8000

# Update frontend HTML with ngrok URL
```
