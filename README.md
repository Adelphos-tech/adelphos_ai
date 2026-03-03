# Adelphos Tech — Website with AI Voice Agent Demo

A modern website for Adelphos Tech featuring a real-time AI Voice Agent demo.

## Features

- **Homepage** — Modern landing page showcasing services, tech stack, and hiring models
- **Voice Agent Demo** — Real-time AI voice assistant powered by:
  - **STT** — Deepgram API (Nova-2 model)
  - **LLM** — Local vLLM (OpenAI-compatible)
  - **TTS** — Local TTS API

## Architecture

```
User speaks → [Deepgram STT] → text → [vLLM] → response → [Local TTS] → audio playback
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Required settings:
- `DEEPGRAM_API_KEY` — Get from https://console.deepgram.com
- `VLLM_BASE_URL` — Your local vLLM endpoint (default: `http://localhost:8000/v1`)
- `VLLM_MODEL` — Model name served by vLLM
- `TTS_API_URL` — Your local TTS API endpoint (default: `http://localhost:8020/tts`)

### 3. Run the server

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8080 --reload
```

Then open http://localhost:8080 in your browser.

## Pages

| Route | Description |
|---|---|
| `/` | Adelphos Tech homepage |
| `/voice-agent` | AI Voice Agent demo |

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/voice` | POST | Full voice pipeline (audio in → audio out) |
| `/chat` | POST | Text chat endpoint |
| `/stt` | POST | Speech-to-text only |
| `/tts` | POST | Text-to-speech only |
| `/tts/voices` | GET | List available TTS voices |
| `/health` | GET | Health check |

## Project Structure

```
├── backend/
│   ├── main.py           # FastAPI server & routes
│   ├── stt_handler.py    # Deepgram STT integration
│   ├── tts_handler.py    # Local TTS integration
│   └── llm_handler.py    # vLLM integration
├── frontend/
│   ├── index.html        # Adelphos Tech homepage
│   └── voice-agent.html  # Voice Agent demo page
├── .env.example
├── requirements.txt
└── README.md
```
