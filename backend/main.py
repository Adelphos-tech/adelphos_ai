import os
import sys
import json
import time
import uuid
import asyncio
import base64
import re
import io
import wave
from typing import Optional
import numpy as np

print("[STARTUP] Loading Adelphos Voice Agent...", flush=True)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

# Check required environment variables
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

if not DEEPGRAM_API_KEY:
    print("[ERROR] DEEPGRAM_API_KEY not set! Please set it in Render environment variables.", flush=True)
if not GROQ_API_KEY:
    print("[ERROR] GROQ_API_KEY not set! Please set it in Render environment variables.", flush=True)

print(f"[STARTUP] DEEPGRAM_API_KEY present: {bool(DEEPGRAM_API_KEY)}", flush=True)
print(f"[STARTUP] GROQ_API_KEY present: {bool(GROQ_API_KEY)}", flush=True)

# Import handlers (with error handling)
try:
    from backend.stt_handler import transcribe_audio
    from backend.tts_handler import tts_sentence, get_available_voices
    from backend.llm_handler import generate_response, build_messages
    from backend.qdrant_handler import search_properties, format_properties_for_llm
    print("[STARTUP] All handlers imported successfully", flush=True)
except Exception as e:
    print(f"[ERROR] Failed to import handlers: {e}", flush=True)
    raise

app = FastAPI()

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "voice-agent")
INDEX = os.path.join(FRONTEND, "index.html")

# In-memory chat storage
chat_store: dict = {}

# ─── Deepgram Live Streaming Config ───
SAMPLE_RATE = 16000
BARGE_ENERGY_DB = -38.0
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
DEEPGRAM_LANG_CODES = {
    "en": "en",
    "zh": "zh",
    "ms": "ms",
    "ta": "ta",
}

# ─── Property Search Triggers ───
PROPERTY_TRIGGERS = [
    "property", "properties", "flat", "apartment", "condo", "hdb", "landed",
    "bedroom", "bedrooms", "br", "bhk", "studio", "penthouse", "bungalow",
    "rent", "rental", "buy", "sale", "for sale", "for rent",
    "district", "location", "price", "budget", "sgd", "house", "home",
    "show me", "find me", "looking for", "available", "listing",
    "sqft", "square feet", "psf", "freehold", "leasehold", "ec ", "mrt",
    "orchard", "marina", "jurong", "tampines", "punggol", "woodlands",
    "bukit timah", "holland", "katong", "bedok", "sentosa", "cbd",
]

_FOLLOWUP_TRIGGERS = [
    "tell me more", "more about", "the first", "the second", "the third",
    "that one", "this one", "which one", "both of them", "all of them",
    "cheaper", "more expensive", "bigger", "smaller", "similar",
    "what about", "how about", "any other", "other option", "different",
    "show more", "more listings", "more options", "another one",
    "compare", "difference", "between them", "is it", "does it have",
    "view details", "more details", "link", "contact", "agent",
]


def _is_property_query(text: str, history: list[dict] = None) -> bool:
    """Detect if user is asking about properties."""
    t = text.lower()
    if any(kw in t for kw in PROPERTY_TRIGGERS):
        return True
    if history and any(kw in t for kw in _FOLLOWUP_TRIGGERS):
        recent = history[-6:]
        for msg in recent:
            if msg.get("role") == "assistant":
                c = (msg.get("content") or "").lower()
                if any(kw in c for kw in ["sgd", "bedroom", "district", "sqft", "listing"]):
                    return True
    return False


def frame_db(i16: np.ndarray) -> float:
    """Calculate dB level of a PCM16 frame."""
    if i16.size == 0:
        return -100.0
    f = i16.astype(np.float32) / 32768.0
    rms = np.sqrt(np.mean(f ** 2))
    if rms < 1e-10:
        return -100.0
    return 20 * np.log10(rms)


def looks_like_noise(text: str) -> bool:
    """Filter out common STT noise artifacts."""
    if not text:
        return True
    t = text.lower().strip()
    if len(t) <= 2:
        return True
    noise_exact = {
        "you", "bye", "the", "a", "hmm", "uh", "um", "oh", "ah",
        "so", "subscribe", "like and subscribe", "thanks for watching",
        "thank you for watching",
    }
    t_clean = re.sub(r'[^\w\s]', '', t).strip()
    if t_clean in noise_exact:
        return True
    noise_substr = {"background music", "applause", "caption", "subtitles"}
    return any(frag in t for frag in noise_substr)


# ─── Voice Session Class ───

class VoiceSession:
    """Manages state for a single WebSocket voice session."""
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.chat_id = str(uuid.uuid4())
        self.is_ai_speaking = False
        self.barged_in = False
        self._cancelled = False
        self.turn_count = 0
        self.voice = os.getenv("TTS_VOICE", "james.wav")
        self.stt_language = "en"
        self.greeting_sent = False

    def cancel(self):
        self._cancelled = True

    @property
    def cancelled(self):
        return self._cancelled


# ─── WebSocket Voice Endpoint ───

@app.websocket("/ws/voice")
async def voice_ws(ws: WebSocket):
    """
    WebSocket voice pipeline with barge-in support.

    Client sends:
      - Binary: raw PCM16 audio frames (Int16Array)
      - JSON: { "type": "text", "text": "..." }
              { "type": "barge_in" }
              { "type": "set_voice", "voice": "test.wav" }
              { "type": "set_language", "language": "en" }

    Server sends:
      { "type": "status", "status": "ready" }
      { "type": "user_text", "text": "..." }
      { "type": "interim_transcript", "text": "..." }
      { "type": "ai_text", "text": "..." }
      { "type": "ai_audio", "data": "base64", "sentence_idx": N, "chunk_index": M, "total_chunks": T, "property_idx": P }
      { "type": "ai_audio_done" }
      { "type": "barge_in_ack" }
      { "type": "properties", "data": [...] }
      { "type": "error", "message": "..." }
    """
    await ws.accept()
    session = VoiceSession(ws)
    print(f"[WS] Voice session connected: {session.chat_id}")

    if session.chat_id not in chat_store:
        chat_store[session.chat_id] = {"messages": []}

    current_task: asyncio.Task | None = None
    TTS_SEM = asyncio.Semaphore(4)
    CHUNK_BYTES = 36000

    # Deepgram STT connection
    stt_ws = None
    stt_task = None
    interim_text = ""
    final_text = ""

    async def send_audio(audio: bytes, sentence_idx: int = 0, property_idx: int = None):
        """Send audio in base64 chunks with optional property highlighting."""
        if not audio:
            return
        total = (len(audio) + CHUNK_BYTES - 1) // CHUNK_BYTES
        for i in range(total):
            chunk = audio[i * CHUNK_BYTES:(i + 1) * CHUNK_BYTES]
            b64 = base64.b64encode(chunk).decode('utf-8')
            try:
                msg = {
                    "type": "ai_audio",
                    "data": b64,
                    "sentence_idx": sentence_idx,
                    "chunk_index": i,
                    "total_chunks": total,
                }
                if property_idx is not None:
                    msg["property_idx"] = property_idx
                await ws.send_json(msg)
            except Exception as e:
                print(f"[WS] send_audio error: {e}")
                break

    def _detect_property_in_sentence(sentence: str, properties: list) -> int:
        """Detect which property index a sentence mentions. Returns index or None."""
        if not properties:
            return None
        s_lower = sentence.lower()
        for idx, prop in enumerate(properties):
            # Check property title keywords
            title = prop.get("title", "").lower()
            # Extract key terms from title (skip common words)
            key_terms = [w for w in title.split() if len(w) > 3 and w not in
                        ["residences", "apartment", "condominium", "house", "flat",
                         "estate", "home", "property", "listing", "for", "rent", "sale", "at", "the"]]
            for term in key_terms[:3]:  # Check first 3 key terms
                if term in s_lower:
                    return idx
            # Check district mention
            district = prop.get("district", "").lower()
            if district and district.split()[0] in s_lower:
                return idx
            # Check address keywords
            address = prop.get("address", "").lower()
            if address:
                addr_parts = address.split()[:2]  # First 2 words of address
                if any(part in s_lower for part in addr_parts if len(part) > 3):
                    return idx
        return None

    async def process_tts_stream(text: str, sentence_idx: int = 0, properties: list = None):
        """Generate TTS and stream audio with property highlighting."""
        async with TTS_SEM:
            if session.cancelled:
                print(f"[TTS] Skipping sentence {sentence_idx} - session cancelled")
                return
            print(f"[TTS] Generating audio for sentence {sentence_idx}: '{text[:50]}...'")
            try:
                audio = await tts_sentence(text, voice=session.voice)
                if audio and not session.cancelled:
                    # Detect which property this sentence mentions
                    prop_idx = _detect_property_in_sentence(text, properties) if properties else None
                    if prop_idx is not None:
                        print(f"[TTS] Sentence {sentence_idx} mentions property {prop_idx}")
                    print(f"[TTS] Sentence {sentence_idx} generated {len(audio)} bytes")
                    await send_audio(audio, sentence_idx, prop_idx)
                elif not audio:
                    print(f"[TTS] No audio generated for sentence {sentence_idx}")
            except Exception as e:
                print(f"[TTS] Error: {e}")

    async def handle_ai_response(user_text: str):
        """Process user text and generate AI response."""
        session.turn_count += 1
        history = chat_store[session.chat_id]["messages"]
        print(f"[LLM] Processing user text: '{user_text[:50]}...'")

        # Check if property query
        properties = []
        if _is_property_query(user_text, history):
            try:
                properties = await search_properties(user_text, limit=4)
                if properties:
                    await ws.send_json({"type": "properties", "data": properties})
            except Exception as e:
                print(f"[SEARCH] Error: {e}")

        # Build messages with property context and generate response
        messages, _ = await build_messages(user_text, history, properties if properties else None)
        try:
            ai_response = generate_response(messages)
            print(f"[LLM] Generated response: '{ai_response[:50]}...'")
        except Exception as e:
            print(f"[LLM] Error: {e}")
            ai_response = "Sorry, I had trouble processing that. Could you try again?"

        # Clean up response
        ai_response = ai_response.replace("**", "").replace("###", "").replace("---", "").strip()

        # Send text response
        await ws.send_json({"type": "ai_text", "text": ai_response})

        # Update history
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": ai_response})

        # Generate TTS and stream audio
        if ai_response:
            session.is_ai_speaking = True
            await ws.send_json({"type": "status", "status": "speaking"})

            # Split into sentences for streaming
            sentences = re.split(r'(?<=[.!?])\s+', ai_response)
            sentences = [s.strip() for s in sentences if s.strip()]

            for idx, sentence in enumerate(sentences):
                if session.cancelled:
                    break
                await process_tts_stream(sentence, sentence_idx=idx, properties=properties)

            if not session.cancelled:
                await ws.send_json({"type": "ai_audio_done"})

            session.is_ai_speaking = False
            if not session.cancelled:
                await ws.send_json({"type": "status", "status": "ready"})

    async def connect_deepgram_stt():
        """Connect to Deepgram live streaming STT."""
        nonlocal stt_ws, stt_task, interim_text, final_text

        if not DEEPGRAM_API_KEY:
            print("[STT] No Deepgram API key")
            return

        lang = DEEPGRAM_LANG_CODES.get(session.stt_language, "en")
        deepgram_url = (
            f"wss://api.deepgram.com/v1/listen?"
            f"model=nova-3&encoding=linear16&sample_rate=16000&channels=1"
            f"&punctuate=true&smart_format=true&filler_words=false"
            f"&interim_results=true&endpointing=200&language={lang}"
        )

        import websockets
        import ssl
        import certifi
        print(f"[STT] Connecting to Deepgram...")
        try:
            # Create SSL context using certifi certificates
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            stt_ws = await websockets.connect(
                deepgram_url,
                additional_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
                ssl=ssl_context
            )
            print(f"[STT] Connected to Deepgram for language: {lang}")

            async def receive_stt():
                nonlocal interim_text, final_text
                try:
                    async for message in stt_ws:
                        if session.cancelled:
                            break
                        data = json.loads(message)
                        channel = data.get("channel", {})
                        alt = channel.get("alternatives", [{}])[0]

                        if data.get("is_final"):
                            transcript = alt.get("transcript", "").strip()
                            if transcript and not looks_like_noise(transcript):
                                final_text = transcript
                                interim_text = ""
                                await ws.send_json({"type": "user_text", "text": transcript})
                                # Trigger AI response
                                await handle_ai_response(transcript)
                        else:
                            transcript = alt.get("transcript", "").strip()
                            if transcript:
                                interim_text = transcript
                                await ws.send_json({"type": "interim_transcript", "text": transcript})
                except Exception as e:
                    print(f"[STT] Receive error: {e}")

            stt_task = asyncio.create_task(receive_stt())

        except Exception as e:
            print(f"[STT] Connection error: {e}")

    async def disconnect_deepgram_stt():
        """Disconnect from Deepgram STT."""
        nonlocal stt_ws, stt_task
        if stt_ws:
            try:
                await stt_ws.close()
            except:
                pass
            stt_ws = None
        if stt_task:
            stt_task.cancel()
            try:
                await stt_task
            except asyncio.CancelledError:
                pass
            stt_task = None

    # Send ready status
    await ws.send_json({"type": "status", "status": "ready", "chat_id": session.chat_id})

    # Connect to Deepgram STT immediately
    await connect_deepgram_stt()

    # Track if AI response is running to avoid concurrent processing
    ai_task = None

    try:
        while True:
            # Receive message with timeout to allow periodic checks
            try:
                data = await asyncio.wait_for(ws.receive(), timeout=0.1)
            except asyncio.TimeoutError:
                # No message received in 100ms, check if we need to forward audio
                continue

            if "text" in data:
                # JSON message
                msg = json.loads(data["text"])
                msg_type = msg.get("type")

                if msg_type == "text":
                    user_text = msg.get("text", "").strip()
                    if user_text:
                        # Cancel any existing AI task
                        if ai_task and not ai_task.done():
                            session.cancel()
                            ai_task.cancel()
                            try:
                                await ai_task
                            except asyncio.CancelledError:
                                pass
                            session._cancelled = False
                        # Start AI response in background so audio can still flow
                        ai_task = asyncio.create_task(handle_ai_response(user_text))

                elif msg_type == "barge_in":
                    session.cancel()
                    if ai_task and not ai_task.done():
                        ai_task.cancel()
                        try:
                            await ai_task
                        except asyncio.CancelledError:
                            pass
                    await ws.send_json({"type": "barge_in_ack"})
                    await disconnect_deepgram_stt()
                    await connect_deepgram_stt()
                    session.barged_in = False
                    session._cancelled = False
                    interim_text = ""
                    final_text = ""
                    ai_task = None

                elif msg_type == "set_voice":
                    voice = msg.get("voice")
                    if voice:
                        session.voice = voice
                        print(f"[WS] Voice set to: {voice}")

                elif msg_type == "set_language":
                    lang = msg.get("language")
                    if lang and lang in DEEPGRAM_LANG_CODES:
                        session.stt_language = lang
                        print(f"[WS] Language set to: {lang}")
                        # Reconnect STT with new language
                        await disconnect_deepgram_stt()
                        await connect_deepgram_stt()

                elif msg_type == "announce_voice":
                    text = msg.get("text", "")
                    if text:
                        await process_tts_stream(text, sentence_idx=0)
                        await ws.send_json({"type": "ai_audio_done"})

            elif "bytes" in data:
                # Binary audio data - forward to Deepgram
                audio_bytes = data["bytes"]
                if stt_ws:
                    try:
                        await stt_ws.send(audio_bytes)
                    except Exception as e:
                        print(f"[STT] Send error: {e}")

    except WebSocketDisconnect:
        print(f"[WS] Disconnected: {session.chat_id}")
    except Exception as e:
        print(f"[WS] Error: {e}")
    finally:
        session.cancel()
        if ai_task and not ai_task.done():
            ai_task.cancel()
        await disconnect_deepgram_stt()
        print(f"[WS] Session ended: {session.chat_id}")


# ─── HTTP Endpoints ───

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
@app.get("/voice-agent")
@app.get("/voice-agent/")
async def index():
    return FileResponse(INDEX)


app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="static")
