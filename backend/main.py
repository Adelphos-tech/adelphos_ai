import os
import json
import time
import uuid
import asyncio
import base64
import random
import re
import io
import wave
import struct
from typing import Optional
import numpy as np

# Fix macOS SSL certificate verification (needed for Deepgram on local dev)
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass
import websockets as ws_lib
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

from backend.stt_handler import transcribe_audio, correct_transcript, TECH_KEYWORDS
from backend.tts_handler import text_to_speech, tts_sentence, get_available_voices
import httpx
from backend.llm_handler import generate_response, generate_response_stream, build_messages
from backend.qdrant_handler import search_properties, format_properties_for_llm
from backend.analytics import (
    init_db as init_analytics_db, create_session as analytics_create_session,
    end_session as analytics_end_session, update_session_voice,
    log_event, log_query, log_barge_in, log_error,
    get_dashboard_stats, get_recent_sessions, get_session_detail,
    get_recent_queries, get_recent_errors,
)

app = FastAPI(title="Adelphos Tech")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Transcribed-Text", "X-Response-Text", "X-Processing-Time"],
)

# In-memory chat storage (replace with DB for production)
chat_store: dict = {}

# ─── Pre-cached filler audio (to fill silence during processing) ───
FILLER_PHRASES_GENERAL = [
    "Hmm, good question, give me a moment.",
    "Let me think about that.",
    "One sec.",
    "Okay, let me get back to you on that.",
    "Let me look into that for you.",
    "That's a great question, hold on.",
]
FILLER_PHRASES = FILLER_PHRASES_GENERAL
filler_cache_general: list[bytes] = []
filler_cache: list[bytes] = []  # all fillers combined (fallback)

@app.on_event("startup")
async def pregenerate_fillers():
    """Pre-generate filler audio at startup so we can send them instantly."""
    import httpx as _httpx
    print("[FILLER] Pre-generating filler audio...")
    async with _httpx.AsyncClient(timeout=30.0) as client:
        for phrase in FILLER_PHRASES_GENERAL:
            try:
                audio = await tts_sentence(phrase, client=client)
                if audio:
                    filler_cache_general.append(audio)
                    filler_cache.append(audio)
                    print(f"[FILLER] General: '{phrase}' ({len(audio)} bytes)")
            except Exception as e:
                print(f"[FILLER] Failed '{phrase}': {e}")
    print(f"[FILLER] {len(filler_cache)} fillers ready ({len(filler_cache_general)} general)")
    init_analytics_db()
    print("[STARTUP] Analytics DB ready")

# Mount frontend
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
async def read_root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/voice-agent")
async def voice_agent_page():
    return FileResponse(os.path.join(FRONTEND_DIR, "voice-agent", "index.html"))


@app.get("/voice-agent/")
async def voice_agent_page_slash():
    return FileResponse(os.path.join(FRONTEND_DIR, "voice-agent", "index.html"))


@app.get("/logo.png")
async def serve_logo():
    return FileResponse(os.path.join(FRONTEND_DIR, "logo.png"), media_type="image/png")


@app.get("/admin")
async def admin_page():
    return FileResponse(os.path.join(FRONTEND_DIR, "admin", "index.html"))

@app.get("/admin/")
async def admin_page_slash():
    return FileResponse(os.path.join(FRONTEND_DIR, "admin", "index.html"))

@app.get("/health")
async def health_check():
    return {"status": "ok"}


# ─── Admin API Endpoints ───

ADMIN_KEY = os.getenv("ADMIN_KEY", "adelphos2024")

def _check_admin(request):
    key = request.headers.get("x-admin-key") or request.query_params.get("key")
    if key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")

@app.get("/api/admin/dashboard")
async def admin_dashboard(request: Request, days: int = 7):
    _check_admin(request)
    return get_dashboard_stats(days)

@app.get("/api/admin/sessions")
async def admin_sessions(request: Request, limit: int = 50, offset: int = 0):
    _check_admin(request)
    return get_recent_sessions(limit, offset)

@app.get("/api/admin/sessions/{session_id}")
async def admin_session_detail(session_id: str, request: Request):
    _check_admin(request)
    return get_session_detail(session_id)

@app.get("/api/admin/queries")
async def admin_queries(request: Request, limit: int = 100, offset: int = 0, failed: bool = False):
    _check_admin(request)
    return get_recent_queries(limit, offset, failed)

@app.get("/api/admin/errors")
async def admin_errors(request: Request, limit: int = 50):
    _check_admin(request)
    return get_recent_errors(limit)

@app.post("/api/admin/upload-properties")
async def upload_properties(request: Request, file: UploadFile = File(...)):
    """Upload Excel file with property data and ingest into Qdrant."""
    _check_admin(request)
    if not file.filename.endswith(('.xlsx', '.xls', '.csv')):
        raise HTTPException(status_code=400, detail="File must be .xlsx, .xls, or .csv")
    
    import tempfile, shutil
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    
    try:
        from backend.qdrant_handler import ingest_properties_from_file
        result = await ingest_properties_from_file(tmp_path)
        return result
    except ImportError:
        raise HTTPException(status_code=501, detail="Property ingestion not yet implemented")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


# ─── Chat Management ───

@app.post("/chats")
async def create_chat():
    chat_id = str(uuid.uuid4())
    chat_store[chat_id] = {"messages": []}
    return {"chat_id": chat_id}


@app.get("/chats/{chat_id}")
async def get_chat(chat_id: str):
    if chat_id not in chat_store:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat_store[chat_id]


# ─── Voice Pipeline: Audio In → Text → LLM → TTS → Audio Out ───

@app.post("/voice")
async def voice_pipeline(
    audio: UploadFile = File(...),
    chat_id: Optional[str] = Form(None),
    voice: Optional[str] = Form(None),
):
    """
    Full voice pipeline:
    1. STT (Deepgram) - audio → text
    2. LLM (vLLM) - text → response
    3. TTS (local) - response → audio
    Returns WAV audio with metadata headers.
    """
    start_time = time.time()

    # 1. Read audio
    audio_bytes = await audio.read()
    print(f"[VOICE] Received audio: {audio.filename}, {len(audio_bytes)} bytes")

    # 2. STT via Deepgram
    try:
        transcribed_text, audio_duration = await transcribe_audio(audio_bytes, audio.filename or "audio.webm")
        if not transcribed_text.strip():
            return Response(
                content=json.dumps({"error": "Could not transcribe audio. Please speak more clearly."}),
                media_type="application/json",
                status_code=400,
            )
        print(f"[VOICE] STT: '{transcribed_text}'")
    except Exception as e:
        print(f"[VOICE] STT Error: {e}")
        return Response(
            content=json.dumps({"error": f"Speech recognition failed: {str(e)}"}),
            media_type="application/json",
            status_code=500,
        )

    # 3. Manage chat history
    if not chat_id or chat_id not in chat_store:
        chat_id = str(uuid.uuid4())
        chat_store[chat_id] = {"messages": []}

    history = chat_store[chat_id]["messages"]
    messages = build_messages(transcribed_text, history)

    # 4. LLM response
    try:
        ai_response = generate_response(messages)
        # Clean response for TTS (strip markdown artifacts)
        ai_response = ai_response.replace("**", "").replace("###", "").replace("---", "")
        print(f"[VOICE] LLM: '{ai_response[:100]}...'")
    except Exception as e:
        print(f"[VOICE] LLM Error: {e}")
        return Response(
            content=json.dumps({
                "transcribed_text": transcribed_text,
                "error": f"LLM failed: {str(e)}",
            }),
            media_type="application/json",
            status_code=500,
        )

    # Save to history
    history.append({"role": "user", "content": transcribed_text})
    history.append({"role": "assistant", "content": ai_response})

    # 5. TTS
    ALLOWED_VOICES = {"otherwavs/habib.wav", "otherwavs/shivang.wav"}
    tts_voice = voice if voice in ALLOWED_VOICES else None
    try:
        tts_audio = await text_to_speech(ai_response, tts_voice)
        if not tts_audio:
            return Response(
                content=json.dumps({
                    "transcribed_text": transcribed_text,
                    "response_text": ai_response,
                    "chat_id": chat_id,
                    "tts_error": "TTS generation failed",
                }),
                media_type="application/json",
            )
    except Exception as e:
        print(f"[VOICE] TTS Error: {e}")
        return Response(
            content=json.dumps({
                "transcribed_text": transcribed_text,
                "response_text": ai_response,
                "chat_id": chat_id,
                "tts_error": str(e),
            }),
            media_type="application/json",
        )

    latency = time.time() - start_time
    print(f"[VOICE] Complete pipeline in {latency:.2f}s")

    # Sanitize headers
    def sanitize(text):
        return (
            text.replace("\n", " ")
            .replace("\u2014", "-")
            .replace("\u2013", "-")
            .replace("\u2018", "'")
            .replace("\u2019", "'")
            .replace("\u201c", '"')
            .replace("\u201d", '"')
            .encode("latin-1", "ignore")
            .decode("latin-1")
        )

    return Response(
        content=tts_audio,
        media_type="audio/wav",
        headers={
            "X-Transcribed-Text": sanitize(transcribed_text[:500]),
            "X-Response-Text": sanitize(ai_response[:2000]),
            "X-Chat-Id": chat_id,
            "X-Processing-Time": f"{latency:.2f}s",
        },
    )


# ─── Text-only chat (for typing mode) ───

PROPERTY_TRIGGERS = [
    "property", "properties", "flat", "apartment", "condo", "hdb", "landed",
    "bedroom", "bedrooms", "br", "bhk", "studio", "penthouse", "bungalow",
    "rent", "rental", "buy", "sale", "for sale", "for rent",
    "district", "location", "price", "budget", "sgd", "house", "home",
    "show me", "find me", "looking for", "available", "listing",
]

def _is_property_query(text: str) -> bool:
    """Detect if user is asking about properties."""
    t = text.lower()
    return any(kw in t for kw in PROPERTY_TRIGGERS)


@app.post("/chat")
async def text_chat(request: dict):
    """Text-based chat endpoint with property search and LLM response."""
    question = request.get("message") or request.get("question", "")
    chat_id = request.get("chat_id")

    if not question.strip():
        raise HTTPException(status_code=400, detail="Question is required")

    if not chat_id or chat_id not in chat_store:
        chat_id = str(uuid.uuid4())
        chat_store[chat_id] = {"messages": []}

    history = chat_store[chat_id]["messages"]

    # Search properties if relevant
    properties = []
    property_context = ""
    if _is_property_query(question):
        try:
            properties = await search_properties(question, limit=4)
            if properties:
                property_context = format_properties_for_llm(properties)
        except Exception as e:
            print(f"[CHAT] Property search error: {e}")

    messages, _ = await build_messages(
        question + (f"\n\n[Available listings:\n{property_context}]" if property_context else ""),
        history
    )

    try:
        ai_response = generate_response(messages)
        ai_response = ai_response.replace("**", "").replace("###", "").replace("---", "")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM failed: {str(e)}")

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": ai_response})

    return {
        "response": ai_response,
        "chat_id": chat_id,
        "properties": properties,
    }


# ─── Individual STT/TTS endpoints ───

@app.post("/stt")
async def stt_endpoint(audio: UploadFile = File(...)):
    """Transcribe audio to text using Deepgram."""
    audio_bytes = await audio.read()
    try:
        text, duration = await transcribe_audio(audio_bytes, audio.filename or "audio.webm")
        return {"text": text, "duration_seconds": duration}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/tts")
async def tts_endpoint(text: str = Form(...), voice: str = Form(None)):
    """Convert text to speech using local TTS API."""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

    audio_bytes = await text_to_speech(text, voice)
    if not audio_bytes:
        raise HTTPException(status_code=500, detail="TTS generation failed")

    return Response(content=audio_bytes, media_type="audio/wav")


@app.get("/tts/voices")
async def list_voices():
    """Get available TTS voices."""
    voices = await get_available_voices()
    return {"voices": voices}


# ─── Deepgram Live Streaming Config ───
SAMPLE_RATE = 16000
BARGE_ENERGY_DB = -38.0      # energy level for barge-in detection
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
# Build keyword boost query params from the shared list in stt_handler
_KW_PARAMS = "&".join(
    f"keywords={kw.replace(' ', '%20')}" for kw in TECH_KEYWORDS
)
DEEPGRAM_WS_URL = (
    "wss://api.deepgram.com/v1/listen?"
    "model=nova-2&language=en&encoding=linear16&sample_rate=16000&channels=1"
    "&punctuate=true&smart_format=true&filler_words=false"
    "&interim_results=true&endpointing=300"
    "&vad_events=true&utterance_end_ms=1000"
    f"&{_KW_PARAMS}"
)


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
    # Very short transcripts (1-2 chars) are noise
    if len(t) <= 2:
        return True
    # Exact-match noise phrases (common STT hallucinations)
    noise_exact = {
        "you", "bye", "the", "a", "hmm", "uh", "um", "oh", "ah",
        "yeah", "okay", "ok", "so", "thank you", "thanks",
        "subscribe", "like and subscribe", "thanks for watching",
        "thank you for watching",
    }
    # Strip trailing punctuation for comparison
    t_clean = re.sub(r'[^\w\s]', '', t).strip()
    if t_clean in noise_exact:
        return True
    # Substring noise (longer phrases that indicate non-speech)
    noise_substr = {"background music", "applause", "caption", "subtitles"}
    return any(frag in t for frag in noise_substr)


# ─── WebSocket Voice Pipeline with Barge-in ───

class VoiceSession:
    """Manages state for a single WebSocket voice session."""
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.chat_id = str(uuid.uuid4())
        self.is_ai_speaking = False
        self.barged_in = False
        self._cancelled = False
        self.turn_count = 0          # track conversation turns for filler logic
        self.voice = os.getenv("TTS_VOICE", "rizwan.wav")  # selected TTS voice

    def cancel(self):
        self._cancelled = True

    @property
    def cancelled(self):
        return self._cancelled


@app.websocket("/ws/voice")
async def voice_ws(ws: WebSocket):
    """
    WebSocket voice pipeline with barge-in support.

    Client sends JSON messages:
      { "type": "audio",  "data": "<base64 webm audio>" }
      { "type": "text",   "text": "typed message" }
      { "type": "barge_in" }                           -- interrupt AI speech
      { "type": "set_chat_id", "chat_id": "..." }

    Server sends JSON messages:
      { "type": "user_text",      "text": "..." }
      { "type": "ai_text",        "text": "...", "chat_id": "..." }
      { "type": "ai_audio",       "data": "<base64 wav>", "chunk_index": N, "total_chunks": M }
      { "type": "ai_audio_done" }
      { "type": "barge_in_ack" }
      { "type": "status",         "status": "..." }
      { "type": "error",          "message": "..." }
    """
    await ws.accept()
    session = VoiceSession(ws)
    print(f"[WS] Voice session connected: {session.chat_id}")
    # Analytics: create session
    _client_ip = ws.client.host if ws.client else ""
    analytics_create_session(session.chat_id, ip=_client_ip, user_agent="", voice=session.voice)

    # Ensure chat store entry exists
    if session.chat_id not in chat_store:
        chat_store[session.chat_id] = {"messages": []}

    # Track the current AI response task so we can cancel on barge-in
    current_task: asyncio.Task | None = None

    # ── Concurrency: allow multiple TTS calls in parallel ──
    TTS_SEM = asyncio.Semaphore(4)

    # ── Send ready status immediately so the frontend unblocks ──
    await ws.send_json({"type": "status", "status": "ready", "chat_id": session.chat_id})

    # ── Helper: send audio in ≤48KB base64 chunks to avoid nginx frame limits ──
    CHUNK_BYTES = 36000  # 36KB raw → ~48KB base64

    async def send_audio(audio: bytes, sentence_idx: int = 0, is_filler: bool = False):
        """Send audio as one or more chunked ai_audio messages."""
        total = len(audio)
        num_chunks = max(1, (total + CHUNK_BYTES - 1) // CHUNK_BYTES)
        for i in range(num_chunks):
            chunk = audio[i * CHUNK_BYTES:(i + 1) * CHUNK_BYTES]
            await ws.send_json({
                "type": "ai_audio",
                "data": base64.b64encode(chunk).decode(),
                "sentence_idx": sentence_idx,
                "chunk_index": i,
                "total_chunks": num_chunks,
                "is_filler": is_filler,
            })

    async def handle_pipeline(user_text: str):
        """
        Pipeline: filler phrase → stream full LLM response → single TTS call → send audio.
        TTS server is single-threaded so one call with the full text is fastest.
        """
        nonlocal session, forwarding_audio
        session.barged_in = False
        session.is_ai_speaking = False
        import time as _time
        t0 = _time.time()

        await ws.send_json({"type": "status", "status": "thinking"})

        # ── Send a filler phrase to cover LLM+TTS latency (skip first turn — greeting) ──
        if not session.barged_in and session.turn_count > 0:
            filler_audio = None
            try:
                filler_text = random.choice(FILLER_PHRASES_GENERAL)
                filler_audio = await tts_sentence(filler_text, voice=session.voice)
            except Exception:
                pass
            if not filler_audio and filler_cache:
                _pool = filler_cache_general if filler_cache_general else filler_cache
                filler_audio = random.choice(_pool)
            if filler_audio:
                await ws.send_json({"type": "status", "status": "speaking"})
                session.is_ai_speaking = True
                await send_audio(filler_audio, sentence_idx=-1, is_filler=True)
                print(f"[WS] Filler sent at {_time.time()-t0:.2f}s")
        session.turn_count += 1
        _query_t0 = _time.time()

        history = chat_store[session.chat_id]["messages"]
        t_build = _time.time()

        # ── Property search in parallel with LLM build ──
        properties = []
        property_context = ""
        is_prop_query = _is_property_query(user_text)
        print(f"[WS] Is property query: {is_prop_query} for '{user_text}'")
        if is_prop_query:
            try:
                print(f"[WS] Searching properties for: '{user_text}'")
                properties = await search_properties(user_text, limit=4)
                print(f"[WS] Search returned {len(properties)} properties")
                if properties:
                    property_context = format_properties_for_llm(properties)
                    print(f"[WS] Formatted property context: {len(property_context)} chars")
            except Exception as e:
                print(f"[WS] Property search error: {e}")
                import traceback
                traceback.print_exc()

        augmented_text = user_text + (f"\n\n[Available listings:\n{property_context}]" if property_context else "")
        messages, _ = await build_messages(augmented_text, history)
        print(f"[WS] build_messages took {int((_time.time()-t_build)*1000)}ms")

        # ── Send property cards to frontend if found ──
        if properties:
            try:
                await ws.send_json({"type": "properties", "data": properties})
            except Exception:
                pass

        # ── Stream full LLM response ──
        full_response = ""
        token_queue = asyncio.Queue()

        def _stream_llm():
            try:
                for token in generate_response_stream(messages):
                    token_queue.put_nowait(token)
                token_queue.put_nowait(None)
            except Exception as e:
                token_queue.put_nowait(e)

        asyncio.get_event_loop().run_in_executor(None, _stream_llm)

        _FILLER_RE = re.compile(
            r'^(certainly|of course|sure thing|absolutely|great|sure|got it|understood)'
            r'[!,.]?\s*',
            re.IGNORECASE
        )

        first_token_logged = False
        while True:
            if session.barged_in:
                print("[WS] Barge-in during LLM streaming, aborting")
                break
            try:
                token = await asyncio.wait_for(token_queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                print("[WS] LLM stream timeout")
                break
            if token is None:
                break
            if isinstance(token, Exception):
                await ws.send_json({"type": "error", "message": f"LLM failed: {token}"})
                return
            if not first_token_logged:
                print(f"[WS] First LLM token at {_time.time()-t0:.2f}s")
                first_token_logged = True
            full_response += token.replace('\n', ' ').replace('\r', '')

        if session.barged_in:
            session.is_ai_speaking = False
            return

        # ── Clean and send full response as single TTS call ──
        clean = full_response.replace("**", "").replace("###", "").replace("---", "").strip()
        clean = _FILLER_RE.sub('', clean).strip()
        if clean:
            clean = clean[0].upper() + clean[1:]

        print(f"[WS] LLM done at {_time.time()-t0:.2f}s, {len(clean)} chars — calling TTS")

        if clean and not session.barged_in:
            tts_task = asyncio.ensure_future(tts_sentence(clean, voice=session.voice))
            while not tts_task.done():
                if session.barged_in:
                    tts_task.cancel()
                    break
                await asyncio.sleep(0.05)
            if not session.barged_in:
                try:
                    audio = tts_task.result()
                except Exception:
                    audio = None
                if audio:
                    try:
                        if not session.is_ai_speaking:
                            await ws.send_json({"type": "status", "status": "speaking"})
                            session.is_ai_speaking = True
                        print(f"[WS] ⚡ Audio ready at {_time.time()-t0:.2f}s ({len(audio)} bytes, {len(clean)} chars)")
                        await send_audio(audio, sentence_idx=0)
                    except Exception:
                        pass

        if session.barged_in:
            session.is_ai_speaking = False
            return

        full_response = full_response.replace("**", "").replace("###", "").replace("---", "")
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": full_response})

        try:
            await ws.send_json({
                "type": "ai_text",
                "text": full_response,
                "chat_id": session.chat_id,
            })
            session.is_ai_speaking = False
            _resp_ms = int((_time.time() - _query_t0) * 1000)
            print(f"[WS] Full pipeline in {_time.time()-t0:.2f}s")
            # Analytics: log query
            try:
                log_query(session.chat_id, user_text, mode="voice",
                          is_property_query=is_prop_query,
                          properties_returned=len(properties),
                          llm_response=full_response, response_time_ms=_resp_ms)
            except Exception:
                pass
            accumulated_transcript.clear()  # reset for next utterance
            forwarding_audio = True  # resume forwarding PCM to Deepgram
            await ws.send_json({"type": "status", "status": "ready"})
        except Exception:
            pass

    # ── Deepgram Live Streaming ──
    dg_ws = None           # Deepgram WebSocket connection
    dg_reader_task = None  # background task reading Deepgram responses
    forwarding_audio = True  # whether to forward PCM to Deepgram
    accumulated_transcript = []  # final segments accumulating before trigger
    trigger_handle = None        # asyncio.Handle for delayed pipeline trigger
    pipeline_start_time = 0.0    # timestamp when pipeline last started (for barge-in cooldown)
    barge_in_time = 0.0          # timestamp when barge-in last fired (discard echo transcripts)

    async def open_deepgram():
        """Open a live streaming connection to Deepgram (with retry)."""
        nonlocal dg_ws
        for attempt in range(3):
            try:
                headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
                dg_ws = await ws_lib.connect(DEEPGRAM_WS_URL, additional_headers=headers)
                print(f"[DG] Deepgram live connection opened (attempt {attempt+1})")
                return
            except Exception as e:
                print(f"[DG] Failed to connect to Deepgram (attempt {attempt+1}): {e}")
                dg_ws = None
                if attempt < 2:
                    await asyncio.sleep(0.5 * (attempt + 1))
        print("[DG] All Deepgram connection attempts failed")

    async def close_deepgram():
        """Gracefully close Deepgram connection."""
        nonlocal dg_ws
        if dg_ws:
            try:
                await dg_ws.send(json.dumps({"type": "CloseStream"}))
                await dg_ws.close()
            except Exception:
                pass
            dg_ws = None
            print("[DG] Deepgram connection closed")

    async def fire_pipeline_delayed(delay: float = 0.30):
        """Fires pipeline after silence timeout (fallback if UtteranceEnd not received)."""
        nonlocal trigger_handle, accumulated_transcript, forwarding_audio, current_task, barge_in_time
        await asyncio.sleep(delay)
        # Discard if transcript was accumulated right after barge-in (AI speaker echo)
        if time.time() - barge_in_time < 1.5:
            print(f"[DG] Echo discard — {time.time()-barge_in_time:.2f}s after barge-in")
            accumulated_transcript = []
            trigger_handle = None
            return
        full_text = correct_transcript(" ".join(accumulated_transcript).strip())
        accumulated_transcript = []
        trigger_handle = None
        if not full_text:
            return
        cleaned = re.sub(r'[^\w]', '', full_text).strip()
        word_count = len(full_text.split())
        if not cleaned or looks_like_noise(full_text):
            print(f"[DG] Noise/empty: '{full_text}' — skipping")
            return
        if word_count < 2 and len(cleaned) < 6:
            print(f"[DG] Too short ({word_count} words): '{full_text}' — skipping")
            return
        print(f"[DG] Triggering pipeline (silence): '{full_text}'")
        forwarding_audio = False
        pipeline_start_time = time.time()
        await ws.send_json({"type": "user_text", "text": full_text})
        current_task = asyncio.create_task(handle_pipeline(full_text))

    async def deepgram_reader():
        """Background task: read transcripts from Deepgram and trigger pipeline."""
        nonlocal current_task, forwarding_audio, dg_ws, accumulated_transcript, trigger_handle, pipeline_start_time, barge_in_time
        while True:
            try:
                if not dg_ws:
                    await asyncio.sleep(0.1)
                    continue
                raw = await dg_ws.recv()
                msg = json.loads(raw)
                msg_type = msg.get("type", "")

                if msg_type == "Results":
                    channel = msg.get("channel", {})
                    alts = channel.get("alternatives", [])
                    if not alts:
                        continue
                    transcript = alts[0].get("transcript", "").strip()
                    is_final = msg.get("is_final", False)
                    speech_final = msg.get("speech_final", False)

                    if not transcript:
                        continue

                    # Send interim transcripts to frontend for display
                    if not is_final:
                        display = " ".join(accumulated_transcript + [transcript])
                        await ws.send_json({"type": "interim_transcript", "text": display})
                        
                        # Immediate barge-in on interim transcripts during AI speech
                        cooldown_elapsed = time.time() - pipeline_start_time
                        if session.is_ai_speaking and cooldown_elapsed > 1.5 and len(transcript) > 2:
                            print(f"[DG] Barge-in on interim: '{transcript}'")
                            session.barged_in = True
                            session.is_ai_speaking = False
                            if current_task and not current_task.done():
                                current_task.cancel()
                            forwarding_audio = True
                            await ws.send_json({"type": "barge_in_ack"})
                            await ws.send_json({"type": "status", "status": "ready"})
                        continue

                    # is_final=True — discard if within barge-in echo window
                    if time.time() - barge_in_time < 1.5:
                        print(f"[DG] Echo segment discarded ({time.time()-barge_in_time:.2f}s after barge-in): '{transcript}'")
                        continue

                    accumulated_transcript.append(transcript)
                    display = " ".join(accumulated_transcript)
                    print(f"[DG] Final segment: '{transcript}' (speech_final={speech_final})")
                    await ws.send_json({"type": "interim_transcript", "text": display})

                    # Cancel any pending silence timer (more speech arriving)
                    if trigger_handle and not trigger_handle.done():
                        trigger_handle.cancel()
                        trigger_handle = None

                    # Always use silence timer — never fire immediately on speech_final
                    # This lets multiple segments accumulate into one complete utterance
                    if speech_final:
                        print(f"[DG] speech_final — starting silence timer (accumulating)")
                    trigger_handle = asyncio.create_task(fire_pipeline_delayed())

                elif msg_type == "SpeechStarted":
                    print("[DG] Speech started")
                    # If a delayed trigger is pending, cancel it (user is still speaking)
                    if trigger_handle and not trigger_handle.done():
                        trigger_handle.cancel()
                        trigger_handle = None
                    # Barge-in: if AI is speaking and user starts talking
                    # Cooldown: ignore SpeechStarted for 1.5s after pipeline started
                    # (prevents residual Deepgram audio from immediately cancelling)
                    cooldown_elapsed = time.time() - pipeline_start_time
                    if session.is_ai_speaking and cooldown_elapsed > 1.5:
                        print("[DG] Barge-in detected (SpeechStarted during AI)")
                        session.barged_in = True
                        session.is_ai_speaking = False
                        accumulated_transcript = []
                        if current_task and not current_task.done():
                            current_task.cancel()
                        forwarding_audio = True
                        await ws.send_json({"type": "barge_in_ack"})
                        await ws.send_json({"type": "status", "status": "ready"})
                    elif session.is_ai_speaking:
                        print(f"[DG] SpeechStarted ignored (barge-in cooldown, {cooldown_elapsed:.2f}s elapsed)")

                elif msg_type == "UtteranceEnd":
                    # Authoritative "user finished speaking" signal from Deepgram VAD
                    # Fire pipeline immediately — no extra delay needed
                    print("[DG] UtteranceEnd — firing pipeline now")
                    if trigger_handle and not trigger_handle.done():
                        trigger_handle.cancel()
                        trigger_handle = None
                    if accumulated_transcript and not (current_task and not current_task.done()):
                        trigger_handle = asyncio.create_task(fire_pipeline_delayed(0.0))

                elif msg_type == "Metadata":
                    print(f"[DG] Connected: request_id={msg.get('request_id', '?')}")

                elif msg_type == "Error":
                    print(f"[DG] Error: {msg}")

            except ws_lib.exceptions.ConnectionClosed:
                print("[DG] Deepgram connection closed")
                dg_ws = None
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[DG] Reader error: {e}")
                await asyncio.sleep(0.5)

    async def deepgram_keepalive():
        """Send KeepAlive to Deepgram every 8s when not forwarding audio."""
        while True:
            await asyncio.sleep(8)
            if dg_ws and not forwarding_audio:
                try:
                    await dg_ws.send(json.dumps({"type": "KeepAlive"}))
                except Exception:
                    pass

    # ── Deepgram opened lazily on first PCM frame (not eagerly on connect) ──
    # Eager open causes Deepgram to timeout+close when no audio arrives (page load),
    # triggering WS onclose → reconnect → infinite loop before user clicks mic.
    dg_reader_task = asyncio.create_task(deepgram_reader())
    asyncio.create_task(deepgram_keepalive())

    # ── Send greeting audio to the user on connect ──
    async def _send_greeting():
        greeting_text = "Hey there, I'm Habib, your Singapore property consultant. What kind of property are you looking for today?"
        try:
            greeting_audio = await tts_sentence(greeting_text, voice=session.voice)
            if greeting_audio:
                await ws.send_json({"type": "status", "status": "speaking"})
                session.is_ai_speaking = True
                await send_audio(greeting_audio, sentence_idx=0)
                await ws.send_json({"type": "ai_text", "text": greeting_text, "chat_id": session.chat_id})
                chat_store[session.chat_id]["messages"].append({"role": "assistant", "content": greeting_text})
                session.is_ai_speaking = False
                await ws.send_json({"type": "status", "status": "ready"})
        except Exception as e:
            print(f"[WS] Greeting failed: {e}")
            await ws.send_json({"type": "status", "status": "ready"})
    asyncio.create_task(_send_greeting())

    _frame_count = 0
    try:
        while True:
            message = await ws.receive()
            msg_type_raw = message.get("type", "")

            # Handle disconnect gracefully
            if msg_type_raw == "websocket.disconnect":
                print(f"[WS] Client disconnected (code={message.get('code', '?')})")
                break

            # ── Binary message: raw PCM16 audio frame from client ──
            if "bytes" in message and message["bytes"]:
                pcm_data = message["bytes"]
                _frame_count += 1
                if _frame_count <= 3 or _frame_count % 500 == 0:
                    chunk = np.frombuffer(pcm_data, dtype=np.int16)
                    print(f"[WS] Frame #{_frame_count}: {chunk.size} samples, {frame_db(chunk):.1f}dB, fwd={forwarding_audio}")

                # Forward PCM to Deepgram for live transcription
                if forwarding_audio:
                    if not dg_ws:
                        # Deepgram dropped — try to reconnect
                        print("[DG] Connection lost, reconnecting...")
                        await open_deepgram()
                    if dg_ws:
                        try:
                            await dg_ws.send(pcm_data)
                        except Exception as e:
                            print(f"[DG] Send error: {e}")
                            dg_ws = None
                            # Attempt immediate reconnect on send failure
                            await open_deepgram()

                continue

            # ── Text message: JSON commands ──
            if "text" in message and message["text"]:
                raw = message["text"]
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                msg_type = msg.get("type")

                if msg_type == "set_chat_id":
                    cid = msg.get("chat_id", "")
                    if cid and cid in chat_store:
                        session.chat_id = cid
                    await ws.send_json({"type": "status", "status": "ready", "chat_id": session.chat_id})

                elif msg_type == "set_voice":
                    voice = msg.get("voice", "").strip()
                    ALLOWED_VOICES = {"test.wav", "test2.wav"}
                    if voice in ALLOWED_VOICES:
                        session.voice = voice
                        print(f"[WS] Voice changed to: {voice}")
                        try: update_session_voice(session.chat_id, voice)
                        except Exception: pass
                    await ws.send_json({"type": "voice_ack", "voice": session.voice})

                elif msg_type == "announce_voice":
                    # Speak voice change announcement
                    announcement = msg.get("text", "").strip()
                    if announcement:
                        try:
                            audio = await tts_sentence(announcement, voice=session.voice)
                            if audio:
                                await ws.send_json({"type": "status", "status": "speaking"})
                                await send_audio(audio, sentence_idx=-2)
                                await ws.send_json({"type": "ai_text", "text": announcement, "chat_id": session.chat_id})
                                await ws.send_json({"type": "status", "status": "ready"})
                        except Exception as e:
                            print(f"[WS] Voice announcement failed: {e}")

                elif msg_type == "barge_in":
                    print("[WS] Barge-in received")
                    try: log_barge_in(session.chat_id)
                    except Exception: pass
                    session.barged_in = True
                    session.is_ai_speaking = False
                    barge_in_time = time.time()
                    # Cancel any pending pipeline trigger and discard echo transcripts
                    if trigger_handle and not trigger_handle.done():
                        trigger_handle.cancel()
                    accumulated_transcript.clear()
                    if current_task and not current_task.done():
                        current_task.cancel()
                    forwarding_audio = True  # resume audio forwarding
                    await ws.send_json({"type": "barge_in_ack"})
                    await ws.send_json({"type": "status", "status": "ready"})

                elif msg_type == "text":
                    text = msg.get("text", "").strip()
                    if not text:
                        continue
                    await ws.send_json({"type": "user_text", "text": text})
                    current_task = asyncio.create_task(handle_pipeline(text))

                elif msg_type == "audio":
                    # Legacy: Client recorded audio blob → transcribe → pipeline
                    audio_b64 = msg.get("data", "")
                    if not audio_b64:
                        continue
                    audio_bytes = base64.b64decode(audio_b64)
                    print(f"[WS] Received legacy audio: {len(audio_bytes)} bytes")
                    await ws.send_json({"type": "status", "status": "transcribing"})
                    try:
                        text, dur = await transcribe_audio(audio_bytes, "recording.webm")
                        cleaned = re.sub(r'[^\w]', '', text).strip()
                        if not cleaned or looks_like_noise(text):
                            await ws.send_json({"type": "status", "status": "ready"})
                            continue
                    except Exception as e:
                        print(f"[WS] STT error: {e}")
                        await ws.send_json({"type": "status", "status": "ready"})
                        continue
                    await ws.send_json({"type": "user_text", "text": text})
                    current_task = asyncio.create_task(handle_pipeline(text))

    except WebSocketDisconnect:
        print(f"[WS] Voice session disconnected: {session.chat_id}")
    except Exception as e:
        print(f"[WS] Error: {e}")
        try: log_error(session.chat_id, "ws_error", str(e))
        except Exception: pass
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        # Analytics: end session
        try: analytics_end_session(session.chat_id)
        except Exception: pass
        # Clean up Deepgram connection
        if dg_reader_task:
            dg_reader_task.cancel()
            try:
                await dg_reader_task
            except asyncio.CancelledError:
                pass
        await close_deepgram()
