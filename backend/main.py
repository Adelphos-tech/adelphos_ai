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
import websockets as ws_lib
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

from backend.stt_handler import transcribe_audio, correct_transcript, TECH_KEYWORDS
from backend.tts_handler import text_to_speech, tts_sentence, get_available_voices
import httpx
from backend.llm_handler import generate_response, generate_response_stream, build_messages

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

# Mount frontend
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
async def read_root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/voice-agent")
async def voice_agent_page():
    return FileResponse(os.path.join(FRONTEND_DIR, "voice-agent.html"))


@app.get("/logo.png")
async def serve_logo():
    return FileResponse(os.path.join(FRONTEND_DIR, "logo.png"), media_type="image/png")


@app.get("/health")
async def health_check():
    return {"status": "ok"}


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
    try:
        tts_audio = await text_to_speech(ai_response)
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

@app.post("/chat")
async def text_chat(request: dict):
    """Text-based chat endpoint with streaming LLM response."""
    question = request.get("message") or request.get("question", "")
    chat_id = request.get("chat_id")

    if not question.strip():
        raise HTTPException(status_code=400, detail="Question is required")

    if not chat_id or chat_id not in chat_store:
        chat_id = str(uuid.uuid4())
        chat_store[chat_id] = {"messages": []}

    history = chat_store[chat_id]["messages"]
    messages, _ = await build_messages(question, history)

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

    # Ensure chat store entry exists
    if session.chat_id not in chat_store:
        chat_store[session.chat_id] = {"messages": []}

    # Track the current AI response task so we can cancel on barge-in
    current_task: asyncio.Task | None = None

    # ── Concurrency: allow multiple TTS calls in parallel ──
    TTS_SEM = asyncio.Semaphore(4)

    # ── Send ready status immediately so the frontend unblocks ──
    await ws.send_json({"type": "status", "status": "ready", "chat_id": session.chat_id})

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
        if filler_cache and not session.barged_in and session.turn_count > 0:
            _pool = filler_cache_general if filler_cache_general else filler_cache
            filler_audio = random.choice(_pool)
            await ws.send_json({"type": "status", "status": "speaking"})
            session.is_ai_speaking = True
            await ws.send_json({
                "type": "ai_audio",
                "data": base64.b64encode(filler_audio).decode(),
                "sentence_idx": -1,
                "is_filler": True,
            })
            print(f"[WS] Filler sent at {_time.time()-t0:.2f}s")
        session.turn_count += 1

        history = chat_store[session.chat_id]["messages"]
        t_build = _time.time()
        messages, _ = await build_messages(user_text, history)
        print(f"[WS] build_messages took {int((_time.time()-t_build)*1000)}ms")

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
            tts_task = asyncio.ensure_future(tts_sentence(clean))
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
                        print(f"[WS] ⚡ Audio ready at {_time.time()-t0:.2f}s ({len(clean)} chars)")
                        await ws.send_json({
                            "type": "ai_audio",
                            "data": base64.b64encode(audio).decode(),
                            "sentence_idx": 0,
                        })
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
            print(f"[WS] Full pipeline in {_time.time()-t0:.2f}s")
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

    # ── Open Deepgram eagerly on connect (not lazily on first PCM frame) ──
    await open_deepgram()

    # Start Deepgram reader + keepalive tasks
    dg_reader_task = asyncio.create_task(deepgram_reader())
    asyncio.create_task(deepgram_keepalive())

    # ── Send greeting audio to the user on connect ──
    async def _send_greeting():
        greeting_text = "Hello! I'm Alex from Adelphos Tech. How can I help you today?"
        try:
            greeting_audio = await tts_sentence(greeting_text)
            if greeting_audio:
                await ws.send_json({"type": "status", "status": "speaking"})
                session.is_ai_speaking = True
                await ws.send_json({
                    "type": "ai_audio",
                    "data": base64.b64encode(greeting_audio).decode(),
                    "sentence_idx": 0,
                })
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

                elif msg_type == "barge_in":
                    print("[WS] Barge-in received")
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
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        # Clean up Deepgram connection
        if dg_reader_task:
            dg_reader_task.cancel()
            try:
                await dg_reader_task
            except asyncio.CancelledError:
                pass
        await close_deepgram()
