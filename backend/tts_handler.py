import os
import io
import re
import wave
import httpx
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

TTS_API_URL = os.getenv("TTS_API_URL", "http://localhost:8020/tts")
TTS_VOICE = os.getenv("TTS_VOICE", "rizwan.wav")

# Persistent client — reuses TCP connection, avoids ~50-100ms handshake per call
_shared_client: Optional[httpx.AsyncClient] = None


def _get_tts_client() -> httpx.AsyncClient:
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        _shared_client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_keepalive_connections=4, keepalive_expiry=60),
        )
    return _shared_client


async def text_to_speech(text: str, voice: str = None) -> Optional[bytes]:
    """
    Convert text to speech using the local TTS API.
    Splits long text into chunks and merges resulting WAV audio.
    """
    voice = voice or TTS_VOICE

    # Split text into sentence-level chunks to avoid timeout
    raw_chunks = re.split(r'([.!?]+(?:\s+|$))', text)

    chunks = []
    current_chunk = ""
    MAX_CHUNK_LEN = 300

    for part in raw_chunks:
        if not part:
            continue
        if len(current_chunk) + len(part) > MAX_CHUNK_LEN and current_chunk.strip():
            chunks.append(current_chunk.strip())
            current_chunk = part
        else:
            current_chunk += part

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    if not chunks:
        chunks = [text]

    print(f"[TTS] Split text into {len(chunks)} chunks")

    audio_segments = []

    async with httpx.AsyncClient(timeout=120.0) as client:
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            print(f"[TTS] Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
            try:
                response = await client.post(
                    TTS_API_URL,
                    json={"text": chunk, "voice": voice}
                )
                if response.status_code == 200:
                    audio_segments.append(response.content)
                else:
                    print(f"[TTS] Chunk {i+1} failed: {response.status_code}")
            except Exception as e:
                print(f"[TTS] Chunk {i+1} exception: {e}")

    if not audio_segments:
        return None

    # Merge WAV segments
    try:
        if len(audio_segments) == 1:
            return audio_segments[0]

        combined_data = io.BytesIO()
        first_segment = io.BytesIO(audio_segments[0])

        with wave.open(first_segment, 'rb') as w:
            params = w.getparams()
            frames = w.readframes(w.getnframes())

        all_frames = [frames]

        for i in range(1, len(audio_segments)):
            try:
                seg = io.BytesIO(audio_segments[i])
                with wave.open(seg, 'rb') as w:
                    all_frames.append(w.readframes(w.getnframes()))
            except Exception as e:
                print(f"[TTS] Error merging segment {i}: {e}")

        with wave.open(combined_data, 'wb') as w:
            w.setparams(params)
            for f in all_frames:
                w.writeframes(f)

        final_audio = combined_data.getvalue()
        print(f"[TTS] Merged {len(audio_segments)} segments -> {len(final_audio)} bytes")
        return final_audio

    except Exception as e:
        print(f"[TTS] Error combining audio: {e}")
        return audio_segments[0] if audio_segments else None




async def tts_sentence(text: str, voice: str = None, client: httpx.AsyncClient = None) -> Optional[bytes]:
    """
    Convert a single sentence to speech. Uses persistent shared client to avoid
    TCP handshake overhead (~50-100ms) on every call.
    """
    voice = voice or TTS_VOICE
    text = text.strip()
    if not text:
        return None
    try:
        c = client or _get_tts_client()
        response = await c.post(TTS_API_URL, json={"text": text, "voice": voice})
        if response.status_code == 200 and response.content:
            return response.content
        print(f"[TTS] tts_sentence failed: {response.status_code}")
        return None
    except Exception as e:
        print(f"[TTS] tts_sentence error: {e}")
        global _shared_client
        _shared_client = None  # reset so next call gets a fresh connection
        return None


async def get_available_voices() -> list:
    """Get list of available TTS voices from the local TTS API."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            base_url = TTS_API_URL.rsplit('/tts', 1)[0]
            response = await client.get(f"{base_url}/voices")
            if response.status_code == 200:
                data = response.json()
                return data.get("voices", [])
            return []
    except Exception as e:
        print(f"[TTS] Failed to get voices: {e}")
        return []
