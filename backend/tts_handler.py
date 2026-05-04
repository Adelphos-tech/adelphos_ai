import os
import io
import wave
import struct
import httpx
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")

# Voice name → Deepgram model/voice mapping
# Deepgram Aura voices - British English closer to Singaporean accent
# Available voices: asteria (British F), luna (British F), orion (British M),
#                   arcas (American M), thalia (American F), helios (British M)
VOICE_MAP = {
    "james.wav":   {"model": "aura-helios-en", "voice": None},   # British Male - professional
    "elena.wav": {"model": "aura-orion-en", "voice": None},  # British Male - warm
    "marcus.wav":  {"model": "aura-asteria-en", "voice": None}, # British Female - clear
    "zara.wav":   {"model": "aura-luna-en", "voice": None},    # British Female - friendly
    "test.wav":   {"model": "aura-helios-en", "voice": None},  # Habib - British Male
    "test2.wav":  {"model": "aura-orion-en", "voice": None},  # Shivang - British Male
}
DEFAULT_VOICE_KEY = os.getenv("TTS_VOICE", "james.wav")

# Persistent client
_shared_client: Optional[httpx.AsyncClient] = None


def _get_tts_client() -> httpx.AsyncClient:
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        _shared_client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_keepalive_connections=4, keepalive_expiry=60),
        )
    return _shared_client


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 24000, channels: int = 1, sampwidth: int = 2) -> bytes:
    """Wrap raw PCM16 bytes in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(sample_rate)
        w.writeframes(pcm_bytes)
    return buf.getvalue()


async def tts_sentence(text: str, voice: str = None, client: httpx.AsyncClient = None) -> Optional[bytes]:
    """Convert text to speech using Deepgram TTS API. Returns WAV bytes."""
    text = text.strip()
    if not text or not DEEPGRAM_API_KEY:
        return None

    voice_key = voice or DEFAULT_VOICE_KEY
    voice_cfg = VOICE_MAP.get(voice_key, VOICE_MAP["james.wav"])
    model = voice_cfg["model"]
    voice_name = voice_cfg["voice"]

    try:
        c = client or _get_tts_client()
        # Build URL with or without voice parameter
        url = f"https://api.deepgram.com/v1/speak?model={model}&encoding=linear16&sample_rate=24000"
        if voice_name:
            url += f"&voice={voice_name}"
        resp = await c.post(
            url,
            headers={
                "Authorization": f"Token {DEEPGRAM_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"text": text},
            timeout=20.0,
        )
        if resp.status_code == 200 and resp.content:
            # Deepgram returns raw PCM16 — wrap in WAV
            wav = _pcm_to_wav(resp.content, sample_rate=24000)
            print(f"[TTS] Deepgram TTS: {len(text)} chars → {len(wav)} bytes WAV")
            return wav
        elif resp.status_code == 403:
            print(f"[TTS] Deepgram TTS 403: Insufficient permissions. Check your API key has TTS access.")
            print(f"[TTS] Response: {resp.text[:300]}")
        else:
            print(f"[TTS] Deepgram TTS failed: {resp.status_code} {resp.text[:200]}")
        return None
    except Exception as e:
        print(f"[TTS] tts_sentence error: {e}")
        global _shared_client
        _shared_client = None
        return None


async def text_to_speech(text: str, voice: str = None) -> Optional[bytes]:
    """Convert text to speech (full text, single call)."""
    return await tts_sentence(text, voice)


async def get_available_voices() -> list:
    """Return available voice options."""
    return [
        {"id": "test.wav", "name": "Habib"},
        {"id": "test2.wav", "name": "Shivang"},
        {"id": "james.wav", "name": "James (British)"},
        {"id": "elena.wav", "name": "Elena (British)"},
        {"id": "marcus.wav", "name": "Marcus (British)"},
        {"id": "zara.wav", "name": "Zara (British)"},
    ]
