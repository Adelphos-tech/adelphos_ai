import os
import re
import httpx
from dotenv import load_dotenv

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")

# ─── Post-processing correction map for common STT mishearings ───
_CORRECTIONS = [
    (r'\biman\b(?!\s+developer|\s+properties|\s+prop)', '2 bedroom'),
    (r'\be[\s-]man\b', '2 bedroom'),
    (r'\bto\s+be\s+are\b', '2 BR'),
    (r'\btube?\s*are\b', '2 BR'),
    (r'\b(three|free)\s+be\s+are\b', '3 BR'),
    (r'\b(\d)\s*b\s*r\b', r'\1 BR'),
    (r'\b(\d)\s*bed\s*room', r'\1 bedroom'),
    (r'\b(\d)\s*bath\s*room', r'\1 bathroom'),
    (r'\bwon\s+bedroom\b', '1 bedroom'),
    (r'\bcon\s*do\b', 'condo'),
    (r'\bh\s*d\s*b\b', 'HDB'),
    (r'\bp\s*s\s*f\b', 'PSF'),
    (r'\bs\s*g\s*d\b', 'SGD'),
    (r'\ba\s*e\s*d\b', 'AED'),
    (r'\badel\s*foss?\b', 'Adelphos'),
    (r'\badel\s*foes?\b', 'Adelphos'),
    (r'\ba\s*delfus\b', 'Adelphos'),
    (r'\bflutter\b', 'Flutter'),
    (r'\breact\b', 'React'),
    (r'\bi\s*o\s*s\b', 'iOS'),
    (r'\ba\s*p\s*i\b', 'API'),
    (r'\bu\s*i\b', 'UI'),
    (r'\bu\s*x\b', 'UX'),
    (r'\bword\s*press\b', 'WordPress'),
    (r'\bm\s*v\s*p\b', 'MVP'),
]

_COMPILED_CORRECTIONS = [(re.compile(p, re.IGNORECASE), r) for p, r in _CORRECTIONS]


def correct_transcript(text: str) -> str:
    if not text:
        return text
    for pattern, replacement in _COMPILED_CORRECTIONS:
        text = pattern.sub(replacement, text)
    return text


async def transcribe_audio(audio_bytes: bytes, filename: str = "audio.webm") -> tuple[str, float]:
    """Transcribe audio using Deepgram REST API directly via httpx."""
    if not DEEPGRAM_API_KEY:
        raise Exception("DEEPGRAM_API_KEY not set.")

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "webm"
    mime_map = {
        "webm": "audio/webm",
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "ogg": "audio/ogg",
        "flac": "audio/flac",
        "m4a": "audio/mp4",
    }
    mimetype = mime_map.get(ext, "audio/webm")

    url = "https://api.deepgram.com/v1/listen"
    params = {
        "model": "nova-3",
        "language": "en",
        "smart_format": "true",
        "punctuate": "true",
        "filler_words": "false",
        "utterances": "true",
        "diarize": "false",
    }

    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": mimetype,
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, params=params, headers=headers, content=audio_bytes, timeout=30.0)
        response.raise_for_status()
        data = response.json()

    transcript = ""
    duration = 0.0

    if data and "results" in data:
        channels = data["results"].get("channels", [])
        if channels:
            alternatives = channels[0].get("alternatives", [])
            if alternatives:
                transcript = alternatives[0].get("transcript", "")
        duration = data.get("metadata", {}).get("duration", 0.0)

    transcript = correct_transcript(transcript)
    return transcript, duration
