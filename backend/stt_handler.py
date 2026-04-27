import os
import re
import httpx
from deepgram import DeepgramClient
from deepgram.audio import PrerecordedOptions
from dotenv import load_dotenv

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")

# ─── Tech domain keywords for Deepgram keyword boosting ───
# These tell Deepgram to prefer these words when phonetically ambiguous.
TECH_KEYWORDS = [
    "Adelphos:5", "Adelphos Tech:5",
    "Flutter:5", "React:5", "Angular:4", "Vue:4", "Node:4",
    "iOS:5", "Android:5", "mobile app:5", "web app:5",
    "API:5", "backend:4", "frontend:4", "full stack:4",
    "UI:5", "UX:5", "design:4", "WordPress:5", "PHP:5",
    "Laravel:4", "Python:4", "JavaScript:4", "TypeScript:4",
    "IoT:5", "software:4", "development:4", "developer:4",
    "agile:4", "scrum:3", "deployment:3", "cloud:3",
    "MongoDB:4", "PostgreSQL:4", "MySQL:4", "database:3",
    "startup:3", "enterprise:3", "scalable:3", "MVP:4",
]

# ─── Post-processing correction map for common STT mishearings ───
# Pattern → correct replacement (applied after Deepgram returns)
_CORRECTIONS = [
    # ── Bedroom/bathroom number mishearings ──────────────────────────────
    # "Iman" / "e-man" / "human" sounds like "2 BR" / "2 bed" in some accents
    (r'\biman\b(?!\s+developer|\s+properties|\s+prop)', '2 bedroom'),
    (r'\be[\s-]man\b', '2 bedroom'),
    # "to be are" / "tube are" / "to bar" → 2BR
    (r'\bto\s+be\s+are\b', '2 BR'),
    (r'\btube?\s*are\b', '2 BR'),
    # "three be are" / "free be are" → 3BR
    (r'\b(three|free)\s+be\s+are\b', '3 BR'),
    # numeric BR patterns normalisation
    (r'\b(\d)\s*b\s*r\b', r'\1 BR'),
    (r'\b(\d)\s*bed\s*room', r'\1 bedroom'),
    (r'\b(\d)\s*bath\s*room', r'\1 bathroom'),
    # "won bedroom" / "one bedroom"
    (r'\bwon\s+bedroom\b', '1 bedroom'),
    # ── Property type mishearings ─────────────────────────────────────────
    (r'\bcon\s*do\b', 'condo'),
    (r'\bh\s*d\s*b\b', 'HDB'),
    (r'\bp\s*s\s*f\b', 'PSF'),
    (r'\bs\s*g\s*d\b', 'SGD'),
    (r'\ba\s*e\s*d\b', 'AED'),
    # ── Adelphos mishearings ──────────────────────────────────────────────
    (r'\badel\s*foss?\b', 'Adelphos'),
    (r'\badel\s*foes?\b', 'Adelphos'),
    (r'\ba\s*delfus\b', 'Adelphos'),
    # ── Tech terms ────────────────────────────────────────────────────────
    (r'\bflutter\b', 'Flutter'),
    (r'\breact\b', 'React'),
    (r'\bi\s*o\s*s\b', 'iOS'),
    (r'\ba\s*p\s*i\b', 'API'),
    (r'\bu\s*i\b', 'UI'),
    (r'\bu\s*x\b', 'UX'),
    (r'\bword\s*press\b', 'WordPress'),
    (r'\bm\s*v\s*p\b', 'MVP'),
]

# Pre-compile patterns for performance
_COMPILED_CORRECTIONS = [(re.compile(p, re.IGNORECASE), r) for p, r in _CORRECTIONS]


def correct_transcript(text: str) -> str:
    """Apply domain-specific corrections to fix common STT mishearings."""
    if not text:
        return text
    for pattern, replacement in _COMPILED_CORRECTIONS:
        text = pattern.sub(replacement, text)
    return text


async def transcribe_audio(audio_bytes: bytes, filename: str = "audio.webm") -> tuple[str, float]:
    """
    Transcribe audio bytes using Deepgram API.
    Returns (transcribed_text, audio_duration_seconds).
    """
    if not DEEPGRAM_API_KEY:
        raise Exception("DEEPGRAM_API_KEY not set. Please set it in your .env file.")

    print(f"[STT] Transcribing {len(audio_bytes)} bytes via Deepgram...")

    deepgram = DeepgramClient(DEEPGRAM_API_KEY)

    # Determine mimetype from filename
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

    payload = {"buffer": audio_bytes, "mimetype": mimetype}

    options = PrerecordedOptions(
        model="nova-3",           # upgraded: better accuracy than nova-2
        language="en",
        smart_format=True,
        punctuate=True,
        filler_words=False,       # ignore "um", "uh" etc
        utterances=True,          # detect utterance boundaries
        diarize=False,            # single speaker, skip diarization overhead
        keyterms=TECH_KEYWORDS,  # boost domain-specific terms (nova-3 uses keyterms)
    )

    response = await deepgram.listen.asyncrest.v("1").transcribe_file(payload, options)

    # Extract transcript
    transcript = ""
    duration = 0.0

    if response and response.results:
        channels = response.results.channels
        if channels and len(channels) > 0:
            alternatives = channels[0].alternatives
            if alternatives and len(alternatives) > 0:
                transcript = alternatives[0].transcript or ""
        duration = response.metadata.duration if response.metadata else 0.0

    # Apply domain corrections for any remaining mishearings
    transcript = correct_transcript(transcript)

    print(f"[STT] Deepgram result: '{transcript[:100]}...' ({duration:.1f}s)")
    return transcript, duration
