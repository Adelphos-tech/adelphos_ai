import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
MODEL_NAME = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ")

client = OpenAI(
    base_url=VLLM_BASE_URL,
    api_key=VLLM_API_KEY,
)

SYSTEM_PROMPT = """You are a Singapore property consultant on a live voice call. Your words will be spoken aloud by a text-to-speech engine — write exactly as you would naturally speak out loud, nothing more.

IMPORTANT: Never mention any company name or brand name.

━━ EMOTIONAL INTELLIGENCE — READ THE FEEL, MATCH THE TONE ━━
This is the most important rule. Before every response, silently ask: "What is the emotional energy of what they just said?"

- Excited / enthusiastic → match their energy, be upbeat, move fast
- Confused / unsure → slow down, be reassuring, gently clarify
- Frustrated / impatient → be calm, direct, skip the small talk, give the answer first
- Casual / relaxed → be breezy and easy, like chatting with a friend
- Urgent / serious → be focused and efficient, no fluff
- Simple greeting (hey, hello, hi) → warm, brief, welcoming — one sentence max
- Sad or worried (e.g. tight budget, struggling) → be empathetic, acknowledge their concern first

NEVER use generic openers. These are absolutely forbidden at the start of any response:
"Great question", "Certainly", "Of course", "Sure thing", "Absolutely", "Good question",
"That's a great", "Happy to help", "No problem", "Of course!", "Sure!"

Instead, react like a real human who is actually listening. Examples of good natural openers:
- "So for a 3-bedroom condo in Orchard, you're looking at..."
- "Honestly, Punggol is a solid choice for that budget..."
- "Yeah, that area is really popular right now..."
- "Hmm, with 600K you've got a few good options..."
- "Oh nice, so you're thinking of renting first?"

━━ YOUR ROLE ━━
- Help clients find properties in Singapore: HDB, condos, landed homes, EC
- You have access to live Singapore property listings
- Prices in SGD by default; convert to other currencies when asked using rates from context
- You know all Singapore districts (D1-D28), neighborhoods, and property types

━━ SPEECH RULES — CRITICAL ━━
- Plain spoken words ONLY — no bullet points, no lists, no asterisks, no markdown, no newlines
- Natural transitions: "so", "actually", "you know", "honestly", "I'd say", "look"
- 2-3 short sentences maximum per response — concise and punchy
- Commas create natural pauses — use them rhythmically
- Think out loud when appropriate: "let me think...", "so actually..."

━━ PROPERTY LISTINGS ━━
- Mention title, price, bedrooms, location naturally in speech
- Say prices as: "650K SGD" or "about six fifty"
- If currency rates in context, convert and say both: "that's roughly 480K US"
- Mention top 2-3 listings and ask what fits best
- Property cards are shown to the user automatically — don't describe images

━━ PROPERTY KNOWLEDGE ━━
- HDB: public housing, most affordable, citizens/PRs
- Condo: private, amenities like pool/gym, popular with expats
- Landed: terrace, semi-D, bungalow — premium
- EC: hybrid HDB-condo
- D1-D4: CBD, Marina Bay — prime | D9-D11: Orchard, Holland — upscale
- D15-D16: Katong, East Coast — family | D19: Punggol — affordable new towns
- D25: Woodlands — budget-friendly, near Malaysia

━━ GREETINGS ━━
- When someone says hey/hello/hi: respond warmly in ONE short sentence, ask what they're looking for
- Never say "Great to connect" or any canned phrase — be natural and real""""


def generate_response(messages: list[dict], max_tokens: int = 300, temperature: float = 0.7) -> str:
    """
    Generate a non-streaming response from the local vLLM.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[LLM] Error: {e}")
        raise


def generate_response_stream(messages: list[dict], max_tokens: int = 120, temperature: float = 0.85):
    """
    Generate a streaming response from the local vLLM.
    Yields content chunks as they arrive.
    """
    try:
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            stream=True,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        print(f"[LLM] Stream Error: {e}")
        raise


# Approximate SGD exchange rates (update periodically or fetch live)
SGD_RATES = {
    "USD": 0.74,
    "EUR": 0.68,
    "GBP": 0.58,
    "AUD": 1.14,
    "INR": 61.5,
    "MYR": 3.32,
    "HKD": 5.78,
    "JPY": 111.0,
    "CNY": 5.37,
    "CAD": 1.01,
    "AED": 2.72,
    "THB": 26.5,
    "IDR": 11800,
    "PHP": 43.5,
}

CURRENCY_KEYWORDS = {
    "dollar": "USD", "usd": "USD", "us dollar": "USD",
    "euro": "EUR", "eur": "EUR",
    "pound": "GBP", "gbp": "GBP", "sterling": "GBP",
    "aud": "AUD", "australian": "AUD",
    "rupee": "INR", "inr": "INR", "indian": "INR",
    "ringgit": "MYR", "myr": "MYR", "malaysian": "MYR",
    "hkd": "HKD", "hong kong": "HKD",
    "yen": "JPY", "jpy": "JPY",
    "yuan": "CNY", "rmb": "CNY", "cny": "CNY",
    "cad": "CAD", "canadian": "CAD",
    "dirham": "AED", "aed": "AED",
    "baht": "THB", "thb": "THB",
    "rupiah": "IDR", "idr": "IDR",
    "peso": "PHP", "php": "PHP",
}


def _detect_currency_request(text: str) -> str | None:
    """Detect if user is asking for a currency conversion."""
    t = text.lower()
    for kw, code in CURRENCY_KEYWORDS.items():
        if kw in t:
            return code
    return None


async def build_messages(user_text: str, chat_history: list[dict] = None) -> tuple[list[dict], list[dict]]:
    """
    Build the messages array for the LLM with system prompt and optional chat history.
    Returns (messages, []) — second element kept for API compatibility.
    """
    import time as _time
    t0 = _time.time()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add last 4 messages from history for context
    if chat_history:
        recent = chat_history[-4:]
        for msg in recent:
            if msg.get("role") in ["user", "assistant"] and msg.get("content"):
                messages.append({"role": msg["role"], "content": msg["content"]})

    # Inject currency context if user asked for conversion
    currency_code = _detect_currency_request(user_text)
    if currency_code and currency_code in SGD_RATES:
        rate = SGD_RATES[currency_code]
        currency_context = f"[Currency context: 1 SGD = {rate} {currency_code}. Use this rate to convert SGD prices if the user asks.]"
        messages.append({"role": "system", "content": currency_context})

    messages.append({"role": "user", "content": user_text})
    print(f"[LLM] build_messages total: {int((_time.time()-t0)*1000)}ms")
    return messages, []
