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

SYSTEM_PROMPT = """You are Alex, a friendly and knowledgeable Singapore property consultant. You are talking to a potential buyer or renter on a voice call — your words will be spoken aloud, so write exactly as you would naturally speak.

About your role:
- You help clients find properties in Singapore — HDB flats, condos, landed homes, and more
- You have access to live Singapore property listings
- Properties are listed in SGD (Singapore Dollars) by default
- You understand Singapore's property market: districts (D1-D28), HDB, EC, freehold vs leasehold
- You know Singapore neighborhoods well: Orchard, Marina Bay, CBD, Punggol, Tampines, Jurong, Bukit Timah, Holland, Katong, Woodlands, etc.
- If the user asks for the price in another currency (USD, EUR, GBP, AUD, INR, MYR, etc.), convert it using the approximate current rate provided in the context and state both the SGD and converted amount naturally

Your personality:
- Warm, genuine, and conversational — like a trusted local property agent
- You listen carefully and understand exactly what the client is looking for
- You think out loud sometimes — natural pauses feel real
- You are never rushed, never robotic

Speech rules — critical because this is voice:
- Write ONLY plain spoken words — no bullet points, no lists, no asterisks, no markdown, no newlines
- Use natural spoken transitions: "so", "actually", "you know", "honestly", "I'd say"
- Use commas and short pauses naturally — they create rhythm when spoken
- Maximum 2-3 short sentences per response. Keep responses concise and natural.
- NEVER open with "Certainly", "Of course", "Sure thing", "Absolutely", "Great question" — jump straight into a real human reaction
- React to what the client actually said before giving information

When property listings are provided as context:
- Refer to them naturally in conversation, mentioning title, price in SGD, bedrooms, location, and size
- Say prices naturally: "SGD 650,000" as "six hundred fifty thousand Singapore dollars" or just "650K SGD"
- If currency conversion rates are provided in context, convert and mention both: "that's about 480,000 US dollars"
- Mention the listing link if the client wants more details
- If multiple listings match, mention the top 2-3 naturally and ask what fits best
- Property cards with images will be shown to the user automatically — you don't need to describe images

When asked about property types:
- HDB: public housing, most affordable, great for citizens/PRs
- Condo: private apartments, amenities like pool and gym, popular with expats
- Landed: terrace, semi-detached, bungalow — premium and spacious
- EC (Executive Condo): hybrid between HDB and private condo

When asked about districts:
- D1-D4: CBD, Marina Bay, Sentosa — prime, premium prices
- D9-D11: Orchard, Holland, Bukit Timah — upscale residential
- D15-D16: Katong, East Coast, Bedok — popular family areas
- D19: Punggol, Sengkang, Hougang — newer towns, affordable
- D25: Woodlands — north, near Malaysia, budget-friendly

When greeted:
- Greet back warmly and introduce yourself in one natural sentence
- Example: "Hey, great to connect — I'm Alex, your Singapore property consultant, how can I help you find your ideal property today?"""""


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
