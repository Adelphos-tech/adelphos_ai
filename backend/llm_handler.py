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

━━ MULTILINGUAL — CRITICAL ━━
Detect the language the user is speaking and ALWAYS reply in that SAME language.
Supported languages: English, Mandarin Chinese (简体/繁體), Malay (Bahasa Melayu), Tamil (தமிழ்).
- If user writes in Chinese → reply fully in Chinese
- If user writes in Malay → reply fully in Malay
- If user writes in Tamil → reply fully in Tamil
- If user writes in English → reply in English
- If mixed, use whichever language dominates
All property knowledge, prices, and advice should be conveyed naturally in whatever language the user is using.
Do NOT translate or explain — just respond natively in their language.

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
- Never say "Great to connect" or any canned phrase — be natural and real"""


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


# ── Intent Classifier ────────────────────────────────────────────────────────
# Runs in <1ms (pure regex/set lookups), zero extra LLM calls.

_INTENT_RULES: list[tuple[str, list[str]]] = [
    ("greeting",         ["^hey$", "^hi$", "^hello$", "^yo$", "^hiya$", "^howdy$",
                          "^hey there", "^hi there", "^hello there", "^good morning",
                          "^good afternoon", "^good evening", "^what's up", "^sup "]),
    ("price_inquiry",    ["how much", "what.s the price", "what is the price",
                          "price range", "cost of", "pricing", "afford", "budget",
                          "cheapest", "most expensive", "price.*condo", "condo.*price",
                          "how expensive", "value", "worth"]),
    ("currency_convert", ["in usd", "in dollar", "in ringgit", "in euro", "in pound",
                          "in rupee", "in yen", "in yuan", "in aud", "convert",
                          "equivalent in", "how much.*usd", "how much.*myr"]),
    ("location_info",    ["tell me about", "what is.*district", "which district",
                          "area like", "neighbourhood", "neighborhood", "near mrt",
                          "near.*station", "good area", "best area", "popular area",
                          "orchard road", "bukit timah", "holland village",
                          "sentosa", "jurong", "tampines", "punggol", "woodlands",
                          "marine parade", "katong", "bedok", "clementi", "bishan",
                          "ang mo kio", "toa payoh", "novena", "marina bay", "cbd"]),
    ("general_advice",   ["should i", "advise", "recommend", "better to", "difference between",
                          "buy or rent", "rent or buy", "hdb vs", "vs condo", "freehold vs",
                          "good investment", "investment potential", "roi", "capital gain",
                          "first time buyer", "first-time", "foreigner.*buy", "pr.*buy",
                          "loan", "mortgage", "stamp duty", "absd", "cpf", "ltvr",
                          "how to buy", "process", "steps to"]),
    ("property_followup", ["tell me more", "more about", "more details", "more info",
                           "the first", "the second", "the third", "that one", "this one",
                           "which one", "both", "all of them", "compare", "versus", " vs ",
                           "cheaper option", "any other", "other option", "show more",
                           "more listing", "another one", "similar", "like that",
                           "view details", "link", "contact", "agent", "is it.*available",
                           "still available", "any discount", "negotiable"]),
    ("property_search",  ["looking for", "find me", "show me", "search for", "i want",
                          "i need", "i.m looking", "any.*bedroom", "\\d.*bed", "bed.*room",
                          "for sale", "for rent", "to rent", "to buy", "available.*condo",
                          "available.*hdb", "available.*landed", "property.*under",
                          "under.*million", "below.*sgd", "around.*sgd"]),
    ("off_topic",        ["weather", "recipe", "sport", "football", "movie", "music",
                          "stock market", "crypto", "politics", "news today",
                          "tell me a joke", "what time is it"]),
]

# Pre-compile all patterns for speed
import re as _re
_COMPILED_RULES: list[tuple[str, list[_re.Pattern]]] = [
    (intent, [_re.compile(p, _re.IGNORECASE) for p in patterns])
    for intent, patterns in _INTENT_RULES
]

_INTENT_CONTEXT_HINTS: dict[str, str] = {
    "greeting":          "[Intent: greeting] Keep reply to ONE warm sentence. Ask what they're looking for.",
    "property_search":   "[Intent: property_search] User wants specific listings. Mention top 2-3 properties from context by name, price, location. Be concise.",
    "property_followup": "[Intent: property_followup] User is asking about a previously mentioned property. Use the conversation history to answer specifically — don't re-list everything.",
    "price_inquiry":     "[Intent: price_inquiry] User is asking about pricing. Give clear price ranges or specific prices. Mention PSF if available.",
    "currency_convert":  "[Intent: currency_convert] User wants prices in another currency. Use the currency context provided and state both SGD and converted amount.",
    "location_info":     "[Intent: location_info] User wants to know about a Singapore area or district. Give a brief, useful description — vibe, MRT access, typical prices.",
    "general_advice":    "[Intent: general_advice] User wants property advice or guidance. Give a clear, opinionated recommendation in 2-3 sentences max.",
    "off_topic":         "[Intent: off_topic] This is outside your domain. Politely redirect to Singapore property topics in one sentence.",
}


def classify_intent(text: str, history: list[dict] = None) -> str:
    """
    Classify user query intent using fast rule-based matching.
    Falls back to 'property_search' if history shows recent property context.
    Zero LLM calls — runs in <1ms.
    """
    t = text.strip()

    for intent, patterns in _COMPILED_RULES:
        for pat in patterns:
            if pat.search(t):
                return intent

    # Contextual fallback: if recent history had property talk, treat as followup
    if history:
        recent = history[-4:]
        for msg in recent:
            c = (msg.get("content") or "").lower()
            if any(kw in c for kw in ["sgd", "bedroom", "district", "listing", "sqft", "condo"]):
                return "property_followup"

    return "general_advice"


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

    # Add last 10 messages from history for context (enough for multi-turn follow-ups)
    if chat_history:
        recent = chat_history[-10:]
        for msg in recent:
            if msg.get("role") in ["user", "assistant"] and msg.get("content"):
                # Trim very long assistant messages (property listings) to keep tokens low
                content = msg["content"]
                if msg["role"] == "assistant" and len(content) > 400:
                    content = content[:400] + "…"
                messages.append({"role": msg["role"], "content": content})

    # Inject currency context if user asked for conversion
    currency_code = _detect_currency_request(user_text)
    if currency_code and currency_code in SGD_RATES:
        rate = SGD_RATES[currency_code]
        currency_context = f"[Currency context: 1 SGD = {rate} {currency_code}. Use this rate to convert SGD prices if the user asks.]"
        messages.append({"role": "system", "content": currency_context})

    # Inject intent context hint so the LLM knows exactly how to respond
    intent = classify_intent(user_text, chat_history)
    hint = _INTENT_CONTEXT_HINTS.get(intent, "")
    if hint:
        messages.append({"role": "system", "content": hint})
    print(f"[LLM] Intent: {intent}")

    messages.append({"role": "user", "content": user_text})
    print(f"[LLM] build_messages total: {int((_time.time()-t0)*1000)}ms")
    return messages, []
