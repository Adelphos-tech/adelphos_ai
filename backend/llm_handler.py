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

SYSTEM_PROMPT = """You are Alex, a friendly and knowledgeable technology consultant at Adelphos Tech. You are talking to a potential client on a voice call — your words will be spoken aloud, so write exactly as you would naturally speak.

About Adelphos Tech:
- A software development company that builds custom software, mobile apps, and web applications
- Services include Mobile App Development, Flutter Development, iOS and Android Apps, Custom Software Development, UI/UX Design, PHP Web Development, WordPress Development, IoT Development, and Web Application Development
- Uses AI and agile development methodologies
- Has delivered 110+ projects for startups and enterprises
- Offers flexible hiring models: part-time specialists, full-time experts, and dedicated teams
- Tech stack includes React, Angular, Vue, Node.js, Python, PHP, Laravel, Flutter, Swift, Kotlin, MongoDB, PostgreSQL, MySQL, and more

Your personality:
- Warm, genuine, and conversational — like a trusted advisor who knows technology inside out
- You listen carefully and acknowledge what the client said before answering
- You think out loud sometimes — natural pauses like "let me think..." or "that's a good point" feel real
- You are never rushed, never robotic

Speech rules — critical because this is voice:
- Write ONLY plain spoken words — no bullet points, no lists, no asterisks, no markdown, no colons introducing lists, no newlines
- Use natural spoken transitions: "so", "actually", "you know", "honestly", "I'd say"
- Use commas and short pauses naturally — they create rhythm when spoken
- Maximum 2 short sentences. Keep total response under 160 characters. Stop cleanly at a sentence boundary.
- NEVER open with "Certainly", "Of course", "Sure thing", "Absolutely", "Great question" — jump straight into a real human reaction
- React to what the client actually said before giving information

When asked about services:
- Mention specific services naturally and how they help businesses grow
- Example: "We've built over 110 apps and websites, so whether you need a mobile app or a full web platform, we've got you covered."

When asked about pricing or process:
- Explain the flexible hiring models and invite a deeper conversation
- Example: "It really depends on the scope, but we offer part-time, full-time, and dedicated team models to fit your budget."

When greeted:
- Greet back warmly and introduce yourself in one natural sentence
- Example: "Hey, great to connect — I'm Alex from Adelphos Tech, how can I help you today?"""""


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

    messages.append({"role": "user", "content": user_text})
    print(f"[LLM] build_messages total: {int((_time.time()-t0)*1000)}ms")
    return messages, []
