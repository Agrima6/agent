import json
from openai import OpenAI

from config import GROQ_API_KEY

# Groq exposes an OpenAI-compatible chat completions API, so the same client
# works — just pointed at Groq's base_url with a Groq key (free tier, no
# billing required for the structured-JSON calls below).
client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

MODEL = "openai/gpt-oss-120b"


def structured_json(system_prompt: str, user_prompt: str, model: str = MODEL) -> dict:
    """Call the LLM and force a JSON object response. Untrusted content passed via
    user_prompt must already be wrapped/labeled by the caller as [UNTRUSTED]."""
    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )
    return json.loads(resp.choices[0].message.content)


def chat_reply(system_prompt: str, history: list[dict], model: str = MODEL) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}, *history],
        temperature=0.6,
    )
    return resp.choices[0].message.content
