"""Diagnostic: why does the ElevenLabs streaming (WebSocket) connection close? Read-only; prints no secrets.

Run inside the worker container so it uses the server's real settings:
  docker compose exec -T worker python - < deploy/diag_elevenlabs_ws.py
Each variant opens the same multi-stream-input socket the interviewer uses, speaks one short Hinglish sentence, and
reports either 'OK (n audio chunks)' or the close code plus whatever reason ElevenLabs sent.
"""
import asyncio
import base64
import json
import os
import uuid

import aiohttp

KEY = os.environ.get("ELEVENLABS_API_KEY", "")
VOICE = os.environ.get("ELEVENLABS_VOICE_ID_MALE") or os.environ.get("ELEVENLABS_VOICE_ID") or "hpp4J3VqNfWAUOO0d1Us"
TEXT = "Namaste, aap apne last project ke baare mein bataiye. "

VARIANTS = {
    "A  exactly like the interviewer (flash v2.5, sync_alignment, auto_mode)": dict(model_id="eleven_flash_v2_5", sync_alignment="true", auto_mode="true"),
    "B  without sync_alignment": dict(model_id="eleven_flash_v2_5", auto_mode="true"),
    "C  without auto_mode": dict(model_id="eleven_flash_v2_5", sync_alignment="true"),
    "D  multilingual_v2 model": dict(model_id="eleven_multilingual_v2", sync_alignment="true", auto_mode="true"),
    "E  minimal (flash v2.5 only)": dict(model_id="eleven_flash_v2_5"),
    "F  pcm_16000 output instead of mp3": dict(model_id="eleven_flash_v2_5", output_format="pcm_16000"),
}


async def try_variant(session: aiohttp.ClientSession, name: str, extra: dict) -> None:
    params = {"output_format": "mp3_22050_32", "enable_ssml_parsing": "false", "enable_logging": "true",
              "inactivity_timeout": "180", "apply_text_normalization": "auto", **extra}
    url = f"wss://api.elevenlabs.io/v1/text-to-speech/{VOICE}/multi-stream-input?" + "&".join(f"{k}={v}" for k, v in params.items())
    ctx = uuid.uuid4().hex[:12]
    audio, notes = 0, []
    try:
        async with session.ws_connect(url, headers={"xi-api-key": KEY}, timeout=10) as ws:
            await ws.send_json({"text": " ", "context_id": ctx})
            await ws.send_json({"text": TEXT, "context_id": ctx, "flush": True})
            await ws.send_json({"context_id": ctx, "close_context": True})
            while True:
                msg = await asyncio.wait_for(ws.receive(), timeout=12)
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if data.get("audio"):
                        audio += 1
                    elif data.get("isFinal"):
                        break
                    else:
                        notes.append(str({k: v for k, v in data.items() if k not in ("audio", "alignment", "normalizedAlignment")})[:200])
                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING):
                    reason = msg.extra if isinstance(msg.extra, str) else ""
                    notes.append(f"CLOSED code={ws.close_code} reason={reason!r}")
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    notes.append(f"ERROR {ws.exception()}")
                    break
    except aiohttp.WSServerHandshakeError as exc:
        notes.append(f"HANDSHAKE REFUSED status={exc.status} message={exc.message!r}")
    except asyncio.TimeoutError:
        notes.append("timed out waiting for ElevenLabs")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"{type(exc).__name__}: {str(exc)[:160]}")
    verdict = f"OK ({audio} audio chunks)" if audio and not any(n.startswith(("CLOSED", "ERROR", "HANDSHAKE")) for n in notes) else "FAILED"
    print(f"{name}\n    -> {verdict}" + "".join(f"\n       {n}" for n in notes))


async def main() -> None:
    if not KEY:
        print("ELEVENLABS_API_KEY is not set in this container"); return
    print(f"voice id in use: {VOICE}")
    async with aiohttp.ClientSession() as session:
        for name, extra in VARIANTS.items():
            await try_variant(session, name, extra)


asyncio.run(main())
