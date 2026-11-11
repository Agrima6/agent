"""The Text to Dialogue plugin against a local fake of ElevenLabs' WebSocket (protocol as documented at
https://elevenlabs.io/docs/api-reference/text-to-dialogue/ttd-websocket)."""
import asyncio
import base64
import json

import pytest
import websockets

import languages
from elevenlabs_dialogue_plugin import DialogueTransportError, ElevenLabsDialogueTTS, parse_format


class FakeServer:
    def __init__(self, behaviours):
        self.behaviours = list(behaviours)      # one per connection: "ok" | "error" | "denied"
        self.connections = []
        self.server = None

    async def handler(self, ws):
        path = ws.request.path
        headers = dict(ws.request.headers)
        init = json.loads(await ws.recv())
        inputs = json.loads(await ws.recv())
        closing = json.loads(await ws.recv())
        self.connections.append({"path": path, "key": headers.get("xi-api-key"), "init": init, "inputs": inputs, "close": closing})
        behaviour = self.behaviours.pop(0) if self.behaviours else "ok"
        if behaviour == "denied":
            await ws.send(json.dumps({"message": "missing the permission text_to_speech", "error": "missing_permissions", "code": 1008}))
            await ws.close(1008, "missing permission")
            return
        if behaviour == "error":
            await ws.send(json.dumps({"message": "try again", "error": "internal_error", "code": 1011}))
            await ws.close(1011, "internal")
            return
        await ws.send(json.dumps({"audio": base64.b64encode(b"\x01\x02\x03").decode()}))    # odd length: half a 16-bit sample
        await ws.send(json.dumps({"audio": base64.b64encode(b"\x04\x05\x06").decode()}))
        await ws.send(json.dumps({"is_final_audio_for_turn": True}))
        await ws.send(json.dumps({"is_final": True}))
        await ws.close()

    async def __aenter__(self):
        self.server = await websockets.serve(self.handler, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc):
        self.server.close()
        await self.server.wait_closed()


def make(server, **kw):
    return ElevenLabsDialogueTTS(api_key="k-test", model="eleven_v3_conversational", voice_id="voice123",
                                 language="te", base_url=f"ws://127.0.0.1:{server.port}", **kw)


async def collect(agen):
    return [chunk async for chunk in agen]


def test_formats():
    assert parse_format("pcm_24000") == (24000, "audio/pcm")
    assert parse_format("mp3_44100_128") == (44100, "audio/mpeg")


def test_speaks_one_sentence_with_the_documented_messages():
    async def run():
        async with FakeServer(["ok"]) as server:
            engine = make(server, stability=0.5)
            audio = b"".join(await collect(engine.speak("నమస్కారం, స్వాగతం.")))
            conn = server.connections[0]
            assert "model_id=eleven_v3_conversational" in conn["path"] and "language_code=te" in conn["path"]
            assert "output_format=pcm_24000" in conn["path"]
            assert conn["key"] == "k-test"
            assert conn["init"] == {"voices": ["voice123"], "voice_settings": {"stability": 0.5}}
            assert conn["inputs"] == {"inputs": [{"text": "నమస్కారం, స్వాగతం.", "voice_id": "voice123"}]}
            assert conn["close"] == {"close_socket": True}
            return audio
    audio = asyncio.run(run())
    assert audio == b"\x01\x02\x03\x04\x05\x06"      # the odd byte is carried into the next chunk, never emitted as half a sample


def test_language_can_be_switched_mid_interview():
    async def run():
        async with FakeServer(["ok"]) as server:
            engine = make(server)
            engine.update_options(language="bn")
            await collect(engine.speak("নমস্কার, স্বাগতম।"))
            assert "language_code=bn" in server.connections[0]["path"]
    asyncio.run(run())


def test_a_transient_failure_is_retried_once_on_a_fresh_connection():
    async def run():
        async with FakeServer(["error", "ok"]) as server:
            engine = make(server)
            audio = b"".join(await collect(engine.speak_with_retry("Hello there, welcome.", "r1", 0)))
            assert len(server.connections) == 2
            return audio
    assert asyncio.run(run()) == b"\x01\x02\x03\x04\x05\x06"


def test_a_missing_permission_is_not_retried():
    async def run():
        async with FakeServer(["denied", "ok"]) as server:
            engine = make(server)
            with pytest.raises(DialogueTransportError) as caught:
                await collect(engine.speak_with_retry("Hello there, welcome.", "r1", 0))
            assert not caught.value.retryable
            assert len(server.connections) == 1
    asyncio.run(run())


def test_dialogue_models_speak_every_language_and_others_are_unchanged():
    everything = frozenset(languages.LANGUAGE_CONFIGS)
    for model in ("eleven_v3_conversational", "eleven_v4_turbo", "eleven_v4"):
        assert languages.languages_for_provider("elevenlabs", model) == everything
    assert languages.languages_for_provider("elevenlabs", "eleven_flash_v2_5") == languages.ELEVENLABS_V25_LANGUAGES
    assert "pa" not in languages.languages_for_provider("elevenlabs", "eleven_v3")
