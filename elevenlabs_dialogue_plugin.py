"""ElevenLabs Eleven v3 Conversational / v4 text-to-speech for LiveKit, over the Text to Dialogue WebSocket.

Why this exists: the plain ElevenLabs streaming models (Flash / Turbo v2.5) speak only Hindi and Tamil of the Indian
languages, and the plugin that ships with livekit-agents drives the Text to Speech WebSocket, which rejects the v3 models
(HTTP 400). The v3 / v4 models cover every Indian language this agent supports (Punjabi, Telugu, Bengali, Marathi, Gujarati,
Kannada, Malayalam as well) and are reachable in real time only through ElevenLabs' Text to Dialogue WebSocket
(wss://.../v1/text-to-dialogue/stream-input), whose protocol is different, hence this plugin.

Delivery follows the same rules as sarvam_tts_plugin.py:
  * One short WebSocket per sentence (connect, voices, text, close_socket, read until is_final), several sentences
    synthesised ahead at once, their audio released strictly in speaking order.
  * A sentence is retried once on a fresh connection, but only while NOTHING of it has been emitted, so no sentence is
    ever spoken twice. Auth / permission / bad-parameter errors are not retried.
  * Barge-in cancels the in-flight sockets, so stale audio can never reach the next utterance.
"""
import asyncio
import base64
import contextlib
import json
import logging
import time
from urllib.parse import urlencode

import websockets
from livekit.agents import APIConnectionError, APIStatusError, tts, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

import turn_trace
from speech_prep import split_for_tts, split_sentences

logger = logging.getLogger("elevenlabs-dialogue-tts")

BASE_URL = "wss://api.elevenlabs.io"
DEFAULT_FORMAT = "pcm_24000"
CONNECT_TIMEOUT = 6.0
FIRST_CHUNK_TIMEOUT = 6.0
SENTENCE_TIMEOUT = 20.0
PREFETCH = 3
RETRY_BACKOFF = 0.2
# Error codes the server sends that retrying cannot fix.
_NON_RETRYABLE = ("authentication", "missing_permissions", "invalid", "quota", "model")


class DialogueTransportError(Exception):
    def __init__(self, message: str, *, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


def parse_format(output_format: str) -> tuple[int, str]:
    """(sample rate, mime type) for an ElevenLabs output_format such as pcm_24000 or mp3_44100_128."""
    kind, _, rest = output_format.partition("_")
    rate = int(rest.split("_")[0]) if rest.split("_")[0].isdigit() else 24000
    return rate, ("audio/pcm" if kind == "pcm" else "audio/mpeg")


def speech_units(text: str) -> list[str]:
    units: list[str] = []
    for part in split_sentences(text, min_chars=16):
        units.extend(split_for_tts(part))
    return units


class ElevenLabsDialogueTTS(tts.TTS):
    def __init__(self, *, api_key: str, model: str, voice_id: str, language: str | None = None,
                 output_format: str = DEFAULT_FORMAT, stability: float | None = None,
                 pronunciation_locators: list[dict] | None = None, base_url: str = BASE_URL):
        rate, self._mime = parse_format(output_format)
        super().__init__(capabilities=tts.TTSCapabilities(streaming=True), sample_rate=rate, num_channels=1)
        self._api_key, self._model, self._voice_id = api_key, model, voice_id
        self._language, self._format, self._base_url = language, output_format, base_url
        self._stability, self._locators = stability, pronunciation_locators

    def update_options(self, *, language: str | None = None) -> None:
        """The candidate switched interview language: pin the new one for the next sentence."""
        if language:
            self._language = language

    def _url(self) -> str:
        params = {"model_id": self._model, "output_format": self._format}
        if self._language:
            params["language_code"] = self._language
        return f"{self._base_url}/v1/text-to-dialogue/stream-input?{urlencode(params)}"

    def _init_message(self) -> dict:
        message: dict = {"voices": [self._voice_id]}
        if self._stability is not None:
            message["voice_settings"] = {"stability": self._stability}
        if self._locators:
            message["pronunciation_dictionary_locators"] = self._locators
        return message

    async def speak(self, text: str, timing: dict | None = None):
        """Yield audio bytes for one sentence as they arrive. Raises DialogueTransportError."""
        started = time.monotonic()
        first = None
        carry = b""
        try:
            async with websockets.connect(self._url(), additional_headers={"xi-api-key": self._api_key},
                                          open_timeout=CONNECT_TIMEOUT, max_size=None) as ws:
                await ws.send(json.dumps(self._init_message()))
                await ws.send(json.dumps({"inputs": [{"text": text, "voice_id": self._voice_id}]}))
                await ws.send(json.dumps({"close_socket": True}))      # flushes the buffered text, then ends with is_final
                while True:
                    remaining = SENTENCE_TIMEOUT - (time.monotonic() - started)
                    if remaining <= 0:
                        raise DialogueTransportError("sentence timed out")
                    wait = min(remaining, FIRST_CHUNK_TIMEOUT) if first is None else remaining
                    try:
                        raw = await asyncio.wait_for(ws.recv(), wait)
                    except asyncio.TimeoutError:
                        raise DialogueTransportError("timed out waiting for audio") from None
                    msg = json.loads(raw)
                    if msg.get("error"):
                        code = str(msg.get("error"))
                        raise DialogueTransportError(f"elevenlabs error: {code}: {str(msg.get('message', ''))[:140]}",
                                                     retryable=not any(k in code for k in _NON_RETRYABLE))
                    if msg.get("audio"):
                        data = carry + base64.b64decode(msg["audio"])
                        if self._mime == "audio/pcm":
                            carry = data[-1:] if len(data) % 2 else b""    # 16-bit samples: never emit half a sample
                            data = data[: len(data) - len(carry)]
                        if first is None:
                            first = time.monotonic() - started
                            if timing is not None:
                                timing["first"] = first
                        if data:
                            yield data
                    if msg.get("is_final"):
                        return
        except DialogueTransportError:
            raise
        except websockets.exceptions.InvalidStatus as exc:
            status = exc.response.status_code
            raise DialogueTransportError(f"connect rejected ({status})", retryable=status >= 500) from exc
        except websockets.exceptions.ConnectionClosed as exc:
            reason = str(getattr(exc.rcvd, "reason", "") or "connection closed")[:160]
            raise DialogueTransportError(f"closed: {reason}", retryable=False if "permission" in reason else True) from exc
        except Exception as exc:  # noqa: BLE001
            raise DialogueTransportError(f"connection failed: {type(exc).__name__}") from exc

    async def speak_with_retry(self, text: str, request_id: str, idx: int):
        """One retry on a fresh connection, only while nothing has been yielded (so nothing is ever spoken twice)."""
        yielded = 0
        last: DialogueTransportError | None = None
        for attempt in (1, 2):
            timing: dict = {}
            turn_trace.mark("tts_request")
            try:
                async with contextlib.aclosing(self.speak(text, timing)) as chunks:
                    async for chunk in chunks:
                        if yielded == 0:
                            turn_trace.mark("tts_first_audio")
                        yielded += len(chunk)
                        yield chunk
                logger.info("tts_sentence request_id=%s sentence=%d attempt=%d chars=%d first_audio_ms=%d bytes=%d",
                            request_id, idx, attempt, len(text), timing.get("first", 0) * 1000, yielded)
                return
            except DialogueTransportError as exc:
                last = exc
                if yielded:
                    logger.warning("tts_sentence_cut request_id=%s sentence=%d error=%s (audio already playing - not replayed)",
                                   request_id, idx, exc)
                    return
                logger.warning("tts_sentence_failed request_id=%s sentence=%d attempt=%d error=%s", request_id, idx, attempt, exc)
                if not exc.retryable:
                    break
                await asyncio.sleep(RETRY_BACKOFF)
        raise last or DialogueTransportError("synthesis failed")

    def synthesize(self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> tts.ChunkedStream:
        return _ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> tts.SynthesizeStream:
        return _SynthesizeStream(tts=self, conn_options=conn_options)


def _as_api_error(exc: DialogueTransportError):
    return APIConnectionError(f"ElevenLabs TTS failed: {exc}", retryable=False)


class _ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: ElevenLabsDialogueTTS, input_text: str, conn_options: APIConnectOptions):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._el = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(request_id=request_id, sample_rate=self._el.sample_rate, num_channels=1,
                                  mime_type=self._el._mime)
        try:
            for idx, sentence in enumerate(split_for_tts(self.input_text)):
                async for chunk in self._el.speak_with_retry(sentence, request_id, idx):
                    output_emitter.push(chunk)
        except DialogueTransportError as exc:
            raise _as_api_error(exc) from exc
        output_emitter.flush()


class _SynthesizeStream(tts.SynthesizeStream):
    def __init__(self, *, tts: ElevenLabsDialogueTTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._el = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(request_id=request_id, sample_rate=self._el.sample_rate, num_channels=1,
                                  mime_type=self._el._mime, stream=True)
        output_emitter.start_segment(segment_id=utils.shortuuid())
        sentences: asyncio.Queue[str | None] = asyncio.Queue()
        pending: asyncio.Queue = asyncio.Queue()
        slots = asyncio.Semaphore(PREFETCH)
        producers: list[asyncio.Task] = []

        async def _input() -> None:
            buffer = ""
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    for unit in split_for_tts(buffer):
                        await sentences.put(unit)
                    buffer = ""
                    continue
                buffer += data
                parts = split_sentences(buffer, min_chars=16)
                if len(parts) > 1:
                    for complete in parts[:-1]:
                        for unit in split_for_tts(complete):
                            await sentences.put(unit)
                    buffer = parts[-1]
            for unit in split_for_tts(buffer):
                await sentences.put(unit)
            await sentences.put(None)

        async def _produce(sentence: str, idx: int, chunks: asyncio.Queue) -> None:
            async with slots:
                try:
                    async with contextlib.aclosing(self._el.speak_with_retry(sentence, request_id, idx)) as audio:
                        async for chunk in audio:
                            chunks.put_nowait(chunk)
                    chunks.put_nowait(None)
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:  # noqa: BLE001  (DialogueTransportError once retries are exhausted)
                    chunks.put_nowait(exc)

        async def _dispatch() -> None:
            idx = 0
            while True:
                sentence = await sentences.get()
                if sentence is None:
                    await pending.put(None)
                    return
                self._mark_started()
                chunks: asyncio.Queue = asyncio.Queue()
                producers.append(asyncio.create_task(_produce(sentence, idx, chunks)))
                await pending.put(chunks)
                idx += 1

        async def _synthesize() -> None:
            pushed = False
            while True:
                chunks = await pending.get()
                if chunks is None:
                    return
                while True:
                    item = await chunks.get()
                    if item is None:
                        break
                    if isinstance(item, BaseException):
                        raise item
                    if not pushed:
                        turn_trace.mark("tts_first_push")
                        pushed = True
                    output_emitter.push(item)

        tasks = [asyncio.create_task(_input()), asyncio.create_task(_dispatch())]
        synth = asyncio.create_task(_synthesize())
        try:
            await synth
        except DialogueTransportError as exc:
            raise _as_api_error(exc) from exc
        finally:
            output_emitter.end_segment()
            await utils.aio.gracefully_cancel(*tasks, synth)
            for task in producers:
                task.cancel()
            for task in producers:
                with contextlib.suppress(BaseException):
                    await task
