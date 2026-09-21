"""Sarvam Bulbul v3 text-to-speech for LiveKit, over WebSocket streaming.

Why this replaces the old REST plugin (all measured against Sarvam's live API):
  * REST took 2-6 s per sentence (2.0 / 5.9 / 3.1 s for the same text) and opened a fresh HTTPS
    connection each time; a slow sentence arriving after the previous one had finished playing is
    an audible gap - the "audio breaking" candidates heard.
  * WebSocket streaming produced the first audio in ~0.3-0.5 s and synthesises ~5x faster than real
    time, on one long-lived connection.
  * Audio is requested as raw 16-bit PCM (linear16) instead of one MP3 file per sentence, which
    removes MP3 encoder padding at every sentence boundary and the decode cost.

Delivery guarantees:
  * SENTENCE-LEVEL COMMIT. A sentence's audio is buffered until Sarvam signals it is complete and
    only then pushed to the audio output - exactly once, in order. LiveKit itself never retries once
    audio has been emitted, so committing whole sentences is what makes every sentence safely
    retryable: a failed attempt has emitted nothing, so nothing can ever be spoken twice.
  * RETRY LADDER per sentence: WebSocket -> a fresh WebSocket -> REST (same voice) -> raise, at which
    point LiveKit's FallbackAdapter can switch to a different provider.
  * Client errors (bad speaker/params, HTTP 4xx) are not retried - retrying cannot fix them.
  * Truncated audio (far shorter than the text needs) is treated as a failure and retried.
  * Cancellation (candidate barge-in) closes the in-flight connection instead of returning it to
    the pool, so leftover chunks from an interrupted sentence can never reach the next utterance.
"""
import asyncio
import base64
import contextlib
import io
import logging
import time
import wave
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import httpx
from livekit.agents import APIConnectionError, APIStatusError, tts, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

import turn_trace
from speech_prep import split_for_tts, split_sentences
from voice_config import VoiceConfig

logger = logging.getLogger("sarvam-tts")

REST_URL = "https://api.sarvam.ai/text-to-speech"
CONNECT_TIMEOUT = 6.0
FIRST_CHUNK_TIMEOUT = 4.0      # a sentence must start producing audio within this long
SENTENCE_TIMEOUT = 12.0        # ...and finish within this long
REST_TIMEOUT = 15.0
PREFETCH = 3                  # sentences synthesised ahead of playback
FIRST_SENTENCE_HEAD_START = 1.5   # max seconds later sentences wait for sentence 0's first audio
PREROLL_SECONDS = 0.25         # audio to hold back at the very start so a slow first chunk can't underrun playback
MAX_CONNECTION_AGE = 240.0     # recycle idle connections well before server-side limits
MIN_SECONDS_PER_WORD = 0.10    # audio shorter than this is treated as truncated (600 wpm: implausibly fast)
RETRY_BACKOFF = 0.15


class SarvamTransportError(Exception):
    def __init__(self, message: str, *, retryable: bool = True, status: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status


@dataclass
class TTSStats:
    """Counters that make TTS delivery measurable (chunks, retries, reconnects, latency)."""
    sentences: int = 0
    ws_ok: int = 0
    ws_failures: int = 0
    ws_connections_opened: int = 0
    ws_reconnects: int = 0
    rest_fallbacks: int = 0
    failures: int = 0
    discarded_in_flight: int = 0
    mid_sentence_failures: int = 0
    cache_hits: int = 0
    chunks_received: int = 0
    audio_seconds: float = 0.0
    first_audio_ms: list[float] = field(default_factory=list)

    def snapshot(self) -> dict:
        ttfb = sorted(self.first_audio_ms)
        return {
            "sentences": self.sentences, "ws_ok": self.ws_ok, "ws_failures": self.ws_failures,
            "ws_connections_opened": self.ws_connections_opened, "ws_reconnects": self.ws_reconnects,
            "rest_fallbacks": self.rest_fallbacks, "failures": self.failures,
            "mid_sentence_failures": self.mid_sentence_failures, "cache_hits": self.cache_hits,
            "discarded_in_flight": self.discarded_in_flight, "chunks_received": self.chunks_received,
            "audio_seconds": round(self.audio_seconds, 2),
            "first_audio_ms_p50": ttfb[len(ttfb) // 2] if ttfb else None,
            "first_audio_ms_max": ttfb[-1] if ttfb else None,
        }


def speech_units(text: str) -> list[str]:
    """The exact sentence units the synthesise stream cuts `text` into (same rule as `_input`), so a
    fixed phrase can be recognised - and cached - by the strings the engine will actually be asked for."""
    units: list[str] = []
    for part in split_sentences(text, min_chars=16):
        units.extend(split_for_tts(part))
    return units


class _PhraseCache:
    """Small LRU of synthesised audio for lines that never change (refusals, closing...). Bounded in
    entries AND bytes so it cannot grow with the interview: a 4 s line is ~190 KB of 24 kHz PCM."""

    def __init__(self, max_entries: int = 48, max_bytes: int = 8_000_000):
        self._items: "OrderedDict[str, bytes]" = OrderedDict()
        self._bytes = 0
        self._max_entries, self._max_bytes = max_entries, max_bytes

    def get(self, text: str) -> bytes | None:
        pcm = self._items.get(text)
        if pcm is not None:
            self._items.move_to_end(text)
        return pcm

    def put(self, text: str, pcm: bytes) -> None:
        if not pcm or len(pcm) > self._max_bytes // 4:
            return
        if text in self._items:
            self._bytes -= len(self._items.pop(text))
        self._items[text] = pcm
        self._bytes += len(pcm)
        while len(self._items) > self._max_entries or self._bytes > self._max_bytes:
            _, evicted = self._items.popitem(last=False)
            self._bytes -= len(evicted)

    def __len__(self) -> int:
        return len(self._items)


class _WSConn:
    """One configured Bulbul WebSocket. Handles exactly one sentence at a time."""

    def __init__(self, cm, ws):
        self._cm = cm
        self._ws = ws
        self.created = time.monotonic()
        self.closed = False

    @classmethod
    async def open(cls, api_key: str, voice: VoiceConfig) -> "_WSConn":
        from sarvamai import AsyncSarvamAI

        client = AsyncSarvamAI(api_subscription_key=api_key)
        cm = client.text_to_speech_streaming.connect(model=voice.model, send_completion_event=True)
        try:
            ws = await asyncio.wait_for(cm.__aenter__(), CONNECT_TIMEOUT)
            await asyncio.wait_for(ws.configure(
                target_language_code=voice.language_code, speaker=voice.speaker, pace=voice.pace,
                speech_sample_rate=voice.sample_rate, output_audio_codec="linear16",
                min_buffer_size=30, max_chunk_length=150,
            ), CONNECT_TIMEOUT)
        except Exception as exc:
            with contextlib.suppress(Exception):
                await cm.__aexit__(None, None, None)
            raise SarvamTransportError(f"websocket connect failed: {type(exc).__name__}") from exc
        return cls(cm, ws)

    async def synth_stream(self, text: str, *, first_chunk_timeout: float = FIRST_CHUNK_TIMEOUT,
                           total_timeout: float = SENTENCE_TIMEOUT, timing: dict | None = None):
        """Synthesize one sentence, yielding PCM chunks AS THEY ARRIVE (16-bit aligned).

        `timing["first"]` is set to the seconds until the first chunk. Raises SarvamTransportError on
        an error message, a timeout or a dropped connection - possibly after some chunks were yielded,
        which is why the caller (not this method) decides whether a retry is still safe.
        """
        from sarvamai import AudioOutput, ErrorResponse, EventResponse

        started = time.monotonic()
        first: float | None = None
        carry = b""
        try:
            await self._ws.convert(text)
            await self._ws.flush()
            while True:
                remaining = total_timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise SarvamTransportError("sentence timed out")
                wait = min(remaining, first_chunk_timeout) if first is None else remaining
                try:
                    msg = await asyncio.wait_for(self._ws.recv(), wait)
                except asyncio.TimeoutError:
                    raise SarvamTransportError("timed out waiting for audio") from None
                if isinstance(msg, AudioOutput):
                    data = carry + base64.b64decode(msg.data.audio)
                    carry = data[-1:] if len(data) % 2 else b""   # 16-bit samples: never emit half a sample
                    data = data[: len(data) - len(carry)]
                    if first is None:
                        first = time.monotonic() - started
                        if timing is not None:
                            timing["first"] = first
                    if data:
                        yield data
                elif isinstance(msg, EventResponse):
                    if getattr(msg.data, "event_type", "") == "final":
                        break
                elif isinstance(msg, ErrorResponse):
                    message = str(getattr(msg.data, "message", "") or "error")
                    client_error = message.lstrip().startswith("4")
                    raise SarvamTransportError(f"sarvam error: {message[:160]}", retryable=not client_error)
        except SarvamTransportError:
            raise
        except Exception as exc:  # connection closed, protocol error, ...
            raise SarvamTransportError(f"connection failed: {type(exc).__name__}") from exc
        if timing is not None and first is None:
            timing["first"] = time.monotonic() - started

    async def synth(self, text: str, *, first_chunk_timeout: float = FIRST_CHUNK_TIMEOUT,
                    total_timeout: float = SENTENCE_TIMEOUT) -> tuple[bytes, int, float]:
        """Buffered variant: the whole sentence as one blob. Returns (pcm, chunk_count, seconds_to_first_chunk)."""
        timing: dict = {}
        pcm = bytearray()
        chunks = 0
        started = time.monotonic()
        async for chunk in self.synth_stream(text, first_chunk_timeout=first_chunk_timeout,
                                             total_timeout=total_timeout, timing=timing):
            pcm += chunk
            chunks += 1
        return bytes(pcm), chunks, timing.get("first", time.monotonic() - started)

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        with contextlib.suppress(Exception):
            await self._cm.__aexit__(None, None, None)


class _ConnectionPool:
    """Keeps at most one idle, already-configured connection ready, so a sentence rarely pays the
    connect + configure round-trips."""

    def __init__(self, open_fn: Callable[[], Awaitable[_WSConn]], max_age: float = MAX_CONNECTION_AGE):
        self._open = open_fn
        self._max_age = max_age
        self._idle: _WSConn | None = None

    async def acquire(self) -> tuple[_WSConn, bool]:
        conn, self._idle = self._idle, None
        if conn is not None and not conn.closed and time.monotonic() - conn.created < self._max_age:
            return conn, True
        if conn is not None:
            await conn.close()
        return await self._open(), False

    async def release(self, conn: _WSConn, healthy: bool) -> None:
        if not healthy or conn.closed or self._idle is not None:
            await conn.close()
            return
        self._idle = conn

    async def warm(self) -> None:
        if self._idle is None:
            with contextlib.suppress(Exception):
                self._idle = await self._open()

    async def aclose(self) -> None:
        conn, self._idle = self._idle, None
        if conn is not None:
            await conn.close()


class _SentenceEngine:
    """Turns one sentence into committed PCM audio using the retry ladder."""

    def __init__(self, api_key: str, voice: VoiceConfig, *, pool: _ConnectionPool | None = None,
                 rest: Callable[[str], Awaitable[bytes]] | None = None,
                 open_ws: Callable[[], Awaitable[_WSConn]] | None = None):
        self.voice = voice
        self._cache = _PhraseCache()
        self._cacheable: set[str] = set()
        self._dedicated_open = open_ws
        self.stats = TTSStats()
        self._api_key = api_key
        self._http: httpx.AsyncClient | None = None
        self._pool = pool or _ConnectionPool(self._open_ws)
        self._rest = rest or self._rest_synth

    async def _open_ws(self) -> _WSConn:
        conn = await _WSConn.open(self._api_key, self.voice)
        self.stats.ws_connections_opened += 1
        return conn

    async def warm(self) -> None:
        await self._pool.warm()

    async def aclose(self) -> None:
        await self._pool.aclose()
        if self._http is not None:
            with contextlib.suppress(Exception):
                await self._http.aclose()
        logger.info("tts_session_summary %s", self.stats.snapshot())

    async def _rest_synth(self, text: str) -> bytes:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=REST_TIMEOUT)
        try:
            resp = await self._http.post(
                REST_URL,
                headers={"api-subscription-key": self._api_key, "Content-Type": "application/json"},
                json={"text": text, "target_language_code": self.voice.language_code, "model": self.voice.model,
                      "speaker": self.voice.speaker, "pace": self.voice.pace,
                      "speech_sample_rate": self.voice.sample_rate, "output_audio_codec": "wav"},
            )
        except httpx.HTTPError as exc:
            raise SarvamTransportError(f"rest request failed: {type(exc).__name__}") from exc
        if resp.status_code >= 400:
            raise SarvamTransportError(f"rest http {resp.status_code}", status=resp.status_code,
                                       retryable=resp.status_code >= 500 or resp.status_code == 429)
        try:
            audio = base64.b64decode(resp.json()["audios"][0])
            with wave.open(io.BytesIO(audio)) as w:
                if w.getsampwidth() != 2 or w.getnchannels() != 1 or w.getframerate() != self.voice.sample_rate:
                    raise SarvamTransportError("rest returned unexpected audio format", retryable=False)
                return w.readframes(w.getnframes())
        except SarvamTransportError:
            raise
        except Exception as exc:
            raise SarvamTransportError(f"rest returned unusable audio: {type(exc).__name__}") from exc

    def _validate(self, pcm: bytes, text: str) -> None:
        seconds = len(pcm) / 2 / self.voice.sample_rate
        words = max(1, len(text.split()))
        if seconds < max(0.25, MIN_SECONDS_PER_WORD * words):
            raise SarvamTransportError(f"truncated audio ({seconds:.2f}s for {words} words)")

    async def _ws_once(self, text: str) -> tuple[bytes, int, float]:
        conn, reused = await self._pool.acquire()
        healthy = False
        try:
            result = await conn.synth(text)
            healthy = True
            return result
        finally:
            # An interrupted or failed sentence leaves the connection in an unknown state: never
            # reuse it, so stale chunks can't bleed into the next sentence.
            await self._pool.release(conn, healthy)
            # Open the NEXT connection now, while the candidate is still listening/answering, so the
            # following sentence starts without a connect + configure round-trip.
            if healthy:
                with contextlib.suppress(RuntimeError):
                    asyncio.get_running_loop().create_task(self._pool.warm())

    async def synthesize(self, text: str, *, request_id: str, idx: int) -> bytes:
        plan = ("ws", "ws", "rest")
        last: SarvamTransportError | None = None
        for attempt, transport in enumerate(plan, 1):
            started = time.monotonic()
            try:
                if transport == "ws":
                    if attempt > 1:
                        self.stats.ws_reconnects += 1
                    pcm, chunks, ttfb = await self._ws_once(text)
                else:
                    self.stats.rest_fallbacks += 1
                    pcm, chunks, ttfb = await self._rest(text), 1, time.monotonic() - started
                self._validate(pcm, text)
            except asyncio.CancelledError:
                self.stats.discarded_in_flight += 1
                raise
            except SarvamTransportError as exc:
                last = exc
                if transport == "ws":
                    self.stats.ws_failures += 1
                logger.warning("tts_sentence_failed request_id=%s sentence=%d attempt=%d transport=%s error=%s",
                               request_id, idx, attempt, transport, exc)
                if not exc.retryable:
                    break
                await asyncio.sleep(RETRY_BACKOFF * attempt)
                continue
            seconds = len(pcm) / 2 / self.voice.sample_rate
            self.stats.sentences += 1
            self.stats.chunks_received += chunks
            self.stats.audio_seconds += seconds
            self.stats.first_audio_ms.append(round(ttfb * 1000))
            if transport == "ws":
                self.stats.ws_ok += 1
            logger.info("tts_sentence request_id=%s sentence=%d transport=%s attempt=%d chars=%d chunks=%d "
                        "first_audio_ms=%d total_ms=%d audio_s=%.2f", request_id, idx, transport, attempt,
                        len(text), chunks, ttfb * 1000, (time.monotonic() - started) * 1000, seconds)
            return pcm
        self.stats.failures += 1
        raise last or SarvamTransportError("synthesis failed")


    # ------------------------------------------------------------------ fixed-phrase cache
    def set_cacheable(self, units) -> None:
        """Lines that never change (refusals, closing...): synthesised once, then replayed from memory."""
        self._cacheable = set(units)

    async def prewarm_phrases(self, units) -> int:
        """Synthesise fixed lines in the background on their OWN connection, so it can never take the warm
        connection a live reply needs. Best effort: any failure just means the line is synthesised on
        first use instead. Returns the number of lines added to the cache."""
        todo = [u for u in dict.fromkeys(units) if u in self._cacheable and self._cache.get(u) is None]
        if not todo:
            return 0
        opener = self._dedicated_open or self._open_ws
        added = 0
        try:
            conn = await opener()
        except Exception as exc:  # noqa: BLE001
            logger.info("phrase prewarm skipped: %s", exc)
            return 0
        try:
            for unit in todo:
                pcm, _chunks, _ttfb = await conn.synth(unit)
                seconds = len(pcm) / 2 / self.voice.sample_rate
                if seconds >= max(0.25, MIN_SECONDS_PER_WORD * max(1, len(unit.split()))):
                    self._cache.put(unit, pcm)
                    added += 1
        except Exception as exc:  # noqa: BLE001
            logger.info("phrase prewarm stopped early: %s", type(exc).__name__)
        finally:
            await conn.close()
        return added

    # ------------------------------------------------------------------ streaming path (used in production)
    async def _ws_stream(self, text: str, timing: dict):
        """One WebSocket attempt as a chunk generator. The connection goes back to the pool only if the
        sentence completed cleanly; an interrupted or failed one is discarded, so stale chunks can never
        bleed into the next sentence."""
        conn, _reused = await self._pool.acquire()
        healthy = False
        try:
            if hasattr(conn, "synth_stream"):
                async for chunk in conn.synth_stream(text, timing=timing):
                    yield chunk
            else:                                    # buffered-only connection (test doubles)
                pcm, _chunks, ttfb = await conn.synth(text)
                timing["first"] = ttfb
                yield pcm
            healthy = True
        finally:
            await self._pool.release(conn, healthy)
            if healthy:
                with contextlib.suppress(RuntimeError):
                    asyncio.get_running_loop().create_task(self._pool.warm())

    async def synthesize_stream(self, text: str, *, request_id: str, idx: int):
        """Yield PCM chunks for one sentence as Sarvam produces them.

        Time to first audio is now the time to the first CHUNK (~0.3 s) instead of the time to the whole
        sentence (0.4-1.7 s measured, more under load). Exactly-once is preserved differently: retries
        (ws -> ws -> rest) are only allowed while NOTHING has been yielded. Once audio is on its way to the
        candidate a failure ends that sentence (logged, counted) instead of replaying it and duplicating speech.
        """
        if text in self._cacheable:
            cached = self._cache.get(text)
            if cached is not None:                      # a fixed line we've already synthesised: no network at all
                self.stats.cache_hits += 1
                turn_trace.mark("tts_request")
                turn_trace.mark("tts_first_audio")
                yield cached
                return
        keep = bytearray() if text in self._cacheable else None
        plan = ("ws", "ws", "rest")
        last: SarvamTransportError | None = None
        yielded = 0
        chunks = 0
        for attempt, transport in enumerate(plan, 1):
            started = time.monotonic()
            timing: dict = {}
            turn_trace.mark("tts_request")
            try:
                if transport == "ws":
                    if attempt > 1:
                        self.stats.ws_reconnects += 1
                    async with contextlib.aclosing(self._ws_stream(text, timing)) as stream:
                        async for chunk in stream:
                            if yielded == 0:
                                turn_trace.mark("tts_first_audio")
                            yielded += len(chunk)
                            chunks += 1
                            if keep is not None:
                                keep += chunk
                            yield chunk
                else:
                    self.stats.rest_fallbacks += 1
                    pcm = await self._rest(text)
                    timing["first"] = time.monotonic() - started
                    turn_trace.mark("tts_first_audio")
                    yielded += len(pcm)
                    chunks += 1
                    if keep is not None:
                        keep += pcm
                    yield pcm
            except asyncio.CancelledError:
                self.stats.discarded_in_flight += 1
                raise
            except SarvamTransportError as exc:
                last = exc
                if transport == "ws":
                    self.stats.ws_failures += 1
                if yielded:
                    self.stats.mid_sentence_failures += 1
                    logger.warning("tts_sentence_cut request_id=%s sentence=%d transport=%s error=%s "
                                   "(audio already playing - not replayed)", request_id, idx, transport, exc)
                    return
                logger.warning("tts_sentence_failed request_id=%s sentence=%d attempt=%d transport=%s error=%s",
                               request_id, idx, attempt, transport, exc)
                if not exc.retryable:
                    break
                await asyncio.sleep(RETRY_BACKOFF * attempt)
                continue
            seconds = yielded / 2 / self.voice.sample_rate
            self.stats.sentences += 1
            self.stats.chunks_received += chunks
            self.stats.audio_seconds += seconds
            first = timing.get("first", time.monotonic() - started)
            self.stats.first_audio_ms.append(round(first * 1000))
            if transport == "ws":
                self.stats.ws_ok += 1
            words = max(1, len(text.split()))
            if seconds < max(0.25, MIN_SECONDS_PER_WORD * words):
                logger.warning("tts_short_audio request_id=%s sentence=%d %.2fs for %d words (already played)",
                               request_id, idx, seconds, words)
            elif keep is not None:
                self._cache.put(text, bytes(keep))          # only complete, plausible audio is ever cached
            logger.info("tts_sentence request_id=%s sentence=%d transport=%s attempt=%d chars=%d chunks=%d "
                        "first_audio_ms=%d total_ms=%d audio_s=%.2f", request_id, idx, transport, attempt,
                        len(text), chunks, first * 1000, (time.monotonic() - started) * 1000, seconds)
            return
        self.stats.failures += 1
        raise last or SarvamTransportError("synthesis failed")


def _as_api_error(exc: SarvamTransportError):
    """Retries are already exhausted inside the engine; mark the error non-retryable so LiveKit
    doesn't multiply them, and so a FallbackAdapter can switch provider promptly."""
    if exc.status is not None:
        return APIStatusError(str(exc), status_code=exc.status, retryable=False)
    return APIConnectionError(f"Sarvam TTS failed: {exc}", retryable=False)


class SarvamTTS(tts.TTS):
    def __init__(self, api_key: str, voice: VoiceConfig, *, engine: _SentenceEngine | None = None):
        super().__init__(capabilities=tts.TTSCapabilities(streaming=True), sample_rate=voice.sample_rate,
                         num_channels=1)
        self.voice = voice
        self._engine = engine or _SentenceEngine(api_key, voice)

    @property
    def stats(self) -> TTSStats:
        return self._engine.stats

    def set_cacheable(self, units) -> None:
        self._engine.set_cacheable(units)

    async def prewarm_phrases(self, units) -> int:
        return await self._engine.prewarm_phrases(units)

    def prewarm(self) -> None:
        with contextlib.suppress(RuntimeError):
            asyncio.get_running_loop().create_task(self._engine.warm())

    def synthesize(self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> tts.ChunkedStream:
        return _SarvamChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> tts.SynthesizeStream:
        return _SarvamSynthesizeStream(tts=self, conn_options=conn_options)

    async def aclose(self) -> None:
        await self._engine.aclose()


class _SarvamChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: SarvamTTS, input_text: str, conn_options: APIConnectOptions):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._sarvam = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        engine = self._sarvam._engine
        request_id = utils.shortuuid()
        output_emitter.initialize(request_id=request_id, sample_rate=engine.voice.sample_rate, num_channels=1,
                                  mime_type="audio/pcm")
        try:
            for idx, sentence in enumerate(split_for_tts(self.input_text)):
                output_emitter.push(await engine.synthesize(sentence, request_id=request_id, idx=idx))
        except SarvamTransportError as exc:
            raise _as_api_error(exc) from exc
        output_emitter.flush()


class _SarvamSynthesizeStream(tts.SynthesizeStream):
    def __init__(self, *, tts: SarvamTTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._sarvam = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        engine = self._sarvam._engine
        request_id = utils.shortuuid()
        output_emitter.initialize(request_id=request_id, sample_rate=engine.voice.sample_rate, num_channels=1,
                                  mime_type="audio/pcm", stream=True)
        output_emitter.start_segment(segment_id=utils.shortuuid())
        sentences: asyncio.Queue[str | None] = asyncio.Queue()

        async def _input() -> None:
            # Group incoming tokens into whole sentences; everything before the last (possibly
            # unfinished) sentence is complete and can be synthesised immediately.
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

        # Sentences are synthesised CONCURRENTLY (up to PREFETCH at once, each on its own connection) but
        # their audio is released strictly in speaking order, and CHUNK BY CHUNK: the first sentence starts
        # playing on its first ~0.3 s chunk instead of after the whole sentence is generated. Sarvam
        # generates audio ~5x faster than it plays (measured), so playback cannot outrun generation; the
        # small pre-roll below only guards against an unusually slow first chunk.
        pending: asyncio.Queue["_Unit | None"] = asyncio.Queue()
        slots = asyncio.Semaphore(PREFETCH)
        producers: list[asyncio.Task] = []
        preroll_bytes = int(engine.voice.sample_rate * 2 * PREROLL_SECONDS)

        first_audio = asyncio.Event()   # set once sentence 0 has produced audio (or ended without any)

        async def _produce(sentence: str, idx: int, unit: "_Unit") -> None:
            if idx > 0:
                # Later sentences wait for sentence 0's first audio: it gets the warm connection and the
                # whole network/CPU for its first chunk (measured: starting all of them at once made a
                # 4-sentence greeting ~130 ms SLOWER to first audio). Generation is ~5x faster than
                # playback, so they still finish well ahead of the speaker.
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(first_audio.wait(), FIRST_SENTENCE_HEAD_START)
            async with slots:
                try:
                    async with contextlib.aclosing(
                            engine.synthesize_stream(sentence, request_id=request_id, idx=idx)) as chunks:
                        async for chunk in chunks:
                            if idx == 0:
                                first_audio.set()
                            unit.chunks.put_nowait(chunk)
                    unit.chunks.put_nowait(None)
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:  # SarvamTransportError once every retry is exhausted
                    unit.chunks.put_nowait(exc)
                finally:
                    if idx == 0:
                        first_audio.set()

        async def _dispatch() -> None:
            idx = 0
            while True:
                sentence = await sentences.get()
                if sentence is None:
                    await pending.put(None)
                    return
                self._mark_started()
                unit = _Unit()
                producers.append(asyncio.create_task(_produce(sentence, idx, unit)))
                await pending.put(unit)
                idx += 1

        async def _synthesize() -> None:
            held = bytearray()          # pre-roll: only ever used before the very first push
            pushed_any = False

            def push(data: bytes) -> None:
                nonlocal pushed_any
                if not pushed_any:
                    turn_trace.mark("tts_first_push")
                    pushed_any = True
                output_emitter.push(data)

            while True:
                unit = await pending.get()
                if unit is None:
                    break
                while True:
                    item = await unit.chunks.get()
                    if item is None:
                        break
                    if isinstance(item, BaseException):
                        raise item
                    if pushed_any:
                        push(item)                       # committed exactly once, in order
                    else:
                        held += item
                        if len(held) >= preroll_bytes:
                            push(bytes(held))
                            held.clear()
                if held and not pushed_any:              # the first sentence was shorter than the pre-roll
                    push(bytes(held))
                    held.clear()

        input_task = asyncio.create_task(_input())
        dispatch_task = asyncio.create_task(_dispatch())
        synth_task = asyncio.create_task(_synthesize())
        try:
            await synth_task
        except SarvamTransportError as exc:
            raise _as_api_error(exc) from exc
        finally:
            output_emitter.end_segment()
            await utils.aio.gracefully_cancel(input_task, dispatch_task, synth_task)
            # Barge-in / failure: drop every sentence still in flight so no stale audio is ever played.
            for task in producers:
                task.cancel()
            for task in producers:
                with contextlib.suppress(BaseException):
                    await task


class _Unit:
    """One sentence's audio on its way to the speaker: chunks, then None (done) or an exception."""

    __slots__ = ("chunks",)

    def __init__(self) -> None:
        self.chunks: asyncio.Queue = asyncio.Queue()
