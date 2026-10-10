"""Sarvam streaming TTS plugin: ordering, exactly-once delivery, retries, fallback, cancellation.

Uses scripted fake connections in place of the network. The real message-handling code
(_WSConn.synth) is exercised separately with real SDK message types.
"""
import asyncio
import base64
import struct

import pytest
from sarvamai import AudioOutput, ErrorResponse, EventResponse
from sarvamai.types import AudioOutputData, ErrorResponseData, EventResponseData

from sarvam_tts_plugin import (
    SarvamTTS, SarvamTransportError, _ConnectionPool, _SentenceEngine, _WSConn, _as_api_error,
)
from voice_config import resolve_voice

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


VOICE = resolve_voice("en", "female")
SR = VOICE.sample_rate


def pcm_for(text: str, marker: int | None = None) -> bytes:
    """Deterministic, text-identifiable PCM: 0.4 s per word, every sample = a per-text marker."""
    marker = marker if marker is not None else (sum(map(ord, text)) % 30000) + 1
    samples = int(SR * 0.4 * max(1, len(text.split())))
    return struct.pack("<h", marker) * samples


class FakeConn:
    """Stands in for _WSConn. `script` holds one outcome per synth() call: bytes | Exception | 'block'."""

    def __init__(self, registry, script=None):
        self.registry = registry
        self.script = list(script or [])
        self.created = __import__("time").monotonic()
        self.closed = False
        self.calls: list[str] = []
        self.started = asyncio.Event()
        registry.opened.append(self)

    async def synth(self, text, **_):
        self.calls.append(text)
        self.registry.all_calls.append(text)
        self.started.set()
        outcome = self.script.pop(0) if self.script else pcm_for(text)
        if outcome == "block":
            await asyncio.Event().wait()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome, 3, 0.05

    async def close(self):
        self.closed = True


class Registry:
    def __init__(self, *scripts):
        self.scripts = list(scripts)
        self.opened: list[FakeConn] = []
        self.all_calls: list[str] = []

    async def open(self):
        return FakeConn(self, self.scripts.pop(0) if self.scripts else None)


def make_engine(registry, rest=None):
    engine = _SentenceEngine("key", VOICE, pool=_ConnectionPool(registry.open), rest=rest)
    return engine


async def speak(tts_obj, text) -> bytes:
    stream = tts_obj.stream()
    stream.push_text(text)
    stream.end_input()
    out = bytearray()
    async for ev in stream:
        out += ev.frame.data.tobytes()
    await stream.aclose()
    return bytes(out)


# ------------------------------------------------------------------ ordering / exactly-once
async def test_units_are_emitted_once_and_in_order():
    reg = Registry()
    tts_obj = SarvamTTS("key", VOICE, engine=make_engine(reg))
    text = "You mentioned hashing. What happens when two keys land in the same bucket, and how is it handled?"
    audio = await speak(tts_obj, text)
    from speech_prep import split_for_tts
    expected = b"".join(pcm_for(u) for u in split_for_tts(text))
    assert audio == expected
    assert reg.all_calls == split_for_tts(text)          # each unit requested exactly once
    await tts_obj.aclose()


async def test_a_single_connection_is_reused_across_units():
    reg = Registry()
    tts_obj = SarvamTTS("key", VOICE, engine=make_engine(reg))
    await speak(tts_obj, "First sentence is right here. Second sentence follows it. Third one ends it.")
    assert len(reg.opened) == 1 and not reg.opened[0].closed
    await tts_obj.aclose()
    assert reg.opened[0].closed


async def test_prewarm_opens_a_connection_before_the_first_sentence():
    reg = Registry()
    tts_obj = SarvamTTS("key", VOICE, engine=make_engine(reg))
    tts_obj.prewarm()
    await asyncio.sleep(0.05)
    assert len(reg.opened) == 1
    await speak(tts_obj, "Hello there, welcome to the interview today.")
    assert len(reg.opened) == 1          # the prewarmed connection was used, no second one opened
    await tts_obj.aclose()


# ------------------------------------------------------------------ retry ladder
async def test_failed_attempt_retries_on_a_fresh_connection_without_duplicate_audio():
    text = "What happens when two keys land in the same bucket?"
    reg = Registry([SarvamTransportError("connection dropped mid-sentence")], [])
    engine = make_engine(reg)
    pcm = await engine.synthesize(text, request_id="r", idx=0)
    assert pcm == pcm_for(text)                       # exactly one copy of the sentence audio
    assert reg.opened[0].closed and not reg.opened[1].closed   # failed connection discarded, never reused
    assert engine.stats.ws_failures == 1 and engine.stats.ws_reconnects == 1 and engine.stats.rest_fallbacks == 0


async def test_two_websocket_failures_fall_back_to_rest_with_the_same_voice():
    text = "Why is constructor injection preferred over field injection?"
    reg = Registry([SarvamTransportError("x")], [SarvamTransportError("y")])
    rest_calls = []

    async def rest(t):
        rest_calls.append(t)
        return pcm_for(t)

    engine = make_engine(reg, rest=rest)
    assert await engine.synthesize(text, request_id="r", idx=0) == pcm_for(text)
    assert rest_calls == [text] and engine.stats.rest_fallbacks == 1 and engine.stats.ws_failures == 2


async def test_client_errors_are_not_retried():
    reg = Registry([SarvamTransportError("sarvam error: 400: bad speaker", retryable=False)])
    rest_calls = []

    async def rest(t):
        rest_calls.append(t)
        return pcm_for(t)

    engine = make_engine(reg, rest=rest)
    with pytest.raises(SarvamTransportError):
        await engine.synthesize("Please answer the question now.", request_id="r", idx=0)
    assert len(reg.all_calls) == 1 and rest_calls == [] and engine.stats.failures == 1


async def test_exhausted_ladder_raises_after_ws_ws_rest():
    reg = Registry([SarvamTransportError("a")], [SarvamTransportError("b")])

    async def rest(t):
        raise SarvamTransportError("rest down", status=503)

    engine = make_engine(reg, rest=rest)
    with pytest.raises(SarvamTransportError):
        await engine.synthesize("Tell me about your last project.", request_id="r", idx=0)
    assert engine.stats.failures == 1 and len(reg.opened) == 2


async def test_truncated_audio_is_treated_as_a_failure_and_retried():
    text = "Could you walk me through how you would approach testing a payment integration?"
    reg = Registry([b"\x01\x00" * 100], [])          # 100 samples for 12 words: obviously truncated
    engine = make_engine(reg)
    assert await engine.synthesize(text, request_id="r", idx=0) == pcm_for(text)
    assert engine.stats.ws_failures == 1


def test_transport_errors_map_to_non_retryable_livekit_errors():
    from livekit.agents import APIConnectionError, APIStatusError
    assert isinstance(_as_api_error(SarvamTransportError("x")), APIConnectionError)
    assert _as_api_error(SarvamTransportError("x")).retryable is False
    assert isinstance(_as_api_error(SarvamTransportError("x", status=400)), APIStatusError)


# ------------------------------------------------------------------ interruption / stale audio
async def test_barge_in_discards_the_inflight_sentence_and_never_reuses_its_connection():
    reg = Registry(["block"], [])
    engine = make_engine(reg)
    tts_obj = SarvamTTS("key", VOICE, engine=engine)

    stream = tts_obj.stream()
    stream.push_text("This sentence will be interrupted by the candidate.")
    stream.end_input()
    reader = asyncio.create_task(_drain(stream))
    for _ in range(200):                             # wait until the stream has actually started synthesising
        if reg.opened and reg.opened[0].started.is_set():
            break
        await asyncio.sleep(0.01)
    assert reg.opened and reg.opened[0].started.is_set()
    await stream.aclose()                            # candidate started speaking
    await asyncio.gather(reader, return_exceptions=True)

    assert reg.opened[0].closed, "interrupted connection must not go back into the pool"
    assert engine.stats.discarded_in_flight == 1

    # The next utterance gets a fresh connection and only its own audio - no stale chunks.
    text = "Thanks. What is your approach?"
    audio = await speak(tts_obj, text)
    from speech_prep import split_for_tts
    assert audio == b"".join(pcm_for(u) for u in split_for_tts(text))
    assert len(reg.opened) == 2 and reg.opened[1] is not reg.opened[0]
    await tts_obj.aclose()


async def _drain(stream):
    async for _ in stream:
        pass


async def test_overlapping_utterances_do_not_share_a_connection():
    reg = Registry()
    tts_obj = SarvamTTS("key", VOICE, engine=make_engine(reg))
    a, b = await asyncio.gather(speak(tts_obj, "The first utterance goes out here."),
                                speak(tts_obj, "A completely different second utterance."))
    assert a == pcm_for("The first utterance goes out here.")
    assert b == pcm_for("A completely different second utterance.")
    await tts_obj.aclose()


# ------------------------------------------------------------------ real message handling
def _audio(pcm: bytes) -> AudioOutput:
    return AudioOutput(type="audio", data=AudioOutputData(audio=base64.b64encode(pcm).decode(), content_type="audio/x-raw"))


def _final() -> EventResponse:
    return EventResponse(type="event", data=EventResponseData(event_type="final"))


def _error(message: str) -> ErrorResponse:
    return ErrorResponse(type="error", data=ErrorResponseData(message=message))


class FakeSocket:
    def __init__(self, messages):
        self.messages = list(messages)
        self.sent: list[tuple] = []

    async def convert(self, text):
        self.sent.append(("convert", text))

    async def flush(self):
        self.sent.append(("flush",))

    async def recv(self):
        if not self.messages:
            await asyncio.Event().wait()             # nothing more will ever arrive
        item = self.messages.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class NullCM:
    async def __aexit__(self, *a):
        return None


async def test_ws_synth_assembles_chunks_in_arrival_order_until_final():
    sock = FakeSocket([_audio(b"\x01\x00\x02\x00"), _audio(b"\x03\x00"), _final()])
    pcm, chunks, ttfb = await _WSConn(NullCM(), sock).synth("hello there")
    assert pcm == b"\x01\x00\x02\x00\x03\x00" and chunks == 2 and ttfb >= 0
    assert sock.sent == [("convert", "hello there"), ("flush",)]


async def test_ws_synth_drops_a_dangling_odd_byte_instead_of_emitting_a_click():
    sock = FakeSocket([_audio(b"\x01\x00\x02"), _final()])
    pcm, _, _ = await _WSConn(NullCM(), sock).synth("x")
    assert pcm == b"\x01\x00"


@pytest.mark.parametrize("message,retryable", [("400: Speaker 'zzz' is not recognized", False),
                                               ("500: internal error", True)])
async def test_ws_error_responses_are_classified(message, retryable):
    with pytest.raises(SarvamTransportError) as info:
        await _WSConn(NullCM(), FakeSocket([_error(message)])).synth("hello")
    assert info.value.retryable is retryable


async def test_ws_synth_times_out_when_no_audio_ever_arrives():
    with pytest.raises(SarvamTransportError, match="timed out"):
        await _WSConn(NullCM(), FakeSocket([])).synth("hello", first_chunk_timeout=0.05, total_timeout=0.2)


async def test_ws_synth_times_out_when_the_final_event_never_arrives():
    sock = FakeSocket([_audio(b"\x01\x00")])          # audio, then silence: 'final' is missing
    with pytest.raises(SarvamTransportError):
        await _WSConn(NullCM(), sock).synth("hello", first_chunk_timeout=0.05, total_timeout=0.2)


async def test_ws_synth_turns_a_dropped_connection_into_a_retryable_error():
    sock = FakeSocket([_audio(b"\x01\x00"), ConnectionResetError("reset")])
    with pytest.raises(SarvamTransportError) as info:
        await _WSConn(NullCM(), sock).synth("hello")
    assert info.value.retryable is True


def test_stats_snapshot_reports_latency_percentiles():
    from sarvam_tts_plugin import TTSStats
    stats = TTSStats(first_audio_ms=[300, 500, 400])
    snap = stats.snapshot()
    assert snap["first_audio_ms_p50"] == 400 and snap["first_audio_ms_max"] == 500


# ------------------------------------------------------------------ concurrent synthesis
class _SlowFirstEngine:
    """Sentence 0 starts audio quickly; the later ones each take 0.5 s - sequential synthesis would need >1.1 s."""

    def __init__(self):
        self.voice = VOICE
        self.stats = type("S", (), {"discarded_in_flight": 0})()
        self.started: list[str] = []

    async def synthesize_stream(self, text, *, request_id, idx):
        self.started.append(text)
        await asyncio.sleep((0.1, 0.5, 0.5, 0.5)[idx])
        yield pcm_for(text)

    async def aclose(self):
        pass


async def test_sentences_are_synthesised_concurrently_but_delivered_in_order():
    engine = _SlowFirstEngine()
    tts_obj = SarvamTTS("key", VOICE, engine=engine)
    text = "First we design the schema carefully. Then we add the indexes we need. Finally we measure the queries."
    started = asyncio.get_event_loop().time()
    audio = await speak(tts_obj, text)
    elapsed = asyncio.get_event_loop().time() - started
    from speech_prep import split_for_tts
    units = split_for_tts(text)
    assert len(units) >= 3
    assert audio == b"".join(pcm_for(u) for u in units)          # strict speaking order, exactly once
    assert elapsed < 0.95, f"sentences were synthesised one after another ({elapsed:.2f}s)"


# ------------------------------------------------------------------ chunk-level streaming
class StreamConn:
    """A connection whose synth_stream() yields scripted chunks with delays.
    script entries: ("chunk", bytes, delay_s) | ("fail", Exception) | ("block",)"""

    def __init__(self, registry, script):
        self.registry, self.script, self.closed = registry, list(script), False
        self.created = __import__("time").monotonic()
        registry.opened.append(self)

    async def synth_stream(self, text, *, timing=None, **_):
        self.registry.all_calls.append(text)
        first = True
        for step in self.script:
            if step[0] == "block":
                await asyncio.Event().wait()
            if step[0] == "fail":
                raise step[1]
            await asyncio.sleep(step[2])
            if first and timing is not None:
                timing["first"] = step[2]
            first = False
            yield step[1]

    async def close(self):
        self.closed = True


class StreamRegistry(Registry):
    async def open(self):
        return StreamConn(self, self.scripts.pop(0))


def stream_engine(reg):
    return _SentenceEngine("key", VOICE, pool=_ConnectionPool(reg.open), rest=None)


async def timed_speak(tts_obj, text):
    """(first_audio_seconds, total_seconds, audio_bytes) for one utterance."""
    loop = asyncio.get_event_loop()
    started = loop.time()
    stream = tts_obj.stream()
    stream.push_text(text)
    stream.end_input()
    first, out = None, bytearray()
    async for ev in stream:
        if first is None:
            first = loop.time() - started
        out += ev.frame.data.tobytes()
    await stream.aclose()
    return first, loop.time() - started, bytes(out)


def half_second(marker):
    return struct.pack("<h", marker) * (SR // 2)          # 0.5 s of audio


async def test_first_audio_is_played_when_the_first_chunk_arrives_not_when_the_sentence_is_done():
    reg = StreamRegistry([("chunk", half_second(7), 0.05), ("chunk", half_second(8), 0.6), ("chunk", half_second(9), 0.6)])
    tts_obj = SarvamTTS("key", VOICE, engine=stream_engine(reg))
    first, total, audio = await timed_speak(tts_obj, "One single sentence that takes a while to generate fully.")
    assert audio == half_second(7) + half_second(8) + half_second(9)      # everything delivered, in order
    assert first < 0.45, f"first audio waited for the whole sentence ({first:.2f}s)"
    assert total > 1.1                                                     # ...which really did take >1s to finish


async def test_a_failure_after_audio_started_is_never_replayed():
    boom = SarvamTransportError("connection dropped")
    reg = StreamRegistry([("chunk", half_second(7), 0.0), ("fail", boom)], [("chunk", half_second(5), 0.0)])
    engine = stream_engine(reg)
    tts_obj = SarvamTTS("key", VOICE, engine=engine)
    text = "First sentence that will be cut short. Second sentence carries on normally."
    audio = await speak(tts_obj, text)
    # sentence 1 stops where it broke (its first chunk was already played once, never again);
    # sentence 2 is unaffected. No duplicate, no retry of sentence 1.
    assert audio == half_second(7) + half_second(5)
    assert engine.stats.mid_sentence_failures == 1
    assert reg.all_calls.count("First sentence that will be cut short.") == 1
    assert reg.opened[0].closed, "the broken connection must be discarded"


async def test_a_failure_before_any_audio_is_retried_cleanly():
    reg = StreamRegistry([("fail", SarvamTransportError("boom"))], [("chunk", half_second(3), 0.0)])
    engine = stream_engine(reg)
    audio = await speak(SarvamTTS("key", VOICE, engine=engine), "A sentence that fails once and then works.")
    assert audio == half_second(3)                         # exactly one copy of the audio
    assert engine.stats.ws_failures == 1 and engine.stats.ws_reconnects == 1


async def test_later_sentences_generated_early_still_play_after_earlier_ones():
    # sentence 0 is slow to finish, sentence 1 finishes long before it - playback order must not change
    reg = StreamRegistry([("chunk", half_second(1), 0.0), ("chunk", half_second(2), 0.5)], [("chunk", half_second(3), 0.0)])
    audio = await speak(SarvamTTS("key", VOICE, engine=stream_engine(reg)),
                        "This is the first sentence of two. And here is the second sentence.")
    assert audio == half_second(1) + half_second(2) + half_second(3)


async def test_cancelling_mid_stream_closes_every_in_flight_connection():
    reg = StreamRegistry([("chunk", half_second(1), 0.0), ("block",)], [("block",)])
    engine = stream_engine(reg)
    tts_obj = SarvamTTS("key", VOICE, engine=engine)
    stream = tts_obj.stream()
    stream.push_text("First sentence goes here for a while. Second sentence is already being prepared.")
    stream.end_input()
    reader = asyncio.create_task(_drain(stream))
    for _ in range(200):
        if len(reg.opened) >= 2:
            break
        await asyncio.sleep(0.01)
    await stream.aclose()                                  # the candidate started talking
    await asyncio.gather(reader, return_exceptions=True)
    assert len(reg.opened) >= 2 and all(c.closed for c in reg.opened), "a connection leaked after barge-in"


# ------------------------------------------------------------------ fixed-phrase cache
from sarvam_tts_plugin import _PhraseCache, speech_units

FIXED = "I can't provide hints during the interview. Please continue with your answer."


async def test_a_fixed_line_is_synthesised_once_then_replayed_from_memory():
    reg = StreamRegistry([("chunk", half_second(4), 0.0), ("chunk", half_second(5), 0.0)])
    engine = stream_engine(reg)
    engine.set_cacheable(speech_units(FIXED))
    tts_obj = SarvamTTS("key", VOICE, engine=engine)
    first = await speak(tts_obj, FIXED)
    calls_after_first = len(reg.all_calls)
    second = await speak(tts_obj, FIXED)
    assert second == first                                   # identical audio
    assert len(reg.all_calls) == calls_after_first           # ...with no further request to Sarvam
    assert engine.stats.cache_hits >= 1


async def test_lines_that_are_not_registered_are_never_cached():
    reg = StreamRegistry([("chunk", half_second(1), 0.0)], [("chunk", half_second(1), 0.0)])
    engine = stream_engine(reg)
    engine.set_cacheable(speech_units(FIXED))
    tts_obj = SarvamTTS("key", VOICE, engine=engine)
    await speak(tts_obj, "Tell me about a project you are proud of.")
    await speak(tts_obj, "Tell me about a project you are proud of.")
    assert len(reg.all_calls) == 2 and engine.stats.cache_hits == 0     # personalised / LLM text is always live


async def test_a_line_cut_short_by_a_failure_is_not_cached():
    reg = StreamRegistry([("chunk", half_second(7), 0.0), ("fail", SarvamTransportError("dropped"))],
                         [("chunk", half_second(7), 0.0), ("chunk", half_second(8), 0.0)])
    engine = stream_engine(reg)
    unit = "Please continue with your answer."
    engine.set_cacheable([unit])
    tts_obj = SarvamTTS("key", VOICE, engine=engine)
    await speak(tts_obj, unit)                                # cut mid-sentence
    assert engine._cache.get(unit) is None                    # a truncated line must never be replayed
    await speak(tts_obj, unit)                                # next time it plays in full...
    assert engine._cache.get(unit) is not None                # ...and only now is it stored


async def test_prewarm_fills_the_cache_on_its_own_connection_without_touching_the_live_pool():
    live = StreamRegistry()
    warm = Registry([pcm_for("Please continue with your answer.")])      # separate registry = separate connections
    engine = _SentenceEngine("key", VOICE, pool=_ConnectionPool(live.open), rest=None, open_ws=warm.open)
    unit = "Please continue with your answer."
    engine.set_cacheable([unit])
    added = await engine.prewarm_phrases([unit, "Not a registered line."])
    assert added == 1 and engine._cache.get(unit) is not None
    assert live.opened == [] and warm.opened[0].closed          # the live pool was never used; warm-up cleaned up


def test_the_cache_is_bounded_by_entries_and_bytes():
    cache = _PhraseCache(max_entries=3, max_bytes=1000)
    for i in range(10):
        cache.put(f"line {i}", b"x" * 100)
    assert len(cache) == 3 and cache.get("line 9") is not None and cache.get("line 0") is None
    cache.put("huge", b"x" * 900)                                 # bigger than a quarter of the budget: refused
    assert cache.get("huge") is None


def test_fixed_phrases_cover_every_refusal_and_stay_free_of_personal_text():
    import refusals
    from interaction_guard import RESTRICTED_INTENTS
    for lang in ("en", "hi", "hinglish"):
        everything = refusals.fixed_phrases(lang)
        essential = refusals.fixed_phrases(lang, essential_only=True)
        assert set(essential) <= set(everything) and 5 <= len(essential) < len(everything)
        for intent in RESTRICTED_INTENTS:
            assert all(v in everything for v in refusals.REFUSALS[lang][intent])
        assert not any("{" in line for line in everything)        # no template placeholders, no candidate name
