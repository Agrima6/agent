"""Latency tracing: spans are computed from real marks, missing marks are null (never guessed), and the
LiveKit event wiring opens a turn when the candidate stops and closes it when the interviewer starts."""
import os

import pytest

import turn_trace
from turn_trace import TurnTrace, TurnTracer


def test_spans_come_from_marks_and_missing_marks_are_null():
    t = TurnTrace("x-1")
    for name, at in [("user_speech_end", 10.0), ("stt_final", 10.3), ("turn_committed", 10.5), ("judge_start", 10.5),
                     ("judge_end", 11.5), ("compose_start", 11.5), ("compose_end", 12.1), ("say_called", 12.2),
                     ("tts_request", 12.2), ("tts_first_audio", 12.5), ("tts_first_push", 12.5), ("agent_speaking", 12.6)]:
        t.mark(name, at)
    r = t.report()
    assert (r["stt_final_ms"], r["judge_ms"], r["compose_ms"], r["tts_ttfa_ms"], r["ttfa_ms"]) == (300, 1000, 600, 300, 2600)
    assert r["tts_buffer_ms"] == 0 and r["publish_ms"] == 100
    partial = TurnTrace("x-2")
    partial.mark("user_speech_end", 1.0)
    assert partial.report()["ttfa_ms"] is None and partial.report()["judge_ms"] is None    # NOT MEASURED, not zero


def test_first_mark_of_a_name_wins():
    t = TurnTrace("x-3")
    t.mark("tts_request", 5.0)
    t.mark("tts_request", 9.0)      # a retry must not move the start of the measurement
    assert t.marks["tts_request"] == 5.0


class FakeSession:
    def __init__(self):
        self.handlers = {}

    def on(self, name, fn):
        self.handlers[name] = fn


class Ev:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_tracer_opens_on_speech_end_and_reports_when_the_interviewer_starts_speaking(caplog):
    tracer = TurnTracer("intv_1", vad_min_silence_s=0.4)
    session = FakeSession()
    tracer.attach(session)
    turn_trace.install(tracer)
    with caplog.at_level("INFO", logger="voice-metrics"):
        session.handlers["user_state_changed"](Ev(old_state="speaking", new_state="listening"))
        session.handlers["user_input_transcribed"](Ev(is_final=True))
        turn_trace.mark("judge_start"); turn_trace.mark("judge_end"); turn_trace.mark("say_called")
        session.handlers["agent_state_changed"](Ev(old_state="thinking", new_state="speaking"))
    line = [r.message for r in caplog.records if "turn_trace" in r.message]
    assert len(line) == 1 and '"turn_id": "intv_1-1"' in line[0] and '"vad_min_silence_ms": 400' in line[0]
    assert tracer.current is None                            # closed: the next turn starts clean
    turn_trace.install(None)


def test_a_final_transcript_that_beats_the_vad_event_is_not_lost():
    tracer = TurnTracer("intv_2")
    tracer.mark("stt_final")                                  # no turn open yet
    trace = tracer.begin()
    assert "stt_final" in trace.marks
    tracer.finish()


def test_agent_initiated_speech_like_the_greeting_gets_its_own_trace():
    tracer = TurnTracer("intv_3")
    turn_trace.install(tracer)
    turn_trace.mark("say_called")
    assert tracer.current is not None and tracer.current.kind == "agent_initiated"
    turn_trace.install(None)


def test_marking_without_a_tracer_is_a_no_op():
    turn_trace.install(None)
    turn_trace.mark("judge_start")       # must not raise


@pytest.mark.parametrize("mode,expected", [("local", "turn-detector-v1-mini"), ("cloud", "turn-detector-v1"), ("vad", "vad")])
def test_the_turn_detector_is_chosen_explicitly(monkeypatch, mode, expected):
    from agent import make_turn_handling
    monkeypatch.setenv("TURN_DETECTOR", mode)
    detector = make_turn_handling()["turn_detection"]
    assert (detector if isinstance(detector, str) else detector.model) == expected


def test_default_turn_detector_is_local_never_the_cloud_one(monkeypatch):
    from agent import make_turn_handling
    monkeypatch.delenv("TURN_DETECTOR", raising=False)
    assert make_turn_handling()["turn_detection"].model == "turn-detector-v1-mini"
