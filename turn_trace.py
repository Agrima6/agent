"""Per-turn latency trace: where does the time between "candidate stopped talking" and "candidate hears
the interviewer" actually go?

One LiveKit job = one process = one interview, so a module-level tracer is safe (each interview has its
own). Stages mark themselves with `mark("name")`; the first mark of each name in a turn wins. A turn
starts when the candidate's speech ends and finishes when the interviewer's first audio starts playing,
at which point ONE structured line is logged:

    turn_trace {"turn_id": "...", "stt_final_ms": ..., "judge_ms": ..., "tts_ttfa_ms": ..., "ttfa_ms": ...}

Marks (all `time.monotonic()`):
    user_speech_end   VAD decided the candidate stopped (LiveKit user_state speaking -> listening)
    stt_final         final transcript arrived
    turn_committed    end-of-turn decided, our turn callback invoked
    judge_start/_end, compose_start/_end        the two LLM calls
    say_called        text handed to the TTS pipeline
    tts_request       first sentence sent to Sarvam
    tts_first_audio   first audio chunk back from Sarvam
    tts_first_push    first audio handed to LiveKit
    agent_speaking    LiveKit reports the interviewer speaking (first audio being played out)

Any span whose marks are missing is reported as null ("not measured") rather than guessed. No transcript
text or personal data is recorded.
"""
import json
import logging
import time

logger = logging.getLogger("voice-metrics")

_SPANS = {
    # name: (from_mark, to_mark)
    "stt_final_ms": ("user_speech_end", "stt_final"),
    "turn_commit_ms": ("user_speech_end", "turn_committed"),
    "judge_ms": ("judge_start", "judge_end"),
    "compose_ms": ("compose_start", "compose_end"),
    "logic_ms": ("turn_committed", "say_called"),          # everything our code does before speaking
    "tts_ttfa_ms": ("tts_request", "tts_first_audio"),      # Sarvam: request -> first audio chunk
    "tts_buffer_ms": ("tts_first_audio", "tts_first_push"),  # our buffering before handing audio to LiveKit
    "publish_ms": ("tts_first_push", "agent_speaking"),     # LiveKit: handed audio -> speaking
    "pre_tts_ms": ("user_speech_end", "tts_request"),
    "ttfa_ms": ("user_speech_end", "agent_speaking"),       # the number that matters
}


class TurnTrace:
    __slots__ = ("turn_id", "marks", "kind")

    def __init__(self, turn_id: str, kind: str = "candidate_turn"):
        self.turn_id = turn_id
        self.kind = kind
        self.marks: dict[str, float] = {}

    def mark(self, name: str, at: float | None = None) -> None:
        self.marks.setdefault(name, time.monotonic() if at is None else at)

    def span_ms(self, start: str, end: str) -> int | None:
        a, b = self.marks.get(start), self.marks.get(end)
        return None if a is None or b is None else max(round((b - a) * 1000), 0)

    def report(self) -> dict:
        out = {"turn_id": self.turn_id, "kind": self.kind}
        out.update({name: self.span_ms(a, b) for name, (a, b) in _SPANS.items()})
        return out


class TurnTracer:
    def __init__(self, interview_id: str, vad_min_silence_s: float | None = None):
        self.interview_id = interview_id
        self.vad_min_silence_ms = None if vad_min_silence_s is None else int(vad_min_silence_s * 1000)
        self.current: TurnTrace | None = None
        self._seq = 0
        self._early: dict[str, float] = {}     # marks that can arrive just BEFORE the turn is opened

    def begin(self, kind: str = "candidate_turn") -> TurnTrace:
        self._seq += 1
        self.current = TurnTrace(f"{self.interview_id}-{self._seq}", kind)
        for name, at in self._early.items():       # e.g. a final transcript that beat the VAD event
            self.current.mark(name, at)
        self._early.clear()
        return self.current

    def mark(self, name: str) -> None:
        if self.current is None:
            if name == "stt_final":
                self._early.setdefault(name, time.monotonic())
                return
            if name != "say_called":
                return
            self.begin("agent_initiated")     # greeting / re-prompt with no candidate turn before it
        self.current.mark(name)

    def finish(self) -> dict | None:
        trace, self.current = self.current, None
        self._early.clear()
        if trace is None:
            return None
        trace.mark("agent_speaking")
        report = trace.report()
        # ttfa_ms is measured from when VAD CONFIRMED the end of speech; the candidate actually stopped
        # about `vad_min_silence_ms` earlier than that.
        report["vad_min_silence_ms"] = self.vad_min_silence_ms
        logger.info("turn_trace %s", json.dumps(report))
        return report

    # -- LiveKit session wiring ---------------------------------------------------------------------
    def attach(self, session) -> None:
        def on_user_state(event) -> None:
            if event.old_state == "speaking" and event.new_state == "listening":
                self.begin().mark("user_speech_end")

        def on_transcript(event) -> None:
            if getattr(event, "is_final", False):
                self.mark("stt_final")

        def on_agent_state(event) -> None:
            if event.new_state == "speaking" and self.current is not None:
                self.finish()

        session.on("user_state_changed", on_user_state)
        session.on("user_input_transcribed", on_transcript)
        session.on("agent_state_changed", on_agent_state)


_active: TurnTracer | None = None


def install(tracer: TurnTracer | None) -> None:
    """Make `tracer` the process-wide tracer (one interview per process). None disables tracing."""
    global _active
    _active = tracer


def mark(name: str) -> None:
    """Record a stage timestamp on the current turn. A no-op when tracing is not installed."""
    if _active is not None:
        _active.mark(name)
