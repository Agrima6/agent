"""Shared test doubles (imported by the test modules; not collected by pytest itself)."""
from llm_provider import LLMError


class FakeProvider:
    """Scripted LLM provider: pops one response (dict or Exception) per call and records the prompts."""
    name = "fake"
    model = "fake-model"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def complete_json(self, system, user, *, temperature=0.2, max_tokens=None, timeout=None, model=None,
                      reasoning_effort=None, retries=1):
        self.calls.append({"system": system, "user": user, "max_tokens": max_tokens, "reasoning_effort": reasoning_effort,
                           "retries": retries})
        if not self.responses:
            raise LLMError("no scripted response left")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def all_prompt_text(self) -> str:
        return "\n".join(c["system"] + "\n" + c["user"] for c in self.calls)


# ---------------------------------------------------------------------------- LiveKit audio sink
from livekit.agents.voice import io as _lk_io  # noqa: E402


class CapturingAudioSink(_lk_io.AudioOutput):
    """Stands in for LiveKit's room audio output: records every PCM byte the agent 'plays'.

    Playback completes when `finish_playout()` is called (default: on flush, i.e. instantly). Set
    `hold_playout=True` to simulate real-time playback that a test can interrupt (barge-in).
    """

    def __init__(self, sample_rate: int = 24000, hold_playout: bool = False):
        super().__init__(label="capturing-sink", capabilities=_lk_io.AudioOutputCapabilities(pause=False),
                         next_in_chain=None, sample_rate=sample_rate)
        self.pcm = bytearray()
        self.cleared = 0
        self.hold_playout = hold_playout
        self._flushed = False

    async def capture_frame(self, frame) -> None:
        await super().capture_frame(frame)
        self.pcm += frame.data.tobytes()

    def flush(self) -> None:
        super().flush()
        self._flushed = True
        if not self.hold_playout:
            self.finish_playout()

    def finish_playout(self) -> None:
        if self._flushed:
            self._flushed = False
            self.on_playback_finished(playback_position=len(self.pcm) / 2 / (self.sample_rate or 24000), interrupted=False)

    def clear_buffer(self) -> None:
        self.cleared += 1
        self._flushed = False
        self.on_playback_finished(playback_position=len(self.pcm) / 2 / (self.sample_rate or 24000), interrupted=True)
