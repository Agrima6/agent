import edge_tts
from livekit.agents import tts, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions


class EdgeTTS(tts.TTS):
    """Free, high quality neural TTS using Edge TTS without requiring paid API keys."""

    def __init__(self, voice: str = "en-US-JennyNeural"):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=24000,
            num_channels=1,
        )
        self._voice = voice

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        return EdgeChunkedStream(
            tts=self, input_text=text, conn_options=conn_options, voice=self._voice
        )


class EdgeChunkedStream(tts.ChunkedStream):
    def __init__(
        self, *, tts: EdgeTTS, input_text: str, conn_options: APIConnectOptions, voice: str
    ):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._voice = voice

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        communicate = edge_tts.Communicate(self.input_text, self._voice)
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=24000,
            num_channels=1,
            mime_type="audio/mp3",
        )
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                output_emitter.push(chunk["data"])
        output_emitter.flush()
