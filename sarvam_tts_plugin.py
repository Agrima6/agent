import base64
import logging
import httpx
from livekit.agents import tts, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

logger = logging.getLogger("sarvam-tts")


class SarvamTTS(tts.TTS):
    """Sarvam AI Bulbul TTS supporting presets and custom cloned voices (svc-...)."""

    def __init__(
        self,
        api_key: str,
        model: str = "bulbul:v3",
        speaker: str = "svc-e6c0f0a8-9386-4eb2-8558-2fc0036f53a4",
        language_code: str = "en-IN",
        pace: float = 1.0,
        sample_rate: int = 22050,
    ):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=1,
        )
        self._api_key = api_key
        self._model = model
        self._speaker = speaker
        self._language_code = language_code
        self._pace = pace
        self._sample_rate = sample_rate

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        return SarvamChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
            api_key=self._api_key,
            model=self._model,
            speaker=self._speaker,
            language_code=self._language_code,
            pace=self._pace,
            sample_rate=self._sample_rate,
        )


class SarvamChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: SarvamTTS,
        input_text: str,
        conn_options: APIConnectOptions,
        api_key: str,
        model: str,
        speaker: str,
        language_code: str,
        pace: float,
        sample_rate: int,
    ):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._api_key = api_key
        self._model = model
        self._speaker = speaker
        self._language_code = language_code
        self._pace = pace
        self._sample_rate = sample_rate

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        clean_text = self.input_text.strip()
        if not clean_text:
            return

        lang = self._language_code
        if lang == "en":
            lang = "en-IN"
        elif lang == "hi":
            lang = "hi-IN"

        url = "https://api.sarvam.ai/text-to-speech"
        headers = {
            "api-subscription-key": self._api_key,
            "Content-Type": "application/json",
        }

        payload = {
            "text": clean_text,
            "language_code": lang,
            "model": self._model,
            "pace": self._pace,
            "speech_sample_rate": self._sample_rate,
            "output_audio_codec": "mp3",
        }

        # Check if the speaker is a cloned voice ID (starts with svc-) or preset name
        if self._speaker.startswith("svc-"):
            payload["voice_id"] = self._speaker
        else:
            payload["speaker"] = self._speaker

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
                if resp.status_code != 200:
                    logger.error(
                        f"Sarvam TTS Error ({resp.status_code}): {resp.text} | Payload: {payload}"
                    )
                    resp.raise_for_status()

                data = resp.json()
                audios = data.get("audios", [])
                if not audios:
                    raise ValueError(f"No audio returned from Sarvam AI TTS: {data}")

                audio_bytes = base64.b64decode(audios[0])

            output_emitter.initialize(
                request_id=utils.shortuuid(),
                sample_rate=self._sample_rate,
                num_channels=1,
                mime_type="audio/mp3",
            )
            output_emitter.push(audio_bytes)
            output_emitter.flush()
        except Exception as e:
            logger.error(f"Sarvam TTS error: {e}", exc_info=True)
            raise
