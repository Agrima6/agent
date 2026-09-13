import base64
import logging
import httpx
from livekit.agents import tts, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

logger = logging.getLogger("sarvam-tts")

# Warm, professional Indian male voice on bulbul:v3.
DEFAULT_PRESET_SPEAKER = "aditya"

# Sarvam's API rejects any other value outright (400) — a bad SARVAM_SAMPLE_RATE in .env would
# otherwise silently break 100% of speech output with no audio at all, which is a much harder
# failure to diagnose than a slightly-off sample rate.
_ALLOWED_SAMPLE_RATES = (8000, 16000, 22050, 24000, 32000, 44100, 48000)
_DEFAULT_SAMPLE_RATE = 22050


def _sanitize_sample_rate(sample_rate: int) -> int:
    if sample_rate in _ALLOWED_SAMPLE_RATES:
        return sample_rate
    fallback = min(_ALLOWED_SAMPLE_RATES, key=lambda r: abs(r - sample_rate))
    logger.warning(
        f"SARVAM_SAMPLE_RATE={sample_rate} is not one of Sarvam's allowed rates "
        f"{_ALLOWED_SAMPLE_RATES} — every TTS call would fail with a 400. Using {fallback} instead."
    )
    return fallback


class SarvamTTS(tts.TTS):
    """Sarvam AI Bulbul TTS supporting presets and custom cloned voices (svc-...)."""

    def __init__(
        self,
        api_key: str,
        model: str = "bulbul:v3",
        speaker: str = DEFAULT_PRESET_SPEAKER,
        language_code: str = "en-IN",
        pace: float = 1.0,
        sample_rate: int = _DEFAULT_SAMPLE_RATE,
    ):
        sample_rate = _sanitize_sample_rate(sample_rate)
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

        # Sarvam's public /text-to-speech API only accepts the documented `speaker` enum
        # (preset names) — it has no field for a Content Studio cloned voice ID (svc-...).
        # Sending an svc- id silently falls back to the API default ("shubh"), so guard
        # against that here rather than making every caller remember it.
        speaker = self._speaker
        if speaker.startswith("svc-"):
            logger.warning(
                f"Speaker '{speaker}' looks like a Content Studio cloned voice ID, which "
                f"the public TTS API does not support (falls back to default 'shubh'). "
                f"Using '{DEFAULT_PRESET_SPEAKER}' instead."
            )
            speaker = DEFAULT_PRESET_SPEAKER
        payload["speaker"] = speaker

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
