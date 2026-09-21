"""Structured voice-pipeline metrics, so audio and latency problems are measurable instead of anecdotal.

Emits one JSON log line per LiveKit metric event with the interview/question context attached:

    STT       transcription duration vs audio duration
    EOU       end-of-utterance delay and how long the candidate-turn callback took
    TTS       time to first byte, synthesis duration, audio duration, characters, cancellations

plus TTS delivery counters (retries, reconnects, first-audio latency percentiles, discarded
in-flight sentences) from the Sarvam plugin. No transcript text or candidate personal data is logged.
"""
import json
import logging
from typing import Callable

logger = logging.getLogger("voice-metrics")


def _ms(seconds) -> int | None:
    return None if seconds is None else int(float(seconds) * 1000)


def describe_metric(metric) -> dict | None:
    """Reduce a LiveKit metrics object to the fields worth logging (None = not interesting)."""
    kind = getattr(metric, "type", "")
    if kind == "stt_metrics":
        return {"metric": "stt", "stt_ms": _ms(metric.duration), "audio_s": round(metric.audio_duration, 2),
                "streamed": metric.streamed}
    if kind == "eou_metrics":
        return {"metric": "eou", "end_of_utterance_ms": _ms(metric.end_of_utterance_delay),
                "transcription_ms": _ms(metric.transcription_delay),
                "turn_callback_ms": _ms(metric.on_user_turn_completed_delay)}
    if kind == "tts_metrics":
        return {"metric": "tts", "ttfb_ms": _ms(metric.ttfb), "tts_ms": _ms(metric.duration),
                "audio_s": round(metric.audio_duration, 2), "chars": metric.characters_count,
                "streamed": metric.streamed, "cancelled": metric.cancelled,
                "connection_reused": metric.connection_reused}
    return None


def attach_metrics_logging(session, interview_id: str, context: Callable[[], dict]) -> None:
    def on_metrics(event) -> None:
        fields = describe_metric(event.metrics)
        if fields:
            logger.info("voice_metric %s", json.dumps({"interview_id": interview_id, **context(), **fields}))

    def on_error(event) -> None:
        error = getattr(event, "error", None)
        # Only the error class and whether it was recoverable: never the message body, which can
        # contain request payloads.
        logger.warning("voice_error %s", json.dumps({
            "interview_id": interview_id, **context(), "source": type(getattr(event, "source", None)).__name__,
            "error": type(error).__name__, "recoverable": getattr(error, "recoverable", None)}))

    session.on("metrics_collected", on_metrics)
    session.on("error", on_error)
