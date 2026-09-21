"""Centralised voice configuration: calibrated defaults, overrides, and safe degradation."""
import pytest

import voice_config
from voice_config import (
    CALIBRATED_PACE, MAX_PACE, MIN_PACE, VALID_V3_SPEAKERS, resolve_voice,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("TTS_SPEAKER_OVERRIDE", "TTS_PACE_OVERRIDE", "TTS_GENDER", "TTS_SAMPLE_RATE", "TTS_PROVIDER",
                 "SARVAM_PACE", "SARVAM_SPEAKER", "SARVAM_LANGUAGE_CODE", "SARVAM_SAMPLE_RATE", "TTS_VOICE"):
        monkeypatch.delenv(name, raising=False)
    voice_config._warned_legacy = False


@pytest.mark.parametrize("language,gender,speaker,code", [
    ("en", "male", "ratan", "en-IN"),
    ("en", "female", "ishita", "en-IN"),
    ("hi", "male", "shubh", "hi-IN"),
    ("hi", "female", "priya", "hi-IN"),
    ("hinglish", "male", "shubh", "hi-IN"),
    ("hinglish", "female", "priya", "hi-IN"),
])
def test_default_speaker_per_language_and_gender(language, gender, speaker, code):
    v = resolve_voice(language, gender)
    assert (v.speaker, v.language_code, v.gender) == (speaker, code, gender)
    assert v.model == "bulbul:v3"


def test_pace_is_calibrated_per_speaker_not_global():
    paces = {s: resolve_voice("en", speaker=s).pace for s in ("ratan", "ishita", "neha")}
    assert paces["ratan"] == CALIBRATED_PACE["ratan"] and paces["ishita"] == CALIBRATED_PACE["ishita"]
    assert paces["neha"] == voice_config.UNCALIBRATED_PACE
    assert len(set(paces.values())) == 3  # speakers have different base speeds


def test_calibrated_paces_hit_the_natural_speaking_rate():
    # The regression: SARVAM_PACE=1.3 made the interviewer race (~215 wpm). Every calibrated default
    # must have been MEASURED inside the calm-interviewer range, and pace must stay in Sarvam's range.
    low, high = voice_config.TARGET_WPM
    for speaker, (pace, wpm) in voice_config.CALIBRATION.items():
        assert low <= wpm <= high, f"{speaker} measured {wpm} wpm"
        assert MIN_PACE <= pace <= MAX_PACE
    # Speakers differ in base speed: ratan (fastest voice) needs a much lower pace than priya.
    assert CALIBRATED_PACE["ratan"] < 0.75 < CALIBRATED_PACE["priya"]
    assert max(CALIBRATED_PACE.values()) < 1.3


def test_legacy_env_vars_are_ignored(monkeypatch):
    monkeypatch.setenv("SARVAM_PACE", "1.3")
    monkeypatch.setenv("SARVAM_SPEAKER", "aditya")
    v = resolve_voice("en", "female")
    assert v.speaker == "ishita" and v.pace == CALIBRATED_PACE["ishita"]


def test_explicit_override_env_vars_win_over_defaults(monkeypatch):
    monkeypatch.setenv("TTS_SPEAKER_OVERRIDE", "priya")
    monkeypatch.setenv("TTS_PACE_OVERRIDE", "0.9")
    v = resolve_voice("en")
    assert v.speaker == "priya" and v.pace == 0.9


def test_per_interview_values_win_over_env(monkeypatch):
    monkeypatch.setenv("TTS_SPEAKER_OVERRIDE", "priya")
    v = resolve_voice("en", speaker="ratan", pace=0.8)
    assert v.speaker == "ratan" and v.pace == 0.8


def test_out_of_range_pace_is_clamped_into_sarvams_accepted_range():
    assert resolve_voice("en", pace=0.1).pace == MIN_PACE
    assert resolve_voice("en", pace=9).pace == MAX_PACE


def test_garbage_input_degrades_instead_of_raising():
    v = resolve_voice("klingon", gender="robot", speaker="not-a-speaker", pace="fast")
    assert v.speaker in VALID_V3_SPEAKERS and MIN_PACE <= v.pace <= MAX_PACE
    assert v.language == "en"


def test_uncalibrated_speaker_gets_conservative_pace():
    v = resolve_voice("en", speaker="neha")
    assert v.pace == voice_config.UNCALIBRATED_PACE < 1.0


def test_sample_rate_is_sanitised(monkeypatch):
    monkeypatch.setenv("TTS_SAMPLE_RATE", "12345")
    assert resolve_voice("en").sample_rate == voice_config.DEFAULT_SAMPLE_RATE
    monkeypatch.setenv("TTS_SAMPLE_RATE", "22050")
    assert resolve_voice("en").sample_rate == 22050


def test_edge_fallback_voice_matches_language_and_gender():
    assert resolve_voice("en", "male").edge_voice.startswith("en-IN")
    assert resolve_voice("hi", "female").edge_voice.startswith("hi-IN")


def test_legacy_env_warning_is_emitted_once(monkeypatch, caplog):
    monkeypatch.setenv("SARVAM_PACE", "1.3")
    with caplog.at_level("WARNING", logger="voice-config"):
        resolve_voice("en")
        resolve_voice("en")
    assert sum("legacy voice env vars" in r.message for r in caplog.records) == 1
