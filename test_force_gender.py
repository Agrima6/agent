from voice_config import resolve_voice


def test_forced_gender_overrides_the_drive_and_candidate_choice(monkeypatch):
    monkeypatch.setenv("TTS_FORCE_GENDER", "male")
    for asked in ("female", "male", None):
        assert resolve_voice("hi", asked).gender == "male"
    assert resolve_voice("en", "female", speaker="ishita").gender == "male"     # a female speaker is dropped
    assert resolve_voice("en", "female", speaker="ishita").speaker == "ratan"


def test_without_the_setting_the_requested_gender_is_used(monkeypatch):
    monkeypatch.delenv("TTS_FORCE_GENDER", raising=False)
    monkeypatch.delenv("TTS_GENDER", raising=False)
    assert resolve_voice("en", "female").gender == "female"
    assert resolve_voice("en", "male").gender == "male"
