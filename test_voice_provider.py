"""Voice provider selection: validation, persistence, audit, per-interview snapshot, no leaking of secrets."""
import json

import pytest

import api
import config
import voice_settings
from db import AppSetting, Interview, SettingAudit
# Reuse the scratch-database fixtures (an isolated SQLite file; the real database is never touched).
from test_api_interview_config import (  # noqa: F401
    _Session, client, create, isolated_db, no_external_calls, role_and_candidate,
)

URL = "/v1/settings/voice-provider"


@pytest.fixture(autouse=True)
def keys_and_clean_state(monkeypatch):
    monkeypatch.setattr(config, "SARVAM_API_KEY", "sarvam-secret-VALUE-123")
    monkeypatch.setattr(config, "ELEVENLABS_API_KEY", "eleven-secret-VALUE-456")
    monkeypatch.setattr(config, "DEFAULT_VOICE_PROVIDER", "")
    monkeypatch.setattr(config, "TTS_PROVIDER", "sarvam")
    db = _Session()
    db.query(SettingAudit).delete(); db.query(AppSetting).delete(); db.commit(); db.close()


def put(client, provider, actor="admin@example.com", **extra):
    return client.put(URL, json={"provider": provider, **extra}, headers={"X-Actor": actor})


def test_reads_the_default_without_changing_production_and_reports_configuration(client):
    body = client.get(URL).json()
    assert body["provider"] == "sarvam" and body["isDefault"] is True and body["version"] == 0
    assert {p["id"]: p["configured"] for p in body["availableProviders"]} == {"sarvam": True, "elevenlabs": True}


def test_default_follows_the_current_production_provider(client, monkeypatch):
    monkeypatch.setattr(config, "TTS_PROVIDER", "elevenlabs")
    assert client.get(URL).json()["provider"] == "elevenlabs"
    monkeypatch.setattr(config, "TTS_PROVIDER", "edge")                 # not one of the two: safe default
    assert client.get(URL).json()["provider"] == "sarvam"


def test_saving_persists_audits_and_versions(client):
    r = put(client, "elevenlabs")
    assert r.status_code == 200 and r.json()["provider"] == "elevenlabs" and r.json()["message"]
    again = client.get(URL).json()
    assert again["provider"] == "elevenlabs" and again["isDefault"] is False and again["updatedBy"] == "admin@example.com"
    put(client, "sarvam", actor="second@example.com")
    db = _Session()
    audit = db.query(SettingAudit).order_by(SettingAudit.at).all()
    assert [(a.old_value, a.new_value, a.actor) for a in audit] == [("sarvam", "elevenlabs", "admin@example.com"), ("elevenlabs", "sarvam", "second@example.com")]
    assert db.get(AppSetting, voice_settings.SETTING_KEY).version == 2
    db.close()


@pytest.mark.parametrize("bad", ["", "edge", "openai", "SARVAM; DROP TABLE", "  "])
def test_only_the_two_providers_are_accepted(client, bad):
    assert put(client, bad).status_code == 400


def test_a_change_must_name_the_admin(client):
    assert client.put(URL, json={"provider": "sarvam"}).status_code == 400


def test_missing_credentials_block_activation_and_keep_the_last_saved_value(client, monkeypatch):
    put(client, "sarvam")
    monkeypatch.setattr(config, "ELEVENLABS_API_KEY", "")
    r = put(client, "elevenlabs")
    assert r.status_code == 422 and "credentials" in r.json()["detail"]
    assert client.get(URL).json()["provider"] == "sarvam"                # the failed save changed nothing
    assert {p["id"]: p["configured"] for p in client.get(URL).json()["availableProviders"]}["elevenlabs"] is False


def test_concurrent_edits_are_detected(client):
    put(client, "sarvam")
    assert put(client, "elevenlabs", expectedVersion=0).status_code == 409   # someone saved since version 0
    assert put(client, "elevenlabs", expectedVersion=1).status_code == 200


def test_secrets_never_appear_in_any_response(client):
    texts = [client.get(URL).text, put(client, "elevenlabs").text, put(client, "bogus").text]
    for text in texts:
        assert "VALUE-123" not in text and "VALUE-456" not in text


def test_the_service_key_protects_the_endpoints(client, monkeypatch):
    monkeypatch.setattr(api, "AGENT_SERVICE_KEY", "the-service-key")
    assert client.get(URL).status_code in (401, 403)
    assert client.put(URL, json={"provider": "sarvam"}, headers={"X-Actor": "a"}).status_code in (401, 403)
    assert client.get(URL, headers={"X-Agent-Key": "the-service-key"}).status_code == 200


def test_each_interview_keeps_the_provider_it_started_with(client, role_and_candidate):
    role_id, cand_id = role_and_candidate
    put(client, "sarvam")
    first = create(client, role_id, cand_id).json()["id"]
    put(client, "elevenlabs")                                            # admin changes it mid-interview
    second = create(client, role_id, cand_id).json()["id"]
    assert client.get(f"/v1/interviews/{first}").json()["voice_provider"] == "sarvam"      # unchanged
    assert client.get(f"/v1/interviews/{second}").json()["voice_provider"] == "elevenlabs"  # next one uses the new value


def test_an_explicit_choice_whose_credentials_vanished_fails_clearly_not_silently(client, role_and_candidate, monkeypatch):
    role_id, cand_id = role_and_candidate
    put(client, "elevenlabs")
    monkeypatch.setattr(config, "ELEVENLABS_API_KEY", "")
    r = create(client, role_id, cand_id)
    assert r.status_code == 503 and "elevenlabs" in r.json()["detail"]


def test_without_any_saved_choice_a_missing_key_does_not_block_local_development(client, role_and_candidate, monkeypatch):
    role_id, cand_id = role_and_candidate
    monkeypatch.setattr(config, "SARVAM_API_KEY", "")
    assert create(client, role_id, cand_id).status_code == 200
