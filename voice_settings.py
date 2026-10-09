"""Which voice provider powers NEW interviews: the admin-selected setting, validated and audited.

Rules (docs: WorkmateIQ voice provider selection):
  * Exactly two providers: sarvam and elevenlabs. Anything else is rejected.
  * The setting is stored by the backend (never trusted from a browser) and is GLOBAL for the deployment: voice
    usage is billed to the platform's own provider keys, so one platform admin decision applies to every
    organization. (Per-tenant choice would need per-tenant keys; it is deliberately not mixed in.)
  * Only the provider NAME is stored. API keys stay in environment variables and are never returned or logged.
  * A provider can only be activated when its credentials exist.
  * Every interview records the provider it started with (Interview.voice_provider); changing the setting never
    changes a session in progress.
  * No silent cross-provider fallback (see agent.build_tts).
"""
import datetime

from sqlalchemy.orm import Session

import config
from db import AppSetting, Interview, SettingAudit
import states

PROVIDERS = ("sarvam", "elevenlabs")
SETTING_KEY = "voice_ai_provider"


class VoiceSettingsError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def provider_configured(provider: str) -> bool:
    """Credentials present for this provider (checked at call time, so a changed environment is honoured)."""
    if provider == "sarvam":
        return bool(config.SARVAM_API_KEY)
    if provider == "elevenlabs":
        return bool(config.ELEVENLABS_API_KEY)
    return False


def default_provider() -> str:
    """Initial value when nothing has been saved: the provider production already uses, so deploying this feature
    never changes the voice by accident."""
    for candidate in (config.DEFAULT_VOICE_PROVIDER, config.TTS_PROVIDER):
        if candidate in PROVIDERS:
            return candidate
    return "sarvam"


def saved_setting(db: Session) -> AppSetting | None:
    row = db.get(AppSetting, SETTING_KEY)
    return row if row is not None and row.value in PROVIDERS else None      # a corrupt value is ignored, not trusted


def active_provider(db: Session) -> str:
    row = saved_setting(db)
    return row.value if row else default_provider()


def resolve_for_new_interview(db: Session) -> str:
    """The provider a brand-new interview will use. If an admin explicitly chose a provider and its credentials have
    since disappeared, creating the interview fails with a clear error instead of quietly using something else."""
    row = saved_setting(db)
    provider = row.value if row else default_provider()
    if row is not None and not provider_configured(provider):
        raise VoiceSettingsError(503, f"The selected voice provider '{provider}' is not configured on the server.")
    return provider


def describe(db: Session) -> dict:
    row = saved_setting(db)
    active_sessions = db.query(Interview).filter(Interview.status == states.IN_PROGRESS).count()
    return {
        "provider": row.value if row else default_provider(),
        "isDefault": row is None,
        "availableProviders": [{"id": p, "configured": provider_configured(p)} for p in PROVIDERS],
        "version": row.version if row else 0,
        "updatedAt": row.updated_at.isoformat() + "Z" if row and row.updated_at else None,
        "updatedBy": row.updated_by if row else None,
        "activeSessions": active_sessions,
    }


def set_provider(db: Session, provider: str, actor: str, expected_version: int | None = None) -> dict:
    provider = (provider or "").strip().lower()
    actor = (actor or "").strip()[:120]
    if provider not in PROVIDERS:
        raise VoiceSettingsError(400, f"Unsupported voice provider. Choose one of: {', '.join(PROVIDERS)}.")
    if not actor:
        raise VoiceSettingsError(400, "The administrator making the change must be identified.")
    if not provider_configured(provider):
        raise VoiceSettingsError(422, f"'{provider}' cannot be activated: its credentials are not configured on the server.")

    row = db.get(AppSetting, SETTING_KEY)
    if expected_version is not None and (row.version if row else 0) != expected_version:
        raise VoiceSettingsError(409, "The setting was changed by someone else. Reload and try again.")

    old = row.value if row else None
    now = datetime.datetime.utcnow()
    if row is None:
        row = AppSetting(key=SETTING_KEY, value=provider, version=1, updated_by=actor, created_at=now, updated_at=now)
        db.add(row)
    else:
        row.value, row.version, row.updated_by, row.updated_at = provider, (row.version or 0) + 1, actor, now
    db.add(SettingAudit(key=SETTING_KEY, old_value=old or default_provider(), new_value=provider, actor=actor, at=now))
    db.commit()          # setting + audit row commit together, or not at all
    return describe(db)
