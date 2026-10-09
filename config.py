import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./workmate.db")
# Render/Heroku-style hosts hand out "postgres://" connection strings, but SQLAlchemy 2.x with
# psycopg2 requires the "postgresql://" scheme — rewrite rather than making every deploy target
# remember this.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
# Shared secret for service-to-service auth between the agent worker (agent.py) and this API
# (api.py). Unset by default so local dev keeps working without extra config; set it before
# exposing the API beyond localhost — see api.py's require_service_key().
AGENT_SERVICE_KEY = os.getenv("AGENT_SERVICE_KEY", "")
# Name the worker registers under in LiveKit and the API dispatches to. Change it only to run a second,
# separate worker (e.g. a test one) against the same LiveKit project without stealing production jobs.
AGENT_NAME = os.getenv("AGENT_NAME", "workmate-interviewer")
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "edge").lower()
TTS_VOICE = os.getenv("TTS_VOICE", "aditya")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
# The two voice providers an admin can choose between (see voice_settings.py). The saved admin setting wins;
# DEFAULT_VOICE_PROVIDER (else TTS_PROVIDER when it is one of the two, else sarvam) is only the initial value.
DEFAULT_VOICE_PROVIDER = os.getenv("DEFAULT_VOICE_PROVIDER", "").lower()
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")        # empty = the plugin's default voice
# Optional voices per interviewer gender (the interviewer is named Aarav / Aarvi accordingly); each falls back to ELEVENLABS_VOICE_ID.
ELEVENLABS_VOICE_ID_MALE = os.getenv("ELEVENLABS_VOICE_ID_MALE", "")
ELEVENLABS_VOICE_ID_FEMALE = os.getenv("ELEVENLABS_VOICE_ID_FEMALE", "")
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5")
# Opt-in only: when the chosen provider fails, speak with Edge TTS instead of staying silent. OFF by default:
# a provider failure is reported, never silently handed to a different provider.
VOICE_PROVIDER_FALLBACK = os.getenv("VOICE_PROVIDER_FALLBACK", "").lower()
# Voice settings (speaker, pace, language code, sample rate) live in voice_config.py, calibrated per
# speaker; override with TTS_SPEAKER_OVERRIDE / TTS_PACE_OVERRIDE / TTS_GENDER / TTS_SAMPLE_RATE.



