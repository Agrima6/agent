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
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "edge").lower()
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-AndrewMultilingualNeural")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_MODEL = os.getenv("SARVAM_MODEL", "bulbul:v3")
SARVAM_SPEAKER = os.getenv("SARVAM_SPEAKER", "svc-e6c0f0a8-9386-4eb2-8558-2fc0036f53a4")
SARVAM_VOICE_ID = os.getenv("SARVAM_VOICE_ID", "svc-e6c0f0a8-9386-4eb2-8558-2fc0036f53a4")
SARVAM_LANGUAGE_CODE = os.getenv("SARVAM_LANGUAGE_CODE", "en-IN")
SARVAM_PACE = float(os.getenv("SARVAM_PACE", "1.0"))
SARVAM_SAMPLE_RATE = int(os.getenv("SARVAM_SAMPLE_RATE", "22050"))



