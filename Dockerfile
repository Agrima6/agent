FROM python:3.12-slim

WORKDIR /app

# ffmpeg/audio libs needed by livekit-agents' audio pipeline (STT/TTS codecs).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m -u 1000 app && chown -R app:app /app
USER app

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# Runs both the FastAPI service and (via RUN_AGENT_INLINE) the LiveKit agent worker as a
# subprocess of it — see api.py's _start_agent_worker(). Set RUN_AGENT_INLINE=false and run
# `python agent.py start` as a separate container/process once scaling the agent workers
# independently of the API matters (plan §37 — stateless, independently scalable agent workers).
ENV RUN_AGENT_INLINE=true
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
