# Workmate.IQ — Interview Agent (local MVP)

A minimal, working slice of the full plan: one FastAPI service (interview domain,
resume parsing, planner, scoring, report) + one LiveKit voice agent (OpenAI
STT/LLM/TTS) that conducts a real-time voice interview with adaptive follow-ups.

Not included yet (see `plan.md` for the full production design): multi-tenancy,
integrity/proctoring, Postgres/Redis/Kafka, microservice split, horizontal scaling,
Hindi/Hinglish. This is Phase 1–7 of the roadmap, single-process, SQLite-backed.

## 1. Prerequisites

- Python 3.10+
- An OpenAI API key
- A free LiveKit Cloud project: sign up at LiveKit Cloud, create a project, copy its
  WebSocket URL + API key + secret

## 2. Setup

```bash
cd workmate-iq-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and fill in OPENAI_API_KEY, LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
```

## 3. Run

Two processes, in two terminals (same venv):

```bash
# Terminal 1 — the API + HR/candidate web pages
uvicorn api:app --reload --port 8000

# Terminal 2 — the voice agent worker (joins interview rooms automatically)
python agent.py dev
```

Then open **http://localhost:8000** — that's the HR console. Walk through:
1. Create a role
2. Create a candidate and upload a resume (PDF or .txt)
3. Create the interview — this generates the plan and gives you a candidate room link
4. Open that link in another tab (or send it to someone) and click "Join Interview" —
   the agent worker will automatically join the same room and start talking
5. After the interview, come back to the HR console and click "Fetch report"

## Notes / known rough edges

- `livekit-agents` has changed its API across versions; if `python agent.py dev`
  errors on import, check the installed version (`pip show livekit-agents`) against
  the LiveKit Agents docs and adjust `agent.py`'s `VoicePipelineAgent`/`FunctionContext`
  usage to match.
- Scoring and coverage judging call OpenAI synchronously per turn — fine for a demo,
  not for the "never block the live loop" production rule in the full plan.
- No auth on the API — do not expose this past localhost as-is.
# agent
