# WorkmateIQ Voice Agent — Architecture (as built)

Verified against `workmate-iq-agent` at the time of writing. Versions installed: `livekit-agents 1.7.0`,
`livekit 1.1.14`, `livekit-api 1.2.0`, `livekit-plugins-{silero,groq,elevenlabs,openai} 1.7.0`, `openai 2.54.0`.

## 1. What this is (and is not)
* **LiveKit Agents 1.7.0** (`AgentSession` + `Agent`), run by a worker registered as `workmate-interviewer`.
* **Not** the ElevenLabs Conversational AI / Agents platform, **not** ElevenLabs Speech Engine. ElevenLabs is used only
  as a **TTS plugin** (`livekit-plugins-elevenlabs`). Therefore settings made in an ElevenLabs *Agents* dashboard
  (agent prompt, turn timeout, language detection) have **no effect** here. The settings that do apply are the TTS
  options passed in `agent.build_tts` (voice id, model, language, pronunciation dictionary).
* The `AgentSession` has **no conversational LLM**. Every spoken line is produced by deterministic code
  (`InterviewConductor`) plus two bounded LLM calls (judge, composer) made through `llm_provider`.

## 2. Component map
```
Candidate browser (portal: client/ InterviewRoomPage.jsx)         HR / platform backend (client-service)
   │  WebRTC audio (Opus) + data channel                              │  X-Agent-Key (service-to-service)
   ▼                                                                  ▼
LiveKit Cloud room  interview-<id>  ───────────────►  Agent API (FastAPI, api.py) ◄── agent worker (agent.py)
   ▲   ▲                                                   │  Postgres/SQLite: interviews, turns, coverage, reports
   │   └── data topics: transcript, progress, commit_turn  │
   │                                                       └── /agent-context (plan), /turns, /progress, /complete
   │
Agent worker process (one interview per job)
   VAD (silero, min_silence 0.4 s)
     → STT: Groq Whisper whisper-large-v3-turbo (batch per VAD segment, NOT streaming)
     → end-of-turn model (LiveKit TurnDetector v1-mini, local) + endpointing 2.0 s .. 6.0 s
     → AgentSession.on_user_turn_completed
     → InterviewConductor (asyncio.Lock: one turn at a time)
          classify (interaction_guard, context-aware) → language policy → judge (LLM) → decide_action (code)
          → composer (LLM) or fixed refusal text → Speech
     → prepare_for_speech → TTS (Sarvam bulbul:v3 | ElevenLabs | Edge opt-in fallback)
     → LiveKit audio track → candidate
```

## 3. Providers and where each is configured
| Stage | Provider | Where configured | Notes |
|---|---|---|---|
| Transport | LiveKit Cloud | `LIVEKIT_URL/KEY/SECRET`; explicit dispatch in `api._ensure_agent_dispatched` | one agent per room guarded in API and worker |
| VAD | Silero | `agent.prewarm`, `VAD_MIN_SILENCE_S` (0.4) | loaded once per worker process |
| STT | Groq Whisper `whisper-large-v3-turbo` | `agent.entrypoint` (`lk_groq.STT`) | `AUTO_LANGUAGE=true` ⇒ `language` is **omitted** (Whisper detects per utterance); the turbo model returns **no language code**, so the agent cannot read Whisper's detection and judges by script |
| End of turn | LiveKit `TurnDetector v1-mini` (local) | `agent.make_turn_handling`, `TURN_DETECTOR`, `TURN_MIN_DELAY_S` (2.0), `TURN_MAX_DELAY_S` (6.0) | `vad` mode available, cuts off thinking pauses |
| Judge / composer | OpenAI-compatible provider via `llm_provider` (`LLM_PROVIDER`, default `groq`; models `JUDGE_MODEL` / `COMPOSER_MODEL`) | `llm_provider.py` | called with deadlines (judge 3.5 s, compose 2.5 s) in threads |
| TTS | Sarvam / ElevenLabs | **admin setting** `app_settings.voice_ai_provider`, fixed per interview at creation (`Interview.voice_provider`) | no silent cross-provider fallback; `VOICE_PROVIDER_FALLBACK=edge` is opt-in |
| ElevenLabs options | model `ELEVENLABS_MODEL` (default `eleven_flash_v2_5`), voice ids, **language pinned** for models that accept a language code, optional pronunciation dictionary | `agent.build_tts` | English-only models are rejected for Hindi/Hinglish |

## 4. Audio and conversation data flow
1. Candidate speech → LiveKit (Opus) → worker. Silero VAD marks speech start/end.
2. Each VAD segment is sent to Groq Whisper as a WAV request (**not streamed**; metrics report `streamed:false`).
3. The final transcript joins the user turn. The end-of-turn model plus endpointing delay decide when the turn is
   complete; `on_user_turn_completed` is then called once with the merged text.
4. `InterviewConductor.handle_candidate_turn`:
   duplicate guard → turn lock → `TURN_FINALIZED` → script/language policy → classify → judge on the *cumulative*
   answer → answer saved → flush pending writes → decision (follow-up / next / complete) → composed or fixed text.
5. `Speech` → `prepare_for_speech` (strips markdown/JSON/emoji, expands a few abbreviations) → `session.say`.
6. TTS audio is published to the room; captions go out on the `transcript` data topic (display only).

The composer's reply is produced **in full** before speaking (it is a blocking call in a worker thread), so the
sequence is serial: end-of-turn → judge → compose → TTS first audio. There is no token streaming from the composer.
TTS itself streams where the plugin streams (ElevenLabs websocket) — read the `streamed` field in `voice_metric` lines.

## 5. Question and transcript lifecycle
* **Plan creation** (`api.create_interview` → `planner.build_interview_plan`): HR questions kept verbatim; role/dynamic
  scenario questions generated once per role and cached (`roles.generated_questions`); the question bank is filtered by
  role then ranked by embedding similarity (`retrieval.py`, in-memory; no vector database) only to choose bank
  questions. The plan is stored in `interviews.plan` and is ordered and fixed.
* **Delivery**: the worker loads the plan with the service key from `GET /v1/interviews/{id}/agent-context` (the
  candidate-facing `GET /v1/interviews/{id}` has no plan). `InterviewRunner` walks the list; follow-ups come from the
  composer under policy limits and never replace a main question. Progress is persisted to `interviews.progress`
  (monotonic) so a restarted worker resumes.
* **Turn lifecycle** (`turn_state.py`, inside the interview-level machine in `states.py`):
  `CONNECTED → QUESTION_DELIVERED → CANDIDATE_SPEAKING → TURN_FINALIZED → ANSWER_SAVED → DECIDING → QUESTION_DELIVERED | COMPLETED`.
  One finalized turn can advance the interview at most once.
* **Persistence**: every candidate/agent line → `interview_turns` (original transcript text, intent, action, question
  id; idempotent within 4 s). Per-answer evaluation → `coverage_results`. `POST /complete` is idempotent and starts a
  background scoring job → `reports` (`report_status`), with `interview_status` and `evaluation_status` separate.
* **Reporting**: HR reads the report through the backend with the service key; the candidate receives only a
  submission receipt.

## 6. Observability (all structured, no transcript text)
| Log line | Source | Content |
|---|---|---|
| `voice_metric {metric:"stt"}` | LiveKit metrics | `stt_ms`, `audio_s`, `streamed` |
| `voice_metric {metric:"eou"}` | LiveKit metrics | end-of-utterance delay, transcription delay, turn-callback delay |
| `voice_metric {metric:"tts"}` | LiveKit metrics | `ttfb_ms`, `tts_ms`, `audio_s`, `chars`, `streamed`, `cancelled`, `connection_reused` |
| `turn_trace` | `turn_trace.py` | `stt_final_ms`, `judge_ms`, `compose_ms`, `logic_ms`, `ttfa_ms` (candidate stopped → agent speaking) |
| `turn_trace_tts` | `turn_trace.py` (new) | TTS first-audio for providers that do not fill the trace marks (ElevenLabs), attached to its turn |
| `event=…` | conductor | `turn_received`, `turn_classified`, `answer_finalized`, `question_transition`, `unsupported_language`, `turn_duplicate_ignored`, `advance_ignored_duplicate`, `turn_phase_rejected`, `interruption`, `commit_turn_received/ignored` |

## 7. Data channels (browser ⇄ agent)
| Topic | Direction | Authority |
|---|---|---|
| `transcript` | agent → browser | display only; never read back as evidence |
| `progress` | agent → browser | display only |
| `commit_turn` | browser → agent | accepted **only** from identity `candidate-*`; flushes the current turn. The portal does not currently send it; only the legacy `static/room.html` has the button |
