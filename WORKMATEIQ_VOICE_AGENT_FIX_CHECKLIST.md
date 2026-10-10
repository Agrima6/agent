# WorkmateIQ Voice Agent — Fix Checklist

Status legend: **Done** (code + test) · **Done, audio-unverified** (logic tested, needs recordings) · **Open**.
Suite: `python -m pytest -q` → **719 passed** (agent). Details: `WORKMATEIQ_VOICE_AGENT_ROOT_CAUSE_REPORT.md`.

## Ordered fixes (dependency order)
| # | Stage | Item | Status | Files | Tests |
|---|---|---|---|---|---|
| 1 | Architecture | Map the real pipeline (LiveKit Agents 1.7.0; ElevenLabs = TTS plugin only; no conversational LLM in session) | Done | `WORKMATEIQ_VOICE_AGENT_ARCHITECTURE.md` | — |
| 2 | Observability | ElevenLabs TTS first-audio attached to its turn (`turn_trace_tts`); existing STT/EOU/TTS metrics documented | Done | `turn_trace.py` | 2 |
| 3 | Transport / streaming | Confirmed ElevenLabs plugin is streaming-capable; STT is batch; composer is not token-streamed. **No change** without measurements | Open (measure first) | — | — |
| 4 | STT/TTS config | ElevenLabs language pinned (models that accept a code); English-only models rejected for Hindi/Hinglish; optional pronunciation dictionary | Done, audio-unverified | `agent.py`, `config.py` | 9 |
| 5 | Language policy | Arabic-script transcripts = non-answer (saved verbatim, not judged/quoted/language-changing) | Done | `language_policy.py`, `conductor.py`, `interaction_guard.py`, `refusals.py` | 8 |
| 6 | Language policy | Unsupported-language requests handled (stay in language, repeat question) | Done | `language_switch.py`, `conductor.py` | 6 |
| 7 | Language policy | Provider/model-aware supported languages; auto-follow respects them | Done | `agent.py`, `conductor.py` | 4 |
| 8 | Language policy | Interviewer output containing Arabic script rejected → deterministic wording | Done | `output_guard.py` | 2 |
| 9 | Turn handling | `commit_turn` accepted only from the candidate identity | Done | `agent.py` | 1 |
| 10 | Turn handling | Explicit per-turn state machine; one finalized turn advances at most once; answer saved before decision | Done | `turn_state.py`, `conductor.py` | 6 |
| 11 | Turn handling | Pause tolerance (earlier work: 2.0 s / 6.0 s configurable) + merged fragments + duplicate guard | Done, audio-unverified | `agent.py`, `conductor.py` | prior + 4 |
| 12 | Hallucination / redirects | Context-aware guard, judge cannot overrule relevant answers (earlier work) | Done | `interaction_guard.py`, `policy.py` | prior |
| 13 | Persistence / evaluation | Coverage by competency, exactly-once answers, distinct report/interview/evaluation states (earlier work) | Done | `evaluation.py`, `intelligence_report.py`, `scoring.py` | prior |
| 14 | Security | Candidate/HR split, authenticated report/plan (earlier work, deployed via PRs) | Done | `api.py`, platform `client-service` | prior |
| 15 | Provider toggle | Verified unchanged; interview keeps its provider | Done | — | existing |

## Not done / open
| Item | Why |
|---|---|
| Real-audio regression harness (1/2/5/10 s pauses, Hindi/Urdu/mixed, acronyms) | needs recordings + provider credentials |
| Latency optimisation (e.g. start TTS on the first composed sentence, parallelise judge/compose) | needs measured `turn_trace` data from a live session first |
| Choppy playback root cause | not reproduced; use `voice_metric tts` (`streamed`, `connection_reused`, `cancelled`) from a bad session |
| Decide `AUTO_LANGUAGE` default | product call — strict request-only switching = `AUTO_LANGUAGE=false` |
| Urdu as a supported language | not supported today; would need STT language, voice and composer instruction changes |
| ElevenLabs key permission | `text_to_speech` must be enabled or the interviewer is silent |

## Deployment steps
1. Merge/deploy the agent PR (`Agrima6/agent#1` plus this work), keeping `AGENT_SERVICE_KEY` equal to the backend's.
2. Ensure `ELEVENLABS_MODEL` is multilingual (default is) and the key has Text to Speech.
3. (Optional) create an ElevenLabs pronunciation dictionary and set its two ids.
4. After deploying, watch `voice_metric`, `turn_trace` and `turn_trace_tts` for one real interview; share them to
   continue with Stage 3 (latency) and V3 (choppiness).
5. No database migration.
