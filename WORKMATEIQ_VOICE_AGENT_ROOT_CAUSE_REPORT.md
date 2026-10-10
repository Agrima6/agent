# WorkmateIQ Voice Agent — Root Cause Report

Companion to `WORKMATEIQ_AI_AGENT_ROOT_CAUSE_REPORT.md` (turn handling, redirects, exposure, evaluation — already fixed).
This report covers latency/choppiness, STT/TTS accuracy, language mixing and the turn lifecycle.

**Labels used:** **CONFIRMED** = established from code/SDK source/stored data, with a regression test where it is a code
defect. **HYPOTHESIS** = plausible mechanism not provable without live audio or logs. **UNVERIFIED** = needs data this
environment does not have.

## What could not be done here
* No recordings, no live microphone session, no ElevenLabs/Sarvam synthesis (the ElevenLabs key lacked the
  `text_to_speech` permission when last checked) and **no production logs**. Acoustic accuracy, pronunciation quality,
  audible choppiness and real latency numbers are therefore **UNVERIFIED**. Nothing below claims a measured improvement.
* The stored interviews in the local database contain no Urdu-script transcripts and no latency logs.

## Issue register
| # | Issue | Status | Root cause / finding | Fix | Test |
|---|---|---|---|---|---|
| V1 | Speech latency | **CONFIRMED structural**, numbers UNVERIFIED | The response path is serial: end-of-turn → judge (≤3.5 s) → composer (≤2.5 s, blocking, not token-streamed) → TTS. STT is batch (`streamed:false`) per VAD segment. The earlier thinking-pause fix adds up to 2.0 s of endpointing. TTS (ElevenLabs websocket) *is* streaming-capable (`capabilities.streaming=True`) | No speculative optimisation made without measurements. Added the missing measurement (V2) and kept the pause tolerance configurable (`TURN_MIN_DELAY_S`) | `test_elevenlabs_tts_latency_is_attached…` |
| V2 | No TTS latency for ElevenLabs turns | **CONFIRMED** | `turn_trace` TTS spans are filled only by the Sarvam plugin; ElevenLabs turns reported `tts_ttfa_ms: null` | `TurnTracer.note_tts_metrics`: attaches LiveKit's TTS time-to-first-byte to its turn once (`turn_trace_tts`) | 2 trace tests |
| V3 | Choppy / incomplete playback | **UNVERIFIED (not reproduced)** | No evidence collected. Candidate mechanisms (HYPOTHESES): per-sentence TTS cold connections (`connection_reused:false`), TTS cancelled by a false interruption, browser autoplay/buffering. Instrumentation to tell them apart already exists: `voice_metric tts` carries `ttfb_ms`, `streamed`, `cancelled`, `connection_reused`; `event=interruption` marks barge-ins | None applied (would be guessing) | — |
| V4 | Unexpected Urdu words | **PARTLY CONFIRMED** | (a) Language-blind transcript path: `classify_language` ignored Arabic script entirely, so an Urdu-script transcript was treated as "no signal" and passed to the judge as an ordinary answer — **CONFIRMED**. (b) With `AUTO_LANGUAGE=true` Whisper's `language` is omitted and the Groq turbo model returns no language code, so a Hindustani/Indian-accent utterance can be written in Urdu script — mechanism **CONFIRMED in plugin source**, occurrence on real audio **UNVERIFIED**. (c) ElevenLabs `language` was **not set** (old `build_tts`: api_key/model/voice only), so the engine infers the language from each sentence — **CONFIRMED**; effect on pronunciation **HYPOTHESIS**. (d) Quote-back of Urdu words via follow-ups — **FALSIFIED**: `ground_concepts` drops Arabic-script concepts. (e) Captions: the candidate's own Urdu-script transcript is shown in the caption stream — **HYPOTHESIS** for "the app mixes Urdu" reports | Arabic-script transcripts are a non-answer (`unsupported_language`): saved verbatim, not judged, not quoted, cannot change the language; fixed "say it again in English/Hindi/Hinglish" reply. Interviewer output containing Arabic script is rejected by `output_guard` (deterministic wording used). ElevenLabs language pinned for `eleven_flash_v2_5/turbo_v2_5/v3` | 14 tests |
| V5 | Unsupported language requests ("Can I speak in Urdu?") | **CONFIRMED defect** | Not recognised: fell through as a technical answer and was judged | `detect_unsupported_language_request`; fixed reply that stays in the current language and repeats the question; nothing echoed | 6 tests |
| V6 | Voice cannot speak the requested language | **CONFIRMED** | A switch to Hindi was attempted even with an English-only ElevenLabs model | `supported_languages_for(provider, model)`; English-only models fail clearly for Hindi sessions at start, and a mid-session switch is refused; auto-follow respects it | 5 tests |
| V7 | Language flips from ambiguous input | **Checked, already guarded** | `LanguageFollower`: ≥4 words (or two utterances), cooldown, English↔Hindi hysteresis. New: Arabic-script and unsupported targets can never trigger a follow | tests above | 2 tests |
| V8 | "Done speaking" early-commit path | **CONFIRMED present, narrowed** | The portal has **no** Done-speaking control (grep of `client/src`); only legacy `static/room.html` publishes `commit_turn`. The agent honoured `commit_turn` from *any* participant | Accepted only from identity `candidate-*`; duplicates absorbed by the in-flight duplicate guard | `is_candidate_identity` test |
| V9 | Other early-commit paths | **Re-inspected** | `TURN_DETECTOR=vad` remains available (documented as cutting off pauses; default is the local model). Endpointing is 2.0–6.0 s. Partial transcripts are never committed: the agent only receives `on_user_turn_completed` | no change | prior 1/2/5/10-s tests: **see gap G1** | 
| V10 | Hallucinated / substituted questions | **Checked** | Main questions come only from the persisted plan via `/agent-context`; the runner index is server-persisted and monotonic; follow-ups are policy-limited and never replace a main question; missing plan → `raise_for_status` fails the job instead of inventing content | no change needed | existing runner/policy tests |
| V11 | Duplicate/racing events advancing the interview | **CONFIRMED gap, fixed** | no per-turn lifecycle; only a text-equality guard | `turn_state.TurnStateMachine`: answer saved before decision; one finalized turn advances at most once; illegal transitions are logged (`turn_phase_rejected`) | 6 tests |
| V12 | Pronunciation of acronyms / names | **UNVERIFIED** | `prepare_for_speech` is intentionally conservative. How ElevenLabs reads "SQL", "TTL", "Redis" can only be judged by listening | Optional ElevenLabs pronunciation dictionary (`ELEVENLABS_PRONUNCIATION_DICTIONARY_ID/VERSION_ID`), used only when both are set | config test |
| V13 | Provider toggle integrity | **Verified, unchanged** | Admin setting persisted in `app_settings`; `Interview.voice_provider` fixed at creation; mid-interview changes do not apply; no silent cross-provider fallback | none | `test_voice_provider.py` (existing) |
| V14 | Candidate data not altered before evaluation | **Verified** | The original transcript is stored as `interview_turns.text`; language policy only changes how the *turn is handled*, never the stored text | unsupported-script turn is saved verbatim | `…_not_evaluated_quoted…` |

## Gap G1 (honest)
Tests for 1/2/5/10-second pauses are **configuration and logic tests**, not audio tests. They assert that the configured
thresholds tolerate those pauses and that fragments are merged. Whether the on-device end-of-turn model plus 2.0 s
endpointing behaves correctly on real speech with a 10 s silence (it relies on `max_delay=6.0` then commits) requires a
recorded-audio replay harness, which does not exist yet. A 10-second silence will be committed at ≈6 s by design.

## Language policy now enforced (code, not prompt)
1. Session language is explicit state on the conductor (`self.language`), initialised from the interview.
2. It changes only through (a) an explicit request for a supported language (`detect_language_request`), or (b) the
   cautious auto-follow (multi-utterance, cooldown) **when `AUTO_LANGUAGE` is enabled and the voice supports it**.
3. Arabic-script transcripts and requests for unsupported languages never change it.
4. The interviewer's own text is rejected if it contains Arabic script.
5. English-only ElevenLabs models are rejected for Hindi/Hinglish.

Product decision to confirm: the earlier commit "Follow the language the candidate speaks" keeps auto-follow on by
default. Your new requirement says to switch "only through an intentional flow". The conservative follower satisfies
"no single ambiguous word", but if you want strictly request-only switching set `AUTO_LANGUAGE=false` (this also pins
Whisper's language to the session language, which is the single most effective guard against Urdu-script output).

## Deployment requirements
* New env (all optional): `ELEVENLABS_PRONUNCIATION_DICTIONARY_ID`, `ELEVENLABS_PRONUNCIATION_VERSION_ID`.
* `ELEVENLABS_MODEL` must be multilingual (default `eleven_flash_v2_5`) for Hindi/Hinglish sessions.
* The ElevenLabs API key needs the **Text to Speech** permission.
* No database migration.
