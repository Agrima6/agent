# WorkmateIQ Voice Agent — Test Plan

Two tiers: **automated** (deterministic, runs in CI, no credentials) and **live** (recordings / a real session; not
automated yet). Automated results: `python -m pytest -q` → 719 passed.

## A. Automated matrix
| Area | Scenario | File / test | Asserts |
|---|---|---|---|
| Turn detection | Thinking pauses 1, 2, 5, 10 s | `test_voice_language_policy.py::test_thinking_pauses…` | configured thresholds tolerate ≤5 s; 10 s commits at max_delay by design (**config-level, not audio**) |
| Turn detection | Fragments of one answer | `test_turn_and_api_security.py::test_a_second_fragment_is_judged_together…` | judge sees both fragments; timer untouched |
| Turn detection | Duplicate end-of-turn | `…::test_duplicate_end_of_turn_events_advance_and_answer_exactly_once` | one judgement, one advance |
| Turn detection | One finalized turn advances once | `test_voice_language_policy.py::test_a_second_finalized_turn_cannot_advance_twice` | duplicate advance ignored |
| Turn detection | Illegal transitions | `…::test_the_turn_machine_…`, `…::test_an_answer_cannot_be_saved_before…` | refused and recorded |
| Turn detection | Candidate-only commit | `…::test_only_the_candidate_identity_may_end_a_turn` | non-candidate identities rejected |
| Multilingual | Arabic-script transcript | `…::test_an_urdu_script_transcript_is_not_evaluated…` | not judged, not quoted, language unchanged, original text saved |
| Multilingual | Script detection matrix | `…::test_arabic_script_transcripts_are_detected…` | Urdu yes; English/Hinglish/Devanagari no |
| Multilingual | Language request (supported) | `test_report_fixes.py`, `test_conductor.py` | switches and repeats current question, no advance, not scored |
| Multilingual | Language request (unsupported) | `…::test_a_request_for_an_unsupported_language…` | stays in language, repeats question |
| Multilingual | Voice can't speak Hindi | `…::test_a_voice_that_cannot_speak_hindi…`, `…::test_an_english_only_model_fails_clearly…` | refused / fails clearly |
| Multilingual | Ambiguous word / auto-follow | `…::test_a_single_ambiguous_word…`, `…::test_auto_follow_never_moves…` | no switch |
| TTS config | Language pinned, model validation, pronunciation dictionary | `…::test_elevenlabs_language_is_pinned…` etc. | options reach the plugin |
| TTS output | Interviewer text in Arabic script | `…::test_interviewer_text_in_arabic_script_is_rejected…` | falls back to deterministic wording |
| Latency | TTS first-audio attached once | `…::test_elevenlabs_tts_latency_is_attached…` | one follow-up line per turn |
| Hallucination | Relevant answers never redirected | `test_turn_and_api_security.py::test_a_relevant_technical_answer…` | 9 realistic answers = answer |
| Hallucination | Off-topic / injection still redirected | `…::test_genuine_off_topic_and_manipulation…` | 11 cases |
| Hallucination | Judge cannot overrule | `…::test_judge_cannot_refuse…` | relevant answers kept |
| Persistence | Answers exactly once; scoring; coverage | `test_evaluation_pipeline.py` | 24 tests |
| Security | Candidate/HR/service boundaries | `test_turn_and_api_security.py`, `test_evaluation_pipeline.py` | 401 without key; DTOs free of rubric/evaluation |
| Isolation | Concurrent sessions | `…::test_two_interviews_in_one_process_never_share_turn_state` | independent state |
| Provider | Toggle persisted, per-interview, no mid-session change | `test_voice_provider.py` | existing |

## B. Live / recorded tests (to build — not automated)
| # | Test | Input | Pass criterion | Needs |
|---|---|---|---|---|
| L1 | Pause matrix on real speech | 5 recordings: answer, pause 1/2/5/10 s, continue | no early commit for ≤5 s; transcript contains both segments | audio files, Groq key |
| L2 | English with Indian accent | 10 technical answers | WER ≤ agreed threshold; **no Arabic-script output** | audio, ground-truth transcripts |
| L3 | Hindi / Hinglish | 10 answers each | script is Devanagari/Latin, never Arabic | audio |
| L4 | Urdu speech | 5 recordings | `unsupported_language` reply, no scoring | audio |
| L5 | Acronyms and names | "SQL, TTL, Redis, Kubernetes, PostgreSQL, JWT" in a script | listener-rated pronunciation; add dictionary entries | ElevenLabs key + listener |
| L6 | Latency | 20 turns, both providers | p50/p95 `ttfa_ms`, `stt_ms`, `judge_ms`, `compose_ms`, `tts_ttfa_ms` recorded | live session |
| L7 | Choppiness | 10-minute session on desktop + mobile + weak network | no gaps in `voice_metric tts` (`cancelled:false`); listener notes | live session |
| L8 | Reconnect | drop network 5 s mid-answer | session resumes at same question, no advance, answer not lost | live session |

## C. Commands
```
python -m pytest -q                                  # agent, 719 tests
cd ../interviewIQ/services/client-service && node --test tests/*.test.js   # platform, 32 tests
```
Lint/type-check: none configured in either repo.
