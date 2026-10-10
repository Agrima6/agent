# WorkmateIQ AI Interview Agent — Fix Checklist

Legend — **Fixed**: code changed and covered by tests · **Mitigated**: risk reduced, residual noted · **Open**: not changed.
Test run: `python -m pytest -q` → **673 passed** (56 in `test_turn_and_api_security.py`, 24 in `test_evaluation_pipeline.py`); client-service `node --test` → 32 passed. Details and evidence:
`WORKMATEIQ_AI_AGENT_ROOT_CAUSE_REPORT.md`.

## P0 — Confirmed critical security vulnerabilities / credential exposure

| # | Item | Status | Root cause | Files changed | Test coverage | Verification evidence |
|---|---|---|---|---|---|---|
| P0-1 | HR evaluation report readable without authentication and rendered on the candidate page | **Fixed** | `GET /v1/interviews/{id}/report` had no `require_service_key`; `room.html` fetched it and called `renderReport()` | `api.py` (report now keyed; new `/completion` receipt), `static/room.html` (report fetch + renderer removed) | `test_the_hr_report_is_not_reachable_without_the_service_key`, `test_the_candidate_completion_receipt_contains_nothing_evaluative`, `test_the_candidate_page_never_requests_or_renders_hr_data` | Before/after payloads in report §6; `/report` → 401 without key; page script passes `node --check` and contains no `/report`/`renderReport` |
| P0-2 | Answer key / scoring configuration returned to unauthenticated callers | **Fixed** | `GET /v1/interviews/{id}` and `POST /v1/interviews` returned the stored plan verbatim (`expected_topics`, `followup_topics`, `competencies`, `focus_priority`, `focus_areas`, `hr_guidance`, `resume_claims`) | `api.py` (`public_plan`, public DTO, `/agent-context`), `agent.py` (reads `/agent-context` with key), `static/room.html` (`question_count`) | `test_the_candidate_facing_interview_endpoint_carries_no_plan_or_rubric`, `test_the_full_plan_is_only_available_to_the_service`, `test_create_interview_response_has_question_wording_but_no_rubric` | Live check on running API: public GET returns 6 fields; `/agent-context` returns plan with rubric fields |
| P0-3 | Production could run with every privileged endpoint open | **Fixed** | `AGENT_SERVICE_KEY` unset ⇒ `require_service_key` is a no-op; nothing stopped that in production | `api.py` (`_is_production`, fail-closed at import, warning when unset, constant-time compare) | `test_production_refuses_to_start_without_a_service_key` | Subprocess import with `APP_ENV=production` and no key exits non-zero |
| P0-4 | Live session language rewritable by anyone holding an interview id | **Fixed** | `POST /v1/interviews/{id}/language` unauthenticated at all stages | `api.py` (`set_interview_language`: free before start, key required after) | `test_language_can_be_chosen_before_start_but_not_changed_mid_interview_without_the_key` | 401 without key when IN_PROGRESS |
| P0-5 | Credentials pasted into chat during setup (OpenAI key, LiveKit secret, ElevenLabs key) | **Open — action for owner** | Secrets shared in plain text outside a secrets manager | none (cannot be fixed in code) | n/a | Revoke and regenerate each; keep in `.env` / secret store only |

## P1 — Premature turn completion, incorrect redirection, confidential response leaks

| # | Item | Status | Root cause | Files changed | Test coverage | Verification evidence |
|---|---|---|---|---|---|---|
| P1-1 | Thinking pause of 1–2 s ends the turn | **Fixed (config + logic verified; live audio not verified)** | `endpointing.min_delay` 0.4 s (below SDK default 0.5 s); `max_delay` 3.0 s while a comment claimed 5 s | `agent.py` (`TURN_MIN_DELAY_S`=2.0, `TURN_MAX_DELAY_S`=6.0, env-adjustable) | `test_turn_handling_tolerates_a_one_to_two_second_thinking_pause` | SDK `turn.py` semantics read; values asserted. Needs a real-mic session to tune |
| P1-2 | Answer split into fragments is judged piecemeal | **Fixed** | `_answer_turn` judged only the latest committed turn | `conductor.py`, `policy.py` (`QuestionState.answer_parts`) | `test_a_second_fragment_is_judged_together_with_the_first` | `judge_calls[1]` contains both fragments |
| P1-3 | "Let's stay focused…" on valid technical answers (deterministic guard) | **Fixed** | Request patterns matched words anywhere in ≤35-word answers; 8 of 12 realistic answers misrouted | `interaction_guard.py` | `test_a_relevant_technical_answer_is_an_answer_not_a_refusal` (9 cases), `test_the_patterns_themselves_no_longer_fire_on_ordinary_engineering_speech`, `test_genuine_off_topic_and_manipulation_are_still_redirected` (11 cases) | Probe table in report §2.2 re-run: all answers now `answer` |
| P1-4 | LLM judge refuses relevant answers (any length for "prompt injection") | **Fixed** | `restricted_intent_from_judge` exempted `prompt_injection` from the 35-word cap and ignored relevance | `policy.py`, `conductor.py` | `test_judge_cannot_refuse_a_long_relevant_answer_as_prompt_injection`, `test_judge_cannot_refuse_a_substantive_on_topic_answer_as_off_topic`, `test_judge_can_still_refuse_a_short_genuinely_unrelated_message` | Off-topic handling retained, not removed |
| P1-5 | Injection-like wording inside a legitimate security answer | **Fixed** | Hard refusal for any ≤35-word text containing the phrase | `interaction_guard.py` | `test_injection_wording_inside_an_on_topic_answer_is_evaluated_but_flagged` | Answer evaluated, flagged `suspicious`, raw text withheld from composer |
| P1-6 | Duplicate end-of-turn answered/advanced twice | **Fixed** | No de-duplication of concurrent identical turns | `conductor.py` (`_inflight_turns`) | `test_duplicate_end_of_turn_events_advance_and_answer_exactly_once` | One judge call, one advance for 3 concurrent calls |
| P1-7 | Candidate sees HR-only completion output | **Fixed** | See P0-1 | see P0-1 | see P0-1 | see P0-1 |

## P2 — Question generation consistency, state management, retries, observability

| # | Item | Status | Root cause / finding | Files changed | Test coverage | Verification evidence |
|---|---|---|---|---|---|---|
| P2-1 | Question advances before the answer is persisted | **Fixed** | `record_turn`/coverage writes are fire-and-forget; advance did not wait | `conductor.py` (`flush_pending(timeout=2.0)` before `advance`) | `test_the_answer_is_saved_before_the_interview_moves_on` | Barrier asserted; bounded so a slow API cannot freeze the interview |
| P2-2 | Duplicate transcript rows | **Fixed** | `POST /turns` not idempotent (5 duplicate groups in local DB) | `api.py` (`record_turn`, 4 s window) | `test_posting_the_same_turn_twice_stores_one_row` | Second POST returns the first id with `duplicate: true` |
| P2-3 | Structured diagnostics for turn lifecycle | **Fixed** | Only free-text logs existed | `conductor.py` (`_event`), `agent.py` (interruption) | `test_diagnostic_events_carry_no_candidate_words` | Events: `turn_received`, `turn_classified`, `answer_finalized`, `question_transition`, `turn_duplicate_ignored`, `interruption` — ids/counts only |
| P2-4 | Question set generated once and consistent | **Verified, no change** | Plan generated once in `create_interview`, stored in `interviews.plan`; role-level dynamic questions cached in `roles.generated_questions`; agent consumes the stored plan | — | existing planner/API tests | DB inspection: 90 stored questions, fields listed in report §4 |
| P2-5 | Server-side progress is authoritative and monotonic | **Verified, no change** | `record_progress` rejects stale writes | — | existing `test_progress_only_moves_forward` | Passing |
| P2-6 | Question timer unaffected by pauses | **Verified, no change** | `started_at` set once per question | — | `test_a_thinking_pause_does_not_reset_or_skip_the_question_timer` | Passing |
| P2-7 | Reconnect restarts the current question's timer | **Open** | `runner.restore()` builds a new `QuestionState` | — | — | Documented; decide whether elapsed time should persist |
| P2-8 | Abandoned `IN_PROGRESS` interviews never expire | **Open** | No sweeper (2 stale rows locally) | — | — | Add a periodic expiry job |
| P2-9 | Platform backend still forwarding plan fields | **Open — other repo** | The `questions[]` bundle in the issue is assembled by the backend wrapping this service | — | — | Backend must switch to `/agent-context` for HR views and send only the sanitised plan to browsers |

## P3 — Performance and additional hardening

| # | Item | Status | Notes |
|---|---|---|---|
| P3-1 | Latency cost of the longer end-of-turn delay | **Open (tuning)** | Replies start ≥2 s after silence. Tune `TURN_MIN_DELAY_S` with real audio; evaluate `endpointing.mode="dynamic"` |
| P3-2 | DB indexes | **Open** | No index on `interview_turns.interview_id`, `coverage_results.interview_id`, `reports.interview_id`; add before volume grows |
| P3-3 | Tenant/org isolation | **Open — other repo** | This service has no org/user model; isolation must be enforced by the backend that holds the service key |
| P3-4 | Per-interview signed candidate tokens instead of bare interview ids | **Open** | Ids are capabilities; short-lived signed tokens would remove id-guessing as a concern for candidate endpoints |
| P3-5 | Rate limiting on public candidate endpoints | **Open** | `/interviews/{id}`, `/completion`, `/language` |
| P3-6 | `hr.html` has no key field | **Open** | Local console works only while the key is unset |
| P3-7 | Replace deprecated `@app.on_event` handlers with lifespan | **Open** | 6 deprecation warnings in test output |
| P3-8 | Live-audio regression harness | **Open** | Turn timing is verified via config/logic only; add recorded-audio replays (pauses of 1, 2, 4 s) |

## Commands run
```
python -m pytest -q                         # 648 passed
python -m pytest -q test_turn_and_api_security.py   # 56 passed
node --check <extracted room.html script>   # OK
```
Not run: linters/type-checkers (none configured in the repo), front-end build (no build step; static HTML), live
microphone session, ElevenLabs TTS (key lacks `text_to_speech` permission).


## Addendum — evaluation pipeline (see Addendum A in the root-cause report)

| # | Item | Pri | Status | Root cause | Files | Tests |
|---|---|---|---|---|---|---|
| E-1 | Candidate received the full HR report from `agent-complete` | P0 | **Fixed** | `ok(res, report)` in the candidate-facing controller | `client-service/controllers/interviewDrive.controller.js`, `utils/candidateQuestions.js` | `candidateQuestions.test.js`, `test_a_candidate_cannot_obtain_any_hr_only_evaluation_data` |
| E-2 | Browser received the agent's rubric fields via `questions` | P0 | **Fixed** | start response forwarded `plan.questions` verbatim | `client-service/services/interviewDrive.service.js`, `utils/candidateQuestions.js` | `candidateQuestions.test.js` |
| E-3 | Coverage ignored dynamic/resume questions (Problem Solving NOT_ASKED) | P1 | **Fixed** | coverage keyed on free-text `focus` tag only; "asked" = planned | `intelligence_report.py` | 6 coverage tests |
| E-4 | Language request scored 0 / not repeated when already in language | P1 | **Fixed** | `"language_request"` not a non-answer intent; same-language request fell to answer path | `interaction_guard.py`, `conductor.py`, `evaluation.py` | 3 language tests + evaluation test |
| E-5 | Final score deflated by unassessed competencies | P1 | **Fixed** | divided by full role weight | `scoring.py` | normalisation + reproduction tests |
| E-6 | Communication score never fed the `communication` competency; mixed 0–10 / 0–100 | P1 | **Fixed** | computed, then unused; prompt omitted scale | `evaluation.py`, `scoring.py` | clamp/feeds test, prompt-scale test |
| E-7 | Skipped questions could raise the average | P1 | **Fixed** | asked-but-unanswered dropped | `evaluation.py` | skip tests |
| E-8 | `READY` read as "evaluation complete" | P1 | **Fixed** | only report-generation status was exposed | `api.py`, `scoring.py` | distinct-states test |
| E-9 | Re-delivered answer line scored twice; intro inflated answered count | P2 | **Fixed** | rows joined without de-duplication; intro counted | `evaluation.py` | 3 exactly-once tests |
| E-10 | Tests inherited `AGENT_SERVICE_KEY` from `.env` | P2 | **Fixed** | non-hermetic suite | `conftest.py` | full suite |
| E-11 | Old reports keep 0–10 communication scores / deflated finals | P3 | **Open** | no retroactive re-scoring | — | — |
| E-12 | Candidate list returns `aiScore` | P3 | **Open (decision needed)** | explicit field in `getMyInterviews` | `client-service/services/interviewDrive.service.js` | — |
