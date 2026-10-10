# WorkmateIQ AI Interview Agent — Root Cause Report

Scope: the `workmate-iq-agent` repository (FastAPI interview service + LiveKit voice agent + static HR/candidate pages).
Installed stack at investigation time: `livekit-agents 1.7.0`, `livekit 1.1.14`, `livekit-api 1.2.0`,
`livekit-plugins-silero 1.7.0`, `livekit-plugins-groq 1.7.0`, `livekit-plugins-elevenlabs 1.7.0`, `openai 2.54.0`,
`fastapi 0.115.14`, `SQLAlchemy 2.0.52`. All findings below were traced in this repo's code and a read-only copy of the
local SQLite database; nothing was inferred from the issue text alone.

## 1. Executive summary

| # | Reported problem | Verdict | Root cause (short) |
|---|---|---|---|
| 1a | Agent cuts candidates off during 1–2 s thinking pauses | **Confirmed** | End-of-turn silence required after a "finished-sounding" sentence was **0.4 s** (SDK default 0.5 s); a stale comment claimed a 5 s wait that was never configured (`max_delay` was 3.0). Each committed fragment was then judged **alone**. |
| 1b | "Let's focus on the interview" on valid technical answers | **Confirmed** | Not the system prompt (the session has no LLM). It is the fixed `PROMPT_INJECTION` / `UNRELATED` refusal text. The deterministic guard matched request phrases **anywhere** inside ≤35-word answers; **8 of 12** realistic technical answers were misrouted. The LLM judge could also refuse an answer of **any length** as "prompt injection". |
| 2 | Pre-generated questions exposed through the API | **Confirmed** | The whole plan (`expected_topics`, `followup_topics`, `competencies`, `focus_priority`, `focus_areas`, `hr_guidance`, `resume_claims`) is stored in one JSON column and returned verbatim by an **unauthenticated** `GET /v1/interviews/{id}` and by the create response. The candidate page itself read it. |
| 3 | Completion leaks HR output to the candidate | **Confirmed** | `GET /v1/interviews/{id}/report` had **no authentication**, and `static/room.html` (the candidate page) fetched it after completion and rendered the full HR report (score, hire/no-hire, strengths/gaps, per-question analysis, evidence). `POST /complete` itself returns only `{report_id, status}`. |
| 4 | Progression / idempotency | **Partly sound, gaps fixed** | Server-side progress is already monotonic and the conductor serialises turns with a lock. Gaps: advancing did not wait for the answer write; duplicate end-of-turn events could be answered twice; `/turns` was not idempotent; `/language` was open to anyone mid-interview. |

All of the above are fixed and covered by regression tests (648 tests pass; 56 are new). Items that cannot be verified from
this repository are listed in §9.

## 2. Reproduction

### 2.1 Premature turn completion
1. Config evidence: `agent.py` (before) `TURN_HANDLING = {"endpointing": {"min_delay": 0.4, "max_delay": 3.0}, ...}`;
   SDK source `livekit/agents/voice/turn.py` documents `min_delay` = "silence required … before declaring the user's turn
   complete" and `max_delay` = "maximum time the agent waits".
2. With the on-device end-of-turn model (`v1-mini`), a sentence it judges complete is committed after `min_delay`
   (0.4 s). A candidate who finishes a sentence and thinks for 1–2 s therefore produces **two committed turns**.
3. Conductor tests reproduce the second half: before the fix each fragment was judged in isolation
   (`h.judge_calls[1]` contained only the second fragment); after the fix it contains both.

### 2.2 False redirect — probe run against the original guard
Realistic technical answers (≤35 words) passed through `interaction_guard.classify` **before** the fix:

| Answer (abridged) | Old verdict | Refusal the candidate hears |
|---|---|---|
| "…we **repeat** the call with exponential backoff…" | `repeat` | question is read out again |
| "I would store the **system prompt** in a config service…" | `prompt_injection` | "Let's stay focused on the interview…" |
| "For the **weather** widget we cache the forecast…" | `unrelated` | "Let's stay focused on the current interview question…" |
| "…check **my score** on the hit-ratio dashboard…" | `request_evaluation` | evaluation refusal |
| "**You are now** looking at a read-heavy workload…" | `prompt_injection` | "Let's stay focused on the interview…" |
| "**what's next** is checking the slow query log…" | `request_upcoming` | "can't reveal upcoming questions" |
| "In the **film** rendering pipeline we shard frames…" | `unrelated` | "Let's stay focused…" |
| "**Explain** the data flow, then I would describe…" | `request_explanation` | explanation refusal |

Control cases (ordinary answers with none of those words) were classified correctly as answers, which is why the bug
looked intermittent.

### 2.3 Exposure
```
curl -s localhost:8000/v1/interviews/<id>                # no credentials
curl -s localhost:8000/v1/interviews/<id>/report         # no credentials
```
Both returned full internal data before the fix (see §6 for before/after).

## 3. Actual architecture and request flow

```
HR/platform backend ──(X-Agent-Key)──► POST /v1/roles, /v1/candidates, /v1/interviews        api.py:243
        │                                   └─ planner builds plan ─► interviews.plan (JSON)
        ├─(X-Agent-Key)──► POST /v1/interviews/{id}/candidate-token  ─► LiveKit JWT + explicit agent dispatch
        └─(X-Agent-Key)──► GET  /v1/interviews/{id}/report            (HR evaluation)  api.py:817

Candidate browser (static/room.html) ── LiveKit room ──► InterviewerAgent (agent.py)
        │ GET  /v1/interviews/{id}            → status, role, language, question_count   api.py:324
        │ POST /v1/interviews/{id}/language   → only before the interview starts        api.py:432
        │ GET  /v1/interviews/{id}/completion → {submitted, message}                    api.py:804

Voice agent (agent.py) ── GET /agent-context (X-Agent-Key) ──► full plan                  api.py:339
   VAD (silero) → STT (Groq Whisper) → end-of-turn model → on_user_turn_completed
   → InterviewConductor.handle_candidate_turn (asyncio.Lock) → classify → judge → decide_action
   → composer/refusal text → TTS (Sarvam / ElevenLabs) ; every write goes through the API
```
Key facts verified in code:
* The `AgentSession` is created **without an LLM**; every spoken line comes from `InterviewConductor`
  (`agent.py` `on_user_turn_completed` ends in `raise StopResponse()`). So "Let's focus on the interview" can only come from
  `refusals.py`, never from a free-form model.
* `INTERVIEW` state machine (`states.py`): CREATED → PLANNED → READY → IN_PROGRESS → FINALIZING → COMPLETED
  (plus TERMINATED / FAILED / EXPIRED), every transition persisted in `interview_events` with a version check.
* Progress is persisted by `POST /progress` and is **monotonic** (`api.py` `record_progress`).

## 4. Database findings (read-only, local `workmate.db`)

Tables: `roles, candidates, interviews, interview_events, interview_turns, coverage_results, reports, integrity_events,
app_settings, setting_audit`. No tenant/organisation column exists in any table.

* `interviews.plan` (JSON) stores, for each question, **all** of: `competencies, difficulty, expected_topics, focus,
  focus_priority, followup_topics, id, question_text, role_tags, time_limit, topic, type` (90 questions inspected). The plan
  is **generated once** at creation (`create_interview`) and stored; it is not regenerated per request, and the agent
  consumes the stored plan (no separate generation in the agent). Role-level dynamic questions are cached on
  `roles.generated_questions` so every candidate for a role gets the same ones.
* `reports.content` top-level keys: `candidate, overall, questions, role` (+ coverage/intelligence keys on newer rows) —
  this is the HR evaluation that the candidate page used to render.
* `interview_turns`: 179 rows; 5 groups of **identical** rows for the same interview/question/speaker (agent
  `GREETING`/`NEXT_QUESTION`/`REDIRECT` lines and one candidate `unrelated` turn). These come from re-delivered events and
  reconnects; `POST /turns` had no idempotency.
* Status distribution: COMPLETED 9, IN_PROGRESS 2, PLANNED 5 — the two IN_PROGRESS rows are abandoned sessions with no
  expiry sweeper.
* Indexes: none on `interview_turns.interview_id`, `coverage_results.interview_id`, `reports.interview_id` (only
  `ix_setting_audit_key` exists).
* Not verifiable here: whether the platform backend's own database (organisations, applications, HR users) matches these
  interview ids — that service is not in this repository.

## 5. Evidence-based root causes

### 5.1 Premature turn completion (P1)
* `agent.py` (old) `endpointing = {min_delay 0.4, max_delay 3.0}` and silero `min_silence_duration=0.4`
  (`prewarm`, now `agent.py:427`). The comment above the dict said "raised from 3 s to 5 s" but the value was 3.0.
* Fragments: `conductor._answer_turn` judged only the current committed text; an early fragment was scored as a thin
  answer and a follow-up (or refusal) could be spoken over a candidate who was still explaining.
* No protection against the same end-of-turn being delivered twice (e.g. detector + the "Done speaking" button).
* Not a cause (checked): the question timer. `QuestionState.started_at` is set once when a question is created
  (`runner._new_question_state`), so pauses neither reset nor skip it. One adjacent behaviour: `runner.restore()` builds a
  fresh state, so a **reconnect** restarts that question's timer (documented in the checklist, not changed).

### 5.2 "Let's focus on the interview" (P1)
1. **Deterministic guard over-matching** (`interaction_guard.py`): `_REPEAT`, `_UNRELATED`, `_EVALUATION`, `_UPCOMING`,
   `_EXPLANATION` matched words anywhere in ≤35-word utterances; `_INJECTION` fired on bare `system prompt`,
   `you are now`, `imagine we …`, at any answer length ≤35 words.
2. **Judge second opinion** (`policy.restricted_intent_from_judge`): the 35-word cap explicitly exempted
   `prompt_injection`, so a long, on-topic security answer could be refused on the model's say-so; `off_topic` only needed
   `relevance ≤ 20` on a possibly truncated fragment.
3. **Fragment judged alone**: combined with 5.1, a mid-answer fragment such as "why we use SQL and NoSQL is important" had
   little lexical overlap with the question and a low relevance score.
4. Not causes (checked): no system prompt/LLM in the session; question metadata (`expected_topics`) is correct; the judge
   already receives the current question text and expected topics.

### 5.3 Question exposure (P0/P1)
* `get_interview` returned `interview.plan` verbatim with no authentication (it is the endpoint the voice agent *and* the
  candidate page both used). `create_interview` returned `"plan": plan` unfiltered.
* `static/room.html` read `data.plan.questions` to show a question count — i.e. the page depended on the leak.
* The `url / token / roomName / interviewId / questions[...]` bundle described in the issue is produced by the platform
  backend that wraps this service; this service now returns a plan with only `id, type, question_text, topic,
  time_limit`, so a backend that forwards the create response can no longer leak rubric fields.

### 5.4 Completion / report disclosure (P0)
* `GET /v1/interviews/{id}/report` had no `Depends(require_service_key)` and no ownership check.
* `static/room.html` `showCompletionAndReport()` polled that endpoint and called `renderReport()`, which rendered
  `recommendation`, `key_strengths`, `key_gaps`, competency scores and per-question evidence to the candidate.
* The completion endpoint (`POST /complete`) was already minimal (`{report_id, status}`) and idempotent.
* Related: `POST /v1/interviews/{id}/language` was open — anyone holding an id could change the language of a live
  session. Interview ids (`intv_` + 12 hex) act as bearer capabilities; before the fix, possession of one was enough to
  read the HR report.

### 5.5 State / idempotency gaps (P2)
* `_advance_and_ask` advanced the question while the answer/coverage writes were still fire-and-forget tasks.
* `POST /turns` accepted exact duplicates (see §4).
* `AGENT_SERVICE_KEY` unset ⇒ every privileged endpoint open; nothing prevented that in production.

## 6. Before / after API responses (synthetic data)

`GET /v1/interviews/intv_demo` (no credentials)

Before:
```json
{"id":"intv_demo","status":"IN_PROGRESS","room_name":"interview-intv_demo","candidate":"Asha","role":"Backend Engineer",
 "language":"en","coverage_threshold":0.7,"max_followups_per_question":2,"generation_id":0,"duration_minutes":30,
 "voice":{"gender":"female","speaker":null,"pace":null},"voice_provider":"sarvam","progress":{"question_index":2},
 "plan":{"questions":[{"id":"q1","type":"scenario","question_text":"How does a HashMap work?","topic":"hashmaps",
   "competencies":["technical_depth"],"expected_topics":["hashing","buckets"],"followup_topics":["load factor"],
   "focus":"Technical Depth","focus_priority":"HIGH","difficulty":"hard","time_limit":120}],
   "focus_areas":[{"name":"Technical Depth","priority":"HIGH"}],"hr_guidance":["probe depth"]}}
```
After:
```json
{"id":"intv_demo","status":"in_progress","role":"Backend Engineer","language":"en","duration_minutes":30,"question_count":1}
```
`GET /v1/interviews/intv_demo/agent-context` — **401** without `X-Agent-Key`; with it, the full former payload.

`GET /v1/interviews/intv_demo/report` (no credentials)

Before: `200` with `final_score`, `competency_scores`, `content.overall.{recommendation,key_strengths,key_gaps}`,
`content.questions[].{content_score,communication_score,candidate_answer}`.
After: `401 {"detail":"missing or invalid X-Agent-Key"}`.

Candidate completion receipt (new): `GET /v1/interviews/intv_demo/completion`
```json
{"interview_id":"intv_demo","submitted":true,
 "message":"Thank you. Your interview has been submitted and will be reviewed by the hiring team."}
```
`POST /v1/interviews` response `plan.questions[]` now: `{"id","type","question_text","topic","time_limit"}` only.

## 7. Fixes implemented

| Area | Change | Files |
|---|---|---|
| End-of-turn | `TURN_MIN_DELAY_S` (default **2.0 s**) and `TURN_MAX_DELAY_S` (default **6.0 s**) are env-adjustable; `max_delay` is never below `min_delay`; the stale comment is corrected. Candidate can still end instantly with "Done speaking". | `agent.py:75-85` |
| Fragments | The judge receives everything said to the current question (`state.answer_parts`, capped at 6 000 chars); the timer is untouched. | `conductor.py:394`, `policy.py:253` |
| Duplicate events | A second end-of-turn with the same words **while the first is queued or running** is ignored; `/turns` is idempotent within 4 s. | `conductor.py:159`, `api.py:542` |
| Guard precision | Request patterns need request framing; `system prompt` needs "your/what is"; `you are now …` needs an identity noun; `imagine/pretend` only with *you*; "once more/again" only as a request. | `interaction_guard.py` |
| Context-aware guard | `classify(text, context=question…)`; a long, answer-shaped utterance that shares vocabulary with the question skips request-style rules; injection-like wording inside such an answer is **evaluated but flagged** `suspicious`. Genuine off-topic / manipulation still redirects. | `interaction_guard.py:313-359` |
| Judge override | 35-word cap for **all** intents; a substantive on-topic answer is never refused on the model's say-so. | `policy.py:220` |
| Progression | The conductor flushes pending writes (≤2 s) before advancing; transitions are logged. | `conductor.py:492` |
| Diagnostics | Structured `event=` lines: `turn_received, turn_classified, answer_finalized, question_transition, turn_duplicate_ignored, interruption`. Ids, counts, decisions only — **no transcript text, no tokens**. | `conductor.py:117`, `agent.py:185` |
| Candidate/HR split | Public `GET /v1/interviews/{id}` DTO; `GET …/agent-context` (service key); sanitised create response; `GET …/completion`; `GET …/report` requires the key; language change gated after start; constant-time key compare. | `api.py:47-82,243,324,339,432,804,817` |
| Candidate page | No plan/report access; shows a submission receipt only. | `static/room.html:231,327` |
| Fail closed | Process refuses to start in production (`APP_ENV=production` or `RENDER`) without `AGENT_SERVICE_KEY`; logs a warning when unset. | `api.py:47-58` |
| Agent | Loads the plan from `/agent-context` with the key. | `agent.py:244` |

Defaults trade latency for safety: replies after a complete-sounding sentence now start ≥2 s after the candidate stops.
Lower `TURN_MIN_DELAY_S` (e.g. 1.2) if that feels slow; do not go below ~1.0 and keep the 1–2 s tolerance requirement in mind.

## 8. Tests executed

```
python -m pytest -q          →  648 passed   (592 pre-existing, adjusted where they pinned the defects; 56 new)
node --check (room.html script) → syntax OK
```
New file `test_turn_and_api_security.py` (56 tests) covers: 9 relevant technical answers incl. keyword-free ones;
genuine off-topic / injection / hint / repeat still redirected; judge cannot overrule relevant answers; fragment
accumulation; timer not reset; duplicate end-of-turn → one judgement, one advance; answer saved before advance;
partially relevant → clarification; silence; repeat request; diagnostics carry no candidate words; candidate DTO has no
rubric fields; agent-context/report need the key; wrong/missing key handled without echo; another interview's data not
reachable without the key; completion receipt has nothing evaluative and is idempotent; language gating; turn
idempotency; production refuses to start without the key; candidate page never requests `/report`.
Pre-existing tests changed: `test_runner_and_agent` (encoded the old 0.4 s value), and five API tests that read internal
fields from the public endpoint now use `/agent-context`.

Live smoke check on the running local stack confirmed the new public and agent-context payloads and worker registration.

**Could not be verified:** real microphone audio end-to-end (the sandboxed browser has no mic) and ElevenLabs TTS
(the configured key lacks the `text_to_speech` permission, returned `401 missing_permissions`). The turn-timing change is
verified at configuration/logic level and against SDK source, not by live speech.

## 9. Remaining uncertainties and follow-ups
1. **Platform backend** (not in this repo) must be updated: use `/agent-context` (or its own copy) wherever it needs the
   full plan, and call `/report` with the service key only after authorising the HR user. Any frontend that previously
   read `plan.expected_topics` from the create/get response will now see them missing — by design.
2. **Tenant/organisation isolation** cannot be enforced here: this service has no org, user or role model, only the
   shared service key and capability-style ids. Cross-organisation checks must live in the backend that holds the key.
3. In local dev with `AGENT_SERVICE_KEY` unset, privileged endpoints remain open (only the browser pages stopped using
   them). Set the key for any shared environment — production now enforces it.
4. Tuning the 2.0 s / 6.0 s defaults needs real interview audio; consider `endpointing.mode = "dynamic"` after measuring.
5. A reconnect restarts the current question's timer (`runner.restore`); abandoned `IN_PROGRESS` sessions have no expiry.
6. `hr.html` (local HR console) calls privileged endpoints without a key — fine for dev, it needs a key field before
   being used against a secured deployment.
7. Rotate any credentials that were pasted into chat during setup (OpenAI, LiveKit secret, ElevenLabs).

---

# Addendum A — Evaluation response investigation

Subject: a completion/evaluation payload (`status: READY`, `final_score: 5.6`, competencies 7.5 / 15, `no_hire`,
Technical Depth `PARTIAL` with one question, Problem Solving `NOT_ASKED`, a zero-scored answer "Can I speak in Hindi?").
Method: each discrepancy was treated as a hypothesis and checked against code and stored records. Every finding below
names the file, the evidence and the regression test.

## A.0 Evidence limitations (read first)
* **The exact 5.6 record is not in the local database.** `SELECT … WHERE round(final_score,1)=5.6` returned no rows; the
  payload was produced by another agent instance. The two nearest local reports (`intv_d68a04b6c7ec` = 5.5,
  `intv_f35ebfd608a1` = 5.4) show the same mechanisms and were used as evidence. Nothing about the 5.6 record's own
  rows is claimed.
* The platform's MongoDB (drives/candidates, remote Atlas) was only queried for the candidate's drive list; no
  evaluation records were read from it. Tenant isolation is verified from code, not from data.
* No live audio or LLM calls were made; scorer outputs are faked at the LLM edge in tests.

## A.1 Which endpoint returned this payload, and to whom (question 5)
The payload's envelope `{success, data:{status, final_score, competency_scores, content, error}, meta:{requestId}}` is
the platform's `ok()` wrapper around the agent's report. In code that is exactly
`POST /api/v1/candidate/me/interviews/:id/rounds/:n/agent-complete`
(`interviewIQ/services/client-service/controllers/interviewDrive.controller.js`, previously `ok(res, report)`), called by
the **candidate's own browser** (`InterviewRoomPage.jsx` `finishInterview`, which ignores the response body).
**Confirmed: the full HR report was returned to the candidate.**

Authorization review of the HR side (read from code):

| Check | Result |
|---|---|
| HR transcript/report/drive routes | `authenticate` + `requirePermission("CLIENT_SELF_READ")` (`interviewDrive.routes.js:20-37`) |
| Tenant isolation | every lookup is `InterviewDrive.findOne({_id, tenantId})` with `tenantId` from the verified token (`getCandidateTranscript`, `getDriveById`) |
| Candidate ownership | `findOwnRosterEntry` matches the roster entry by **token email + tenant**, never a client-supplied id |
| Candidate role | `CANDIDATE` only; not granted `CLIENT_SELF_READ`, so HR routes return 403 |
| Candidate list DTO | picks fields explicitly (no `agentReport`); **does include the candidate's own `aiScore`** — left unchanged, flagged |
| Agent `GET /report` | was unauthenticated; now requires the service key (Addendum, §5.4) |
| Duplicate/unauthorised completion | `/complete` idempotent (one report row); response now a receipt |

Fix: `agent-complete` returns `candidateCompletionReceipt()` = `{submitted:true}`; the report is still persisted to
`candidate.agentReport` for HR (`interviewDrive.service.js:954`). Tests: `candidateQuestions.test.js` (receipt, safe
questions); agent side `test_a_candidate_cannot_obtain_any_hr_only_evaluation_data`.

## A.2 Coverage mismatch: Problem Solving NOT_ASKED (question 1)
**Hypotheses:** (a) dynamic questions excluded, (b) question ids mapped wrongly, (c) counters updated inconsistently.
**Verdict:** (a) and (c); (b) ruled out.
* `intelligence_report.build_intelligence` related questions to a focus area with `q.get("focus") == area.name`. `focus`
  is a free-text tag set by lexical overlap at plan time. Dynamic (`q_dynamic_*`) and resume questions carry
  `competencies` but **no `focus` tag**, so they were invisible to coverage. In the stored plans examined
  (`intv_d68a…`, `intv_f35e…`) **all** questions have `focus = None` and no `focus_areas`.
* The one question that *was* tagged was the behavioural "…technical decision" question (shared word), which is why
  Technical Depth showed 1 asked / 1 answered with evidence "Every time I disagree with my team members…".
* `questions_asked` was `len(related)` — **planned**, not asked — so the counters did not mean what their names said.
* Question ids are consistent: turns, coverage rows and report entries all use the plan's `id` (verified in the stored
  rows), so mapping (b) is not the cause.

Reproduction (before the fix, intro + 2 dynamic + behavioural + HR question; 4 answered, 3 declaring `problem_solving`):
```
Technical Depth  PARTIAL     asked=1 answered=1 score=15.0
Problem Solving  NOT_ASKED   asked=0 answered=0 score=None     <-- three answered questions declare it
unasked_high_priority: ['Problem Solving']
```
Fix (`intelligence_report.py:62` `serves_focus`, `:76` `build_intelligence`; `evaluation.py:52` `asked_question_ids`):
a question serves an area when tagged **or** its `competencies` contain the area; counters are now
`questions_planned / questions_asked / questions_answered` with `answered ≤ asked ≤ planned`; "asked" comes from what the
interviewer actually put to the candidate (agent `NEXT_QUESTION/FOLLOW_UP/CLARIFICATION/REPEAT/LANGUAGE_SWITCH` turns or
any candidate turn); asked-but-unanswered is `NOT_ANSWERED` (not `NOT_ASKED`); introductions never count.
Plan-time tagging was already fixed to be competency-first (`focus.focus_for_question`).
Tests: `test_a_competency_tagged_dynamic_question_counts_toward_its_focus_area`,
`test_counts_are_planned_asked_answered_and_consistent_with_each_other`,
`test_a_planned_question_that_was_never_asked_is_not_counted_as_asked`,
`test_asked_but_unanswered_is_not_answered_rather_than_not_asked`,
`test_plans_stored_without_focus_tags_still_map_by_competency`,
`test_the_introduction_never_counts_toward_a_focus_area_or_the_answered_total`.

## A.3 "Can I speak in Hindi?" scored as a technical answer (question 2)
* Support exists: `language_switch.detect_language_request` → `conductor._switch_language` confirms, retunes STT/TTS,
  persists the language and asks the **same** question again (`idx` unchanged).
* Defect 1: the turn was stored with intent `"language_request"`, a string absent from `NON_ANSWER_INTENTS`, so the
  scoring job treated it as an answer (score 0). Fixed earlier: `Intent.LANGUAGE_REQUEST` is a non-answer intent.
* Defect 2 (this addendum): the scorer now also never builds an answer from non-answer turns *and* lists the question as
  `NOT_ANSWERED` instead of fabricating a score (`evaluation.is_answer_turn`).
* Defect 3: a request for the language **already in use** fell through to the answer path; now routed to `REPEAT` of the
  current question (`conductor._handle`).
* Not a cause: the agent did **not** advance on a recognised request — `_switch_language` never calls `advance()`
  (asserted by the tests below). The two stored local interviews examined contain no language-switch turns, so this is
  established from code and tests, not from stored data.
Tests: `test_a_language_request_on_a_technical_question_repeats_that_question_and_does_not_advance` (3 phrasings; asserts
`idx`, follow-up count and `started_at` unchanged, no judge call, no coverage row, not in `answer_parts`),
`test_a_request_for_the_language_already_in_use_repeats_the_question_too`,
`test_a_language_request_is_never_an_answer_in_the_evaluation_either`.

## A.4 Scoring pipeline: scales, weights, aggregation (question 3)
Documented from `scoring.py`, `evaluation.py` and the stored `reports` rows.

| Step | Rule | Scale |
|---|---|---|
| 1. Answer text | candidate turns for the question with an answer intent, de-duplicated, joined | — |
| 2. Content score | LLM judge per question → `content_score.score` (+ dimensions) | **0–100** |
| 3. Communication score | LLM per answered question → `communication_score.score` | **0–100 (was unstated; model used 0–10)** |
| 4. Per-competency score | mean of the content scores of questions whose `competencies` contain it; mean of communication scores → `communication` | 0–100 |
| 5. Unasked competency | absent from `competency_scores`, excluded from the final | — |
| 6. Asked, not answered | counted as **0** for its competencies; flagged `NOT_ANSWERED` | 0 |
| 7. Final score | `Σ(score_c × weight_c) / Σ(weight_c over assessed c)` (weights from the role, not hard-coded) | 0–100 |
| 8. Assessed share | `Σ weight(assessed) / Σ weight(all)` reported alongside | 0–1 |

Reconstruction of stored report `intv_f35ebfd608a1` (5.4): question scores intro 0, resume 15, dynamic 0 and 8,
behavioural 12 ⇒ technical_depth = problem_solving = mean(15, 0, 8) = 7.67, ownership = 12 ✓ (matches stored
`competency_scores`). Old formula `7.67×.30 + 7.67×.25 + 12×.10 = 5.4` divides by the **full** weight (1.0), i.e. the two
unassessed competencies (practical_application .20, communication .15) were silently scored 0 → **5.4**. Normalised over
assessed weight (.65) the same evidence gives **8.3**; the 35% unassessed share is surfaced, not buried.

Defects found and fixed:
1. **Unassessed competencies counted as zero** in the final (fixed in the previous round; documented here).
2. **Communication never reached the role's `communication` competency** — scores were computed per answer and then
   unused, so that 15% weight could never be assessed. Now wired in (`evaluation.collect_question_results`).
3. **Inconsistent scale**: content 0–100, communication returned 0–10 (e.g. `score 3, dims 2/2/2`; `clarity 8` for a clear
   sentence). The prompt now states 0–100 with anchors; values are clamped to [0, 100]. Older stored reports keep 0–10
   communication scores (no retroactive rescaling, to avoid mis-scaling genuine low scores).
4. **Asked-but-unanswered questions were dropped**, letting a candidate raise their average by skipping — now 0.
5. **Competencies not in the role** (e.g. `collaboration`) are ignored by design; they appear in `competency_scores` but
   carry no weight.
Tests: `test_scores_are_clamped_to_0_100_and_communication_feeds_the_communication_competency`,
`test_the_reported_numbers_reproduce_and_the_normalisation_is_documented_by_the_test`,
`test_unasked_competencies_are_excluded_and_asked_unanswered_ones_count_as_zero`,
`test_skipping_hard_questions_cannot_raise_the_average`, `test_an_incomplete_interview_is_flagged_not_presented_as_complete`,
`test_the_communication_prompt_states_the_scale`.

## A.5 `status: READY` with a NOT_ASKED high-priority area (question 4)
`READY` is `Report.status` (`db.py` `REPORT_PENDING/PROCESSING/READY/FAILED`): **report-generation** state. The interview's
own state is `Interview.status` (`COMPLETED`). The payload exposed only the first, so READY read as "evaluation complete".
Distinct fields now (`api.py:833`, `scoring.evaluation_status`):

| Field | Meaning | Values |
|---|---|---|
| `report_status` (alias `status`, kept for compatibility) | report generation | PENDING, PROCESSING, READY, FAILED |
| `interview_status` | the interview session | … FINALIZING, COMPLETED, TERMINATED, FAILED, EXPIRED |
| `evaluation_status` / `content.evaluation.status` | how much evidence the evaluation rests on | COMPLETE, PARTIAL, INSUFFICIENT_EVIDENCE |

`content.evaluation` also carries `reasons[]`, `answered_questions`, `asked_questions`, `assessed_weight_share`,
`ended_early`. A READY report can legitimately be INSUFFICIENT_EVIDENCE. Test:
`test_report_status_interview_status_and_evaluation_status_are_distinct_fields`.

## A.6 Exactly-once answers, duplicate completion, correct ids (question 6)
* Stored data: 5 groups of identical `interview_turns` rows exist (agent lines on resume/reconnect, one candidate
  non-answer). `_run_scoring_job` joined every matching row, so a re-delivered **answer** line would have been scored
  twice. Fixed: consecutive identical candidate lines per question are dropped (`evaluation.answer_text_for`);
  `POST /turns` is idempotent within 4 s (earlier round).
* Duplicate completion: `/complete` returns the existing report; one `reports` row per interview; scores and content are
  byte-identical on re-read (test below). Scoring runs once per report row.
* Correct ids: the job loads `Interview` by id, then **that** interview's `plan`, `role.competencies` and
  `candidate.name`; turns are filtered by `interview_id`. A second interview's answers never appear
  (`test_the_evaluation_uses_this_interviews_own_turns_plan_and_candidate_only`). The agent service holds no
  organisation id; the platform scopes by `tenantId` and resolves the agent interview id from the caller's own roster
  entry (A.1). Cross-organisation isolation therefore rests on the platform, which was reviewed in code only.
* Introduction rows are reported but excluded from `answered_questions` (they previously inflated the count used by the
  recommendation guard).
Tests: `test_a_redelivered_answer_line_is_scored_once`, `test_two_distinct_answers_to_the_same_question_are_both_kept_in_order`,
`test_non_answer_turns_never_reach_the_scorer`, `test_repeated_completion_does_not_change_scores_or_create_another_report`.

## A.7 Test run
```
workmate-iq-agent:  python -m pytest -q   -> 673 passed   (24 new in test_evaluation_pipeline.py)
client-service:     node --test tests/*.test.js -> 32 passed (3 new in candidateQuestions.test.js)
```
Test-suite hygiene fix: `conftest.py` now neutralises `AGENT_SERVICE_KEY` per test (it was being inherited from the
developer's `.env`, which caused 58 spurious 401 failures once the key was shared with the backend).

## A.8 Still open
* The stored 5.6 record could not be examined (A.0); if it exists in another agent database, re-run the A.2 and A.4
  queries there.
* Old reports keep 0–10 communication scores and the old (deflated) final score; re-scoring them is a deliberate,
  separate step.
* `getMyInterviews` still returns the candidate's own `aiScore`; confirm that is intended.
* Platform tenant isolation was reviewed in code, not tested end to end.
