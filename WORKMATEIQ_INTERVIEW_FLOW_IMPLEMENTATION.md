# WorkmateIQ interview agent - structured opening and personalised questions

## 1. What an interview now looks like

| # | Stage (`interview_flow.py`) | What the agent does | Moves on when |
|---|---|---|---|
| 1 | `INTRODUCTION` | Says the candidate's name, its own configured name, the role, and - only if verified against the resume text - a project/skill, an employer and the years of experience. Ends by asking whether anything needs clarifying. | the introduction has been spoken |
| 2 | `CLARIFICATION` | Answers each doubt only from approved facts (duration, format, role title, how results are communicated). Anything else: "I can't confirm that", never a guess. After every answer asks "any other questions or doubts?". | the candidate says there are no more |
| 3 | `READINESS_CONFIRMATION` | "Is everything clear, and are you ready to begin?" A question puts it back to stage 2; "not yet" gets "need a moment, or something else to ask?" | an explicit yes. Silence, noise or an unclear reply never counts as readiness |
| 4 | `RESUME_QUESTIONS` | One or two questions taken from the resume (existing claim-validation questions), one at a time, at most one follow-up each. No resume: the single generic background question. | the planned question(s) are answered or skipped |
| 5 | `ENVIRONMENT_CHECKS` | The existing "quiet place / can you hear me" check (it used to be a line inside the greeting turns). | a clear yes; after `MAX_ENV_RETRIES` "no" answers it continues and logs `environment_check_unresolved` (the old flow continued after any reply) |
| 6-7 | `MAIN_INTERVIEW` | Existing question / follow-up policy, unchanged. | plan finished |
| 8-9 | completion | Existing `/complete` -> scoring -> report. Unchanged. | |

Stage and readiness are persisted with the existing `progress` record (`stage`, `readiness`), only ever move forward, and are
restored by a restarted worker, which continues where it was and says "welcome back" instead of introducing itself again.
Progress saved before this change (no `stage`) resumes in `MAIN_INTERVIEW`.

Silence during the opening: one gentle re-ask, never an advance. Interrupted/ended interviews are not marked completed:
the existing rules are untouched (closing the browser only ends the session; `/complete` is the only completion path,
is idempotent, and creates one report).

## 2. Root causes found

* **Everyone got the same questions.** `api.create_interview` generated the role's scenario questions once and cached them
  on the role for every candidate ("standardised for fairness"); the question bank was ranked deterministically per role.
  Only the single resume question differed. No seed or caching bug - it was designed that way.
* **Greeting was generic and the "environment check" was a side effect.** The greeting never used the resume; the quiet-place
  question lived in the second small-talk line, so it could not be ordered after resume questions.
* **A restart repeated the introduction.** Progress at question 0 was treated as "barely started" and discarded.

## 3. What changed

| File | Change |
|---|---|
| `interview_flow.py` (new) | stage model, plan ordering (`arrange_questions`), resume-fact verification (`profile_facts`), reply interpretation (`interpret`), clarification topics |
| `refusals.py` | fixed lines in English / Hindi / Hinglish for every stage, `introduction()` builder (also added to the TTS cache list) |
| `runner.py` | stage state, plan arrangement at start, restore of stage, opening-question follow-up cap |
| `conductor.py` | `_preflight_turn` and stage handlers replace the old small-talk turns; environment check inserted before the first main question; silence nudge; language switch during the opening |
| `api.py` | `/progress` stores `stage`/`readiness` (monotonic); `agent-context` carries `candidate_profile` (verified facts only); per-interview question generation with history and fallback |
| `planner.py` | `generate_personalized_questions`, bank rotation (`avoid_ids`), duplicate-aware plan building |
| `agent.py` | passes `candidate_profile` to the runner |
| `db.py`, `conftest.py` | comment on the role cache; hermetic stub for the new LLM call |
| tests | `test_structured_opening.py` (69), `test_question_personalization.py` (22); 4 old small-talk assertions in `test_conductor.py` and one progress assertion rewritten for the new flow |

Configuration: `PERSONALIZED_QUESTIONS` (default on), `QUESTION_HISTORY_INTERVIEWS` (default 3), `MAX_VALIDATION_QUESTIONS`
(resume questions available, default 2). `small_talk_rounds=0` still means "no opening conversation" (used by tests).

## 4. Question personalisation and isolation

* Each interview gets a fresh set from `generate_personalized_questions` (role, experience level, this candidate's own
  evidence, same competencies and difficulty rule as everyone else, the recent questions as "do not repeat"). Output is
  validated: length, prompt-injection check, near-duplicate check against recent and sibling questions.
* Repeat control: questions from the last `QUESTION_HISTORY_INTERVIEWS` interviews of the same candidate record (or the same
  email on the same role) are passed to the generator and excluded from bank selection (topped up from them only if the
  pool is too small). HR-written questions are intentionally asked verbatim every time.
* Failure: logged, and the interview uses the **role-level** set. A personalised set is stored only in its own interview
  plan; it is never written to the role cache, so it cannot reach another candidate.
* The agent receives, per interview, only `candidate_profile`: items that literally appear in that candidate's resume text.
  The resume itself is never sent; the candidate-facing `GET /v1/interviews/{id}` carries none of it.

## 5. Verified

`pytest`: 882 passed (791 before this work). New tests cover the stage order, clarification loop and its limits, readiness
never inferred, resume questions before the environment check, follow-up cap, environment retry limit, restart at each
stage, persistence ordering, language switch mid-opening, introduction wording with and without facts in all three
languages, fact verification (including injection), personalised questions differing per resume, fallback, repeat
avoidance, no leakage between candidates, and stage progress being monotonic.

## 6. Limitations - not verified or not done

* **No live voice test.** Nothing here was run against a real LiveKit room, STT or TTS. The wording of the new Hindi/Hinglish
  lines has not been reviewed by a native speaker.
* **Reply interpretation is rule-based.** It handles common English/Hinglish phrasings; an unusual reply is treated as unclear
  (asked once more) or as a doubt, never as readiness.
* **Candidate-stated experience is not sent to the agent**, so "around N years" is spoken only if the resume says it.
* **Resume parsing of `.docx`** files is unchanged (the agent decodes them as text), so those resumes often yield no facts.
* **Question diversity is bounded by the question bank** for a given role: when its pool is small, recent bank questions repeat.
* **History is per candidate record / email on the same role.** The same person under a different email is not recognised.
* A candidate who says "no" to the environment check three times continues anyway (as before, after any reply).
