import json
import logging
import os
import subprocess
import sys
import uuid

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Header, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from livekit import api as lk_api

from config import LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, AGENT_SERVICE_KEY

logger = logging.getLogger("workmate-api")

# Must match agent.py's WorkerOptions(agent_name=...) — LiveKit Cloud requires explicit
# dispatch (rather than implicit any-worker dispatch) for projects with the Agents feature
# enabled, otherwise the worker never receives a job for the room.
AGENT_NAME = "workmate-interviewer"
import states
from db import (
    init_db, get_db, SessionLocal, Role, Candidate, Interview, InterviewTurn, Report, CoverageResult,
    IntegrityEvent, transition_interview, ConcurrentTransitionError,
    REPORT_PENDING, REPORT_PROCESSING, REPORT_READY, REPORT_FAILED,
)
from resume_parser import extract_text_from_pdf, parse_resume
from planner import build_interview_plan, generate_dynamic_questions, warm_question_bank_embeddings
from scoring import score_answer_content, score_communication, aggregate_final_score, generate_overall_summary
from roles import competencies_for_role, detect_role_type
from interaction_guard import NON_ANSWER_INTENTS
from voice_config import VALID_V3_SPEAKERS, clamp_pace, resolve_voice

init_db()
app = FastAPI(title="Workmate.IQ Interview Agent (MVP)")
app.mount("/static", StaticFiles(directory="static"), name="static")


def require_service_key(x_agent_key: str | None = Header(default=None)):
    """Service-to-service auth for the agent API (plan §10 — 'no authentication on the agent
    API' was flagged as a pre-public-exposure blocker).

    Enabled only when AGENT_SERVICE_KEY is configured — this keeps local/dev setups that
    haven't set it working unchanged, but any deployment that sets the env var gets every
    mutating/internal endpoint locked down. Set AGENT_SERVICE_KEY before exposing this service
    beyond localhost.
    """
    if not AGENT_SERVICE_KEY:
        return
    if x_agent_key != AGENT_SERVICE_KEY:
        raise HTTPException(401, "missing or invalid X-Agent-Key")

_agent_process: subprocess.Popen | None = None


@app.on_event("startup")
def _start_agent_worker():
    """Run the LiveKit voice agent as a background subprocess of this same web process.

    Render's (and most PaaS) free tier only runs "web" service types (ones that bind to a
    port) — a separate background worker requires a paid plan. Since a free web service's
    container can run whatever additional processes it wants internally, spawning the agent
    worker here means the whole app (API + voice agent) fits inside one free web service.
    Guarded by RUN_AGENT_INLINE so local dev can keep running `python agent.py dev` separately
    (the default local flow) without spawning a duplicate worker.
    """
    global _agent_process
    if os.getenv("RUN_AGENT_INLINE", "false").lower() != "true":
        return
    logger.info("Starting LiveKit agent worker as a subprocess (RUN_AGENT_INLINE=true)")
    _agent_process = subprocess.Popen([sys.executable, "agent.py", "start"])


@app.on_event("startup")
def _warm_embeddings():
    """Embed the question bank in the background so the first interview created after a restart
    doesn't wait on the embedding provider (it took 15-24s cold, past the web app's request timeout)."""
    import threading
    threading.Thread(target=warm_question_bank_embeddings, name="warm-embeddings", daemon=True).start()


@app.on_event("shutdown")
def _stop_agent_worker():
    if _agent_process is not None and _agent_process.poll() is None:
        _agent_process.terminate()


@app.get("/")
def index():
    return FileResponse("static/hr.html")


@app.post("/v1/roles", dependencies=[Depends(require_service_key)])
def create_role(name: str = Form(...), db: Session = Depends(get_db)):
    # Competencies (and, via planner.py, which question-bank questions are eligible) are
    # derived from the role name — "Frontend Developer" and "Product Manager" get different
    # interviews, not the same generic backend-flavored one.
    role = Role(name=name, competencies=competencies_for_role(name))
    db.add(role)
    db.commit()
    return {"id": role.id, "name": role.name, "competencies": role.competencies}


@app.post("/v1/candidates", dependencies=[Depends(require_service_key)])
def create_candidate(name: str = Form(...), email: str = Form(""), db: Session = Depends(get_db)):
    cand = Candidate(name=name, email=email)
    db.add(cand)
    db.commit()
    return {"id": cand.id, "name": cand.name}


@app.post("/v1/candidates/{candidate_id}/resume", dependencies=[Depends(require_service_key)])
async def upload_resume(candidate_id: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    cand = db.get(Candidate, candidate_id)
    if not cand:
        raise HTTPException(404, "candidate not found")
    raw = await file.read()
    text = extract_text_from_pdf(raw) if file.filename.lower().endswith(".pdf") else raw.decode("utf-8", "ignore")
    cand.resume_text = text
    cand.evidence_profile = parse_resume(text)
    db.commit()
    return {"id": cand.id, "evidence_profile": cand.evidence_profile}


MAX_HR_QUESTIONS = 20


def _parse_hr_questions(raw: str, role_competencies: list[dict]) -> list[dict]:
    """HR-configured questions arrive as a JSON list of {id?, text, topic?, time_limit?}. They are
    validated and normalised into plan questions here; a malformed payload is rejected outright
    rather than silently dropped, since HR-defined questions must be respected."""
    if not raw or not raw.strip():
        return []
    try:
        items = json.loads(raw)
    except ValueError:
        raise HTTPException(422, "hr_questions must be a JSON list")
    if not isinstance(items, list) or len(items) > MAX_HR_QUESTIONS:
        raise HTTPException(422, f"hr_questions must be a list of at most {MAX_HR_QUESTIONS} questions")
    top = [c["key"] for c in sorted(role_competencies or [], key=lambda c: -c.get("weight", 0))[:2]] \
        or ["practical_application", "problem_solving"]
    questions = []
    for i, item in enumerate(items, 1):
        text = str((item or {}).get("text") or "").strip() if isinstance(item, dict) else ""
        if not text or len(text) > 1500:
            raise HTTPException(422, f"hr_questions[{i - 1}] needs non-empty text of at most 1500 characters")
        try:
            time_limit = int(item.get("time_limit") or item.get("timeLimit") or 0)
        except (TypeError, ValueError):
            time_limit = 0
        questions.append({
            "id": f"hr_{i}", "type": "hr", "difficulty": "medium", "competencies": top,
            "question_text": text, "topic": str(item.get("topic") or "").strip()[:80],
            "expected_topics": [], "followup_topics": [],
            **({"time_limit": max(30, min(600, time_limit))} if time_limit else {}),
        })
    return questions


def _validated_voice(gender: str, speaker: str, pace: float | None) -> dict:
    gender = (gender or "").strip().lower()
    speaker = (speaker or "").strip().lower()
    if gender and gender not in ("male", "female"):
        raise HTTPException(422, "voice_gender must be 'male' or 'female'")
    if speaker and speaker not in VALID_V3_SPEAKERS:
        raise HTTPException(422, f"voice_speaker must be one of {sorted(VALID_V3_SPEAKERS)}")
    return {"voice_gender": gender or None, "voice_speaker": speaker or None,
            "voice_pace": clamp_pace(pace) if pace is not None else None}


@app.post("/v1/interviews", dependencies=[Depends(require_service_key)])
def create_interview(candidate_id: str = Form(...), role_id: str = Form(...),
                      duration_minutes: int = Form(30),
                      hr_questions: str = Form(""), questions_json: str = Form(""),
                      experience_level: str = Form(""),
                      max_followups_per_question: int | None = Form(None),
                      coverage_threshold: float | None = Form(None),
                      depth_probe_enabled: bool = Form(True),
                      voice_gender: str = Form(""), voice_speaker: str = Form(""),
                      voice_pace: float | None = Form(None),
                      db: Session = Depends(get_db)):
    cand = db.get(Candidate, candidate_id)
    role = db.get(Role, role_id)
    if not cand or not role:
        raise HTTPException(404, "candidate or role not found")

    hr_question_list = _parse_hr_questions(hr_questions or questions_json, role.competencies)
    if max_followups_per_question is not None and not 0 <= max_followups_per_question <= 5:
        raise HTTPException(422, "max_followups_per_question must be between 0 and 5")
    if coverage_threshold is not None and not 0.3 <= coverage_threshold <= 0.95:
        raise HTTPException(422, "coverage_threshold must be between 0.3 and 0.95")
    voice_fields = _validated_voice(voice_gender, voice_speaker, voice_pace)

    # Generate this role's scenario questions once and cache them, so every candidate who
    # interviews for the same role faces the same questions — standardized questions are a
    # core fairness practice for comparing candidates against each other on equal footing.
    if role.generated_questions is None:
        try:
            role.generated_questions = generate_dynamic_questions(role.name, role.competencies, count=2)
        except Exception:
            role.generated_questions = []
        db.commit()

    plan = build_interview_plan(
        role_competencies=role.competencies,
        role_name=role.name,
        evidence_profile=cand.evidence_profile or {},
        min_questions=4,
        max_questions=7,
        cached_dynamic_questions=role.generated_questions,
        hr_questions=hr_question_list,
    )

    extra = {}
    if max_followups_per_question is not None:
        extra["max_followups_per_question"] = max_followups_per_question
    if coverage_threshold is not None:
        extra["coverage_threshold"] = coverage_threshold
    interview = Interview(
        candidate_id=candidate_id,
        role_id=role_id,
        duration_minutes=duration_minutes,
        plan=plan,
        status=states.CREATED,
        experience_level=(experience_level.strip()[:80] or None),
        depth_probe_enabled=int(depth_probe_enabled),
        **voice_fields,
        **extra,
    )
    db.add(interview)
    db.commit()
    interview.room_name = f"interview-{interview.id}"
    db.commit()
    transition_interview(db, interview, states.PLANNED, reason="plan generated")
    return {"id": interview.id, "room_name": interview.room_name, "plan": plan}


@app.get("/v1/interviews/{interview_id}")
def get_interview(interview_id: str, db: Session = Depends(get_db)):
    interview = db.get(Interview, interview_id)
    if not interview:
        raise HTTPException(404, "not found")
    return {
        "id": interview.id, "status": interview.status, "room_name": interview.room_name,
        "plan": interview.plan, "candidate": interview.candidate.name, "role": interview.role.name,
        "language": interview.language,
        # Exposed so the agent can wire the HR-configured threshold instead of a hardcoded
        # default (Workmate_production_scalable_fix_plan_v2.md #5), and so it can fence
        # in-flight LLM/TTS results against the authoritative server-side state (#2).
        "coverage_threshold": interview.coverage_threshold,
        "max_followups_per_question": interview.max_followups_per_question,
        "generation_id": interview.generation_id,
        "duration_minutes": interview.duration_minutes,
        "experience_level": interview.experience_level,
        "depth_probe_enabled": bool(interview.depth_probe_enabled if interview.depth_probe_enabled is not None else 1),
        "voice": {"gender": interview.voice_gender, "speaker": interview.voice_speaker, "pace": interview.voice_pace},
        "progress": interview.progress,
    }


SUPPORTED_LANGUAGES = {"en", "hi", "hinglish"}


@app.post("/v1/interviews/{interview_id}/voice", dependencies=[Depends(require_service_key)])
def set_interview_voice(interview_id: str, voice_gender: str = Form(""), voice_speaker: str = Form(""),
                         voice_pace: float | None = Form(None), db: Session = Depends(get_db)):
    """Per-interview voice override (e.g. from HR/admin settings). Empty values fall back to the
    calibrated defaults in voice_config.py. Returns the voice that will actually be used."""
    interview = db.get(Interview, interview_id)
    if not interview:
        raise HTTPException(404, "not found")
    fields = _validated_voice(voice_gender, voice_speaker, voice_pace)
    for name, value in fields.items():
        setattr(interview, name, value)
    db.commit()
    return {"id": interview.id,
            "voice": resolve_voice(interview.language, interview.voice_gender, interview.voice_speaker,
                                    interview.voice_pace).as_dict()}


@app.post("/v1/interviews/{interview_id}/progress", dependencies=[Depends(require_service_key)])
def record_progress(interview_id: str, question_index: int = Form(...), followup_count: int = Form(0),
                     small_talk_done: int = Form(0), phase: str = Form(""), db: Session = Depends(get_db)):
    """The agent reports where the interview is after every transition, so the server holds the
    authoritative progress and a restarted agent worker resumes instead of starting over. Progress
    only ever moves forward: a late write from a stale worker can't rewind a newer one."""
    interview = db.get(Interview, interview_id)
    if not interview:
        raise HTTPException(404, "not found")
    if states.is_terminal(interview.status):
        raise HTTPException(409, "INTERVIEW_ALREADY_COMPLETED")
    current = interview.progress or {}
    prev_q = current.get("question_index", 0)
    stale = current and (
        question_index < prev_q
        or (question_index == prev_q and (small_talk_done < current.get("small_talk_done", 0)
                                          or followup_count < current.get("followup_count", 0)))
    )
    if stale:
        return {"accepted": False, "progress": current}
    interview.progress = {"question_index": max(0, question_index), "followup_count": max(0, followup_count),
                          "small_talk_done": max(0, small_talk_done), "phase": phase[:40]}
    db.commit()
    return {"accepted": True, "progress": interview.progress}


@app.post("/v1/interviews/{interview_id}/tts-metadata", dependencies=[Depends(require_service_key)])
def record_tts_metadata(interview_id: str, metadata: str = Form(...), db: Session = Depends(get_db)):
    """Voice actually used + TTS delivery statistics (retries, reconnects, first-audio latency),
    merged into Interview.voice_used so audio problems are measurable per interview afterwards."""
    interview = db.get(Interview, interview_id)
    if not interview:
        raise HTTPException(404, "not found")
    try:
        data = json.loads(metadata)
    except ValueError:
        raise HTTPException(422, "metadata must be a JSON object")
    if not isinstance(data, dict) or len(metadata) > 4000:
        raise HTTPException(422, "metadata must be a small JSON object")
    interview.voice_used = {**(interview.voice_used or {}), **data}
    db.commit()
    return {"ok": True}


@app.post("/v1/interviews/{interview_id}/language")
def set_interview_language(interview_id: str, language: str = Form(...), db: Session = Depends(get_db)):
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(422, f"language must be one of {sorted(SUPPORTED_LANGUAGES)}")
    interview = db.get(Interview, interview_id)
    if not interview:
        raise HTTPException(404, "not found")
    interview.language = language
    db.commit()
    return {"id": interview.id, "language": interview.language}


def _mint_token(room_name: str, identity: str, name: str) -> str:
    token = lk_api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
    token.with_identity(identity).with_name(name).with_grants(
        lk_api.VideoGrants(room_join=True, room=room_name)
    )
    return token.to_jwt()


async def _agent_already_active(lk, room_name: str) -> bool:
    """True if an interviewer agent is already running (or about to run) in this room.

    Every candidate-token request used to create ANOTHER dispatch, so a page refresh put two
    interviewers in one room - two voices speaking over each other. Checks LiveKit's own dispatch
    state instead of guessing. Fails open: if the state can't be read (room doesn't exist yet, or
    LiveKit is unreachable) it reports False, because a missing interviewer is worse than a rare
    duplicate - and the agent also refuses to run beside another agent (agent.py).
    """
    from livekit.protocol import agent as lk_agent
    try:
        dispatches = await lk.agent_dispatch.list_dispatch(room_name=room_name)
    except Exception:
        return False
    live = (lk_agent.JobStatus.JS_PENDING, lk_agent.JobStatus.JS_RUNNING)
    for dispatch in dispatches:
        if dispatch.agent_name != AGENT_NAME or dispatch.state.deleted_at:
            continue
        jobs = list(dispatch.state.jobs)
        if not jobs or any(job.state.status in live for job in jobs):
            return True
    return False


async def _ensure_agent_dispatched(room_name: str):
    """Explicitly dispatch the interviewer agent to this room, exactly once. Without dispatch, a
    LiveKit Cloud project with the Agents feature enabled never routes a job to the worker (it
    silently sits idle waiting for a job that never comes)."""
    lk = lk_api.LiveKitAPI(LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
    try:
        if await _agent_already_active(lk, room_name):
            logger.info("agent already active in %s - not dispatching a second interviewer", room_name)
            return
        await lk.agent_dispatch.create_dispatch(
            lk_api.CreateAgentDispatchRequest(agent_name=AGENT_NAME, room=room_name)
        )
    except Exception as e:
        logger.warning(f"agent dispatch create failed (may already exist): {e}")
    finally:
        await lk.aclose()


@app.post("/v1/interviews/{interview_id}/candidate-token", dependencies=[Depends(require_service_key)])
async def candidate_token(interview_id: str, db: Session = Depends(get_db)):
    interview = db.get(Interview, interview_id)
    if not interview:
        raise HTTPException(404, "not found")
    if states.is_terminal(interview.status):
        raise HTTPException(409, f"interview already {interview.status}")
    if interview.status == states.PLANNED:
        transition_interview(db, interview, states.READY, reason="candidate token issued")
    identity = f"candidate-{interview.candidate_id}"
    jwt = _mint_token(interview.room_name, identity, interview.candidate.name)
    await _ensure_agent_dispatched(interview.room_name)
    return {"token": jwt, "url": LIVEKIT_URL, "room_name": interview.room_name}


@app.post("/v1/interviews/{interview_id}/start", dependencies=[Depends(require_service_key)])
def start_interview(interview_id: str, db: Session = Depends(get_db)):
    """Called by the agent worker once it actually joins the room and starts the session —
    marks the interview IN_PROGRESS so /complete has a real ACTIVE state to finalize from."""
    interview = db.get(Interview, interview_id)
    if not interview:
        raise HTTPException(404, "not found")
    if interview.status == states.READY:
        transition_interview(db, interview, states.IN_PROGRESS, reason="agent session started")
    return {"id": interview.id, "status": interview.status, "generation_id": interview.generation_id}


@app.post("/v1/interviews/{interview_id}/turns", dependencies=[Depends(require_service_key)])
def record_turn(interview_id: str, question_id: str = Form(...), question_text: str = Form(""),
                 speaker: str = Form(...), text: str = Form(...), is_followup: bool = Form(False),
                 intent: str = Form(""), action: str = Form(""),
                 db: Session = Depends(get_db)):
    interview = db.get(Interview, interview_id)
    if not interview:
        raise HTTPException(404, "not found")
    if states.is_terminal(interview.status):
        # Generation-fence enforcement (plan §2): a turn from a stale in-flight call must never
        # be persisted once the interview has ended.
        raise HTTPException(409, "INTERVIEW_ALREADY_COMPLETED")
    turn = InterviewTurn(interview_id=interview_id, question_id=question_id, question_text=question_text,
                         speaker=speaker, text=text, is_followup=int(is_followup),
                         intent=(intent[:40] or None), action=(action[:40] or None))
    db.add(turn)
    db.commit()
    return {"id": turn.id}


@app.post("/v1/interviews/{interview_id}/coverage-results", dependencies=[Depends(require_service_key)])
def record_coverage_result(interview_id: str, question_id: str = Form(...),
                            coverage_score: float = Form(...), covered_topics: str = Form("[]"),
                            missing_topics: str = Form("[]"), evaluation: str = Form(""),
                            action: str = Form(""), followup_count: int | None = Form(None),
                            db: Session = Depends(get_db)):
    try:
        evaluation_json = json.loads(evaluation) if evaluation else None
    except ValueError:
        evaluation_json = None
    cov = CoverageResult(
        interview_id=interview_id, question_id=question_id, coverage_score=coverage_score,
        covered_topics=json.loads(covered_topics), missing_topics=json.loads(missing_topics),
        evaluation=evaluation_json, action=(action[:40] or None), followup_count=followup_count,
    )
    db.add(cov)
    db.commit()
    return {"id": cov.id}


# Signals that count toward the three-flag policy (plan §12). Purely informational events
# (tab becoming visible again, fullscreen re-entered, a benign reconnect) are recorded for the
# audit trail but never increment the flag counter — only the "something went wrong" half of
# each event pair does.
_FLAG_WORTHY_EVENT_TYPES = {
    "TAB_HIDDEN", "FOCUS_LOST", "FULLSCREEN_EXIT", "CAMERA_OFF", "MICROPHONE_OFF",
    "FACE_NOT_DETECTED", "MULTIPLE_FACES",
}


@app.post("/v1/interviews/{interview_id}/integrity-events", dependencies=[Depends(require_service_key)])
def record_integrity_event(interview_id: str, event_type: str = Form(...), db: Session = Depends(get_db)):
    """Candidate frontend posts browser-observable signals here (visibilitychange,
    fullscreenchange, camera/mic state, etc — plan §12). These are signals, never proof of
    cheating on their own — kept as a separate `integrityEvent` record from any
    `policyDecision`, per the plan's explicit warning against converting one signal directly
    into a fraud verdict. This endpoint only tracks the count and reports the current policy
    tier; the caller (candidate app) decides how to react to `policy_action`.
    """
    interview = db.get(Interview, interview_id)
    if not interview:
        raise HTTPException(404, "not found")
    if states.is_terminal(interview.status):
        raise HTTPException(409, "INTERVIEW_ALREADY_COMPLETED")

    is_flag = event_type in _FLAG_WORTHY_EVENT_TYPES
    event = IntegrityEvent(interview_id=interview_id, event_type=event_type, counted_as_flag=int(is_flag))
    db.add(event)

    if is_flag:
        interview.integrity_flag_count += 1
    db.commit()

    max_flags = interview.max_integrity_flags or 0
    flag_count = interview.integrity_flag_count
    if max_flags <= 0 or flag_count == 0:
        policy_action = "none"
    elif flag_count >= max_flags:
        policy_action = "terminate"
    else:
        policy_action = "warning"

    return {
        "id": event.id, "flag_count": flag_count, "max_flags": max_flags,
        "policy_action": policy_action,
    }


@app.get("/v1/interviews/{interview_id}/integrity-events", dependencies=[Depends(require_service_key)])
def list_integrity_events(interview_id: str, db: Session = Depends(get_db)):
    events = (db.query(IntegrityEvent).filter(IntegrityEvent.interview_id == interview_id)
              .order_by(IntegrityEvent.created_at).all())
    return [{"id": e.id, "event_type": e.event_type, "counted_as_flag": bool(e.counted_as_flag),
             "created_at": e.created_at.isoformat()} for e in events]


@app.get("/v1/interviews/{interview_id}/transcript", dependencies=[Depends(require_service_key)])
def get_transcript(interview_id: str, db: Session = Depends(get_db)):
    turns = db.query(InterviewTurn).filter(InterviewTurn.interview_id == interview_id).order_by(InterviewTurn.started_at).all()
    return [{"question_id": t.question_id, "speaker": t.speaker, "text": t.text, "is_followup": bool(t.is_followup),
             "intent": t.intent, "action": t.action} for t in turns]


def _run_scoring_job(interview_id: str, report_id: str):
    """Runs off the request thread (plan §7 — 'do not perform final scoring synchronously
    inside /complete'). Opens its own DB session since the request-scoped one from Depends(get_db)
    is closed by the time a FastAPI BackgroundTask actually runs.

    This is a pragmatic first step toward the plan's target (a real SQS queue + separate scoring
    worker process, so a crashed API process can't lose a queued job). A same-process background
    task is not durable across restarts — acceptable at current volume, and isolated behind this
    one function so swapping it for an SQS-publish + separate worker later doesn't touch any
    caller. Flagged explicitly as a scale follow-up in changes.md.
    """
    db = SessionLocal()
    try:
        interview = db.get(Interview, interview_id)
        report = db.get(Report, report_id)
        if not interview or not report:
            logger.error("scoring job: interview or report vanished (interview_id=%s report_id=%s)",
                         interview_id, report_id)
            return
        report.status = REPORT_PROCESSING
        db.commit()

        turns = db.query(InterviewTurn).filter(InterviewTurn.interview_id == interview_id).order_by(InterviewTurn.started_at).all()
        questions_by_id = {q["id"]: q for q in interview.plan["questions"] if q.get("question_text")}

        per_question_scores = []
        question_reports = []
        for qid, q in questions_by_id.items():
            # Turns that were not an attempt to answer (hint/answer requests, prompt-injection,
            # repeat requests, skips ...) are kept in the transcript but never scored as answers.
            candidate_text = "\n".join(
                t.text for t in turns
                if t.question_id == qid and t.speaker == "candidate"
                and (t.intent or "answer") not in {i.value for i in NON_ANSWER_INTENTS}
            )
            if not candidate_text.strip():
                continue
            content = score_answer_content(q["question_text"], q.get("expected_topics", []), candidate_text)
            comm = score_communication(candidate_text)
            per_question_scores.append({"competencies": q.get("competencies", []), "content_score": content["score"]})
            question_reports.append({
                "question_id": qid, "question": q["question_text"], "candidate_answer": candidate_text,
                "content_score": content, "communication_score": comm,
            })

        agg = aggregate_final_score(per_question_scores, interview.role.competencies)

        overall = {}
        if question_reports:
            try:
                overall = generate_overall_summary(
                    interview.candidate.name, interview.role.name, agg["final_score"],
                    agg["competency_scores"], question_reports,
                )
            except Exception:
                logger.exception("overall summary generation failed")

        report.final_score = agg["final_score"]
        report.competency_scores = agg["competency_scores"]
        report.content = {
            "questions": question_reports, "candidate": interview.candidate.name,
            "role": interview.role.name, "overall": overall,
        }
        report.status = REPORT_READY
        db.commit()
    except Exception as e:
        logger.exception("scoring job failed for interview %s", interview_id)
        try:
            report = db.get(Report, report_id)
            if report:
                report.status = REPORT_FAILED
                report.error = str(e)
                db.commit()
        except Exception:
            logger.exception("failed to record scoring failure")
    finally:
        # The interview's own lifecycle completes once the live session ended, independent of
        # whether async scoring succeeds — a candidate should never see the interview itself
        # stuck "in progress" just because the scorer errored. Report.status carries the
        # separate PENDING/PROCESSING/READY/FAILED signal for the HR-facing report screen.
        try:
            interview = db.get(Interview, interview_id)
            if interview and interview.status == states.FINALIZING:
                transition_interview(db, interview, states.COMPLETED, reason="scoring job finished")
        except Exception:
            logger.exception("failed to transition interview to COMPLETED after scoring")
        db.close()


@app.post("/v1/interviews/{interview_id}/complete", dependencies=[Depends(require_service_key)])
def complete_interview(interview_id: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Idempotent by design (plan §3): a retried request (e.g. after a network blip on the
    agent's side) must return the already-queued/finalized result rather than schedule scoring a
    second time — running the LLM-backed scorer twice would double the cost and could produce a
    different report each time for the same interview.

    Returns immediately (plan §7): scoring runs in the background, and the caller polls
    GET /report for status (PENDING/PROCESSING/READY/FAILED) instead of blocking on it.
    """
    interview = db.get(Interview, interview_id)
    if not interview:
        raise HTTPException(404, "not found")

    if interview.status in (states.COMPLETED, states.FINALIZING):
        existing = db.query(Report).filter(Report.interview_id == interview_id).order_by(Report.id.desc()).first()
        if existing:
            return {"report_id": existing.id, "status": existing.status, "already_completed": True}
        # No report row yet (race with the request that's creating one) — nothing more to do;
        # the caller should poll GET /report.
        return {"report_id": None, "status": REPORT_PENDING, "already_completed": True}
    if states.is_terminal(interview.status):
        raise HTTPException(409, f"INTERVIEW_ALREADY_COMPLETED: status={interview.status}")

    try:
        # Generation fence: bump generation_id as part of the SAME atomic transition that
        # leaves ACTIVE, so any LLM/TTS call already in flight on the agent side (which
        # captured the pre-bump generation_id) can recognize it is now stale (plan §2).
        transition_interview(db, interview, states.FINALIZING, reason="completion requested",
                              bump_generation=True)
    except ConcurrentTransitionError:
        db.refresh(interview)
        # Someone else already moved it past ACTIVE in the same instant — re-enter this
        # handler's idempotent path instead of erroring.
        return complete_interview(interview_id, background_tasks, db)
    except states.InvalidTransition:
        # e.g. an interview that was created but never started (still PLANNED): there is nothing to
        # complete. A clean 409 instead of an unhandled 500.
        raise HTTPException(409, f"INTERVIEW_NOT_STARTED: status={interview.status}")

    report = Report(interview_id=interview_id, status=REPORT_PENDING)
    db.add(report)
    db.commit()
    background_tasks.add_task(_run_scoring_job, interview_id, report.id)
    return {"report_id": report.id, "status": report.status}


@app.get("/v1/interviews/{interview_id}/report")
def get_report(interview_id: str, db: Session = Depends(get_db)):
    report = db.query(Report).filter(Report.interview_id == interview_id).order_by(Report.id.desc()).first()
    if not report:
        raise HTTPException(404, "no report yet")
    return {
        "status": report.status,
        "final_score": report.final_score,
        "competency_scores": report.competency_scores,
        "content": report.content,
        "error": report.error,
    }
