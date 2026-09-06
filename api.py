import uuid

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from livekit import api as lk_api

from config import LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
from db import init_db, get_db, Role, Candidate, Interview, InterviewTurn, Report
from resume_parser import extract_text_from_pdf, parse_resume
from planner import build_interview_plan
from scoring import score_answer_content, score_communication, aggregate_final_score

init_db()
app = FastAPI(title="Workmate.IQ Interview Agent (MVP)")
app.mount("/static", StaticFiles(directory="static"), name="static")

DEFAULT_ROLE_COMPETENCIES = [
    {"key": "technical_depth", "weight": 0.30},
    {"key": "problem_solving", "weight": 0.25},
    {"key": "practical_application", "weight": 0.20},
    {"key": "communication", "weight": 0.15},
    {"key": "ownership", "weight": 0.10},
]


@app.get("/")
def index():
    return FileResponse("static/hr.html")


@app.post("/v1/roles")
def create_role(name: str = Form(...), db: Session = Depends(get_db)):
    role = Role(name=name, competencies=DEFAULT_ROLE_COMPETENCIES)
    db.add(role)
    db.commit()
    return {"id": role.id, "name": role.name, "competencies": role.competencies}


@app.post("/v1/candidates")
def create_candidate(name: str = Form(...), email: str = Form(""), db: Session = Depends(get_db)):
    cand = Candidate(name=name, email=email)
    db.add(cand)
    db.commit()
    return {"id": cand.id, "name": cand.name}


@app.post("/v1/candidates/{candidate_id}/resume")
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


@app.post("/v1/interviews")
def create_interview(candidate_id: str = Form(...), role_id: str = Form(...),
                      duration_minutes: int = Form(30), db: Session = Depends(get_db)):
    cand = db.get(Candidate, candidate_id)
    role = db.get(Role, role_id)
    if not cand or not role:
        raise HTTPException(404, "candidate or role not found")

    plan = build_interview_plan(
        role_competencies=role.competencies,
        role_name=role.name,
        evidence_profile=cand.evidence_profile or {},
        min_questions=4,
        max_questions=7,
    )

    interview = Interview(
        candidate_id=candidate_id,
        role_id=role_id,
        duration_minutes=duration_minutes,
        plan=plan,
        status="planned",
    )
    db.add(interview)
    db.commit()
    interview.room_name = f"interview-{interview.id}"
    db.commit()
    return {"id": interview.id, "room_name": interview.room_name, "plan": plan}


@app.get("/v1/interviews/{interview_id}")
def get_interview(interview_id: str, db: Session = Depends(get_db)):
    interview = db.get(Interview, interview_id)
    if not interview:
        raise HTTPException(404, "not found")
    return {
        "id": interview.id, "status": interview.status, "room_name": interview.room_name,
        "plan": interview.plan, "candidate": interview.candidate.name, "role": interview.role.name,
    }


def _mint_token(room_name: str, identity: str, name: str) -> str:
    token = lk_api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
    token.with_identity(identity).with_name(name).with_grants(
        lk_api.VideoGrants(room_join=True, room=room_name)
    )
    return token.to_jwt()


@app.post("/v1/interviews/{interview_id}/candidate-token")
def candidate_token(interview_id: str, db: Session = Depends(get_db)):
    interview = db.get(Interview, interview_id)
    if not interview:
        raise HTTPException(404, "not found")
    identity = f"candidate-{interview.candidate_id}"
    jwt = _mint_token(interview.room_name, identity, interview.candidate.name)
    return {"token": jwt, "url": LIVEKIT_URL, "room_name": interview.room_name}


@app.post("/v1/interviews/{interview_id}/turns")
def record_turn(interview_id: str, question_id: str = Form(...), question_text: str = Form(""),
                 speaker: str = Form(...), text: str = Form(...), is_followup: bool = Form(False),
                 db: Session = Depends(get_db)):
    turn = InterviewTurn(interview_id=interview_id, question_id=question_id, question_text=question_text,
                         speaker=speaker, text=text, is_followup=int(is_followup))
    db.add(turn)
    db.commit()
    return {"id": turn.id}


@app.get("/v1/interviews/{interview_id}/transcript")
def get_transcript(interview_id: str, db: Session = Depends(get_db)):
    turns = db.query(InterviewTurn).filter(InterviewTurn.interview_id == interview_id).order_by(InterviewTurn.started_at).all()
    return [{"question_id": t.question_id, "speaker": t.speaker, "text": t.text, "is_followup": bool(t.is_followup)} for t in turns]


@app.post("/v1/interviews/{interview_id}/complete")
def complete_interview(interview_id: str, db: Session = Depends(get_db)):
    interview = db.get(Interview, interview_id)
    if not interview:
        raise HTTPException(404, "not found")
    interview.status = "completed"
    db.commit()

    turns = db.query(InterviewTurn).filter(InterviewTurn.interview_id == interview_id).order_by(InterviewTurn.started_at).all()
    questions_by_id = {q["id"]: q for q in interview.plan["questions"] if q.get("question_text")}

    per_question_scores = []
    question_reports = []
    for qid, q in questions_by_id.items():
        candidate_text = "\n".join(t.text for t in turns if t.question_id == qid and t.speaker == "candidate")
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

    report = Report(
        interview_id=interview_id,
        final_score=agg["final_score"],
        competency_scores=agg["competency_scores"],
        content={"questions": question_reports, "candidate": interview.candidate.name, "role": interview.role.name},
    )
    db.add(report)
    db.commit()
    return {"report_id": report.id, "final_score": agg["final_score"], "competency_scores": agg["competency_scores"]}


@app.get("/v1/interviews/{interview_id}/report")
def get_report(interview_id: str, db: Session = Depends(get_db)):
    report = db.query(Report).filter(Report.interview_id == interview_id).order_by(Report.id.desc()).first()
    if not report:
        raise HTTPException(404, "no report yet")
    return {
        "final_score": report.final_score,
        "competency_scores": report.competency_scores,
        "content": report.content,
    }
