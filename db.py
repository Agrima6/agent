import datetime
import uuid

from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Text, ForeignKey, JSON, update
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.orm import Session as SASession

from config import DATABASE_URL
import states

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Role(Base):
    __tablename__ = "roles"
    id = Column(String, primary_key=True, default=lambda: gen_id("role"))
    name = Column(String, nullable=False)
    competencies = Column(JSON, nullable=False)  # [{"key": "...", "weight": 0.3}, ...]
    # LLM-generated scenario questions tailored to this exact role title, generated once and
    # reused for every candidate who interviews for it — every candidate for the same role gets
    # the same questions, which is what makes their scores fairly comparable to each other.
    # (Regenerating fresh questions per-candidate would make cross-candidate comparison unfair.)
    generated_questions = Column(JSON, nullable=True)


class Candidate(Base):
    __tablename__ = "candidates"
    id = Column(String, primary_key=True, default=lambda: gen_id("cand"))
    name = Column(String, nullable=False)
    email = Column(String)
    resume_text = Column(Text)
    evidence_profile = Column(JSON)  # structured extraction: skills/projects/claims


class Interview(Base):
    __tablename__ = "interviews"
    id = Column(String, primary_key=True, default=lambda: gen_id("intv"))
    candidate_id = Column(String, ForeignKey("candidates.id"))
    role_id = Column(String, ForeignKey("roles.id"))
    language = Column(String, default="en")
    duration_minutes = Column(Integer, default=30)
    min_questions = Column(Integer, default=4)
    max_questions = Column(Integer, default=7)
    max_followups_per_question = Column(Integer, default=2)
    coverage_threshold = Column(Float, default=0.7)
    max_integrity_flags = Column(Integer, default=3)  # plan §12 three-flag policy; 0 disables it
    integrity_flag_count = Column(Integer, default=0, nullable=False)
    status = Column(String, default=states.CREATED)  # see states.py for the legal state machine
    # Optimistic-concurrency guard for state transitions (incremented on every successful
    # transition) and a generation fence for realtime cancellation (incremented whenever the
    # interview ends, so any LLM/TTS call started before that point can recognize it is stale).
    state_version = Column(Integer, default=0, nullable=False)
    generation_id = Column(Integer, default=0, nullable=False)
    plan = Column(JSON)  # generated InterviewPlan
    room_name = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    candidate = relationship("Candidate")
    role = relationship("Role")


class InterviewEvent(Base):
    """Append-only audit trail of every state transition (plan.md #4: "append an event")."""
    __tablename__ = "interview_events"
    id = Column(String, primary_key=True, default=lambda: gen_id("evt"))
    interview_id = Column(String, ForeignKey("interviews.id"))
    from_status = Column(String)
    to_status = Column(String)
    state_version = Column(Integer)
    reason = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class InterviewTurn(Base):
    __tablename__ = "interview_turns"
    id = Column(String, primary_key=True, default=lambda: gen_id("turn"))
    interview_id = Column(String, ForeignKey("interviews.id"))
    question_id = Column(String)
    question_text = Column(Text)
    speaker = Column(String)  # "agent" | "candidate"
    text = Column(Text)
    is_followup = Column(Integer, default=0)
    started_at = Column(DateTime, default=datetime.datetime.utcnow)


class CoverageResult(Base):
    __tablename__ = "coverage_results"
    id = Column(String, primary_key=True, default=lambda: gen_id("cov"))
    interview_id = Column(String, ForeignKey("interviews.id"))
    question_id = Column(String)
    covered_topics = Column(JSON)
    missing_topics = Column(JSON)
    coverage_score = Column(Float)


# Report processing states (Workmate_production_scalable_fix_plan_v2.md §7). The frontend
# should show "Report is being generated..." for PENDING/PROCESSING rather than blocking the
# candidate's own completion on how long LLM-backed scoring takes.
REPORT_PENDING = "PENDING"
REPORT_PROCESSING = "PROCESSING"
REPORT_READY = "READY"
REPORT_FAILED = "FAILED"


class IntegrityEvent(Base):
    """Browser-observable integrity/proctoring signals (Workmate_production_scalable_fix_plan_v2.md
    §12) — a candidate frontend posts these as they happen (tab hidden, fullscreen exit, camera
    off, etc). These are observable signals, not proof of cheating — see policy §12's own caveat."""
    __tablename__ = "integrity_events"
    id = Column(String, primary_key=True, default=lambda: gen_id("ie"))
    interview_id = Column(String, ForeignKey("interviews.id"))
    event_type = Column(String, nullable=False)  # e.g. TAB_HIDDEN, FULLSCREEN_EXIT, CAMERA_OFF
    counted_as_flag = Column(Integer, default=0)  # 1 if this event incremented the flag counter
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class Report(Base):
    __tablename__ = "reports"
    id = Column(String, primary_key=True, default=lambda: gen_id("rep"))
    interview_id = Column(String, ForeignKey("interviews.id"))
    status = Column(String, default=REPORT_PENDING, nullable=False)
    final_score = Column(Float)
    competency_scores = Column(JSON)
    content = Column(JSON)  # full structured report
    error = Column(Text, nullable=True)


def init_db():
    Base.metadata.create_all(engine)
    # create_all only creates missing tables, not missing columns on an existing table — patch
    # in any new columns by hand. Only needed for the legacy sqlite file this repo used to ship
    # with; a fresh Postgres database already gets the column from create_all above.
    if engine.dialect.name == "sqlite":
        with engine.connect() as conn:
            from sqlalchemy import text
            existing_cols = {row[1] for row in conn.execute(text("PRAGMA table_info(roles)"))}
            if "generated_questions" not in existing_cols:
                conn.execute(text("ALTER TABLE roles ADD COLUMN generated_questions JSON"))
                conn.commit()
            interview_cols = {row[1] for row in conn.execute(text("PRAGMA table_info(interviews)"))}
            if "state_version" not in interview_cols:
                conn.execute(text("ALTER TABLE interviews ADD COLUMN state_version INTEGER DEFAULT 0"))
                conn.commit()
            if "generation_id" not in interview_cols:
                conn.execute(text("ALTER TABLE interviews ADD COLUMN generation_id INTEGER DEFAULT 0"))
                conn.commit()
            report_cols = {row[1] for row in conn.execute(text("PRAGMA table_info(reports)"))}
            if "status" not in report_cols:
                conn.execute(text(f"ALTER TABLE reports ADD COLUMN status TEXT DEFAULT '{REPORT_READY}'"))
                conn.commit()
            if "error" not in report_cols:
                conn.execute(text("ALTER TABLE reports ADD COLUMN error TEXT"))
                conn.commit()
            if "max_integrity_flags" not in interview_cols:
                conn.execute(text("ALTER TABLE interviews ADD COLUMN max_integrity_flags INTEGER DEFAULT 3"))
                conn.commit()
            if "integrity_flag_count" not in interview_cols:
                conn.execute(text("ALTER TABLE interviews ADD COLUMN integrity_flag_count INTEGER DEFAULT 0"))
                conn.commit()


class ConcurrentTransitionError(Exception):
    """Raised when a transition's optimistic-concurrency check loses a race — the caller should
    reload the interview and decide whether the winning transition already satisfies the request
    (see api.py's idempotent /complete handler)."""


def transition_interview(db: SASession, interview: "Interview", target_status: str, reason: str | None = None,
                          bump_generation: bool = False) -> "Interview":
    """Atomically move `interview` to `target_status`, guarded by state_version.

    Validates the move is legal (states.validate_transition), then does a single UPDATE ...
    WHERE id = :id AND state_version = :expected — if another request already transitioned the
    row first, expected won't match, 0 rows are updated, and we raise so the caller can reload
    and treat the request as already-satisfied instead of corrupting state.
    """
    states.validate_transition(interview.status, target_status)
    if interview.status == target_status:
        return interview  # no-op re-entry into the same state — this is what makes /complete idempotent

    expected_version = interview.state_version
    new_version = expected_version + 1
    values = {"status": target_status, "state_version": new_version}
    if bump_generation:
        values["generation_id"] = Interview.generation_id + 1

    result = db.execute(
        update(Interview)
        .where(Interview.id == interview.id, Interview.state_version == expected_version)
        .values(**values)
    )
    if result.rowcount == 0:
        db.rollback()
        raise ConcurrentTransitionError(interview.id)

    db.add(InterviewEvent(
        interview_id=interview.id, from_status=interview.status, to_status=target_status,
        state_version=new_version, reason=reason,
    ))
    db.commit()
    db.refresh(interview)
    return interview


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
