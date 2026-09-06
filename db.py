import datetime
import uuid

from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

from config import DATABASE_URL

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
    status = Column(String, default="created")  # created/planned/in_progress/completed
    plan = Column(JSON)  # generated InterviewPlan
    room_name = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    candidate = relationship("Candidate")
    role = relationship("Role")


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


class Report(Base):
    __tablename__ = "reports"
    id = Column(String, primary_key=True, default=lambda: gen_id("rep"))
    interview_id = Column(String, ForeignKey("interviews.id"))
    final_score = Column(Float)
    competency_scores = Column(JSON)
    content = Column(JSON)  # full structured report


def init_db():
    Base.metadata.create_all(engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
