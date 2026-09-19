"""Regression tests for the critical End Interview / idempotent completion bug
(Workmate_production_scalable_fix_plan_v2.md §2-4, §20 P0-1..5).

Run with: pytest test_end_interview.py -v

Isolation note: this does NOT rely on setting the DATABASE_URL env var before import. config.py
resolves DATABASE_URL to a module-level constant the first time ANYTHING imports it (which can
happen transitively, e.g. via another test module collected first in the same pytest process,
or via embeddings.py's `from config import OPENAI_API_KEY`) — an env var set afterwards has no
effect on that already-cached value. Relying on import order silently pointed an earlier version
of this file at the real workmate.db instead of a scratch file. Instead, this file creates its
own throwaway SQLite engine explicitly and overrides FastAPI's `get_db` dependency to use it,
which is correct regardless of what has or hasn't been imported already.
"""
import tempfile

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import states
import api
from db import Base, Role, Candidate, Interview, transition_interview, ConcurrentTransitionError

_tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
test_engine = create_engine(f"sqlite:///{_tmp_db.name}", connect_args={"check_same_thread": False})
TestSessionLocal = sessionmaker(bind=test_engine, autoflush=False, autocommit=False)
Base.metadata.create_all(test_engine)


def _override_get_db():
    session = TestSessionLocal()
    try:
        yield session
    finally:
        session.close()


api.app.dependency_overrides[api.get_db] = _override_get_db
# _run_scoring_job (the background task complete_interview schedules) opens its own session
# directly via api.SessionLocal rather than through Depends(get_db) — since FastAPI's
# dependency_overrides only intercepts Depends() calls, the background task's session factory
# has to be redirected separately, or it silently talks to the real default-engine DB instead
# of this test's scratch one.
api.SessionLocal = TestSessionLocal


@pytest.fixture()
def client():
    return TestClient(api.app)


@pytest.fixture()
def interview_id(client):
    db = TestSessionLocal()
    role = Role(name="Backend Engineer", competencies=[{"key": "backend", "weight": 1.0}])
    cand = Candidate(name="Test Candidate", email="t@example.com")
    db.add_all([role, cand])
    db.commit()
    interview = Interview(
        candidate_id=cand.id, role_id=role.id, duration_minutes=30,
        plan={"questions": [{"id": "p_intro", "type": "introduction", "question_text": None}]},
        status=states.CREATED,
    )
    db.add(interview)
    db.commit()
    interview.room_name = f"interview-{interview.id}"
    db.commit()
    transition_interview(db, interview, states.PLANNED)
    transition_interview(db, interview, states.READY)
    transition_interview(db, interview, states.IN_PROGRESS)
    iid = interview.id
    db.close()
    return iid


def test_complete_is_idempotent_on_double_call(client, interview_id):
    """Candidate clicks End twice (double-submit / slow network retry): the second call must
    return the SAME report (scoring runs in the background — see api._run_scoring_job — not
    inline, so the first response only guarantees a report was queued, not that it's scored
    yet), and must not schedule a second scoring job."""
    r1 = client.post(f"/v1/interviews/{interview_id}/complete")
    assert r1.status_code == 200
    assert r1.json()["status"] in ("PENDING", "PROCESSING", "READY")
    r2 = client.post(f"/v1/interviews/{interview_id}/complete")
    assert r2.status_code == 200
    assert r1.json()["report_id"] == r2.json()["report_id"]
    assert r2.json().get("already_completed") is True

    # Exactly one report row must exist for this interview no matter how many times /complete
    # is called — proves the background scoring job was not scheduled twice.
    db = TestSessionLocal()
    from db import Report
    report_count = db.query(Report).filter(Report.interview_id == interview_id).count()
    db.close()
    assert report_count == 1


def test_turn_rejected_after_completion(client, interview_id):
    """A turn submission that arrives after the interview is terminal must be rejected with
    409 INTERVIEW_ALREADY_COMPLETED, not silently persisted (plan §26)."""
    client.post(f"/v1/interviews/{interview_id}/complete")
    resp = client.post(
        f"/v1/interviews/{interview_id}/turns",
        data={"question_id": "p_intro", "speaker": "candidate", "text": "late answer", "is_followup": False},
    )
    assert resp.status_code == 409
    assert "INTERVIEW_ALREADY_COMPLETED" in resp.text


def test_report_reaches_ready_status(client, interview_id):
    """End-to-end: after /complete, GET /report must eventually reflect the async scoring job's
    result (plan §7 report status states) rather than staying PENDING forever."""
    client.post(f"/v1/interviews/{interview_id}/complete")
    resp = client.get(f"/v1/interviews/{interview_id}/report")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "READY"
    assert "final_score" in body


def test_generation_id_bumps_on_completion(client, interview_id):
    db = TestSessionLocal()
    before = db.get(Interview, interview_id).generation_id
    db.close()
    client.post(f"/v1/interviews/{interview_id}/complete")
    db = TestSessionLocal()
    after = db.get(Interview, interview_id).generation_id
    db.close()
    assert after == before + 1


def test_invalid_transition_rejected():
    """A CREATED interview cannot jump straight to COMPLETED — must go through the real
    lifecycle. Guards against a future bug re-introducing an unguarded status write."""
    db = TestSessionLocal()
    role = Role(name="X", competencies=[])
    cand = Candidate(name="Y")
    db.add_all([role, cand])
    db.commit()
    interview = Interview(candidate_id=cand.id, role_id=role.id, status=states.CREATED)
    db.add(interview)
    db.commit()
    with pytest.raises(states.InvalidTransition):
        transition_interview(db, interview, states.COMPLETED)
    db.close()


def test_no_transition_out_of_terminal_state():
    db = TestSessionLocal()
    role = Role(name="X", competencies=[])
    cand = Candidate(name="Y")
    db.add_all([role, cand])
    db.commit()
    interview = Interview(candidate_id=cand.id, role_id=role.id, status=states.COMPLETED)
    db.add(interview)
    db.commit()
    with pytest.raises(states.InvalidTransition):
        transition_interview(db, interview, states.IN_PROGRESS)
    db.close()


def test_reconnect_after_completion_is_rejected(client, interview_id):
    """A candidate reconnecting (e.g. page refresh) after the interview already completed must
    not be able to mint a fresh session against it (plan §29 — 'never restart the agent')."""
    client.post(f"/v1/interviews/{interview_id}/complete")
    resp = client.post(f"/v1/interviews/{interview_id}/candidate-token")
    assert resp.status_code == 409


def test_integrity_events_three_flag_policy(client, interview_id):
    """plan §12: flag 1/2 -> warning, flag 3 (== max_integrity_flags default) -> terminate.
    Non-flag-worthy events (e.g. TAB_VISIBLE) must not move the counter."""
    r = client.post(f"/v1/interviews/{interview_id}/integrity-events", data={"event_type": "TAB_VISIBLE"})
    assert r.json()["flag_count"] == 0
    assert r.json()["policy_action"] == "none"

    r = client.post(f"/v1/interviews/{interview_id}/integrity-events", data={"event_type": "FULLSCREEN_EXIT"})
    assert r.json()["flag_count"] == 1
    assert r.json()["policy_action"] == "warning"

    client.post(f"/v1/interviews/{interview_id}/integrity-events", data={"event_type": "TAB_HIDDEN"})
    r = client.post(f"/v1/interviews/{interview_id}/integrity-events", data={"event_type": "CAMERA_OFF"})
    assert r.json()["flag_count"] == 3
    assert r.json()["policy_action"] == "terminate"

    events = client.get(f"/v1/interviews/{interview_id}/integrity-events").json()
    assert len(events) == 4
    assert sum(1 for e in events if e["counted_as_flag"]) == 3


def test_integrity_events_rejected_after_completion(client, interview_id):
    client.post(f"/v1/interviews/{interview_id}/complete")
    resp = client.post(f"/v1/interviews/{interview_id}/integrity-events", data={"event_type": "TAB_HIDDEN"})
    assert resp.status_code == 409


class _FakeRunner:
    """Mirrors agent.InterviewRunner's generation-fence surface without needing a real
    httpx client / LiveKit session, to unit-test the late-LLM-response race directly."""

    def __init__(self):
        self.generation = 0
        self.terminal = False

    def is_stale(self, captured_generation):
        return self.terminal or captured_generation != self.generation

    def complete(self):
        self.terminal = True
        self.generation += 1


def test_late_llm_response_is_discarded_after_end():
    """Reproduces the exact race in Workmate_production_scalable_fix_plan_v2.md §2:
    LLM/coverage-judge call starts -> candidate clicks End -> call resolves late -> result must
    be recognized as stale."""
    runner = _FakeRunner()
    captured_generation = runner.generation  # captured at the start of the slow call
    runner.complete()  # candidate ends the interview while the call is still "in flight"
    assert runner.is_stale(captured_generation) is True


def test_response_from_current_generation_is_not_discarded():
    runner = _FakeRunner()
    captured_generation = runner.generation
    assert runner.is_stale(captured_generation) is False
