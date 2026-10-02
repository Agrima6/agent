"""Report sections that show whether the interview covered what the hiring team asked for.

Pure functions over the persisted plan, the scored questions and the per-answer evaluations - no LLM, no I/O -
so the numbers are reproducible and testable. Everything here is DECISION SUPPORT: it never changes a score and
nothing in it rejects a candidate. Inconsistencies are surfaced for a human to review.
"""
import os

from focus import FocusArea

NOTICE = ("Decision support only. These findings come from an automated evaluation of what the candidate said; "
          "a person should review flagged items before any hiring decision.")


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name) or default)
    except ValueError:
        return default


def _covered_score() -> int:        # average answer score at/above which a focus area counts as covered
    return _int_env("FOCUS_COVERED_SCORE", 60)


def _verified_score() -> int:       # answer score at/above which a resume claim counts as substantiated
    return _int_env("CLAIM_VERIFIED_SCORE", 60)


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 1) if values else None


def _by_question(coverage_rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in coverage_rows or []:
        grouped.setdefault(row.get("question_id"), []).append(row)
    return grouped


def _score(question_report: dict | None) -> float | None:
    score = ((question_report or {}).get("content_score") or {}).get("score")
    return float(score) if isinstance(score, (int, float)) else None


def _confidence(evals: list[dict], answered: int) -> str:
    if answered == 0 or not evals:
        return "NONE"
    usable = [e for e in evals if not e.get("failed") and float(e.get("uncertainty") or 0) < 0.7]
    if len(usable) < len(evals) / 2:
        return "LOW"
    return "HIGH" if len(usable) >= 2 else "MEDIUM"


def build_intelligence(plan: dict, question_reports: list[dict], coverage_rows: list[dict]) -> dict:
    """`coverage_rows`: the persisted per-answer records, each {question_id, followup_count, evaluation}."""
    questions = [q for q in (plan or {}).get("questions", []) if q.get("question_text")]
    reports = {r["question_id"]: r for r in question_reports}
    rows = _by_question(coverage_rows)

    def evaluations(qid: str) -> list[dict]:
        return [r["evaluation"] for r in rows.get(qid, []) if isinstance(r.get("evaluation"), dict)]

    def depth(qid: str) -> int:
        return max((int(r.get("followup_count") or 0) for r in rows.get(qid, [])), default=0)

    focus_coverage = []
    for raw in (plan or {}).get("focus_areas") or []:
        area = FocusArea.from_dict(raw)
        related = [q for q in questions if q.get("focus") == area.name]
        answered = [q for q in related if q["id"] in reports]
        evals = [e for q in answered for e in evaluations(q["id"])]
        score = _mean([s for q in answered if (s := _score(reports[q["id"]])) is not None])
        if not related:
            status = "NOT_ASKED"
        elif not answered:
            status = "NOT_ANSWERED"
        else:
            status = "COVERED" if score is not None and score >= _covered_score() else "PARTIAL"
        focus_coverage.append({
            "focus": area.name, "priority": area.priority, "sources": area.sources, "status": status, "score": score,
            "questions_asked": len(related), "questions_answered": len(answered),
            "follow_up_depth": sum(depth(q["id"]) for q in answered),
            "evidence": [e for ev in evals for e in (ev.get("evidence") or [])][:5],
            "confidence": _confidence(evals, len(answered)),
        })

    resume_validation, contradictions = [], []
    for claim in (plan or {}).get("resume_claims") or []:
        qid = f"q_validate_{claim['id']}"
        evals = evaluations(qid)
        score = _score(reports.get(qid))
        flags = [e["inconsistency"] for e in evals if e.get("inconsistency")]
        if qid not in reports:
            status = "NOT_ANSWERED"
        elif flags:
            status = "REVIEW_INCONSISTENCY"
        elif score is not None and score >= _verified_score() and any(e.get("ownership") == "own" for e in evals):
            status = "SUBSTANTIATED"
        else:
            status = "UNVERIFIED"
        resume_validation.append({"claim": claim["claim"], "focus": claim.get("focus"), "status": status,
                                  "score": score, "evidence": [x for e in evals for x in (e.get("evidence") or [])][:3],
                                  "confidence": _confidence(evals, 1 if qid in reports else 0)})
        for note in flags:
            contradictions.append({"claim": claim["claim"], "note": note, "question_id": qid,
                                   "needs_human_review": True})

    # Flags on non-resume questions (e.g. an answer that conflicts with an earlier one) are also surfaced.
    claim_qids = {f"q_validate_{c['id']}" for c in (plan or {}).get("resume_claims") or []}
    for qid, group in rows.items():
        if qid in claim_qids:
            continue
        for e in (r["evaluation"] for r in group if isinstance(r.get("evaluation"), dict)):
            if e.get("inconsistency"):
                contradictions.append({"claim": None, "note": e["inconsistency"], "question_id": qid,
                                       "needs_human_review": True})

    return {"hr_focus_coverage": focus_coverage, "resume_validation": resume_validation,
            "contradictions": contradictions,
            "unasked_high_priority": [f["focus"] for f in focus_coverage
                                      if f["priority"] == "HIGH" and f["status"] in ("NOT_ASKED", "NOT_ANSWERED")],
            "decision_support_only": True, "notice": NOTICE}
