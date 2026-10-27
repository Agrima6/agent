"""The structured opening of every interview, as pure, unit-tested code (no network, no LLM, no LiveKit).

Stages, in order (each one is entered only from the previous one):

    INTRODUCTION  ->  CLARIFICATION  ->  READINESS  ->  RESUME_QUESTIONS  ->  ENVIRONMENT_CHECK  ->  MAIN_INTERVIEW
                          ^    |             |  |
                          +----+             +--+--> back to CLARIFICATION when the candidate asks something

Follow-ups happen inside RESUME_QUESTIONS (at most one) and MAIN_INTERVIEW (the existing per-question policy);
completion and report generation stay with the existing interview lifecycle (states.py -> /complete -> scoring).

This module decides three things and nothing else:
  * how the plan is ordered (arrange_questions) - the resume-grounded questions come before the environment check;
  * what may be said about the candidate (profile_facts) - only facts that literally appear in the resume text;
  * what a candidate's short reply means at a given stage (interpret) - deterministic, so readiness is never inferred
    from silence and a question is never mistaken for "yes".
"""
import re

from interaction_guard import contains_injection

INTRODUCTION = "INTRODUCTION"
CLARIFICATION = "CLARIFICATION"
READINESS = "READINESS_CONFIRMATION"
RESUME_QUESTIONS = "RESUME_QUESTIONS"
ENVIRONMENT_CHECK = "ENVIRONMENT_CHECKS"
MAIN_INTERVIEW = "MAIN_INTERVIEW"

STAGE_ORDER = [INTRODUCTION, CLARIFICATION, READINESS, RESUME_QUESTIONS, ENVIRONMENT_CHECK, MAIN_INTERVIEW]
PREFLIGHT_STAGES = frozenset({INTRODUCTION, CLARIFICATION, READINESS, ENVIRONMENT_CHECK})

# How many resume-grounded questions open the interview, and how long each stage may keep asking.
MAX_OPENING_QUESTIONS = 2
MAX_CLARIFICATIONS = 6          # answered doubts before the interviewer moves on to readiness
MAX_UNCLEAR = 2                 # unintelligible/ambiguous replies tolerated per stage before the fallback applies
MAX_ENV_RETRIES = 2             # "not a quiet place" replies before the interview continues anyway (and says so)

RESUME_TYPES = ("resume_validation", "resume")


def stage_rank(stage: str | None) -> int:
    return STAGE_ORDER.index(stage) if stage in STAGE_ORDER else -1


# ------------------------------------------------------------------------------------------ plan order
def arrange_questions(questions: list[dict]) -> tuple[list[dict], int]:
    """Order a stored plan for the new opening and return (ordered questions, number of opening questions).

    p_intro (the spoken introduction) stays first. The opening block follows it: up to MAX_OPENING_QUESTIONS questions
    grounded in the candidate's resume. When the plan has none (no resume, or an unreadable one) the existing generic
    "tell me about yourself" question is the single background question - honest, and it never pretends to have read a
    resume. When resume questions open the interview that generic question is dropped (they cover the same ground).
    Everything else keeps its order, so older stored plans keep working unchanged.
    """
    questions = list(questions)
    intro = [q for q in questions if q.get("type") == "introduction"]
    background = [q for q in questions if q.get("type") == "candidate_introduction"]
    resume = [q for q in questions if q.get("type") in RESUME_TYPES][:MAX_OPENING_QUESTIONS]
    opening = resume or background[:1]
    used = {id(q) for q in intro + opening}
    dropped = {id(q) for q in background} if resume else set()
    rest = [q for q in questions if id(q) not in used and id(q) not in dropped]
    return intro + opening + rest, len(opening)


# ----------------------------------------------------------------------------------- candidate facts
_MAX_FACT_CHARS = 60


def _clean_fact(value) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip().strip("\"'`.,;:")
    return text[:_MAX_FACT_CHARS].strip()


def _in_resume(fact: str, resume_text: str) -> bool:
    """The fact must literally appear in the resume text (case-insensitive, whitespace-collapsed)."""
    if not fact or not resume_text:
        return False
    return re.sub(r"\s+", " ", fact).lower() in re.sub(r"\s+", " ", resume_text).lower()


def profile_facts(evidence_profile: dict | None, resume_text: str | None) -> dict:
    """What the interviewer may say it noticed in the CV. Every item must be present in the resume text itself (the parser
    is a language model; its output is only a pointer, the resume is the evidence). Anything unverifiable or
    instruction-like is left out - a missing fact is simply not mentioned. Returns {} when there is no usable resume."""
    profile = evidence_profile or {}
    text = resume_text or ""
    if not text.strip() or not isinstance(profile, dict):
        return {}

    def keep(value) -> str:
        fact = _clean_fact(value)
        return fact if fact and not contains_injection(fact) and _in_resume(fact, text) else ""

    projects = [keep(p.get("name")) for p in (profile.get("projects") or []) if isinstance(p, dict)]
    companies = [keep(c.get("name")) for c in (profile.get("companies") or []) if isinstance(c, dict)]
    skills = [keep(s) for s in (profile.get("skills") or []) if isinstance(s, str)]
    facts = {
        "project": next((p for p in projects if p), ""),
        "company": next((c for c in companies if c), ""),
        "skill": next((s for s in skills if s), ""),
    }

    years = (profile.get("candidate") or {}).get("years_experience") if isinstance(profile.get("candidate"), dict) else None
    try:
        years = float(years)
    except (TypeError, ValueError):
        years = 0
    # Only stated if the resume itself says so ("5 years", "3+ yrs"): the parser can estimate, the CV cannot be argued with.
    if 0 < years <= 50:
        shown = int(years) if years == int(years) else years
        if re.search(rf"\b{shown}\+?\s*(?:years?|yrs?)\b", text, re.I):
            facts["experience"] = f"{shown} year{'s' if shown != 1 else ''}"
    return {k: v for k, v in facts.items() if v}


# ------------------------------------------------------------------------------------- reply meaning
_WORD = re.compile(r"[a-z0-9']+|[ऀ-ॿ]+")

QUESTION_WORDS = (
    r"what|why|how|when|where|which|who|whom|whose|is|are|am|was|were|do|does|did|can|could|would|will|shall|should|may|"
    r"might|will\s+there|is\s+there|are\s+there|kya|kaise|kitna|kitne|kitni|kab|kahan|kaun|kyun|kyon|क्या|कैसे|कितना|कितने|कब|कहाँ|कौन|क्यों"
)
_QUESTION = re.compile(rf"^\s*(?:{QUESTION_WORDS})\b|\?\s*$|\b(?:i\s+(?:have|had)\s+(?:a|one|another)\s+(?:question|doubt|query))\b", re.I)

# "No questions" / "all clear" - the answer to "is there anything you'd like me to clarify?"
_NO_QUESTIONS = re.compile(
    r"^\s*(?:no|nope|nah|none|nothing|not really|nahi|nahin|नहीं|नही)\b[\s,.!]*(?:thanks|thank you|i'm good|im good|all good|it's clear|its clear|that's all|thats all|dhanyavaad|धन्यवाद)?[\s.!]*$"
    r"|\b(?:no|nothing|zero)\s+(?:more\s+|other\s+|further\s+)?(?:questions?|doubts?|queries|query)\b"
    r"|\b(?:all|everything|it(?:'s| is)|sab(?:\s+kuch)?|everything's)\s+(?:is\s+)?(?:clear|good|fine|theek|clear hai)\b"
    r"|\bi(?:'m| am)\s+(?:clear|good|fine)\b|\bcrystal clear\b|\bclear hai\b"
    r"|\b(?:koi|कोई)\s+(?:sawal|saval|question|doubt|सवाल|doubt)\s+(?:nahi|nahin|नहीं|नही)\b|\bसब\s+(?:clear|साफ़|ठीक)\b|\bसब\s+कुछ\s+(?:clear|साफ़|ठीक)\b", re.I)

_NOT_READY = re.compile(
    r"\b(?:not\s+(?:yet|ready|quite|sure)|wait|hold\s+on|hang\s+on|give\s+me|one\s+(?:sec|second|minute|moment)|just\s+a\s+(?:sec|second|minute|moment)|"
    r"need\s+(?:a\s+)?(?:moment|minute|second|time)|not\s+now|ruko|ruk\s+jao|abhi\s+nahi|अभी\s+नहीं|रुकिए|रुको|एक\s+मिनट)\b"
    r"|^\s*(?:no|nope|nah|nahi|nahin|नहीं|नही)\b[\s,.!]*$", re.I)

_READY = re.compile(
    r"\b(?:ready|let'?s\s+(?:start|begin|go|do\s+it)|go\s+ahead|start|begin|proceed|carry\s+on|continue|shuru|shuru\s+karo|"
    r"chalo|chaliye|chalte\s+hain|tayyar|taiyaar|तैयार|शुरू|चलिए|चलो|चलते\s+हैं)\b"
    r"|^\s*(?:yes|yeah|yep|yup|yea|sure|ok|okay|alright|all\s+right|absolutely|definitely|of\s+course|fine|good|haan|han|ha|ji|"
    r"jee|theek\s+hai|thik\s+hai|bilkul|हाँ|हां|जी|ठीक\s+है|बिलकुल|बिल्कुल)\b", re.I)

_QUIET_NO = re.compile(
    r"\b(?:noisy|noise|loud|crowded|not\s+(?:very\s+|really\s+)?(?:a\s+)?quiet|traffic|construction|people\s+around|"
    r"shor|bahut\s+shor|शोर|not\s+in\s+a\s+quiet|nahi|nahin|नहीं|नही)\b|^\s*(?:no|nope|nah)\b", re.I)


def looks_like_question(text: str) -> bool:
    return bool(_QUESTION.search(text or ""))


def interpret(stage: str, text: str) -> str:
    """What a candidate's reply means at `stage`. One of:
        QUESTION      they are asking something (never mistaken for readiness, even if it contains "ready")
        NO_QUESTIONS  nothing more to clarify          (CLARIFICATION)
        HAS_QUESTION  "yes" to "any questions?" with no question yet
        READY         explicit go-ahead
        NOT_READY     explicit not-yet
        ENV_OK / ENV_NO   the environment answer
        UNCLEAR       none of the above - the caller asks again, within its retry limit
    """
    t = " ".join((text or "").split())
    if not _WORD.findall(t.lower()):
        return "UNCLEAR"
    ready = bool(_READY.search(t))
    asks = looks_like_question(t)
    explicit_ready = bool(re.search(r"\b(?:ready|let'?s\s+(?:start|begin|go)|go\s+ahead|start|begin|proceed|shuru|chalo|चलिए|चलो|तैयार|शुरू)\b", t, re.I))

    if stage == ENVIRONMENT_CHECK:
        if re.search(r"\b(?:not|isn'?t|is\s+not)\s+(?:very\s+)?(?:noisy|loud|crowded)\b|\bno\s+(?:noise|background)\b", t, re.I):
            return "ENV_OK"
        if asks and not re.search(r"\b(?:yes|yeah|yep|haan|हाँ|quiet)\b", t, re.I):
            return "QUESTION"
        if _QUIET_NO.search(t):
            return "ENV_NO"
        quiet = re.search(r"\b(?:quiet|shaant|शांत|alone|private|closed\s+room|my\s+room|at\s+home|in\s+my\s+office)\b", t, re.I)
        return "ENV_OK" if ready or quiet else "UNCLEAR"

    if stage == READINESS:
        # "Wait, how does this work?" is a doubt, not a refusal; a bare "not yet" / "no" is a refusal.
        if asks and not explicit_ready and len(_WORD.findall(t.lower())) >= 3:
            return "QUESTION"
        if _NOT_READY.search(t):
            return "NOT_READY"
        if asks and not explicit_ready:
            return "QUESTION"
        return "READY" if ready else "UNCLEAR"

    # CLARIFICATION (and any other free-text stage)
    if asks and not _NO_QUESTIONS.search(t):
        return "QUESTION"
    if _NO_QUESTIONS.search(t):
        return "READY" if explicit_ready else "NO_QUESTIONS"
    if _NOT_READY.search(t):
        return "NOT_READY"
    if explicit_ready:
        return "READY"
    if re.match(r"^\s*(?:yes|yeah|yep|yup|yea|haan|han|ha|ji|हाँ|हां|जी)\b[\s,.!]*$", t, re.I):
        return "HAS_QUESTION"
    # A statement of substance ("I wanted to know about the timing") is treated as a pending question, never as a go-ahead.
    return "QUESTION" if len(_WORD.findall(t.lower())) >= 4 else "UNCLEAR"


# --------------------------------------------------------------------------------- clarification topics
_DURATION = re.compile(r"\b(how long|how much time|duration|how many minutes|kitna time|kitne (?:minute|der)|कितना (?:समय|टाइम)|कितनी देर)\b", re.I)
_FORMAT = re.compile(r"\b(how (?:does|will|is) (?:this|the interview|it) (?:work|go|going|run)|what (?:is|are) the (?:format|process|rules|instructions)|"
                     r"how does the interview|what (?:will|do) (?:you|we) (?:do|ask)|can you repeat|can i (?:skip|repeat)|kaise (?:hoga|chalega)|कैसे (?:होगा|चलेगा))\b", re.I)
_ROLE = re.compile(r"\b(?:what(?:'s| is)?\s+the\s+(?:role|position|job|post)|which\s+(?:role|position|job)|about\s+the\s+(?:role|position|job)|"
                   r"role\s+(?:details|description)|job\s+description)\b", re.I)
_RESULT = re.compile(r"\b(?:result|results|outcome|feedback|selected|selection|when\s+will\s+i\s+(?:know|hear|get))\b", re.I)


def clarification_topic(text: str) -> str | None:
    """Which approved subject a question is about: duration | format | role | result, or None (not something we can confirm)."""
    t = text or ""
    for topic, pattern in (("duration", _DURATION), ("format", _FORMAT), ("role", _ROLE), ("result", _RESULT)):
        if pattern.search(t):
            return topic
    return None


# ------------------------------------------------------------------------------------ introduction text
def introduction_parts(facts: dict) -> dict:
    """Which optional sentences the introduction may contain, given the verified facts."""
    skill_or_project = facts.get("project") or facts.get("skill") or ""
    return {
        "has_resume": bool(facts),
        "highlight": skill_or_project,
        "company": facts.get("company", ""),
        "experience": facts.get("experience", ""),
    }
