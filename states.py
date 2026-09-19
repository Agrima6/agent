"""Interview lifecycle state machine (per Workmate_production_scalable_fix_plan_v2.md #4).

Kept deliberately small: a linear happy path plus a fixed set of terminal states, and a
whitelist of legal transitions so `transition()` can reject anything else instead of silently
overwriting `status` with a free string (the bug the plan calls out — previously only
"created"/"planned"/"completed" were ever actually set, with no guard against re-entering a
terminal state).
"""

CREATED = "CREATED"
PLANNED = "PLANNED"
READY = "READY"
IN_PROGRESS = "IN_PROGRESS"
FINALIZING = "FINALIZING"
COMPLETED = "COMPLETED"
TERMINATED = "TERMINATED"
FAILED = "FAILED"
EXPIRED = "EXPIRED"

TERMINAL_STATES = frozenset({COMPLETED, TERMINATED, FAILED, EXPIRED})

# Allowed source -> destination transitions. Anything not listed here is rejected by transition().
_ALLOWED = {
    CREATED: {PLANNED, FAILED, EXPIRED},
    PLANNED: {READY, FAILED, EXPIRED},
    # READY -> FINALIZING covers the edge case where /start (called by the agent right after it
    # joins the room) never landed — e.g. the candidate ended the interview in the few hundred
    # milliseconds before that request completed. Completion must still succeed in that case.
    READY: {IN_PROGRESS, FINALIZING, FAILED, EXPIRED, TERMINATED},
    IN_PROGRESS: {FINALIZING, TERMINATED, FAILED, EXPIRED},
    FINALIZING: {COMPLETED, FAILED},
    # Terminal states have no outgoing transitions.
}


class InvalidTransition(Exception):
    def __init__(self, current: str, target: str):
        super().__init__(f"cannot transition interview from {current!r} to {target!r}")
        self.current = current
        self.target = target


def is_terminal(status: str) -> bool:
    return status in TERMINAL_STATES


def validate_transition(current: str, target: str) -> None:
    """Raises InvalidTransition if current -> target is not a legal move.

    A transition to the SAME state that current already is (e.g. re-requesting FINALIZING while
    already FINALIZING) is treated as a no-op, not an error — this is what makes idempotent
    completion possible (see api.py's /complete handler).
    """
    if current == target:
        return
    if is_terminal(current):
        raise InvalidTransition(current, target)
    if target not in _ALLOWED.get(current, set()):
        raise InvalidTransition(current, target)
