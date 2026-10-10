"""The per-turn lifecycle of one interview question, as an explicit, validated state machine.

This sits UNDER the interview-level machine in states.py (CREATED -> PLANNED -> READY -> IN_PROGRESS -> FINALIZING ->
COMPLETED) and describes what is happening inside IN_PROGRESS:

    CONNECTED -> QUESTION_DELIVERED -> CANDIDATE_SPEAKING -> TURN_FINALIZED -> ANSWER_SAVED -> DECIDING
                       ^                                                                          |
                       +-------------- (follow-up or next question delivered) <-------------------+
                                                                       DECIDING -> COMPLETED

The rules it enforces (each one was a way a duplicate or racing event used to be able to corrupt the interview):
  * an answer can only be saved after its turn was finalized, and only once;
  * a decision (follow-up / next question / complete) can only be taken after the answer was saved;
  * the next question can only be delivered from DECIDING - so one finalized answer advances the interview at most once.

Illegal transitions are reported to the caller (and logged), never raised into the live conversation.
"""
from enum import Enum


class TurnPhase(str, Enum):
    CONNECTED = "CONNECTED"
    QUESTION_DELIVERED = "QUESTION_DELIVERED"
    CANDIDATE_SPEAKING = "CANDIDATE_SPEAKING"
    TURN_FINALIZED = "TURN_FINALIZED"
    ANSWER_SAVED = "ANSWER_SAVED"
    DECIDING = "DECIDING"
    COMPLETED = "COMPLETED"


_ALLOWED = {
    TurnPhase.CONNECTED: {TurnPhase.QUESTION_DELIVERED, TurnPhase.TURN_FINALIZED, TurnPhase.COMPLETED},
    TurnPhase.QUESTION_DELIVERED: {TurnPhase.CANDIDATE_SPEAKING, TurnPhase.TURN_FINALIZED, TurnPhase.QUESTION_DELIVERED,
                                   TurnPhase.COMPLETED},
    TurnPhase.CANDIDATE_SPEAKING: {TurnPhase.TURN_FINALIZED, TurnPhase.QUESTION_DELIVERED, TurnPhase.COMPLETED},
    TurnPhase.TURN_FINALIZED: {TurnPhase.ANSWER_SAVED, TurnPhase.QUESTION_DELIVERED, TurnPhase.DECIDING, TurnPhase.COMPLETED},
    TurnPhase.ANSWER_SAVED: {TurnPhase.DECIDING, TurnPhase.COMPLETED},
    TurnPhase.DECIDING: {TurnPhase.QUESTION_DELIVERED, TurnPhase.COMPLETED},
    TurnPhase.COMPLETED: set(),
}


class TurnStateMachine:
    def __init__(self) -> None:
        self.phase = TurnPhase.CONNECTED
        self.turn_number = 0          # finalized candidate turns so far
        self.rejected: list[tuple[str, str]] = []     # (from, to) transitions that were refused, for diagnostics

    def can(self, target: TurnPhase) -> bool:
        return target in _ALLOWED[self.phase]

    def go(self, target: TurnPhase) -> bool:
        """Move to `target` if legal and return True; otherwise leave the phase unchanged and return False."""
        if target == self.phase and target in (TurnPhase.QUESTION_DELIVERED, TurnPhase.DECIDING, TurnPhase.COMPLETED):
            return True        # re-asking / repeating a question, or re-entering the decision, is idempotent
        if not self.can(target):
            self.rejected.append((self.phase.value, target.value))
            return False
        if target == TurnPhase.TURN_FINALIZED:
            self.turn_number += 1
        self.phase = target
        return True
