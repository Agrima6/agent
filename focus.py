"""Interview focus areas: what the hiring team wants validated, and how much depth each deserves.

Sources (docs: workmateiq-bug-review-and-implementation-plan.md, sections 3-7):

    HR      the hiring team's own weighted skills / priorities for this round
    ROLE    the role's evaluated competencies
    RESUME  claims on the resume that need verifying (linked to a focus area when they match one)

Each focus area has a PRIORITY (HIGH / MEDIUM / LOW). Priority decides how much interview time it earns:

    HIGH    -> the most questions and follow-ups, and a weak answer is probed further
    MEDIUM  -> a normal share
    LOW     -> a brief check

None of the depth numbers are hard-coded: they come from environment settings with sensible defaults,
so an organization can tune them without a code change:

    FOCUS_BUDGETS          "HIGH:4,MEDIUM:2,LOW:1"   max questions + follow-ups spent on one focus area
    FOCUS_FOLLOWUP_DELTA   "HIGH:1,MEDIUM:0,LOW:-1"  follow-ups a question on that focus gets, vs the interview's base
    FOCUS_HIGH_SHARE       0.25                      share of the total weight at/above which a skill is HIGH
    FOCUS_MEDIUM_SHARE     0.10                      ...and at/above which it is MEDIUM (below: LOW)

Everything here is deterministic code. No LLM decides a priority, a budget or a state change.
"""
import os
import re
from dataclasses import dataclass, field

from topic_guard import stems

PRIORITIES = ("HIGH", "MEDIUM", "LOW")
SOURCES = ("HR", "ROLE", "RESUME")
_RANK = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}

DEFAULT_BUDGETS = {"HIGH": 4, "MEDIUM": 2, "LOW": 1}
DEFAULT_FOLLOWUP_DELTA = {"HIGH": 1, "MEDIUM": 0, "LOW": -1}
MAX_FOLLOWUPS = 5          # hard ceiling regardless of configuration (matches the API's own limit)
MAX_FOCUS_AREAS = 12       # a round with dozens of "priorities" has none: keep the plan usable


def _env_map(name: str, default: dict[str, int]) -> dict[str, int]:
    """Parse "HIGH:4,MEDIUM:2,LOW:1". Anything malformed falls back to the default for that key."""
    result = dict(default)
    for part in (os.getenv(name) or "").split(","):
        key, _, value = part.partition(":")
        key = key.strip().upper()
        if key in PRIORITIES:
            try:
                result[key] = int(value.strip())
            except ValueError:
                pass
    return result


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name) or default)
    except ValueError:
        return default


def budgets() -> dict[str, int]:
    return {k: max(0, v) for k, v in _env_map("FOCUS_BUDGETS", DEFAULT_BUDGETS).items()}


def followup_deltas() -> dict[str, int]:
    return _env_map("FOCUS_FOLLOWUP_DELTA", DEFAULT_FOLLOWUP_DELTA)


@dataclass
class FocusArea:
    id: str
    name: str
    priority: str = "MEDIUM"
    sources: list[str] = field(default_factory=lambda: ["HR"])
    weight: float | None = None
    note: str = ""          # what the hiring team wants validated, in their own words (never spoken verbatim)

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "priority": self.priority, "sources": list(self.sources),
                "weight": self.weight, "note": self.note}

    @classmethod
    def from_dict(cls, raw: dict) -> "FocusArea":
        return cls(id=str(raw.get("id") or ""), name=str(raw.get("name") or ""),
                   priority=raw.get("priority") if raw.get("priority") in PRIORITIES else "MEDIUM",
                   sources=[s for s in raw.get("sources", []) if s in SOURCES] or ["HR"],
                   weight=_number(raw.get("weight")), note=str(raw.get("note") or ""))


def _clean_name(value) -> str:
    text = re.sub(r"<[^>]*>", "", str(value or ""))                      # drop markup, keep what was really written
    return re.sub(r"\s+", " ", re.sub(r"[<>{}\[\]`]", "", text)).strip(" .,;:-_")[:80]


def _number(value) -> float | None:
    """A finite non-negative number, or None. Weights arrive from forms and APIs: never trust their type."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and number not in (float("inf"), float("-inf")) and number >= 0 else None


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def priority_from_share(share: float) -> str:
    """HIGH / MEDIUM / LOW from a skill's share (0-1) of the total weight."""
    if share >= _env_float("FOCUS_HIGH_SHARE", 0.25):
        return "HIGH"
    if share >= _env_float("FOCUS_MEDIUM_SHARE", 0.10):
        return "MEDIUM"
    return "LOW"


def _prioritise(items: list[dict]) -> list[str | None]:
    """An explicit, valid priority always wins; otherwise it comes from the item's share of the total weight."""
    weights = [_number(item.get("weight")) or 0.0 for item in items]
    total = sum(weights)
    out: list[str | None] = []
    for item, weight in zip(items, weights):
        explicit = str(item.get("priority") or "").upper()
        if explicit in PRIORITIES:
            out.append(explicit)
        elif total > 0:
            out.append(priority_from_share(weight / total))
        else:
            out.append(None)
    return out


def _merge(into: dict[str, FocusArea], area: FocusArea) -> None:
    existing = into.get(_key(area.name))
    if existing is None:
        into[_key(area.name)] = area
        return
    existing.sources = [s for s in SOURCES if s in set(existing.sources) | set(area.sources)]
    if _RANK[area.priority] < _RANK[existing.priority]:
        existing.priority = area.priority                       # the more demanding priority wins
    existing.note = existing.note or area.note
    if existing.weight is None:
        existing.weight = area.weight


def build_focus_areas(hr_focus: list[dict] | None = None, role_competencies: list[dict] | None = None,
                      *, max_areas: int = MAX_FOCUS_AREAS) -> list[FocusArea]:
    """Merge HR and ROLE focus areas (RESUME claims are tracked separately, see planner). HR wins on conflict."""
    merged: dict[str, FocusArea] = {}

    hr_items = [i for i in (hr_focus or []) if isinstance(i, dict) and _clean_name(i.get("name"))]
    for item, priority in zip(hr_items, _prioritise(hr_items)):
        name = _clean_name(item.get("name"))
        _merge(merged, FocusArea(id="", name=name, priority=priority or "MEDIUM", sources=["HR"],
                                 weight=_number(item.get("weight")), note=str(item.get("note") or "")[:300]))

    role_items = [{"name": _clean_name(c.get("label") or str(c.get("key", "")).replace("_", " ").title()),
                   "weight": c.get("weight")} for c in (role_competencies or []) if isinstance(c, dict)]
    role_items = [i for i in role_items if i["name"]]
    for item, priority in zip(role_items, _prioritise(role_items)):
        _merge(merged, FocusArea(id="", name=item["name"], priority=priority or "LOW", sources=["ROLE"],
                                 weight=_number(item.get("weight"))))

    areas = sorted(merged.values(), key=lambda a: (_RANK[a.priority], -(a.weight or 0)))[:max_areas]
    for index, area in enumerate(areas, 1):
        area.id = f"f_{index}"
    return areas


def match_focus(text: str, areas: list[FocusArea]) -> FocusArea | None:
    """The focus area a question/topic belongs to: the one sharing the most meaningful words with `text`.
    HR areas win ties. Returns None when nothing genuinely overlaps (no forced, misleading tag)."""
    words = stems(text or "")
    if not words:
        return None
    best, best_score = None, 0.0
    for area in areas:
        area_words = stems(area.name)
        if not area_words:
            continue
        overlap = len(words & area_words)
        if overlap == 0:
            continue
        score = overlap / len(area_words) + (0.01 if "HR" in area.sources else 0)
        if score > best_score:
            best, best_score = area, score
    return best if best is not None and best_score >= 0.5 else None


def followup_limit(base: int, priority: str | None) -> int:
    """Follow-ups a question may get: the interview's base limit adjusted for its focus priority, clamped."""
    delta = followup_deltas().get(priority or "", 0)
    return max(0, min(MAX_FOLLOWUPS, int(base) + delta))


def focus_budget(priority: str | None) -> int:
    """Most questions + follow-ups worth spending on one focus area of this priority."""
    return budgets().get(priority or "", budgets()["MEDIUM"])
