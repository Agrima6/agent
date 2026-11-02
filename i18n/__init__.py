"""Fixed interviewer lines for the Indian languages beyond English / Hindi / Hinglish.

One module per language code (i18n/pa.py, ta.py, ...), each defining LINES: {table name: text}. The table names are
those in refusals.py (CLOSING, END_CONFIRM, ...), plus "PROCESS_ANSWERS.duration", "PROCESS_ANSWERS.format",
"REFUSALS" ({intent name: [lines]}) and "FALLBACK_FOLLOWUP" ({standard, depth, clarify, generic}). A line a language does
not define is spoken in English (see refusals._Table) rather than failing, and tests/test_i18n_catalogs.py checks every
catalog for missing keys and broken placeholders. Hindi-script text is phrased without gendered first-person verbs where
the language allows, so it reads correctly with either voice. NOTE: written for review by a native speaker.
"""
import importlib

CODES = ("pa", "ta", "te", "bn", "mr", "gu", "kn", "ml")


def load() -> dict[str, dict]:
    catalogs = {}
    for code in CODES:
        try:
            catalogs[code] = importlib.import_module(f"i18n.{code}").LINES
        except ModuleNotFoundError:
            continue
    return catalogs


CATALOGS = load()
