"""`re` with a boundary that understands Devanagari.

Python's \\b treats Devanagari vowel signs and nasal marks (ा ि ी ँ ं ...) as non-word characters, so a pattern such as
r"हाँ\\b" never matches the word "हाँ" at the end of an utterance ("haan" in Hindi was ignored, "नहीं" too). Modules that
match Hindi text use `import re_dev as re`: every \\b in a pattern becomes a boundary over "letters, digits and any
Devanagari code point". Everything else on the module is the standard library `re`, unchanged.
"""
import functools
import re as _re
from re import *            # noqa: F401,F403  (re-export the public API: I, escape, sub, findall, ...)

_W = r"\wऀ-ॿ"
_BOUNDARY = rf"(?:(?<=[{_W}])(?![{_W}])|(?<![{_W}])(?=[{_W}]))"


@functools.lru_cache(maxsize=1024)
def _translate(pattern: str) -> str:
    out, i, in_class = [], 0, False
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\" and i + 1 < len(pattern):
            nxt = pattern[i + 1]
            out.append(_BOUNDARY if (nxt == "b" and not in_class) else ch + nxt)
            i += 2
            continue
        if ch == "[" and not in_class:
            in_class = True
        elif ch == "]" and in_class:
            in_class = False
        out.append(ch)
        i += 1
    return "".join(out)


def _p(pattern):
    return _translate(pattern) if isinstance(pattern, str) else pattern


def compile(pattern, flags=0):
    return _re.compile(_p(pattern), flags)


def search(pattern, string, flags=0):
    return _re.search(_p(pattern), string, flags)


def match(pattern, string, flags=0):
    return _re.match(_p(pattern), string, flags)


def fullmatch(pattern, string, flags=0):
    return _re.fullmatch(_p(pattern), string, flags)


def sub(pattern, repl, string, count=0, flags=0):
    return _re.sub(_p(pattern), repl, string, count, flags)


def findall(pattern, string, flags=0):
    return _re.findall(_p(pattern), string, flags)


def finditer(pattern, string, flags=0):
    return _re.finditer(_p(pattern), string, flags)


def split(pattern, string, maxsplit=0, flags=0):
    return _re.split(_p(pattern), string, maxsplit, flags)
