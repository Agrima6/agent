"""LLM provider abstraction for every structured (JSON) call the interview engine makes.

The engine (judge, composer, planner, scoring) depends only on LLMProvider.complete_json(); which
vendor answers is a deployment decision (LLM_PROVIDER=groq|openai). Groq and OpenAI both speak the
OpenAI chat-completions protocol, so one implementation covers them; a Gemini or Claude adapter
only needs to implement the same one method.

The realtime voice pipeline's own STT/LLM plugins (LiveKit) are configured separately in agent.py
- this module is for the request/response JSON calls where we need to validate what comes back.
"""
import json
import logging
import os
import re
import time
from typing import Protocol

from openai import NotFoundError, OpenAI

from config import GROQ_API_KEY, OPENAI_API_KEY

logger = logging.getLogger("llm-provider")


class LLMError(Exception):
    """Raised when a provider call fails or returns something that is not usable JSON."""


class LLMProvider(Protocol):
    name: str
    model: str

    def complete_json(self, system: str, user: str, *, temperature: float = 0.2,
                      max_tokens: int | None = None, timeout: float | None = None,
                      model: str | None = None, reasoning_effort: str | None = None,
                      retries: int = 1) -> dict: ...


_JSON_OBJECT = re.compile(r"\{.*\}", re.S)


def parse_json_object(text: str) -> dict:
    """Parse a model reply into a dict, tolerating code fences or stray prose around the object."""
    text = (text or "").strip()
    try:
        data = json.loads(text)
    except ValueError:
        match = _JSON_OBJECT.search(text)
        if not match:
            raise LLMError("model did not return a JSON object")
        try:
            data = json.loads(match.group(0))
        except ValueError as exc:
            raise LLMError("model returned malformed JSON") from exc
    if not isinstance(data, dict):
        raise LLMError("model returned JSON that is not an object")
    return data


# Reasoning models spend hidden "reasoning" tokens BEFORE the visible answer, and those count against
# max_tokens. On gpt-oss-120b a 25-word JSON reply used ~410 tokens at default effort but only ~130 at
# "low" (0.83s vs 2.23s measured on Groq) - and a max_tokens of 300 was exhausted before the JSON was
# finished, which Groq reports as a 400 json_validate_failed.
_REASONING_MODELS = ("gpt-oss", "o1", "o3", "o4", "gpt-5")


class OpenAICompatibleProvider:
    def __init__(self, name: str, api_key: str, base_url: str | None, model: str,
                 default_timeout: float = 20.0):
        self.name = name
        self.model = model
        self._default_timeout = default_timeout
        self._client = OpenAI(api_key=api_key or "missing", base_url=base_url,
                              timeout=default_timeout, max_retries=1)

    def complete_json(self, system: str, user: str, *, temperature: float = 0.2,
                      max_tokens: int | None = None, timeout: float | None = None,
                      model: str | None = None, reasoning_effort: str | None = None,
                      retries: int = 1) -> dict:
        """retries: SDK-level retries (with backoff that honours 429 retry-after). Live-conversation
        callers pass 0: a request that has to wait seconds is worse than falling back immediately."""
        started = time.perf_counter()
        chosen_model = model or self.model
        kwargs: dict = {}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if reasoning_effort and any(tag in chosen_model for tag in _REASONING_MODELS):
            kwargs["reasoning_effort"] = reasoning_effort
        try:
            resp = self._client.with_options(timeout=timeout or self._default_timeout,
                                             max_retries=retries).chat.completions.create(
                model=chosen_model,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=temperature,
                **kwargs,
            )
            content = resp.choices[0].message.content
        except NotFoundError:
            if chosen_model != self.model:
                # A per-role model this key can't use: fall back to the provider's default model
                # rather than failing every call for that role.
                logger.warning("model %s unavailable on %s - falling back to %s", chosen_model, self.name, self.model)
                return self.complete_json(system, user, temperature=temperature, max_tokens=max_tokens,
                                          timeout=timeout, model=self.model, reasoning_effort=reasoning_effort,
                                          retries=retries)
            raise LLMError(f"{self.name} model not found") from None
        except Exception as exc:
            # The wrapper message stays short and safe to show; the real cause goes to the log so a
            # 400 (bad request), 429 (rate limit) or timeout is diagnosable instead of anonymous.
            logger.warning("llm_call_failed provider=%s model=%s latency_ms=%d error=%s: %s", self.name, chosen_model,
                           int((time.perf_counter() - started) * 1000), type(exc).__name__, str(exc)[:300])
            raise LLMError(f"{self.name} request failed: {type(exc).__name__}") from exc
        logger.info("llm_call provider=%s model=%s latency_ms=%d", self.name, chosen_model,
                    int((time.perf_counter() - started) * 1000))
        return parse_json_object(content)


# Per-role model choice. Groq rate-limits PER MODEL, and the two live jobs have different needs: the
# judge decides follow-up routing (accuracy matters) while the composer only phrases one sentence
# (speed matters: gpt-oss-20b 0.53s vs gpt-oss-120b 0.83s, equal quality on this task). Splitting them
# also spreads token usage across two rate-limit buckets. Override with JUDGE_MODEL / COMPOSER_MODEL.
_ROLE_DEFAULTS = {"groq": {"composer": "openai/gpt-oss-20b"}}


def role_model(role: str) -> str | None:
    """The model to use for `role` ('judge' | 'composer'); None means the provider's default model."""
    explicit = os.getenv(f"{role.upper()}_MODEL")
    if explicit:
        return explicit
    return _ROLE_DEFAULTS.get(get_provider().name, {}).get(role)


_provider: LLMProvider | None = None


def get_provider() -> LLMProvider:
    global _provider
    if _provider is None:
        choice = (os.getenv("LLM_PROVIDER") or "groq").lower()
        if choice == "openai" and OPENAI_API_KEY:
            _provider = OpenAICompatibleProvider("openai", OPENAI_API_KEY, None,
                                                 os.getenv("LLM_MODEL", "gpt-4o-mini"))
        else:
            _provider = OpenAICompatibleProvider("groq", GROQ_API_KEY, "https://api.groq.com/openai/v1",
                                                 os.getenv("LLM_MODEL", "openai/gpt-oss-120b"))
    return _provider


def set_provider(provider: LLMProvider | None) -> None:
    """Swap the provider (tests inject a fake; None resets to the configured default)."""
    global _provider
    _provider = provider
