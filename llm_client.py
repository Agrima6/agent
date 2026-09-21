from llm_provider import get_provider


def structured_json(system_prompt: str, user_prompt: str, model: str | None = None, *,
                    reasoning_effort: str | None = None) -> dict:
    """Call the configured LLM provider and force a JSON object response. Untrusted content passed
    via user_prompt must already be wrapped/labeled by the caller as [UNTRUSTED].

    reasoning_effort="low" is for callers a person is waiting on (interview creation): on gpt-oss it
    cut a comparable call from 2.2s to 0.8s. Scoring keeps the default (thorough) effort.
    """
    return get_provider().complete_json(system_prompt, user_prompt, temperature=0.3, model=model,
                                        reasoning_effort=reasoning_effort, max_tokens=2500 if reasoning_effort else None)
