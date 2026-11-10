"""Helpers shared by all framework adapters."""
from __future__ import annotations

import time
from typing import Any

from tracing.collector import TraceCollector


def text_of(msg: Any) -> str:
    """Message content as plain text (Gemini may return a list of parts)."""
    c = getattr(msg, "content", msg)
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for p in c:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and p.get("type", "text") == "text":
                parts.append(str(p.get("text", "")))
        return "".join(parts)
    return str(c)


def is_retryable(e: Exception) -> bool:
    s = str(e).lower()
    return any(k in s for k in ("429", "rate limit", "rate_limit", "resource_exhausted", "quota", "503", "overloaded", "timeout"))


def invoke_with_retry(model_with_tools, messages, collector: TraceCollector, parent: str, max_retries: int,
                      backoff_s: float, agent_id: str = "agent"):
    """Call the LLM. Retries only transient provider errors, and logs each retry.
    Raises the last exception if all attempts fail."""
    last: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return model_with_tools.invoke(messages)
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt < max_retries and is_retryable(e):
                collector.emit(
                    "retry", parent=parent, agent_id=agent_id, status="error",
                    error={"code": "PROVIDER_RETRY", "message": str(e)[:200]},
                    meta={"layer": "provider", "attempt": attempt + 1},
                )
                time.sleep(backoff_s * (2 ** attempt))
                continue
            raise
    raise last  # pragma: no cover
