"""Trace schemas. Frozen contract: every framework adapter must emit these events."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

EventType = Literal[
    "run_start",
    "llm_request",
    "llm_response",
    "tool_call",
    "tool_result",
    "retry",
    "error",
    "final_answer",
    "run_end",
]


class Event(BaseModel):
    run_id: str
    event_id: str
    sequence_number: int
    parent_event_id: Optional[str] = None
    agent_id: str = "agent"
    timestamp: str
    type: EventType
    tool_name: Optional[str] = None
    arguments: Optional[Any] = None
    result: Optional[Any] = None
    content: Optional[str] = None
    status: Optional[Literal["ok", "error"]] = None
    error: Optional[dict] = None
    duration_ms: Optional[float] = None
    meta: dict = Field(default_factory=dict)


class RunMetadata(BaseModel):
    run_id: str
    task_id: str
    task_version: str
    framework: str
    framework_version: str
    provider: str
    model_name: str = ""
    model_id: str
    repetition: int = 1
    condition: str = "base"
    prompt_version: str
    prompt_hash: str
    max_steps: int
    max_retries: int
    tool_versions: dict = Field(default_factory=dict)
    dependency_versions: dict = Field(default_factory=dict)
    started_at: str
    ended_at: Optional[str] = None
    status: str = "running"  # running | completed | max_steps | provider_error | crashed
    steps: int = 0
    final_answer: Optional[str] = None


class Annotation(BaseModel):
    """Failure annotation. Rules create candidates, humans confirm them."""

    failure_id: str
    evidence_event_ids: list[str]
    category: str
    subcategory: Optional[str] = None
    severity: Optional[Literal["low", "medium", "high", "critical"]] = None
    attribution_confidence: Literal["low", "medium", "high", "unknown"] = "unknown"
    source: Literal["rule", "human"] = "rule"
    reviewer: Optional[str] = None
    reviewer_verdict: Optional[Literal["confirmed", "corrected", "rejected"]] = None
    note: Optional[str] = None
    annotation_version: str = "1"
