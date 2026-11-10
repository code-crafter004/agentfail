"""Rule based failure CANDIDATES. These are pre-labels for human review, not proven causes."""
from __future__ import annotations

import json

from tracing.schema import Annotation, Event

REPEAT_THRESHOLD = 2  # identical failed calls in a row


def detect_candidates(events: list[Event], validation: dict, run_status: str) -> list[Annotation]:
    anns: list[Annotation] = []
    n = 0

    def add(evidence, category, sub, severity, conf="low", note=None):
        nonlocal n
        n += 1
        anns.append(
            Annotation(
                failure_id=f"F-{n:02d}", evidence_event_ids=evidence, category=category, subcategory=sub,
                severity=severity, attribution_confidence=conf, note=note,
            )
        )

    calls = {e.event_id: e for e in events if e.type == "tool_call"}
    last_key, streak, streak_ids = None, 0, []
    for e in events:
        if e.type != "tool_result":
            continue
        call = calls.get(e.parent_event_id or "")
        code = (e.error or {}).get("code")
        if e.status == "error":
            if e.meta.get("injected_fault"):
                add([e.event_id], "environment_system", "injected_fault", "low", "high", "Injected by benchmark, not an agent error")
            elif code == "SCHEMA_INVALID":
                add([e.event_id], "tool_use", "schema_invalid_arguments", "medium", "medium")
            elif code in ("UNKNOWN_TOOL", "TOOL_NOT_ALLOWED"):
                add([e.event_id], "tool_use", "wrong_tool", "medium", "medium")
            elif code == "PATH_DENIED":
                add([e.event_id], "safety_policy", "blocked_path_or_action", "critical", "medium")
            else:
                add([e.event_id], "tool_use", f"tool_error_{(code or 'unknown').lower()}", "low", "low",
                    "Check if agent supplied wrong args or the environment failed")
            key = json.dumps([call.tool_name, call.arguments], sort_keys=True, default=str) if call else None
            if key and key == last_key:
                streak += 1
                streak_ids.append(e.event_id)
                if streak == REPEAT_THRESHOLD - 1:
                    add(streak_ids[-2:], "verification_recovery", "repeated_identical_failed_call", "medium", "medium")
            else:
                last_key, streak, streak_ids = key, 0, [e.event_id]
        else:
            last_key, streak, streak_ids = None, 0, []

    if run_status == "max_steps":
        ids = [e.event_id for e in events if e.type == "error"][-1:]
        add(ids, "planning_reasoning", "step_limit_exceeded", "high", "low")
    if run_status == "provider_error":
        ids = [e.event_id for e in events if e.type == "error"][-1:]
        add(ids, "environment_system", "provider_error", "high", "medium")
    if validation.get("status") == "fail":
        fa = [e.event_id for e in events if e.type == "final_answer"]
        add(fa[-1:] or [events[-1].event_id] if events else [], "unknown_uncertain", "output_validation_failed",
            "high", "unknown", "Final output failed validation. Needs manual review to find the cause.")
    if validation.get("false_success"):
        fa = [e.event_id for e in events if e.type == "final_answer"]
        add(fa[-1:], "verification_recovery", "false_success_claim", "high", "high",
            "Agent claimed completion but validator failed")
    return anns
