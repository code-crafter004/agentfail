"""Shared prompts. Same text for every framework and model. Changing it = new version."""
from __future__ import annotations

import hashlib

PROMPT_VERSION = "1.0"

SYSTEM_PROMPT = (
    "You are an autonomous agent that completes tasks using the tools provided. "
    "Work step by step. Call tools to get real data instead of guessing. "
    "If a tool returns an error, read it and decide how to proceed. "
    "When the task is complete, reply with a short final summary and do not call more tools."
)


def build_task_prompt(task) -> str:
    lines = [f"Task: {task.instructions}"]
    if task.input_files:
        lines.append("Input files: " + ", ".join(task.input_files))
    if task.constraints:
        lines.append("Constraints: " + "; ".join(task.constraints))
    return "\n".join(lines)


def prompt_hash(task) -> str:
    return hashlib.sha256((SYSTEM_PROMPT + "\n" + build_task_prompt(task)).encode()).hexdigest()[:12]
