"""Load and validate versioned task definitions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str
    version: str
    title: str
    domain: str
    difficulty: str
    instructions: str
    input_files: list[str]
    allowed_tools: list[str]
    constraints: list[str] = []
    expected_output_ref: str
    validator: str
    fault_schedule: dict = {}
    maximum_steps: int = 10
    maximum_retries: int = 2


def load_task(root: Path, task_id: str) -> TaskSpec:
    for p in sorted((Path(root) / "benchmark_tasks").glob("*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("task_id") == task_id:
            return TaskSpec(**data)
    raise FileNotFoundError(f"Task {task_id} not found in {Path(root) / 'benchmark_tasks'}")


def list_tasks(root: Path) -> list[TaskSpec]:
    return [TaskSpec(**json.loads(p.read_text(encoding="utf-8"))) for p in sorted((Path(root) / "benchmark_tasks").glob("*.json"))]
