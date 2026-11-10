"""Run controller: one run = fresh workspace + trace + validation + failure candidates."""
from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from agents.prompts import PROMPT_VERSION, prompt_hash
from experiments.task_loader import load_task
from failures.rules import detect_candidates
from tools.gateway import TOOL_VERSION, ToolGateway
from tracing.collector import TraceCollector, read_events, utc_now
from tracing.schema import RunMetadata
from validation.validators import VALIDATORS

FRAMEWORKS = ("langgraph", "crewai")


def _ver(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "not-installed"


def run_one(task_id: str, framework: str, model_name: str, root: str | Path = ".", model=None,
            model_info: dict | None = None, backoff_s: float = 3.0, rep: int = 1) -> dict:
    root = Path(root).resolve()
    if framework not in FRAMEWORKS:
        raise ValueError(f"framework must be one of {FRAMEWORKS}")
    task = load_task(root, task_id)

    if model is None:
        from models.providers import get_chat_model

        model, model_info = get_chat_model(model_name)
    model_info = model_info or {"provider": "custom", "model_id": model_name}

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = f"{task_id}_{framework}_{model_name}_{stamp}_{uuid.uuid4().hex[:4]}"
    ws_in = root / "workspaces" / run_id / "input"
    ws_out = root / "workspaces" / run_id / "output"
    ws_in.mkdir(parents=True)
    ws_out.mkdir(parents=True)
    for f in task.input_files:
        shutil.copy(root / "sample_data" / f, ws_in / f)  # fresh copy per run

    run_dir = root / "results" / run_id
    collector = TraceCollector(run_id, run_dir)
    meta = RunMetadata(
        run_id=run_id, task_id=task.task_id, task_version=task.version, framework=framework,
        framework_version=_ver(framework if framework != "langgraph" else "langgraph"),
        provider=model_info["provider"], model_name=model_name, model_id=model_info["model_id"], repetition=rep,
        prompt_version=PROMPT_VERSION, prompt_hash=prompt_hash(task),
        max_steps=task.maximum_steps, max_retries=task.maximum_retries,
        tool_versions={t: TOOL_VERSION for t in task.allowed_tools},
        dependency_versions={p: _ver(p) for p in ("langgraph", "langchain-core", "crewai", "pydantic")},
        started_at=utc_now(),
    )
    collector.write_json("metadata.json", meta)

    gateway = ToolGateway(collector, ws_in, ws_out, task.allowed_tools, task.fault_schedule)
    outcome: dict = {"status": "crashed", "steps": 0, "final_answer": None}
    try:
        if framework == "langgraph":
            from agents.langgraph_agent import run_langgraph_agent

            outcome = run_langgraph_agent(model, gateway, collector, task, backoff_s=backoff_s)
        else:
            from agents.crewai_agent import run_crewai_agent  # written in the next stage

            outcome = run_crewai_agent(model, gateway, collector, task, backoff_s=backoff_s)
    except Exception as e:  # noqa: BLE001  a crash must not lose the trace
        collector.emit("error", status="error", error={"code": "RUN_CRASHED", "message": str(e)[:300]})
        outcome["status"] = "crashed"
    collector.emit("run_end", content=outcome["status"], status="ok" if outcome["status"] == "completed" else "error")

    truth = json.loads((root / task.expected_output_ref).read_text(encoding="utf-8"))
    events = read_events(collector.path)
    validation = VALIDATORS[task.validator](ws_out, truth, outcome.get("final_answer"), events)
    annotations = detect_candidates(events, validation, outcome["status"])

    meta.status = outcome["status"]
    meta.steps = outcome.get("steps", 0)
    meta.final_answer = outcome.get("final_answer")
    meta.ended_at = utc_now()
    collector.write_json("metadata.json", meta)
    collector.write_json("validation.json", validation)
    collector.write_json("annotations.json", [a.model_dump() for a in annotations])
    return {"run_id": run_id, "run_dir": str(run_dir), "status": outcome["status"], "validation": validation["status"],
            "false_success": validation["false_success"], "events": len(events), "failure_candidates": len(annotations)}
