"""Batch runner: tasks x frameworks x models x repetitions.

Sequential, interleaved (shuffled with a fixed seed), resumable. A run counts as done
when its status is completed or max_steps. Provider errors and crashes are re-run on the
next invocation, and their old folders stay in results/ (metrics treat them as infra errors).
"""
from __future__ import annotations

import json
import random
import time
import traceback
from pathlib import Path
from typing import Callable, Optional

from experiments.controller import run_one
from experiments.task_loader import list_tasks

DONE_STATUSES = {"completed", "max_steps"}


def done_keys(results_dir: Path) -> set[tuple]:
    keys = set()
    for p in Path(results_dir).glob("*/metadata.json"):
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if m.get("status") in DONE_STATUSES:
            keys.add((m["task_id"], m["framework"], m.get("model_name", ""), m.get("repetition", 1)))
    return keys


def plan(root: Path, tasks: list[str], frameworks: list[str], models: list[str], reps: int, seed: int = 42) -> list[tuple]:
    all_ids = [t.task_id for t in list_tasks(root)] if tasks == ["all"] else tasks
    combos = [(t, fw, m, r) for t in all_ids for fw in frameworks for m in models for r in range(1, reps + 1)]
    random.Random(seed).shuffle(combos)  # interleave so no config runs at a different time of day
    return combos


def run_batch(root: str | Path, tasks: list[str], frameworks: list[str], models: list[str], reps: int = 3,
              sleep_s: float = 4.0, backoff_s: float = 3.0, seed: int = 42, dry_run: bool = False,
              model_factory: Optional[Callable] = None, log: Callable = print) -> dict:
    root = Path(root).resolve()
    combos = plan(root, tasks, frameworks, models, reps, seed)
    done = done_keys(root / "results")
    todo = [c for c in combos if c not in done]
    log(f"planned={len(combos)} already_done={len(combos) - len(todo)} to_run={len(todo)}")
    if dry_run:
        for c in todo:
            log("  would run:", *c)
        return {"planned": len(combos), "to_run": len(todo), "ran": 0}
    ran = errors = 0
    for i, (task_id, fw, model_name, rep) in enumerate(todo, 1):
        kwargs = {}
        if model_factory:
            kwargs["model"], kwargs["model_info"] = model_factory(model_name)
        try:
            res = run_one(task_id, fw, model_name, root=root, rep=rep, backoff_s=backoff_s, **kwargs)
            log(f"[{i}/{len(todo)}] {task_id} {fw} {model_name} rep{rep} -> {res['status']} / {res['validation']}"
                f"{' / FALSE SUCCESS' if res['false_success'] else ''}")
            ran += 1
        except Exception as e:  # noqa: BLE001  one bad run must never stop the batch
            errors += 1
            log(f"[{i}/{len(todo)}] {task_id} {fw} {model_name} rep{rep} -> BATCH ERROR: {e!r}")
            traceback.print_exc()
        if sleep_s and i < len(todo):
            time.sleep(sleep_s)
    return {"planned": len(combos), "to_run": len(todo), "ran": ran, "errors": errors}
