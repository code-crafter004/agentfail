"""Metrics. Reads results/*/ and writes reports/*.csv and reports/summary.md.

Run:  python3 -m evaluation.metrics

Definitions (denominators are explicit):
- infra error run: status provider_error or crashed. Excluded from success rate, counted separately.
- success rate: pass / (pass + fail), over non infra runs. 95 percent Wilson interval given.
- false success rate: false_success / runs with an explicit completion claim and a determinate validation.
- recovery rate: resolved failure episodes / eligible episodes. An episode is a streak of error results
  for the same tool. It is resolved if a later call to that tool succeeded. Injected faults are included.
- schema validity rate: calls without SCHEMA_INVALID / all tool calls (does not imply semantic correctness).
- repetition rate: tool calls identical (name + arguments) to an earlier call in the run / all tool calls.
- first error step: sequence number of the first NON injected tool error. First error, not proven first critical failure.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from tracing.collector import read_events

INFRA = {"provider_error", "crashed"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def _load(run_dir: Path) -> dict | None:
    try:
        meta = json.loads((run_dir / "metadata.json").read_text())
        val = json.loads((run_dir / "validation.json").read_text())
    except Exception:  # noqa: BLE001  run still in progress or interrupted before analysis
        return None
    ann = []
    if (run_dir / "annotations.json").exists():
        ann = json.loads((run_dir / "annotations.json").read_text())
    return {"meta": meta, "val": val, "events": read_events(run_dir / "events.jsonl"), "ann": ann}


def recovery_episodes(events) -> tuple[int, int]:
    """(eligible, resolved) over streaks of error results per tool."""
    calls = [e for e in events if e.type == "tool_result"]
    eligible = resolved = 0
    i = 0
    while i < len(calls):
        e = calls[i]
        if e.status == "error":
            tool = e.tool_name
            j = i
            while j < len(calls) and calls[j].status == "error" and calls[j].tool_name == tool:
                j += 1
            eligible += 1
            if any(c.status == "ok" and c.tool_name == tool for c in calls[j:]):
                resolved += 1
            i = j
        else:
            i += 1
    return eligible, resolved


def run_row(r: dict) -> dict:
    m, v, ev = r["meta"], r["val"], r["events"]
    calls = [e for e in ev if e.type == "tool_call"]
    results = [e for e in ev if e.type == "tool_result"]
    seen, repeats = set(), 0
    for c in calls:
        key = json.dumps([c.tool_name, c.arguments], sort_keys=True, default=str)
        repeats += key in seen
        seen.add(key)
    schema_bad = sum(1 for e in results if (e.error or {}).get("code") == "SCHEMA_INVALID")
    agent_errs = [e for e in results if e.status == "error" and not e.meta.get("injected_fault")]
    injected = sum(1 for e in results if e.meta.get("injected_fault"))
    elig, res = recovery_episodes(ev)
    try:
        t0 = datetime.fromisoformat(ev[0].timestamp)
        t1 = datetime.fromisoformat(ev[-1].timestamp)
        dur = round((t1 - t0).total_seconds(), 2)
    except Exception:  # noqa: BLE001
        dur = ""
    cats = Counter(a["category"] for a in r["ann"])
    infra = m["status"] in INFRA
    return {
        "run_id": m["run_id"], "task_id": m["task_id"], "framework": m["framework"],
        "model_name": m.get("model_name", ""), "model_id": m["model_id"], "repetition": m.get("repetition", 1),
        "status": m["status"], "infra_error": infra,
        "validation": "infra_error" if infra else v["status"],
        "completion_claim": v["completion_claim"],
        "false_success": v["false_success"] and not infra,
        "llm_steps": m.get("steps", 0), "tool_calls": len(calls),
        "schema_invalid_calls": schema_bad, "repeated_calls": repeats,
        "agent_tool_errors": len(agent_errs), "injected_faults": injected,
        "first_error_step": agent_errs[0].sequence_number if agent_errs else "",
        "recovery_eligible": elig, "recovery_resolved": res,
        "provider_retries": sum(1 for e in ev if e.type == "retry"),
        "duration_s": dur, "failure_candidates": len(r["ann"]),
        "categories": ";".join(f"{k}:{n}" for k, n in sorted(cats.items())),
    }


def summarize(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    out = []
    for g, rs in sorted(groups.items()):
        ok = [r for r in rs if not r["infra_error"]]
        passed = sum(r["validation"] == "pass" for r in ok)
        failed = sum(r["validation"] == "fail" for r in ok)
        det = passed + failed
        claims = [r for r in ok if r["completion_claim"] and r["validation"] in ("pass", "fail")]
        fs = sum(r["false_success"] for r in claims)
        calls = sum(r["tool_calls"] for r in ok)
        elig = sum(r["recovery_eligible"] for r in ok)
        lo, hi = wilson(passed, det)
        cats = Counter()
        for r in ok:
            for part in filter(None, r["categories"].split(";")):
                k, n = part.rsplit(":", 1)
                cats[k] += int(n)
        row = dict(zip(keys, g))
        row.update({
            "runs": len(rs), "infra_error_runs": len(rs) - len(ok), "determinate_runs": det,
            "pass": passed, "fail": failed,
            "success_rate": round(passed / det, 3) if det else "",
            "success_ci95_low": round(lo, 3) if det else "", "success_ci95_high": round(hi, 3) if det else "",
            "runs_with_claim": len(claims), "false_success": fs,
            "false_success_rate": round(fs / len(claims), 3) if claims else "",
            "recovery_eligible": elig, "recovery_resolved": sum(r["recovery_resolved"] for r in ok),
            "recovery_rate": round(sum(r["recovery_resolved"] for r in ok) / elig, 3) if elig else "",
            "schema_validity_rate": round(1 - sum(r["schema_invalid_calls"] for r in ok) / calls, 3) if calls else "",
            "repetition_rate": round(sum(r["repeated_calls"] for r in ok) / calls, 3) if calls else "",
            "step_limit_runs": sum(r["status"] == "max_steps" for r in ok),
            "avg_llm_steps": round(sum(r["llm_steps"] for r in ok) / len(ok), 2) if ok else "",
            "avg_tool_calls": round(calls / len(ok), 2) if ok else "",
            "avg_duration_s": round(sum(float(r["duration_s"] or 0) for r in ok) / len(ok), 2) if ok else "",
        })
        for k, n in cats.items():
            row[f"cat_{k}"] = n
        out.append(row)
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval=0)
        w.writeheader()
        w.writerows(rows)


def _md_table(rows: list[dict], cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def build(results_dir: str | Path = "results", reports_dir: str | Path = "reports") -> dict:
    results_dir, reports_dir = Path(results_dir), Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    runs = [x for d in sorted(results_dir.glob("*/")) if (x := _load(d))]
    rows = [run_row(r) for r in runs]
    by_cfg = summarize(rows, ("framework", "model_id"))
    by_task = summarize(rows, ("task_id", "framework", "model_id"))
    _write_csv(reports_dir / "run_metrics.csv", rows)
    _write_csv(reports_dir / "summary_by_config.csv", by_cfg)
    _write_csv(reports_dir / "summary_by_task.csv", by_task)
    cols = ["framework", "model_id", "runs", "infra_error_runs", "determinate_runs", "success_rate",
            "success_ci95_low", "success_ci95_high", "false_success_rate", "recovery_rate", "schema_validity_rate",
            "repetition_rate", "avg_llm_steps", "avg_duration_s"]
    tcols = ["task_id", "framework", "model_id", "determinate_runs", "pass", "fail", "success_rate", "false_success"]
    cat_cols = sorted({k for r in by_cfg for k in r if k.startswith("cat_")})
    md = [
        "# AgentFail results summary", "",
        f"Runs found: {len(rows)}. Infra error runs (provider error or crash) are excluded from success rate.", "",
        "## By framework and model", "", _md_table(by_cfg, cols), "",
        "## Failure candidate categories (rule based, not human reviewed)", "",
        _md_table(by_cfg, ["framework", "model_id"] + cat_cols) if cat_cols else "No failure candidates yet.", "",
        "## By task", "", _md_table(by_task, tcols), "",
        "Notes: small numbers of runs give wide intervals. Rule based categories are candidates for manual review.",
    ]
    (reports_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    return {"runs": len(rows), "configs": len(by_cfg)}


if __name__ == "__main__":
    out = build()
    print(out, "-> reports/run_metrics.csv, reports/summary_by_config.csv, reports/summary_by_task.csv, reports/summary.md")
