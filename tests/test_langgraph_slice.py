import json
from pathlib import Path

from experiments.controller import run_one
from tests.conftest import ScriptedModel, tool_msg
from langchain_core.messages import AIMessage


def _report(truth, wrong=False):
    lines = ["# Sales report"]
    for m, v in truth["monthly_totals"].items():
        lines.append(f"- {m}: {v + (1000 if wrong else 0):.2f}")
    lines.append(f"Highest month: {'2025-01' if wrong else truth['highest_month']}")
    return "\n".join(lines)


def _script(truth, wrong=False):
    return [
        tool_msg("read_csv", {"file_name": "sales_data.csv"}, "a"),
        tool_msg("aggregate_revenue", {"file_name": "sales_data.csv", "group_column": "month", "value_column": "revenue"}, "b"),
        tool_msg("write_report", {"file_name": "sales_report.md", "content": _report(truth, wrong)}, "c"),
        AIMessage(content="Done. The report has been written successfully."),
    ]


def _events(res):
    return [json.loads(l) for l in (Path(res["run_dir"]) / "events.jsonl").read_text().splitlines()]


def test_success_run(project):
    root, truth = project
    res = run_one("TASK-001", "langgraph", "fake", root=root, model=ScriptedModel(_script(truth)),
                  model_info={"provider": "fake", "model_id": "fake-1"}, backoff_s=0)
    assert res["status"] == "completed" and res["validation"] == "pass" and not res["false_success"]
    d = Path(res["run_dir"])
    for f in ("metadata.json", "events.jsonl", "validation.json", "annotations.json"):
        assert (d / f).exists()
    types = [e["type"] for e in _events(res)]
    assert types[0] == "run_start" and types[-1] == "run_end" and "final_answer" in types
    assert json.loads((d / "metadata.json").read_text())["model_id"] == "fake-1"


def test_false_success_detected(project):
    root, truth = project
    res = run_one("TASK-001", "langgraph", "fake", root=root, model=ScriptedModel(_script(truth, wrong=True)), backoff_s=0)
    assert res["validation"] == "fail" and res["false_success"] is True
    ann = json.loads((Path(res["run_dir"]) / "annotations.json").read_text())
    assert any(a["subcategory"] == "false_success_claim" for a in ann)


def test_max_steps(project):
    root, _ = project
    loop = [tool_msg("read_csv", {"file_name": "sales_data.csv"}, str(i)) for i in range(30)]
    res = run_one("TASK-001", "langgraph", "fake", root=root, model=ScriptedModel(loop), backoff_s=0)
    assert res["status"] == "max_steps" and res["validation"] == "fail"


def test_provider_error_keeps_partial_trace(project):
    root, _ = project
    script = [tool_msg("read_csv", {"file_name": "sales_data.csv"}), RuntimeError("401 bad key")]
    res = run_one("TASK-001", "langgraph", "fake", root=root, model=ScriptedModel(script), backoff_s=0)
    assert res["status"] == "provider_error"
    assert any(e["type"] == "tool_result" for e in _events(res))  # partial trace survived


def test_rate_limit_retry_then_success(project):
    root, truth = project
    script = [RuntimeError("429 RESOURCE_EXHAUSTED")] + _script(truth)
    res = run_one("TASK-001", "langgraph", "fake", root=root, model=ScriptedModel(script), backoff_s=0)
    assert res["status"] == "completed" and res["validation"] == "pass"
    assert any(e["type"] == "retry" for e in _events(res))


def test_bad_args_then_recovery(project):
    root, truth = project
    script = [
        tool_msg("aggregate_revenue", {"file_name": "sales_data.csv", "group_column": "month", "value_column": "revenuee"}, "a"),
        tool_msg("read_csv", {"file_name": "sales_data.csv"}, "b"),
    ] + _script(truth)[1:]
    res = run_one("TASK-001", "langgraph", "fake", root=root, model=ScriptedModel(script), backoff_s=0)
    assert res["validation"] == "pass"
    ann = json.loads((Path(res["run_dir"]) / "annotations.json").read_text())
    assert any(a["subcategory"] == "tool_error_unknown_column" for a in ann)


def test_empty_reply_is_flagged_as_premature_termination(project):
    root, _ = project
    script = [tool_msg("read_csv", {"file_name": "sales_data.csv"}), AIMessage(content="", response_metadata={"finish_reason": "stop"})]
    res = run_one("TASK-001", "langgraph", "fake", root=root, model=ScriptedModel(script), backoff_s=0)
    assert res["status"] == "completed" and res["validation"] == "fail"
    ann = json.loads((Path(res["run_dir"]) / "annotations.json").read_text())
    assert any(a["subcategory"] == "premature_termination_empty_reply" for a in ann)
    assert any(e.get("meta", {}).get("finish_reason") == "stop" for e in _events(res))
