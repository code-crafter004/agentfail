"""CrewAI adapter tests. A scripted fake LLM drives a REAL CrewAI crew (ReAct text path), so no API key is needed."""
import json
from pathlib import Path

import pytest

pytest.importorskip("crewai")

from crewai import BaseLLM  # noqa: E402

from experiments.controller import run_one  # noqa: E402
from tests.test_langgraph_slice import _report  # noqa: E402


class ScriptedCrewLLM(BaseLLM):
    llm_type: str = "scripted"
    script: list = []

    def call(self, messages, tools=None, callbacks=None, available_functions=None, from_task=None,
             from_agent=None, response_model=None):
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return True

    def get_context_window_size(self) -> int:
        return 8192


def act(tool, args):
    return f"Thought: I should use a tool.\nAction: {tool}\nAction Input: {json.dumps(args)}"


def final(text):
    return f"Thought: I now know the final answer\nFinal Answer: {text}"


def run(root, script, task="TASK-001"):
    llm = ScriptedCrewLLM(model="scripted", script=script)
    return run_one(task, "crewai", "fake", root=root, model=llm, model_info={"provider": "fake", "model_id": "fake-crew"}, backoff_s=0)


def events(res):
    return [json.loads(l) for l in (Path(res["run_dir"]) / "events.jsonl").read_text().splitlines()]


def test_success_run_has_same_event_shape_as_langgraph(project):
    root, truth = project
    res = run(root, [
        act("read_csv", {"file_name": "sales_data.csv"}),
        act("aggregate_revenue", {"file_name": "sales_data.csv", "group_column": "month", "value_column": "revenue"}),
        act("write_report", {"file_name": "sales_report.md", "content": _report(truth)}),
        final("Done. The report has been written successfully."),
    ])
    assert res["status"] == "completed" and res["validation"] == "pass" and not res["false_success"]
    ev = events(res)
    types = [e["type"] for e in ev]
    assert types[0] == "run_start" and types[-1] == "run_end"
    assert types.count("llm_request") == types.count("llm_response") == 4
    assert [e["tool_name"] for e in ev if e["type"] == "tool_call"] == ["read_csv", "aggregate_revenue", "write_report"]
    assert "final_answer" in types
    meta = json.loads((Path(res["run_dir"]) / "metadata.json").read_text())
    assert meta["framework"] == "crewai" and meta["model_id"] == "fake-crew" and meta["steps"] == 4


def test_bad_arguments_reach_the_gateway_and_are_logged(project):
    root, truth = project
    res = run(root, [
        act("read_csv", {"wrong_name": "sales_data.csv"}),          # missing required arg, extra arg
        act("read_csv", {"file_name": "sales_data.csv"}),
        final("I could not finish."),
    ])
    codes = [e.get("error", {}).get("code") for e in events(res) if e["type"] == "tool_result"]
    assert codes[0] == "SCHEMA_INVALID" and len(codes) == 2
    ann = json.loads((Path(res["run_dir"]) / "annotations.json").read_text())
    assert any(a["subcategory"] == "schema_invalid_arguments" for a in ann)


def test_identical_calls_are_not_cached(project):
    root, _ = project
    call = act("mock_api_request", {"endpoint": "/sales/summary"})
    res = run(root, [call, call, call, final("gave up")], task="TASK-003")
    results = [e for e in events(res) if e["type"] == "tool_result"]
    assert len(results) == 3  # all three reached the gateway
    assert sum(1 for e in results if e.get("meta", {}).get("injected_fault")) == 2  # 2 injected timeouts, then success


def test_task3_recovery_passes(project):
    root, _ = project
    call = act("mock_api_request", {"endpoint": "/sales/summary"})
    res = run(root, [call, call, call,
                     act("write_report", {"file_name": "summary.md", "content": "Total orders: 1240. Currency: INR."}),
                     final("Done.")], task="TASK-003")
    assert res["validation"] == "pass"


def test_step_limit(project):
    root, _ = project
    loop = [act("read_csv", {"file_name": "sales_data.csv"}) for _ in range(40)]
    res = run(root, loop)
    assert res["status"] == "max_steps" and res["validation"] == "fail"
    assert any(e.get("error", {}).get("code") == "MAX_STEPS_EXCEEDED" for e in events(res))
    assert sum(1 for e in events(res) if e["type"] == "llm_request") == 10  # TASK-001 limit


def test_provider_error_keeps_partial_trace(project):
    root, _ = project
    res = run(root, [act("read_csv", {"file_name": "sales_data.csv"}), RuntimeError("401 invalid api key")])
    assert res["status"] == "provider_error"
    assert any(e["type"] == "tool_result" for e in events(res))


def test_hooks_are_removed_after_each_run(project):
    from crewai.hooks.llm_hooks import get_after_llm_call_hooks, get_before_llm_call_hooks

    root, _ = project
    run(root, [final("nothing to do")])
    assert not get_before_llm_call_hooks() and not get_after_llm_call_hooks()


def test_tools_advertise_our_exact_schema(project):
    from agents.crewai_agent import build_tools
    from tools.gateway import TOOL_SPECS, ToolGateway
    from tracing.collector import TraceCollector
    from types import SimpleNamespace

    root, _ = project
    gw = ToolGateway(TraceCollector("R", root / "r"), root / "in", root / "out", ["filter_csv", "read_csv"])
    for t in build_tools(gw, SimpleNamespace(last_event=None)):
        assert t.args_schema.model_json_schema()["properties"] == TOOL_SPECS[t.name]["parameters"]["properties"]
        assert t.args_schema.model_json_schema()["required"] == TOOL_SPECS[t.name]["parameters"]["required"]
