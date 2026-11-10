import json

import pytest

from tools.gateway import ToolGateway, safe_eval, ToolError
from tracing.collector import TraceCollector, read_events


@pytest.fixture
def gw(tmp_path):
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "d.csv").write_text("month,revenue\na,1.10\na,2.20\nb,5.00\n")
    (tmp_path / "secret.txt").write_text("hidden")
    c = TraceCollector("RUN-T", tmp_path / "res")
    g = ToolGateway(c, tmp_path / "in", tmp_path / "out", ["read_csv", "aggregate_revenue", "calculate", "write_report", "mock_api_request"],
                    fault_schedule={"mock_api_request": {"fail_first_n": 2, "error_code": "TIMEOUT"}})
    return g, c, tmp_path


def test_aggregate_ok_and_exact_decimal(gw):
    g, _, _ = gw
    r = g.call("aggregate_revenue", {"file_name": "d.csv", "group_column": "month", "value_column": "revenue"})
    assert r["status"] == "ok" and r["result"]["groups"] == {"a": 3.3, "b": 5.0}


def test_unknown_column(gw):
    g, _, _ = gw
    r = g.call("aggregate_revenue", {"file_name": "d.csv", "group_column": "month", "value_column": "revenuee"})
    assert r["error"]["code"] == "UNKNOWN_COLUMN"


def test_path_traversal_and_symlink_denied(gw):
    g, _, tmp = gw
    assert g.call("read_csv", {"file_name": "../secret.txt"})["error"]["code"] == "PATH_DENIED"
    assert g.call("read_csv", {"file_name": "/etc/passwd"})["error"]["code"] == "PATH_DENIED"
    (tmp / "in" / "link.csv").symlink_to(tmp / "secret.txt")
    assert g.call("read_csv", {"file_name": "link.csv"})["error"]["code"] == "PATH_DENIED"
    assert g.call("write_report", {"file_name": "../evil.md", "content": "x"})["error"]["code"] == "PATH_DENIED"
    assert not (tmp / "evil.md").exists()


def test_schema_invalid_logged_before_execution(gw):
    g, c, _ = gw
    assert g.call("read_csv", {})["error"]["code"] == "SCHEMA_INVALID"
    assert g.call("read_csv", {"file_name": "d.csv", "extra": 1})["error"]["code"] == "SCHEMA_INVALID"
    types = [e.type for e in c.events()]
    assert types == ["tool_call", "tool_result", "tool_call", "tool_result"]


def test_unknown_and_disallowed_tool(gw):
    g, _, tmp = gw
    assert g.call("delete_everything", {})["error"]["code"] == "UNKNOWN_TOOL"
    g2 = ToolGateway(TraceCollector("R2", tmp / "r2"), tmp / "in", tmp / "out", ["read_csv"])
    assert g2.call("calculate", {"expression": "1+1"})["error"]["code"] == "TOOL_NOT_ALLOWED"


def test_calculator_safe():
    assert safe_eval("(12.5 + 3) * 2") == 31.0
    for bad in ["__import__('os').system('ls')", "2**999", "1/0", "open('x')"]:
        with pytest.raises(ToolError):
            safe_eval(bad)


def test_fault_injection_then_success(gw):
    g, c, _ = gw
    a = g.call("mock_api_request", {"endpoint": "/sales/summary"})
    b = g.call("mock_api_request", {"endpoint": "/sales/summary"})
    ok = g.call("mock_api_request", {"endpoint": "/sales/summary"})
    assert a["error"]["code"] == b["error"]["code"] == "TIMEOUT" and ok["status"] == "ok"
    injected = [e for e in c.events() if e.type == "tool_result" and e.meta.get("injected_fault")]
    assert len(injected) == 2


def test_trace_is_incremental_and_sequential(tmp_path):
    c = TraceCollector("R", tmp_path)
    c.emit("run_start")
    assert len(read_events(c.path)) == 1  # already on disk before the run ends
    c.emit("error", status="error", error={"code": "X", "message": "m"})
    ev = read_events(c.path)
    assert [e.event_id for e in ev] == ["EV-001", "EV-002"]
    assert [e.sequence_number for e in ev] == [1, 2]
