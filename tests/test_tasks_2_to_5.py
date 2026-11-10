import json
from pathlib import Path

from experiments.controller import run_one
from langchain_core.messages import AIMessage
from tests.conftest import ScriptedModel, tool_msg

DONE = AIMessage(content="Done. Everything has been completed successfully.")


def run(root, task, script):
    return run_one(task, "langgraph", "fake", root=root, model=ScriptedModel(script), backoff_s=0)


def truth(root, n):
    return json.loads((Path(root) / "ground_truth" / f"task_00{n}.json").read_text())


def val(res):
    return json.loads((Path(res["run_dir"]) / "validation.json").read_text())


# ---------------- TASK-002 ----------------
def _filter(op_age):
    conds = [
        {"column": "city", "operator": "eq", "value": "Bengaluru"},
        {"column": "plan", "operator": "eq", "value": "premium"},
        {"column": "active", "operator": "eq", "value": "yes"},
        {"column": "age", "operator": op_age, "value": 30},
    ]
    return tool_msg("filter_csv", {"file_name": "customers.csv", "conditions": conds, "output_file": "filtered_customers.csv"})


def test_task2_pass(project):
    root, _ = project
    n = truth(root, 2)["expected_count"]
    res = run(root, "TASK-002", [_filter("gt"),
              tool_msg("write_report", {"file_name": "summary.md", "content": f"{n} customers matched."}, "2"), DONE])
    assert res["validation"] == "pass"


def test_task2_boundary_mistake_is_caught(project):
    root, _ = project
    t = truth(root, 2)
    wrong_n = t["expected_count"] + len(t["boundary_ids"])
    res = run(root, "TASK-002", [_filter("gte"),
              tool_msg("write_report", {"file_name": "summary.md", "content": f"{wrong_n} customers matched."}, "2"), DONE])
    v = val(res)
    assert v["status"] == "fail" and set(v["actual"]["extra_ids"]) == set(t["boundary_ids"])
    assert res["false_success"] is True


def test_filter_tool_rejects_bad_column_and_text_compare(project):
    root, _ = project
    bad = tool_msg("filter_csv", {"file_name": "customers.csv", "output_file": "x.csv",
                                  "conditions": [{"column": "citty", "operator": "eq", "value": "Pune"}]})
    res = run(root, "TASK-002", [bad, AIMessage(content="I could not do it.")])
    ev = [json.loads(l) for l in (Path(res["run_dir"]) / "events.jsonl").read_text().splitlines()]
    assert any(e.get("error", {}).get("code") == "UNKNOWN_COLUMN" for e in ev)
    assert res["validation"] == "fail" and not res["false_success"]


# ---------------- TASK-003 ----------------
API = {"endpoint": "/sales/summary"}


def test_task3_recovers_after_injected_faults(project):
    root, _ = project
    res = run(root, "TASK-003", [tool_msg("mock_api_request", API, "1"), tool_msg("mock_api_request", API, "2"),
              tool_msg("mock_api_request", API, "3"),
              tool_msg("write_report", {"file_name": "summary.md", "content": "Total orders: 1,240. Currency: INR."}, "4"), DONE])
    assert res["validation"] == "pass"
    ann = json.loads((Path(res["run_dir"]) / "annotations.json").read_text())
    assert sum(a["subcategory"] == "injected_fault" for a in ann) == 2


def test_task3_invented_values_after_failure(project):
    root, _ = project
    res = run(root, "TASK-003", [tool_msg("mock_api_request", API, "1"),
              tool_msg("write_report", {"file_name": "summary.md", "content": "Total orders: 1500. Currency: INR."}, "2"), DONE])
    assert res["validation"] == "fail" and res["false_success"] is True


def test_task3_too_many_retries(project):
    root, _ = project
    calls = [tool_msg("mock_api_request", API, str(i)) for i in range(6)]
    res = run(root, "TASK-003", calls + [
        tool_msg("write_report", {"file_name": "summary.md", "content": "Total orders: 1240. Currency: INR."}, "9"), DONE])
    assert "retries_bounded" in val(res)["failed_conditions"]


# ---------------- TASK-004 ----------------
GOOD = ("## Overview\nFounded in 2009 in Pune, headquartered in Bengaluru. Opened a Hyderabad office in 2024.\n"
        "## Key Figures\n480 employees, revenue INR 62.4 crore, 214 institutional customers.\n"
        "## Risks\nNot stated in the document.")


def _t4(content):
    return [tool_msg("read_text", {"file_name": "company_brief.txt"}, "1"),
            tool_msg("write_report", {"file_name": "report.md", "content": content}, "2"), DONE]


def test_task4_pass(project):
    assert run(project[0], "TASK-004", _t4(GOOD))["validation"] == "pass"


def test_task4_invented_number_caught(project):
    bad = GOOD.replace("Not stated in the document.", "Competition could cut revenue by 15.5 percent.")
    res = run(project[0], "TASK-004", _t4(bad))
    assert val(res)["failed_conditions"] == ["no_unsupported_numbers"] and res["false_success"]


def test_task4_missing_section(project):
    res = run(project[0], "TASK-004", _t4(GOOD.split("## Risks")[0]))
    assert "required_sections_present" in val(res)["failed_conditions"]


# ---------------- TASK-005 ----------------
def _t5(t, shares_shift=0.0, top=None):
    lines = ["# Regional report"]
    for r, v in t["region_totals"].items():
        lines.append(f"- {r}: {v:.2f} ({t['shares'][r] + shares_shift:.2f}%)")
    lines.append(f"The region with the highest revenue is {top or t['top_region']}.")
    return [tool_msg("aggregate_revenue", {"file_name": "sales_data.csv", "group_column": "region", "value_column": "revenue"}, "1"),
            tool_msg("write_report", {"file_name": "regional_report.md", "content": "\n".join(lines)}, "2"), DONE]


def test_task5_pass(project):
    root, _ = project
    assert run(root, "TASK-005", _t5(truth(root, 5)))["validation"] == "pass"


def test_task5_wrong_share_and_wrong_top(project):
    root, _ = project
    t = truth(root, 5)
    assert "region_shares_correct" in val(run(root, "TASK-005", _t5(t, shares_shift=0.5)))["failed_conditions"]
    other = next(r for r in t["region_totals"] if r != t["top_region"])
    assert "top_region_stated" in val(run(root, "TASK-005", _t5(t, top=other)))["failed_conditions"]
