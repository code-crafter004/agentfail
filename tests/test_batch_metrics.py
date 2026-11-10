import csv
import json
from pathlib import Path

from langchain_core.messages import AIMessage

from evaluation.metrics import build, recovery_episodes, wilson
from experiments.batch import plan, run_batch
from tests.conftest import ScriptedModel, tool_msg
from tests.test_langgraph_slice import _script


def test_wilson_interval():
    lo, hi = wilson(2, 2)
    assert 0.3 < lo < 0.4 and hi == 1.0 or abs(hi - 1.0) < 1e-9
    assert wilson(0, 0)[0] != wilson(0, 0)[0]  # nan for empty


def test_plan_is_deterministic_and_complete(project):
    root, _ = project
    a = plan(root, ["all"], ["langgraph"], ["gemini", "groq"], 3, seed=1)
    b = plan(root, ["all"], ["langgraph"], ["gemini", "groq"], 3, seed=1)
    assert a == b and len(a) == 5 * 2 * 3 and len(set(a)) == len(a)


def test_batch_resume_and_metrics(project):
    root, truth = project

    def factory(name):
        script = _script(truth) if name == "fake_ok" else [RuntimeError("401 bad key")]
        return ScriptedModel(script), {"provider": "fake", "model_id": name}

    logs = []
    out = run_batch(root, ["TASK-001"], ["langgraph"], ["fake_ok", "fake_bad"], reps=2, sleep_s=0, backoff_s=0,
                    model_factory=factory, log=lambda *a: logs.append(" ".join(map(str, a))))
    assert out["ran"] == 4 and out["errors"] == 0
    # resume: the 2 good runs are done, the 2 provider errors run again
    out2 = run_batch(root, ["TASK-001"], ["langgraph"], ["fake_ok", "fake_bad"], reps=2, sleep_s=0, backoff_s=0,
                     model_factory=factory, log=lambda *a: None)
    assert out2["to_run"] == 2 and out2["ran"] == 2

    build(root / "results", root / "reports")
    rows = list(csv.DictReader(open(root / "reports" / "summary_by_config.csv")))
    ok = next(r for r in rows if r["model_id"] == "fake_ok")
    bad = next(r for r in rows if r["model_id"] == "fake_bad")
    assert ok["success_rate"] == "1.0" and ok["determinate_runs"] == "2"
    assert bad["determinate_runs"] == "0" and bad["success_rate"] == ""  # infra errors do not count as failures
    assert (root / "reports" / "summary.md").read_text().startswith("# AgentFail results summary")


def test_recovery_episode_logic(tmp_path):
    from tracing.collector import TraceCollector

    c = TraceCollector("R", tmp_path)
    for status, tool in [("error", "a"), ("error", "a"), ("ok", "a"), ("error", "b")]:
        c.emit("tool_result", tool_name=tool, status=status)
    assert recovery_episodes(c.events()) == (2, 1)  # a resolved, b never recovered
