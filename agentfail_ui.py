"""AgentFail dashboard.  Run:  streamlit run agentfail_ui.py
Tabs: Results (framework comparison), Failures (inspect any run's trace), Run (launch a new run)."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from evaluation.metrics import _load, run_row, summarize

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
SAMPLE = ROOT / "sample_results"   # small committed demo runs, shown next to real runs
load_dotenv(ROOT / ".env")
st.set_page_config(page_title="AgentFail", layout="wide")
st.title("AgentFail")
st.caption("Benchmarking and categorizing failure patterns of LLM agents on multi step tasks. "
           "Same tasks, tools and prompt on every framework, only the framework changes.")


@st.cache_data(ttl=10)
def load_rows():
    dirs = sorted(RESULTS.glob("*/")) + sorted(SAMPLE.glob("*/"))
    runs = [x for d in dirs if (x := _load(d))]
    return [run_row(r) for r in runs], {r["meta"]["run_id"]: r for r in runs}


rows, runs = load_rows()
df = pd.DataFrame(rows)
tab_res, tab_fail, tab_run = st.tabs(["Results", "Failures and traces", "Run a task"])

# ---------------------------------------------------------------- Results
with tab_res:
    if df.empty:
        st.info("No runs yet. Use the Run tab or: python3 batch.py --frameworks langgraph crewai --models groq --reps 3")
    else:
        cfg = pd.DataFrame(summarize(rows, ("framework", "model_id")))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Runs", len(df))
        c2.metric("Infra errors (excluded)", int(df["infra_error"].sum()))
        det = df[~df["infra_error"]]
        c3.metric("Overall success", f"{(det['validation'] == 'pass').mean():.0%}" if len(det) else "n/a")
        c4.metric("False successes", int(det["false_success"].sum()))

        st.subheader("By framework and model")
        show = cfg[[c for c in ["framework", "model_id", "determinate_runs", "success_rate", "success_ci95_low",
                                "success_ci95_high", "false_success_rate", "recovery_rate", "schema_validity_rate",
                                "repetition_rate", "avg_llm_steps", "avg_duration_s"] if c in cfg]]
        st.dataframe(show, use_container_width=True, hide_index=True)

        task = pd.DataFrame(summarize(rows, ("task_id", "framework", "model_id")))
        task["config"] = task["framework"] + " / " + task["model_id"]
        task["success_rate"] = pd.to_numeric(task["success_rate"], errors="coerce")
        st.subheader("Success rate by task")
        st.bar_chart(task.pivot_table(index="task_id", columns="config", values="success_rate"))

        cat_cols = [c for c in cfg.columns if c.startswith("cat_")]
        if cat_cols:
            st.subheader("Failure candidates (rule based, to be human reviewed)")
            cats = cfg.set_index(cfg["framework"] + " / " + cfg["model_id"])[cat_cols].fillna(0)
            cats.columns = [c[4:] for c in cats.columns]
            st.bar_chart(cats.T)
        st.caption("Success rate uses 95% Wilson intervals. Provider errors and crashes are excluded. "
                   "Failure categories are rule based candidates, not proven causes.")

# ---------------------------------------------------------------- Failures / traces
ICON = {"run_start": "▶", "llm_request": "🧠→", "llm_response": "🧠", "tool_call": "🔧", "tool_result": "📦",
        "retry": "🔁", "error": "⚠", "final_answer": "✅", "run_end": "■"}

with tab_fail:
    if df.empty:
        st.info("No runs yet.")
    else:
        f1, f2, f3 = st.columns(3)
        only_bad = f1.checkbox("Only failed or flagged runs", value=True)
        fw = f2.multiselect("Framework", sorted(df["framework"].unique()), default=sorted(df["framework"].unique()))
        tk = f3.multiselect("Task", sorted(df["task_id"].unique()), default=sorted(df["task_id"].unique()))
        v = df[df["framework"].isin(fw) & df["task_id"].isin(tk)]
        if only_bad:
            v = v[(v["validation"] != "pass") | (v["failure_candidates"] > 0)]
        st.dataframe(v[["run_id", "task_id", "framework", "model_id", "status", "validation", "false_success",
                        "categories"]], use_container_width=True, hide_index=True)
        if len(v):
            rid = st.selectbox("Open a run", v["run_id"].tolist())
            r = runs[rid]
            val = r["val"]
            a, b = st.columns([1, 2])
            with a:
                st.markdown(f"**Status:** {r['meta']['status']}  \n**Validation:** {val['status']}  \n"
                            f"**Agent claimed done:** {val['completion_claim']}  \n**False success:** {val['false_success']}")
                st.markdown("**Validator checks**")
                st.json(val.get("checks", val), expanded=False)
                st.markdown("**Failure candidates**")
                for an in r["ann"]:
                    st.warning(f"{an['category']} / {an['subcategory']}")
                if not r["ann"]:
                    st.success("none")
                if r["meta"].get("final_answer"):
                    st.markdown("**Final answer**")
                    st.write(r["meta"]["final_answer"])
            with b:
                st.markdown("**Trace**")
                for e in r["events"]:
                    label = f"{ICON.get(e.type, '')} {e.sequence_number}. {e.type}"
                    if e.tool_name:
                        label += f"  {e.tool_name}"
                    if e.status == "error":
                        label += "  ERROR"
                    with st.expander(label, expanded=e.status == "error"):
                        if e.arguments is not None:
                            st.code(json.dumps(e.arguments, indent=1, default=str)[:1500], language="json")
                        if e.content:
                            st.text(e.content[:1500])
                        if e.result is not None:
                            st.code(json.dumps(e.result, indent=1, default=str)[:1500], language="json")
                        if e.error:
                            st.error(f"{e.error.get('code')}: {e.error.get('message')}")

# ---------------------------------------------------------------- Run
with tab_run:
    st.write("Launch one run. It uses your keys from .env and writes to results/ like any other run.")
    tasks = sorted(p.stem.replace("task_", "TASK-") for p in (ROOT / "benchmark_tasks").glob("task_*.json"))
    c1, c2, c3 = st.columns(3)
    t = c1.selectbox("Task", tasks)
    fwk = c2.selectbox("Framework", ["langgraph", "crewai"])
    mdl = c3.selectbox("Model", ["groq", "gemini", "ollama"])
    if st.button("Run", type="primary"):
        from experiments.controller import run_one

        with st.spinner("Agent running..."):
            out = run_one(t, fwk, mdl, root=ROOT)
        load_rows.clear()
        (st.success if out["validation"] == "pass" else st.error)(
            f"{out['status']} | validation: {out['validation']} | false success: {out['false_success']}")
        st.caption(f"Run id: {out['run_id']}. Open it in the Failures tab (untick the filter).")