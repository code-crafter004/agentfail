"""AgentFail dashboard.  Run from the repo root:  streamlit run agentfail_ui.py

Read-only over results/ and reports/ (plus a Run tab that calls run.py).
Field names are looked up defensively, so small schema differences do not crash it.
"""
import json, math, re, subprocess, sys
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
except Exception:  # plotly is optional
    px = None
ROOT = Path(__file__).resolve().parent.parent
RESULTS, REPORTS = ROOT / "results", ROOT / "reports"
st.set_page_config(page_title="AgentFail", layout="wide")


# ---------------------------------------------------------------- helpers
def find(obj, *names):
    """First value whose key matches one of `names` (case-insensitive), searched recursively."""
    names = [n.lower() for n in names]
    stack = [obj]
    while stack:
        cur = stack.pop(0)
        if isinstance(cur, dict):
            for n in names:
                for k, v in cur.items():
                    if str(k).lower() == n and v not in (None, ""):
                        return v
            stack.extend(v for v in cur.values() if isinstance(v, (dict, list)))
        elif isinstance(cur, list):
            stack.extend(x for x in cur if isinstance(x, (dict, list)))
    return None


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def read_jsonl(path):
    out = []
    try:
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    out.append({"raw": line})
    except Exception:
        pass
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def norm_status(v):
    s = str(v or "").lower()
    if "pass" in s or s in ("ok", "success", "true"):
        return "pass"
    if "fail" in s or s in ("false", "invalid"):
        return "fail"
    return "unknown"


def labels_of(ann):
    """Normalise annotations into a list of dicts: category, evidence, text."""
    items = []
    if isinstance(ann, list):
        items = [a for a in ann if isinstance(a, dict)]
    elif isinstance(ann, dict):
        for key in ("failures", "candidates", "annotations", "labels", "flags"):
            if isinstance(ann.get(key), list):
                items = [a for a in ann[key] if isinstance(a, dict)]
                break
        else:
            items = [ann] if find(ann, "category", "label", "failure_type") else []
    out = []
    for a in items:
        ev = find(a, "evidence_event_ids", "evidence", "event_ids", "events") or []
        out.append({
            "category": str(find(a, "category", "label", "failure_type", "rule", "type") or "unlabelled"),
            "evidence": ev if isinstance(ev, list) else [ev],
            "detail": a,
        })
    return out


# ---------------------------------------------------------------- loading
@st.cache_data(ttl=10, show_spinner=False)
def load_runs():
    rows = []
    if not RESULTS.exists():
        return pd.DataFrame()
    for d in sorted(p for p in RESULTS.iterdir() if p.is_dir()):
        meta = None
        for name in ("metadata.json", "meta.json", "run.json"):
            meta = read_json(d / name)
            if meta:
                break
        meta = meta or {}
        val = read_json(d / "validation.json") or {}
        ann = read_json(d / "annotations.json")
        events = read_jsonl(d / "events.jsonl")
        if not (meta or val or events):
            continue
        task = find(meta, "task_id", "task") or (re.search(r"TASK-\d+", d.name) or [None])[0]
        fw = find(meta, "framework") or next((f for f in ("langgraph", "crewai") if f in d.name.lower()), None)
        model = find(meta, "model", "model_name", "model_id")
        rep = find(meta, "rep", "repetition", "repeat")
        vstat = norm_status(find(val, "status", "verdict", "result"))
        agent_status = str(find(meta, "agent_status", "run_status", "status") or "")
        labs = labels_of(ann)
        cats = [l["category"].lower() for l in labs]
        blob = (json.dumps(ann) if ann else "").lower() + " ".join(str(find(e, "event_type", "type", "kind") or "") for e in events).lower()
        provider = "provider" in agent_status.lower() or any("provider" in c for c in cats) or "provider_error" in blob
        done_like = bool(re.search(r"complet|success|done|finish|ok", agent_status.lower())) or not agent_status
        false_success = ("false_success" in blob) or (vstat == "fail" and not provider and done_like and not labs)
        rows.append({
            "run": d.name, "task": task or "?", "framework": fw or "?", "model": str(model or "?"),
            "rep": rep, "validation": vstat, "agent_status": agent_status or "-",
            "provider_error": provider, "false_success": bool(false_success and vstat == "fail"),
            "labels": ", ".join(sorted(set(l["category"] for l in labs))) or "-",
            "events": len(events), "path": str(d),
        })
    return pd.DataFrame(rows)


def summarise(df, by):
    det = df[(df.validation.isin(["pass", "fail"])) & (~df.provider_error)]
    rows = []
    for key, g in det.groupby(by):
        n, k = len(g), int((g.validation == "pass").sum())
        lo, hi = wilson(k, n)
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)), "runs": n, "passed": k, "success_rate": k / n,
                     "ci_low": lo, "ci_high": hi,
                     "false_success": int(g.false_success.sum()),
                     "false_success_rate": g.false_success.sum() / n})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- UI
st.title("AgentFail dashboard")
c1, c2 = st.columns([6, 1])
with c2:
    if st.button("Refresh"):
        st.cache_data.clear()
        st.rerun()

runs = load_runs()
tab_res, tab_fail, tab_run = st.tabs(["Results", "Failures and traces", "Run a task"])

# ------------------------------------------------------------ Results
with tab_res:
    if runs.empty:
        st.info("No runs found in results/. Use the Run tab, or: python3 batch.py --frameworks langgraph --models groq --reps 1")
    else:
        det = runs[runs.validation.isin(["pass", "fail"]) & ~runs.provider_error]
        a, b, c, d = st.columns(4)
        a.metric("Runs found", len(runs))
        b.metric("Determinate runs", len(det))
        c.metric("Provider errors (not agent failures)", int(runs.provider_error.sum()))
        d.metric("False successes", int(runs.false_success.sum()))
        st.caption("Success rate has a 95% Wilson interval. Provider errors are excluded. Small run counts mean wide intervals.")

        by = st.radio("Group by", ["framework", "model", "task", "framework + model", "framework + task"], horizontal=True)
        keys = {"framework": ["framework"], "model": ["model"], "task": ["task"],
                "framework + model": ["framework", "model"], "framework + task": ["framework", "task"]}[by]
        summ = summarise(runs, keys)
        if summ.empty:
            st.warning("No determinate runs yet.")
        else:
            st.dataframe(summ.style.format({"success_rate": "{:.0%}", "ci_low": "{:.0%}", "ci_high": "{:.0%}",
                                            "false_success_rate": "{:.0%}"}), hide_index=True)
            if px is not None:
                s = summ.copy()
                s["label"] = s[keys].astype(str).agg(" / ".join, axis=1)
                s["err_plus"] = s.ci_high - s.success_rate
                s["err_minus"] = s.success_rate - s.ci_low
                fig = px.bar(s, x="label", y="success_rate", error_y="err_plus", error_y_minus="err_minus",
                             text=s.runs.map(lambda n: f"n={n}"), range_y=[0, 1.05],
                             labels={"success_rate": "Success rate", "label": ""})
                fig.update_layout(height=380, margin=dict(t=20))
                st.plotly_chart(fig)

        t3 = runs[(runs.task == "TASK-003") & runs.validation.isin(["pass", "fail"]) & ~runs.provider_error]
        if len(t3):
            k = int((t3.validation == "pass").sum())
            lo, hi = wilson(k, len(t3))
            st.metric("TASK-003 pass rate (2 injected timeouts, proxy for recovery)", f"{k}/{len(t3)}",
                      help=f"95% interval {lo:.0%} to {hi:.0%}")

        cats = runs[runs.labels != "-"].labels.str.split(", ").explode().value_counts()
        st.subheader("Failure labels (rule-based candidates)")
        if len(cats):
            st.bar_chart(cats)
        else:
            st.write("No failure labels yet.")

        for name in ("summary_by_config.csv", "summary_by_task.csv"):
            f = REPORTS / name
            if f.exists():
                with st.expander(f"reports/{name}"):
                    st.dataframe(pd.read_csv(f), hide_index=True)

# ------------------------------------------------------------ Failures
with tab_fail:
    if runs.empty:
        st.info("No runs yet.")
    else:
        f1, f2, f3, f4 = st.columns(4)
        task_f = f1.multiselect("Task", sorted(runs.task.unique()))
        fw_f = f2.multiselect("Framework", sorted(runs.framework.unique()))
        st_f = f3.multiselect("Validation", ["pass", "fail", "unknown"], default=["fail"])
        only_prov = f4.checkbox("Only provider errors")
        v = runs
        if task_f: v = v[v.task.isin(task_f)]
        if fw_f: v = v[v.framework.isin(fw_f)]
        if st_f: v = v[v.validation.isin(st_f)]
        if only_prov: v = v[v.provider_error]
        st.dataframe(v.drop(columns=["path"]), hide_index=True)

        if len(v):
            pick = st.selectbox("Open a run", v.run.tolist())
            row = runs[runs.run == pick].iloc[0]
            d = Path(row.path)
            left, right = st.columns([1, 2])
            with left:
                st.subheader("Verdict")
                st.write(f"**{row.task}** | {row.framework} | {row.model} | validation: **{row.validation}**")
                if row.false_success: st.error("False success: agent finished but the result is wrong.")
                if row.provider_error: st.warning("Provider error: not an agent failure.")
                st.json(read_json(d / "validation.json") or {})
                ann = read_json(d / "annotations.json")
                labs = labels_of(ann)
                st.subheader("Failure candidates")
                if not labs: st.write("None.")
                for l in labs:
                    st.markdown(f"**{l['category']}**  evidence events: `{l['evidence']}`")
                    with st.expander("details"):
                        st.json(l["detail"])
            with right:
                st.subheader("Trace")
                events = read_jsonl(d / "events.jsonl")
                evidence = {str(x) for l in labs for x in l["evidence"]}
                if not events: st.write("No events.jsonl in this run.")
                for i, e in enumerate(events):
                    typ = str(find(e, "event_type", "type", "kind") or "event")
                    tool = find(e, "tool", "tool_name")
                    eid = str(find(e, "event_id", "id", "seq", "step") or i)
                    bad = bool(re.search(r"error|fail|timeout", (typ + json.dumps(e)).lower())) and "error\": null" not in json.dumps(e).lower()
                    flag = " ⚑ evidence" if eid in evidence else ""
                    head = f"{i:>3}  {typ}" + (f"  ·  {tool}" if tool else "") + flag
                    with st.expander(head, expanded=bad or bool(flag)):
                        st.json(e)

# ------------------------------------------------------------ Run
with tab_run:
    tasks = sorted({p.name[:8] for p in (ROOT / "benchmark_tasks").glob("TASK-*")}) if (ROOT / "benchmark_tasks").exists() else []
    tasks = tasks or [f"TASK-00{i}" for i in range(1, 6)]
    r1, r2, r3 = st.columns(3)
    t = r1.selectbox("Task", tasks, index=min(2, len(tasks) - 1))
    fw = r2.selectbox("Framework", ["langgraph", "crewai"])
    mdl = r3.selectbox("Model", ["gemini", "groq"])
    st.caption("Runs python run.py and writes to results/ like any other run. Needs your keys in .env.")
    if st.button("Run", type="primary"):
        with st.spinner("Running... this can take a minute on free tiers"):
            try:
                p = subprocess.run([sys.executable, "run.py", "--task", t, "--framework", fw, "--model", mdl],
                                   cwd=ROOT, capture_output=True, text=True, timeout=900)
                st.code((p.stdout or "") + (p.stderr or ""), language="text")
                (st.success if p.returncode == 0 else st.error)(f"exit code {p.returncode}")
            except Exception as ex:
                st.error(f"Could not run: {ex}")
        st.cache_data.clear()
    if st.button("Rebuild reports (python -m evaluation.metrics)"):
        p = subprocess.run([sys.executable, "-m", "evaluation.metrics"], cwd=ROOT, capture_output=True, text=True)
        st.code((p.stdout or "") + (p.stderr or ""), language="text")
