# AgentFail dashboard: context for teammates and Claude

Paste this file into Claude (or Claude Code) when you work on `agentfail_ui.py`.

## Project
AgentFail shows where and why LLM agents fail on multi-step tasks. 5 tasks, 2 frameworks (LangGraph, CrewAI), 2 models (Gemini 2.5 Flash, Groq openai/gpt-oss-120b). Same prompt and 7 tools for every run, so only framework and model change. Every step is traced, every result is checked against hidden ground truth, failures get rule-based labels (candidates, not proven causes). Free tiers only. Source of truth: System Design Document, 15 Oct 2026. We are NOT building a database, REST API or code sandbox.

Pipeline: run controller -> agent adapter -> tool gateway -> trace (events.jsonl) -> validator -> failure rules -> metrics -> reports -> Streamlit dashboard.

## What agentfail_ui.py is
A Streamlit dashboard, read-only over `results/` and `reports/`, with 3 tabs:
- Results: run counts, success rate with 95% Wilson interval per framework/model/task, false successes, provider errors (excluded from agent failures), TASK-003 pass rate (proxy for recovery with 2 injected timeouts), failure label counts, and the CSVs in reports/.
- Failures and traces: filter runs, open one, see validation.json, failure candidates with evidence event ids, and the event timeline (error events open automatically).
- Run a task: runs `python run.py --task X --framework Y --model Z`, then refreshes.

It looks up field names defensively (task_id/task, framework, model, status/verdict, event_type/type, evidence_event_ids). It was tested only on fake runs. If real files use other field names, fix the `find(...)` calls in `load_runs()`.

## Repo layout it expects (repo root)
results/<run_folder>/{metadata.json, events.jsonl, validation.json, annotations.json}; reports/summary_by_config.csv, summary_by_task.csv; run.py; batch.py; evaluation/metrics.py; benchmark_tasks/.

## Team rules (do not break)
- Frozen contracts: tracing/schema.py, ToolGateway.call, validator signature, agent function signature.
- One branch per part (this one: `streamlit-app`), pull request into main, Khushi reviews. Run `python3 -m pytest -q` before every push.
- Never commit .env. results/ and workspaces/ are git-ignored on purpose.
- Do not change a model ID silently: a new ID is a new configuration.
- Report raw counts with number of runs. Provider errors are not agent failures. Rule labels are candidates until a human confirms them.
- Python 3.12 (older fails to install CrewAI on macOS).

## Known gaps
- Never opened in a browser with real runs. First check: the Results tab numbers match reports/summary_by_config.csv.
- Name clash: an old unrelated dashboard.py/app.py can exist locally. That is why this file is called agentfail_ui.py.
- Spec for the Part 2 page set (Runs, Trace, Comparison with charts from the CSVs and evidence events) is in the team handoff; this dashboard covers it in 3 tabs.
