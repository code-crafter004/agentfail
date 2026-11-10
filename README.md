# AgentFail

Benchmarking and categorizing failure patterns of LLM agents on multi step tasks.

## Setup data and tests
```bash
python3 -m sample_data.make_all      # all CSV/text data + hidden ground truth (seeded)
python3 -m pytest -q                 # 29 tests, no API key needed
```

## Run
```bash
python3 run.py --task TASK-001 --framework langgraph --model gemini      # one run
python3 batch.py --dry-run                                               # show the plan
python3 batch.py --frameworks langgraph --models gemini groq --reps 3    # full batch, resumable
python3 -m evaluation.metrics                                            # writes reports/*.csv and reports/summary.md
```
Batch is sequential and shuffled with a fixed seed. Run the same command again to resume: finished runs are skipped,
provider errors are re-run. Use `--sleep 6` if you hit free tier rate limits.

## Tasks
| ID | Task | Tools | Main failure traps |
|---|---|---|---|
| TASK-001 | Monthly sales analysis | read_csv, aggregate_revenue, calculate, write_report | wrong column, wrong totals |
| TASK-002 | Customer filtering | read_csv, filter_csv, calculate, write_report | "older than 30" vs "at least 30" (boundary rows) |
| TASK-003 | API tool recovery | mock_api_request, write_report | 2 injected timeouts, invented values, unbounded retries |
| TASK-004 | Constraint report from a document | read_text, write_report | invented numbers, missing section, no info for Risks |
| TASK-005 | Regional revenue share | read_csv, aggregate_revenue, calculate, write_report | percentage arithmetic, top region |

(The design doc's optional multi agent task is replaced by TASK-005. Multi agent is out of scope for now.)

## Contracts (frozen, do not change without telling the team)
- `tracing/schema.py`: Event, RunMetadata, Annotation
- `tools/gateway.py`: every tool call goes through `ToolGateway.call(...)`
- `agents/<framework>_agent.py`: `run_xxx_agent(model, gateway, collector, task, backoff_s) -> {status, steps, final_answer}`
- `validation/validators.py`: `validator(output_dir, truth, final_answer, events) -> dict`. Ground truth is never shown to agents.

## Metric definitions
See the docstring at the top of `evaluation/metrics.py`.
