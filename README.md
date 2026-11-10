# AgentFail

Benchmarking and categorizing failure patterns of LLM agents on multi step tasks.

## Run one task
```bash
python3 sample_data/make_sales_data.py          # makes data + hidden ground truth
python3 -m pytest -q                            # tests, no API key needed
python3 run.py --task TASK-001 --framework langgraph --model gemini
python3 run.py --task TASK-001 --framework langgraph --model groq
```
Each run writes `results/<run_id>/` (metadata.json, events.jsonl, validation.json, annotations.json)
and `workspaces/<run_id>/` (input, output). Both folders are git ignored.

## Contracts (frozen, do not change without telling the team)
- `tracing/schema.py`: Event, RunMetadata, Annotation
- `tools/gateway.py`: every tool call goes through `ToolGateway.call(...)`
- `agents/<framework>_agent.py`: `run_xxx_agent(model, gateway, collector, task, backoff_s) -> {status, steps, final_answer}`
- `validation/validators.py`: validators read ground truth, agents never see it
