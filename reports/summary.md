# AgentFail results summary

Runs found: 14. Infra error runs (provider error or crash) are excluded from success rate.

## By framework and model

| framework | model_id | runs | infra_error_runs | determinate_runs | success_rate | success_ci95_low | success_ci95_high | false_success_rate | recovery_rate | schema_validity_rate | repetition_rate | avg_llm_steps | avg_duration_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| langgraph | gemini-2.5-flash | 8 | 4 | 4 | 1.0 | 0.51 | 1.0 | 0.0 |  | 1.0 | 0.0 | 4.25 | 20.59 |
| langgraph | openai/gpt-oss-120b | 6 | 0 | 6 | 0.833 | 0.436 | 0.97 | 0.25 | 1.0 | 1.0 | 0.087 | 4.83 | 9.59 |

## Failure candidate categories (rule based, not human reviewed)

| framework | model_id | cat_environment_system | cat_unknown_uncertain | cat_verification_recovery |
|---|---|---|---|---|
| langgraph | gemini-2.5-flash |  |  |  |
| langgraph | openai/gpt-oss-120b | 2 | 1 | 2 |

## By task

| task_id | framework | model_id | determinate_runs | pass | fail | success_rate | false_success |
|---|---|---|---|---|---|---|---|
| TASK-001 | langgraph | gemini-2.5-flash | 1 | 1 | 0 | 1.0 | 0 |
| TASK-001 | langgraph | openai/gpt-oss-120b | 2 | 2 | 0 | 1.0 | 0 |
| TASK-002 | langgraph | gemini-2.5-flash | 1 | 1 | 0 | 1.0 | 0 |
| TASK-002 | langgraph | openai/gpt-oss-120b | 1 | 1 | 0 | 1.0 | 0 |
| TASK-003 | langgraph | gemini-2.5-flash | 0 | 0 | 0 |  | 0 |
| TASK-003 | langgraph | openai/gpt-oss-120b | 1 | 1 | 0 | 1.0 | 0 |
| TASK-004 | langgraph | gemini-2.5-flash | 1 | 1 | 0 | 1.0 | 0 |
| TASK-004 | langgraph | openai/gpt-oss-120b | 1 | 0 | 1 | 0.0 | 1 |
| TASK-005 | langgraph | gemini-2.5-flash | 1 | 1 | 0 | 1.0 | 0 |
| TASK-005 | langgraph | openai/gpt-oss-120b | 1 | 1 | 0 | 1.0 | 0 |

Notes: small numbers of runs give wide intervals. Rule based categories are candidates for manual review.