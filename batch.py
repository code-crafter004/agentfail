"""Batch CLI.

Examples:
  python3 batch.py --dry-run
  python3 batch.py --frameworks langgraph --models gemini groq --reps 3 --sleep 5
Re-run the same command to resume. Finished runs are skipped.
"""
import argparse

from dotenv import load_dotenv

from experiments.batch import run_batch

if __name__ == "__main__":
    load_dotenv(".env")
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["all"], help="e.g. TASK-001 TASK-003, or all")
    p.add_argument("--frameworks", nargs="+", default=["langgraph"], choices=["langgraph", "crewai"])
    p.add_argument("--models", nargs="+", default=["gemini", "groq"], choices=["gemini", "groq", "ollama"])
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--sleep", type=float, default=4.0, help="seconds between runs (free tier rate limits)")
    p.add_argument("--backoff", type=float, default=3.0, help="base seconds for 429 retry backoff")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    out = run_batch(".", a.tasks, a.frameworks, a.models, a.reps, a.sleep, a.backoff, a.seed, a.dry_run)
    print(out)
