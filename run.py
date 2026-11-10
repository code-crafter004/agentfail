"""CLI:  python3 run.py --task TASK-001 --framework langgraph --model gemini"""
import argparse
import json

from dotenv import load_dotenv

from experiments.controller import run_one

if __name__ == "__main__":
    load_dotenv(".env")
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="TASK-001")
    p.add_argument("--framework", choices=["langgraph", "crewai"], default="langgraph")
    p.add_argument("--model", choices=["gemini", "groq", "ollama"], default="gemini")
    a = p.parse_args()
    out = run_one(a.task, a.framework, a.model, root=".")
    print(json.dumps(out, indent=2))
    print(f"\nTrace:      {out['run_dir']}/events.jsonl")
    print(f"Validation: {out['run_dir']}/validation.json")
