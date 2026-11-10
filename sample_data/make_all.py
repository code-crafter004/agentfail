"""Generate ALL sample data and hidden ground truth (seeded, reproducible).

Usage (from the project root):  python3 -m sample_data.make_all
"""
from __future__ import annotations

import csv
import json
import random
import re
from decimal import Decimal
from pathlib import Path

from sample_data.make_sales_data import generate as generate_sales

CITIES = ["Bengaluru", "Mumbai", "Delhi", "Chennai", "Pune"]
PLANS = ["free", "basic", "premium"]
FIRST = ["Aarav", "Diya", "Kabir", "Meera", "Rohan", "Isha", "Vihaan", "Anaya", "Arjun", "Sara", "Neel", "Tara"]
LAST = ["Rao", "Shah", "Iyer", "Nair", "Gupta", "Khan", "Reddy", "Das", "Joshi", "Mehta"]

COMPANY_BRIEF = """Northwind Learning Pvt Ltd: Company Brief

Northwind Learning was founded in 2009 in Pune and is headquartered in Bengaluru.
As of March 2025 the company employs 480 people across 3 offices.
Revenue for fiscal year 2024-25 was INR 62.4 crore, up from INR 51.8 crore the previous year.
The company serves 214 institutional customers. Its main product is an adaptive learning platform used in schools.
In 2024 the company opened a new office in Hyderabad.
The board of directors meets quarterly.
"""

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def numbers_in(text: str) -> list[float]:
    out = []
    for m in NUM.findall(text):
        try:
            out.append(abs(float(m.replace(",", "").rstrip("."))))
        except ValueError:
            pass
    return out


def gen_customers(data_dir: Path, truth_dir: Path, seed: int = 7) -> dict:
    rng = random.Random(seed)
    rows = []
    for _ in range(110):  # random background rows
        rows.append({
            "name": f"{rng.choice(FIRST)} {rng.choice(LAST)}", "city": rng.choice(CITIES),
            "age": rng.randint(18, 60), "plan": rng.choice(PLANS), "active": rng.choice(["yes", "no"]),
            "signup_date": f"202{rng.randint(2, 5)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
        })
    base = {"city": "Bengaluru", "plan": "premium", "active": "yes"}
    injected = [dict(base, age=a) for a in (31, 34, 38, 42, 47, 55)]            # true matches
    injected += [dict(base, age=30), dict(base, age=30)]                         # boundary: age exactly 30
    injected += [dict(base, age=40, active="no"), dict(base, age=45, plan="basic")]  # decoys
    for r in injected:
        r.update(name=f"{rng.choice(FIRST)} {rng.choice(LAST)}", signup_date="2024-06-15")
    rows += injected
    rng.shuffle(rows)
    for i, r in enumerate(rows, 1):
        r["customer_id"] = f"C{i:03d}"
    cols = ["customer_id", "name", "city", "age", "plan", "active", "signup_date"]
    with open(data_dir / "customers.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows({c: r[c] for c in cols} for r in rows)
    ids = sorted(r["customer_id"] for r in rows
                 if r["city"] == "Bengaluru" and r["plan"] == "premium" and r["active"] == "yes" and r["age"] > 30)
    boundary = sorted(r["customer_id"] for r in rows if r["age"] == 30 and r["city"] == "Bengaluru"
                      and r["plan"] == "premium" and r["active"] == "yes")
    truth = {"task_id": "TASK-002", "expected_ids": ids, "expected_count": len(ids), "boundary_ids": boundary}
    (truth_dir / "task_002.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return truth


def gen_api_truth(truth_dir: Path) -> dict:
    truth = {"task_id": "TASK-003", "total_orders": 1240, "currency": "INR", "max_api_calls": 5}
    (truth_dir / "task_003.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return truth


def gen_brief(data_dir: Path, truth_dir: Path) -> dict:
    (data_dir / "company_brief.txt").write_text(COMPANY_BRIEF, encoding="utf-8")
    truth = {
        "task_id": "TASK-004",
        "required_sections": ["Overview", "Key Figures", "Risks"],
        "required_facts": ["2009", "480", "62.4", "214"],
        "source_numbers": sorted(set(numbers_in(COMPANY_BRIEF))),
    }
    (truth_dir / "task_004.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return truth


def gen_regional(data_dir: Path, truth_dir: Path) -> dict:
    totals: dict[str, Decimal] = {}
    with open(data_dir / "sales_data.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            totals[r["region"]] = totals.get(r["region"], Decimal("0")) + Decimal(r["revenue"])
    grand = sum(totals.values())
    shares = {k: float((v / grand * 100).quantize(Decimal("0.01"))) for k, v in totals.items()}
    top = max(totals, key=lambda k: totals[k])
    assert sorted(totals.values())[-1] != sorted(totals.values())[-2]
    truth = {
        "task_id": "TASK-005",
        "region_totals": {k: float(v) for k, v in sorted(totals.items())},
        "total": float(grand),
        "shares": dict(sorted(shares.items())),
        "top_region": top,
    }
    (truth_dir / "task_005.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return truth


def generate_all(data_dir: Path, truth_dir: Path) -> dict:
    data_dir, truth_dir = Path(data_dir), Path(truth_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)
    t1 = generate_sales(data_dir, truth_dir)
    return {
        "TASK-001": t1,
        "TASK-002": gen_customers(data_dir, truth_dir),
        "TASK-003": gen_api_truth(truth_dir),
        "TASK-004": gen_brief(data_dir, truth_dir),
        "TASK-005": gen_regional(data_dir, truth_dir),
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    out = generate_all(root / "sample_data", root / "ground_truth")
    print("generated:", ", ".join(out))
    print("TASK-002 expected count:", out["TASK-002"]["expected_count"], "| TASK-005 top region:", out["TASK-005"]["top_region"])
