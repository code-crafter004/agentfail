"""Generate sales_data.csv and the hidden ground truth (seeded, reproducible).

Usage:  python3 sample_data/make_sales_data.py
"""
from __future__ import annotations

import csv
import json
import random
from decimal import Decimal
from pathlib import Path

MONTHS = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06"]
PRODUCTS = {"Notebook": 4.50, "Pen Pack": 2.25, "Backpack": 18.90, "Desk Lamp": 12.40, "Headphones": 29.99}
REGIONS = ["North", "South", "East", "West"]


def generate(data_dir: Path, truth_dir: Path, seed: int = 42, rows_per_month: int = 25) -> dict:
    rng = random.Random(seed)
    data_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    totals = {m: Decimal("0") for m in MONTHS}
    for m in MONTHS:
        for _ in range(rows_per_month):
            product = rng.choice(list(PRODUCTS))
            units = rng.randint(1, 40)
            revenue = (Decimal(str(PRODUCTS[product])) * units).quantize(Decimal("0.01"))
            day = rng.randint(1, 28)
            rows.append(
                {
                    "date": f"{m}-{day:02d}",
                    "month": m,
                    "product": product,
                    "region": rng.choice(REGIONS),
                    "units": units,
                    "revenue": f"{revenue}",
                }
            )
            totals[m] += revenue
    rng.shuffle(rows)
    with open(data_dir / "sales_data.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["date", "month", "product", "region", "units", "revenue"])
        w.writeheader()
        w.writerows(rows)
    top = max(totals, key=lambda k: totals[k])
    assert sorted(totals.values())[-1] != sorted(totals.values())[-2], "tie for top month"
    truth = {
        "task_id": "TASK-001",
        "monthly_totals": {m: float(v) for m, v in totals.items()},
        "highest_month": top,
        "highest_total": float(totals[top]),
    }
    (truth_dir / "task_001.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return truth


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    t = generate(root / "sample_data", root / "ground_truth")
    print("generated. highest month:", t["highest_month"])
