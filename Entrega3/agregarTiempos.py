from __future__ import annotations

import csv
from pathlib import Path

RESULTS = Path("Entrega3/results")
RESULTS.mkdir(parents=True, exist_ok=True)

PANDAS_TIMING = RESULTS / "pandas_timing.csv"
PYSPARK_TIMING = RESULTS / "pyspark_timing.csv"
OUT = RESULTS / "timing_comparison.csv"


def read_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row:
                rows.append({
                    "tool": row.get("tool", ""),
                    "metric": row.get("metric", ""),
                    "time_s": row.get("time_s", "")
                })
    return rows


def main() -> str:
    rows = []
    rows.extend(read_rows(PANDAS_TIMING))
    rows.extend(read_rows(PYSPARK_TIMING))
    # Write combined
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tool", "metric", "time_s"])  # header
        for r in rows:
            w.writerow([r["tool"], r["metric"], r["time_s"]])
    return str(OUT)


if __name__ == "__main__":
    print(main())
