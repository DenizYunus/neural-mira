"""Gate Neural MIRA evaluation results against conservative hardening thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_THRESHOLDS = {
    "random": {"recall_at_1": 0.40, "recall_at_5": 0.60, "mrr": 0.50},
    "month_holdout": {"recall_at_1": 0.40, "recall_at_5": 0.60, "mrr": 0.50},
    "style_holdout": {"recall_at_1": 0.25, "recall_at_5": 0.60, "mrr": 0.40},
}


def load_metrics(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"metrics file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def metric(split: dict, model: str, name: str) -> float:
    return float(split.get("test_metrics", {}).get(model, {}).get(name, 0.0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=Path("data/neural_mira_metrics.json"))
    parser.add_argument("--min-delta-vs-bm25", type=float, default=0.08)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = load_metrics(args.metrics)
    failures: list[str] = []
    rows: list[dict] = []

    for split in payload.get("splits", []):
        name = split.get("split")
        if name not in DEFAULT_THRESHOLDS:
            continue

        row = {
            "split": name,
            "neural_r1": metric(split, "neural_mira", "recall_at_1"),
            "neural_r5": metric(split, "neural_mira", "recall_at_5"),
            "neural_mrr": metric(split, "neural_mira", "mrr"),
            "bm25_mrr": metric(split, "bm25", "mrr"),
            "scalar_mrr": metric(split, "scalar_htema", "mrr"),
        }
        rows.append(row)

        for key, threshold in DEFAULT_THRESHOLDS[name].items():
            value = metric(split, "neural_mira", key)
            if value < threshold:
                failures.append(f"{name} neural_mira {key} {value:.3f} < {threshold:.3f}")

        if row["neural_mrr"] < row["bm25_mrr"] + args.min_delta_vs_bm25:
            failures.append(
                f"{name} neural_mira mrr {row['neural_mrr']:.3f} is not at least "
                f"{args.min_delta_vs_bm25:.3f} above bm25 {row['bm25_mrr']:.3f}"
            )

    result = {
        "created_at": payload.get("created_at"),
        "memory_count": payload.get("memory_count"),
        "example_count": payload.get("example_count"),
        "rows": rows,
        "failures": failures,
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"created_at={result['created_at']} memories={result['memory_count']} examples={result['example_count']}")
        for row in rows:
            print(
                f"{row['split']}: neural R@1={row['neural_r1']:.3f} "
                f"R@5={row['neural_r5']:.3f} MRR={row['neural_mrr']:.3f} "
                f"bm25_MRR={row['bm25_mrr']:.3f} scalar_MRR={row['scalar_mrr']:.3f}"
            )
        if failures:
            print("\nFAIL")
            for failure in failures:
                print(f"- {failure}")
        else:
            print("\nPASS")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
