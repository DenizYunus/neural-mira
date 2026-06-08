#!/usr/bin/env python3
"""Run reconstruction benchmark across query modes and collect one table."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "benchmarks" / "mira_synthetic" / "generated"
DEFAULT_MODES = ("date_only", "date_plus_public_metadata", "sparse_user_hint", "legacy_hidden_label")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diary-root", type=Path, default=GENERATED / "diaries")
    parser.add_argument("--whatsapp-root", type=Path, default=GENERATED / "whatsapp")
    parser.add_argument("--photo-metadata", type=Path, default=GENERATED / "photos" / "photo_metadata.jsonl")
    parser.add_argument("--ground-truth-events", type=Path, default=GENERATED / "ground_truth" / "events.jsonl")
    parser.add_argument("--target-dates-file", type=Path, default=GENERATED / "queries" / "reconstruction_targets.jsonl")
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES), choices=DEFAULT_MODES)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--summary-json", type=Path, default=GENERATED / "reconstruction_sweep.json")
    parser.add_argument("--summary-md", type=Path, default=GENERATED / "reconstruction_sweep.md")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def run_mode(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    metrics_path = GENERATED / f"reconstruction_metrics.{mode}.json"
    report_path = GENERATED / f"reconstruction_report.{mode}.md"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "reconstruction_benchmark.py"),
        "--diary-root",
        str(args.diary_root),
        "--whatsapp-root",
        str(args.whatsapp_root),
        "--photo-metadata",
        str(args.photo_metadata),
        "--ground-truth-events",
        str(args.ground_truth_events),
        "--target-dates-file",
        str(args.target_dates_file),
        "--query-mode",
        mode,
        "--report",
        str(report_path),
        "--metrics",
        str(metrics_path),
        "--top-k",
        str(args.top_k),
        "--no-reflections",
        "--json",
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise SystemExit(
            f"reconstruction benchmark failed for {mode}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


def metric(metrics: dict[str, Any], name: str) -> float:
    value = metrics.get(name)
    return float(value) if value is not None else 0.0


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Reconstruction Query-Mode Sweep",
        "",
        f"Generated at: {payload['created_at']}",
        f"Top-k evidence: {payload['top_k']}",
        "",
        "| Mode | Model | Fact Recall | Fact Accuracy | Event F1 | Confuser | Confidence Error | Audit Trail |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {mode} | {model} | {fact_recall:.3f} | {fact_accuracy:.3f} | {event_f1:.3f} | {confuser:.3f} | {confidence_error:.3f} | {audit_trail:.3f} |".format(
                **row
            )
        )
    lines.extend([
        "",
        "## Per-Slot Fact Recovery",
        "",
        "| Mode | Model | Slot | Recall | Accuracy | N |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ])
    for row in payload.get("slot_rows", []):
        lines.append(
            "| {mode} | {model} | {slot} | {recall:.3f} | {accuracy:.3f} | {n:.0f} |".format(**row)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    slot_rows: list[dict[str, Any]] = []
    by_mode: dict[str, Any] = {}
    for mode in args.modes:
        result = run_mode(args, mode)
        by_mode[mode] = result
        for model, metrics in result["summary"].items():
            rows.append({
                "mode": mode,
                "model": model,
                "fact_recall": metric(metrics, "reconstruction_recall"),
                "fact_accuracy": metric(metrics, "fact_slot_accuracy"),
                "event_f1": metric(metrics, "event_f1"),
                "confuser": metric(metrics, "confuser_intrusion_rate"),
                "confidence_error": metric(metrics, "confidence_error"),
                "audit_trail": metric(metrics, "audit_trail_coverage"),
            })
        for model, slots in result.get("slot_summary", {}).items():
            for slot, metrics in slots.items():
                slot_rows.append({
                    "mode": mode,
                    "model": model,
                    "slot": slot,
                    "recall": metric(metrics, "recall"),
                    "accuracy": metric(metrics, "accuracy"),
                    "n": metric(metrics, "n"),
                })

    payload = {
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "top_k": args.top_k,
        "modes": list(args.modes),
        "rows": rows,
        "slot_rows": slot_rows,
        "raw": by_mode,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(args.summary_md, payload)

    if args.json:
        print(json.dumps({"rows": rows, "slot_rows": slot_rows, "modes": list(args.modes)}, indent=2))
    else:
        print(f"wrote {args.summary_md}")
        print(f"wrote {args.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
