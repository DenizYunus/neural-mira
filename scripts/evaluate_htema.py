#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from htema_core import load_jsonl, parse_diary_memories
from train_htema import evaluate


LAB_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING = LAB_ROOT / "data" / "generated_queries.jsonl"
DEFAULT_MODEL = LAB_ROOT / "data" / "htema_model.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained HTEMA ranker on generated query examples.")
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.training)
    memories = parse_diary_memories()
    model = json.loads(args.model.read_text(encoding="utf-8"))
    print(json.dumps(evaluate(rows, memories, model["weights"]), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
