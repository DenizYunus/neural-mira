#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from htema_core import build_query_spec, compact_text, parse_diary_memories, score_memories


LAB_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = LAB_ROOT / "data" / "htema_model.json"


def load_model(path: Path) -> dict:
    if not path.exists():
        return {
            "weights": {
                "bias": 0.0,
                "semantic": 1.0,
                "temporal": 1.0,
                "emotion": 1.0,
                "entity": 1.0,
                "hierarchy": 0.5,
                "continuity": 0.8,
                "importance": 0.35,
                "unresolved": 0.25,
                "contradiction": 0.25,
            }
        }
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search diary memory with the trained HTEMA ranker.")
    parser.add_argument("query", nargs="+")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--temporal-floor", type=float, default=0.35)
    parser.add_argument("--include-out-of-window", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    query_text = " ".join(args.query)
    model = load_model(args.model)
    weights = model["weights"]
    memories = parse_diary_memories()
    query = build_query_spec(query_text)
    ranked = score_memories(query, memories, weights)
    if query.time_window and not args.include_out_of_window:
        filtered = [item for item in ranked if item["features"].get("temporal", 0.0) >= args.temporal_floor]
        if filtered:
            ranked = filtered
    ranked = ranked[: args.limit]

    if args.json:
        print(
            json.dumps(
                {
                    "query": query_text,
                    "model": model.get("name", "baseline"),
                    "weights": weights,
                    "results": [
                        {
                            "date": item["memory"].date,
                            "entry_id": item["memory"].entry_id,
                            "mood": item["memory"].mood,
                            "icons": item["memory"].icons,
                            "score": item["score"],
                            "features": item["features"],
                            "trust": item["memory"].trust_payload(),
                            "text": compact_text(item["memory"].text, 900),
                        }
                        for item in ranked
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    print(f"Query: {query_text}")
    print(f"Model: {model.get('name', 'baseline')}")
    print("")
    for index, item in enumerate(ranked, start=1):
        memory = item["memory"]
        features = " ".join(f"{name}:{value:.2f}" for name, value in item["features"].items() if name != "bias")
        print(
            f"{index}. {memory.date} mood={memory.mood or 'n/a'} "
            f"score={item['score']:.3f} trust={memory.trust_level:.2f} evidence={memory.evidence_type}"
        )
        print(memory.entry_id)
        print(features)
        print(compact_text(memory.text, 700))
        print("\n---\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
