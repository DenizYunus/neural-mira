#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from htema_core import load_jsonl, parse_diary_memories
from neural_htema_common import (
    DEFAULT_EMBEDDINGS,
    DEFAULT_MODEL,
    DEFAULT_TRAINING,
    choose_device,
    make_model,
    memory_feature_values,
)
from train_neural_htema import evaluate, torch_load


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained neural HTEMA model.")
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--oracle-window",
        action="store_true",
        help="Use positive_window labels as query time windows. Diagnostic only; leaks labels into evaluation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.training)
    if not rows:
        raise SystemExit(f"No training rows found at {args.training}")
    if not args.model.exists():
        raise SystemExit(f"No neural model found at {args.model}. Run train_neural_htema.py first.")
    if not args.embeddings.exists():
        raise SystemExit(f"No embedding cache found at {args.embeddings}. Run build_neural_embeddings.py first.")

    device = choose_device(args.device)
    checkpoint = torch_load(args.model)
    cache = torch_load(args.embeddings)
    memories = parse_diary_memories()
    if [memory.entry_id for memory in memories] != list(cache["memory_ids"]):
        raise SystemExit("Diary memories no longer match the embedding cache. Rebuild neural embeddings.")

    config = checkpoint["config"]
    model = make_model(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    stats = dict(config.get("stats") or cache["stats"])
    memory_features = torch.tensor([memory_feature_values(memory, stats) for memory in memories], dtype=torch.float32)
    memories_by_id = {memory.entry_id: memory for memory in memories}
    memories_by_date = {memory.date: memory for memory in memories}
    query_index = {text: index for index, text in enumerate(cache["query_texts"])}

    metrics = evaluate(
        model,
        rows,
        memories,
        memories_by_id,
        memories_by_date,
        cache["memory_embeddings"].to(torch.float32),
        memory_features,
        query_index,
        cache["query_embeddings"].to(torch.float32),
        stats,
        device,
        oracle_window=args.oracle_window,
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
