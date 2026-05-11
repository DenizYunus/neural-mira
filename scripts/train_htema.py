#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

from htema_core import (
    FEATURE_NAMES,
    build_query_spec,
    feature_vector,
    load_jsonl,
    parse_diary_memories,
    score_memories,
    split_train_test,
)


LAB_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING = LAB_ROOT / "data" / "generated_queries.jsonl"
DEFAULT_MODEL = LAB_ROOT / "data" / "htema_model.json"


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


def positive_memories(example: dict[str, Any], memories_by_id: dict[str, Any], memories_by_date: dict[str, Any]) -> list[Any]:
    positives = []
    for entry_id in example.get("positive_ids") or []:
        memory = memories_by_id.get(entry_id)
        if memory:
            positives.append(memory)
    for day in example.get("positive_dates") or []:
        memory = memories_by_date.get(day)
        if memory and memory not in positives:
            positives.append(memory)
    if not positives and example.get("target_entry_id"):
        memory = memories_by_id.get(example["target_entry_id"])
        if memory:
            positives.append(memory)
    return positives


def candidate_negatives(example: dict[str, Any], positives: list[Any], memories: list[Any], rng: random.Random, count: int) -> list[Any]:
    positive_ids = {memory.entry_id for memory in positives}
    positive_dates = {memory.date for memory in positives}
    target_mood = example.get("target_mood")
    window = example.get("positive_window") or []
    start, end = (window + [None, None])[:2]
    strategy = str(example.get("hard_negative_strategy") or "").lower()

    pool = [memory for memory in memories if memory.entry_id not in positive_ids and memory.date not in positive_dates]
    hard = []

    if "same emotion" in strategy or "same mood" in strategy or target_mood:
        hard.extend([memory for memory in pool if target_mood and memory.mood == target_mood and not (start and end and start <= memory.date <= end)])

    if start and end:
        hard.extend([memory for memory in pool if start <= memory.date <= end])

    query_terms = set(str(example.get("query") or "").lower().split())
    if query_terms:
        hard.extend([memory for memory in pool if query_terms.intersection(set(memory.tokens))])

    dedup: dict[str, Any] = {memory.entry_id: memory for memory in hard if memory.entry_id not in positive_ids}
    hard = list(dedup.values())
    rng.shuffle(hard)
    rng.shuffle(pool)
    return (hard[: count // 2] + pool[: count])[:count]


def examples_to_pairs(rows: list[dict[str, Any]], memories: list[Any], negatives_per_positive: int, seed: int) -> list[tuple[dict[str, float], dict[str, float]]]:
    rng = random.Random(seed)
    memories_by_id = {memory.entry_id: memory for memory in memories}
    memories_by_date = {memory.date: memory for memory in memories}
    pairs = []

    for row in rows:
        positives = positive_memories(row, memories_by_id, memories_by_date)
        if not positives:
            continue
        override_window = None
        window = row.get("positive_window")
        if isinstance(window, list) and len(window) == 2 and all(isinstance(item, str) for item in window):
            override_window = (window[0], window[1])
        query = build_query_spec(str(row.get("query") or ""), override_window)
        anchors = [memory.ordinal for memory in positives]
        negatives = candidate_negatives(row, positives, memories, rng, negatives_per_positive * len(positives))

        for positive in positives:
            positive_features = feature_vector(positive, query, anchors)
            for negative in negatives[:negatives_per_positive]:
                pairs.append((positive_features, feature_vector(negative, query, anchors)))

    rng.shuffle(pairs)
    return pairs


def evaluate(rows: list[dict[str, Any]], memories: list[Any], weights: dict[str, float], top_k: int = 5) -> dict[str, float]:
    if not rows:
        return {"queries": 0, "recall_at_1": 0.0, "recall_at_5": 0.0, "mrr": 0.0}

    memories_by_id = {memory.entry_id: memory for memory in memories}
    memories_by_date = {memory.date: memory for memory in memories}
    recall_1 = 0
    recall_k = 0
    reciprocal = 0.0
    used = 0

    for row in rows:
        positives = positive_memories(row, memories_by_id, memories_by_date)
        if not positives:
            continue
        positive_ids = {memory.entry_id for memory in positives}
        override_window = None
        window = row.get("positive_window")
        if isinstance(window, list) and len(window) == 2 and all(isinstance(item, str) for item in window):
            override_window = (window[0], window[1])
        query = build_query_spec(str(row.get("query") or ""), override_window)
        ranked = score_memories(query, memories, weights)
        ranked_ids = [item["memory"].entry_id for item in ranked]
        rank = next((index + 1 for index, entry_id in enumerate(ranked_ids) if entry_id in positive_ids), None)
        if rank is None:
            continue
        used += 1
        recall_1 += int(rank <= 1)
        recall_k += int(rank <= top_k)
        reciprocal += 1 / rank

    return {
        "queries": used,
        "recall_at_1": recall_1 / used if used else 0.0,
        "recall_at_5": recall_k / used if used else 0.0,
        "mrr": reciprocal / used if used else 0.0,
    }


def train_pairwise(pairs: list[tuple[dict[str, float], dict[str, float]]], epochs: int, lr: float, l2: float) -> dict[str, float]:
    weights = {
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

    for epoch in range(epochs):
        random.shuffle(pairs)
        total_loss = 0.0
        for positive, negative in pairs:
            diff = {name: positive.get(name, 0.0) - negative.get(name, 0.0) for name in FEATURE_NAMES}
            margin = sum(weights.get(name, 0.0) * diff[name] for name in FEATURE_NAMES)
            probability = sigmoid(margin)
            total_loss += -math.log(max(probability, 1e-9))
            gradient_scale = 1 - probability
            for name in FEATURE_NAMES:
                weights[name] = weights.get(name, 0.0) + lr * (gradient_scale * diff[name] - l2 * weights.get(name, 0.0))

        if epoch == epochs - 1 or epoch % max(1, epochs // 5) == 0:
            print(f"epoch={epoch + 1}/{epochs} loss={total_loss / max(len(pairs), 1):.4f}")

    return weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the lightweight HTEMA pairwise reranker.")
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--negatives", type=int, default=18)
    parser.add_argument("--test-ratio", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.training)
    if not rows:
        raise SystemExit(f"No training rows found at {args.training}")

    memories = parse_diary_memories()
    train_rows, test_rows = split_train_test(rows, args.test_ratio, args.seed)
    pairs = examples_to_pairs(train_rows, memories, args.negatives, args.seed)
    if not pairs:
        raise SystemExit("No pairwise training pairs could be built.")

    print(f"training_rows={len(train_rows)} test_rows={len(test_rows)} memories={len(memories)} pairs={len(pairs)}")
    baseline = {
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
    print("baseline", json.dumps(evaluate(test_rows, memories, baseline), indent=2))
    weights = train_pairwise(pairs, args.epochs, args.lr, args.l2)
    train_metrics = evaluate(train_rows, memories, weights)
    test_metrics = evaluate(test_rows, memories, weights)

    model = {
        "name": "HTEMA lightweight pairwise ranker",
        "version": 1,
        "created_at": int(time.time()),
        "feature_names": FEATURE_NAMES,
        "weights": weights,
        "training_rows": len(train_rows),
        "test_rows": len(test_rows),
        "pairs": len(pairs),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.model.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    print("trained", json.dumps(test_metrics, indent=2))
    print(f"wrote {args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

