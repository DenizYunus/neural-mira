#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from tqdm import tqdm

from htema_core import anchor_ordinals, load_jsonl, parse_diary_memories, split_train_test
from neural_htema_common import (
    DEFAULT_EMBEDDINGS,
    DEFAULT_METRICS,
    DEFAULT_MODEL,
    DEFAULT_TRAINING,
    FEATURE_NAMES,
    MEMORY_FEATURE_NAMES,
    QUERY_FEATURE_NAMES,
    NeuralHTEMA,
    build_query_from_row,
    checkpoint_config,
    choose_device,
    make_model,
    memory_feature_values,
    pair_feature_values,
    query_feature_values,
)
from train_htema import candidate_negatives, positive_memories


def torch_load(path: Path, device: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train neural HTEMA Q/K/V adapters over pretrained embeddings.")
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--negatives", type=int, default=48)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--adapter-dim", type=int, default=160)
    parser.add_argument("--hidden-dim", type=int, default=320)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--test-ratio", type=float, default=0.18)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--oracle-window",
        action="store_true",
        help="Use positive_window labels as query time windows. Diagnostic only; leaks labels into evaluation.",
    )
    return parser.parse_args()


def query_embedding_for(row: dict[str, Any], query_index: dict[str, int], query_embeddings: torch.Tensor) -> torch.Tensor:
    text = str(row.get("query") or "").strip()
    index = query_index.get(text)
    if index is None:
        raise KeyError(f"Query missing from embedding cache: {text[:120]}")
    return query_embeddings[index]


def make_candidate_batch(
    rows: list[dict[str, Any]],
    memories: list[Any],
    memories_by_id: dict[str, Any],
    memories_by_date: dict[str, Any],
    memory_id_to_index: dict[str, int],
    memory_embeddings: torch.Tensor,
    memory_features: torch.Tensor,
    query_index: dict[str, int],
    query_embeddings: torch.Tensor,
    stats: dict[str, float],
    rng: random.Random,
    negatives: int,
    device: torch.device,
    oracle_window: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    q_embs = []
    q_features = []
    m_embs = []
    m_features = []
    pair_features = []

    for row in rows:
        positives = positive_memories(row, memories_by_id, memories_by_date)
        if not positives:
            continue
        positive = rng.choice(positives)
        negatives_for_row = candidate_negatives(row, positives, memories, rng, negatives)
        if len(negatives_for_row) < negatives:
            continue
        candidates = [positive, *negatives_for_row[:negatives]]
        query = build_query_from_row(row, oracle_window=oracle_window)
        anchors = anchor_ordinals(query, memories)
        indices = [memory_id_to_index[memory.entry_id] for memory in candidates]

        q_embs.append(query_embedding_for(row, query_index, query_embeddings))
        q_features.append(query_feature_values(query, stats))
        m_embs.append(memory_embeddings[indices])
        m_features.append(memory_features[indices])
        pair_features.append([pair_feature_values(memory, query, anchors) for memory in candidates])

    if not q_embs:
        return None

    return (
        torch.stack(q_embs).to(device),
        torch.tensor(q_features, dtype=torch.float32, device=device),
        torch.stack(m_embs).to(device),
        torch.stack(m_features).to(device),
        torch.tensor(pair_features, dtype=torch.float32, device=device),
        torch.zeros(len(q_embs), dtype=torch.long, device=device),
    )


@torch.no_grad()
def evaluate(
    model: NeuralHTEMA,
    rows: list[dict[str, Any]],
    memories: list[Any],
    memories_by_id: dict[str, Any],
    memories_by_date: dict[str, Any],
    memory_embeddings: torch.Tensor,
    memory_features: torch.Tensor,
    query_index: dict[str, int],
    query_embeddings: torch.Tensor,
    stats: dict[str, float],
    device: torch.device,
    top_k: int = 5,
    oracle_window: bool = False,
) -> dict[str, float]:
    if not rows:
        return {"queries": 0, "recall_at_1": 0.0, "recall_at_5": 0.0, "mrr": 0.0}

    model.eval()
    recall_1 = 0
    recall_k = 0
    reciprocal = 0.0
    used = 0
    all_memory_embeddings = memory_embeddings.unsqueeze(0).to(device)
    all_memory_features = memory_features.unsqueeze(0).to(device)

    for row in rows:
        positives = positive_memories(row, memories_by_id, memories_by_date)
        if not positives:
            continue
        positive_ids = {memory.entry_id for memory in positives}
        query = build_query_from_row(row, oracle_window=oracle_window)
        anchors = anchor_ordinals(query, memories)
        pair_features = torch.tensor(
            [[pair_feature_values(memory, query, anchors) for memory in memories]],
            dtype=torch.float32,
            device=device,
        )
        q_embedding = query_embedding_for(row, query_index, query_embeddings).unsqueeze(0).to(device)
        q_features = torch.tensor([query_feature_values(query, stats)], dtype=torch.float32, device=device)
        scores, _ = model(q_embedding, q_features, all_memory_embeddings, all_memory_features, pair_features)
        ranked_indices = torch.argsort(scores.squeeze(0), descending=True).tolist()
        ranked_ids = [memories[index].entry_id for index in ranked_indices]
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


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = choose_device(args.device)

    rows = load_jsonl(args.training)
    if not rows:
        raise SystemExit(f"No training rows found at {args.training}")
    if not args.embeddings.exists():
        raise SystemExit(f"No embedding cache found at {args.embeddings}. Run build_neural_embeddings.py first.")

    cache = torch_load(args.embeddings)
    memories = parse_diary_memories()
    if [memory.entry_id for memory in memories] != list(cache["memory_ids"]):
        raise SystemExit("Diary memories no longer match the embedding cache. Rebuild neural embeddings.")

    query_index = {text: index for index, text in enumerate(cache["query_texts"])}
    memory_id_to_index = {memory.entry_id: index for index, memory in enumerate(memories)}
    memories_by_id = {memory.entry_id: memory for memory in memories}
    memories_by_date = {memory.date: memory for memory in memories}
    stats = dict(cache["stats"])

    memory_embeddings = cache["memory_embeddings"].to(torch.float32)
    query_embeddings = cache["query_embeddings"].to(torch.float32)
    memory_features = torch.tensor(
        [memory_feature_values(memory, stats) for memory in memories],
        dtype=torch.float32,
    )

    config = checkpoint_config(
        embedding_model=str(cache["embedding_model"]),
        embedding_dim=int(cache["embedding_dim"]),
        adapter_dim=args.adapter_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        stats=stats,
    )
    model = make_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_rows, test_rows = split_train_test(rows, args.test_ratio, args.seed)

    print(
        "training_rows={train} test_rows={test} memories={memories} embedding_dim={dim} device={device}".format(
            train=len(train_rows),
            test=len(test_rows),
            memories=len(memories),
            dim=cache["embedding_dim"],
            device=device,
        )
    )
    print("feature_dims", json.dumps({
        "query": len(QUERY_FEATURE_NAMES),
        "memory": len(MEMORY_FEATURE_NAMES),
        "pair": len(FEATURE_NAMES),
    }))

    best_mrr = -1.0
    best_state = None
    history = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(train_rows)
        losses = []
        progress = tqdm(range(0, len(train_rows), args.batch_size), desc=f"epoch {epoch}/{args.epochs}")
        for offset in progress:
            batch_rows = train_rows[offset : offset + args.batch_size]
            batch = make_candidate_batch(
                batch_rows,
                memories,
                memories_by_id,
                memories_by_date,
                memory_id_to_index,
                memory_embeddings,
                memory_features,
                query_index,
                query_embeddings,
                stats,
                rng,
                args.negatives,
                device,
                args.oracle_window,
            )
            if batch is None:
                continue
            q_emb, q_feat, m_emb, m_feat, pair_feat, labels = batch
            optimizer.zero_grad(set_to_none=True)
            scores, _ = model(q_emb, q_feat, m_emb, m_feat, pair_feat)
            loss = F.cross_entropy(scores, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            value = float(loss.detach().cpu())
            losses.append(value)
            progress.set_postfix(loss=f"{value:.4f}")

        should_eval = epoch == 1 or epoch == args.epochs or epoch % max(1, args.eval_every) == 0
        epoch_record: dict[str, Any] = {
            "epoch": epoch,
            "loss": sum(losses) / max(len(losses), 1),
        }
        if should_eval:
            test_metrics = evaluate(
                model,
                test_rows,
                memories,
                memories_by_id,
                memories_by_date,
                memory_embeddings,
                memory_features,
                query_index,
                query_embeddings,
                stats,
                device,
                oracle_window=args.oracle_window,
            )
            epoch_record["test_metrics"] = test_metrics
            print(f"epoch={epoch} loss={epoch_record['loss']:.4f} test={json.dumps(test_metrics)}")
            if test_metrics["mrr"] > best_mrr:
                best_mrr = test_metrics["mrr"]
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history.append(epoch_record)

    if best_state is not None:
        model.load_state_dict(best_state)

    train_metrics = evaluate(
        model,
        train_rows,
        memories,
        memories_by_id,
        memories_by_date,
        memory_embeddings,
        memory_features,
        query_index,
        query_embeddings,
        stats,
        device,
        oracle_window=args.oracle_window,
    )
    test_metrics = evaluate(
        model,
        test_rows,
        memories,
        memories_by_id,
        memories_by_date,
        memory_embeddings,
        memory_features,
        query_index,
        query_embeddings,
        stats,
        device,
        oracle_window=args.oracle_window,
    )
    full_metrics = evaluate(
        model,
        rows,
        memories,
        memories_by_id,
        memories_by_date,
        memory_embeddings,
        memory_features,
        query_index,
        query_embeddings,
        stats,
        device,
        oracle_window=args.oracle_window,
    )

    checkpoint = {
        "config": config,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "created_at": int(time.time()),
        "training_rows": len(train_rows),
        "test_rows": len(test_rows),
        "oracle_window": bool(args.oracle_window),
        "history": history,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "full_metrics": full_metrics,
    }
    args.model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.model)

    metrics_payload = {
        "created_at": checkpoint["created_at"],
        "duration_seconds": round(time.time() - start_time, 2),
        "model": str(args.model),
        "embeddings": str(args.embeddings),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "full_metrics": full_metrics,
        "history": history,
    }
    args.metrics.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print("trained", json.dumps({"train": train_metrics, "test": test_metrics, "full": full_metrics}, indent=2))
    print(f"wrote_model={args.model}")
    print(f"wrote_metrics={args.metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
