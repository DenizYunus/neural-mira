#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Callable

import torch

from htema_core import (
    FEATURE_NAMES,
    anchor_ordinals,
    build_query_spec,
    cosine,
    dot,
    emotion_score,
    entity_score,
    feature_vector,
    load_jsonl,
    parse_diary_memories,
    temporal_score,
)
from neural_htema_common import (
    DEFAULT_EMBEDDINGS,
    DEFAULT_MODEL as DEFAULT_NEURAL_MODEL,
    DEFAULT_SCALAR_MODEL,
    DEFAULT_TRAINING,
    choose_device,
    make_model,
    memory_feature_values,
    pair_feature_values,
    query_feature_values,
    scalar_prior_score,
)
from train_htema import positive_memories
from train_neural_htema import torch_load


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "baseline_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare retrieval baselines for HTEMA.")
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--scalar-model", type=Path, default=DEFAULT_SCALAR_MODEL)
    parser.add_argument("--neural-model", type=Path, default=DEFAULT_NEURAL_MODEL)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-oracle-window", action="store_true", help="Use generated positive_window labels as query time windows.")
    parser.add_argument("--include-neural", action="store_true", help="Also evaluate trained neural HTEMA variants.")
    parser.add_argument("--details", action="store_true", help="Print the full JSON report instead of a compact table.")
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args()


def query_for_row(row: dict[str, Any], use_oracle_window: bool):
    override_window = None
    if use_oracle_window:
        window = row.get("positive_window")
        if isinstance(window, list) and len(window) == 2 and all(isinstance(item, str) for item in window):
            override_window = (window[0], window[1])
    return build_query_spec(str(row.get("query") or ""), override_window)


def load_scalar_weights(path: Path) -> dict[str, float]:
    if not path.exists():
        return {
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
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {name: float(value) for name, value in payload["weights"].items()}


class BM25:
    def __init__(self, tokenized_docs: list[tuple[str, ...]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_lengths = [len(doc) for doc in tokenized_docs]
        self.avgdl = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.term_frequencies = [Counter(doc) for doc in tokenized_docs]
        document_frequency: Counter[str] = Counter()
        for doc in tokenized_docs:
            document_frequency.update(set(doc))
        total_docs = len(tokenized_docs)
        self.idf = {
            term: math.log(1 + (total_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def scores(self, query_tokens: tuple[str, ...]) -> list[float]:
        scores = []
        query_terms = set(query_tokens)
        for freqs, doc_len in zip(self.term_frequencies, self.doc_lengths):
            score = 0.0
            for term in query_terms:
                freq = freqs.get(term, 0)
                if not freq:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = freq + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1e-9))
                score += idf * freq * (self.k1 + 1) / denom
            scores.append(score)
        return scores


def positive_ids_for(row: dict[str, Any], memories_by_id: dict[str, Any], memories_by_date: dict[str, Any]) -> set[str]:
    return {memory.entry_id for memory in positive_memories(row, memories_by_id, memories_by_date)}


def rank_metrics(ranked_ids: list[str], positive_ids: set[str], top_values: tuple[int, ...] = (1, 3, 5, 10)) -> dict[str, float] | None:
    rank = next((index + 1 for index, entry_id in enumerate(ranked_ids) if entry_id in positive_ids), None)
    if rank is None:
        return None
    result = {f"recall_at_{k}": 1.0 if rank <= k else 0.0 for k in top_values}
    result["mrr"] = 1 / rank
    result["rank"] = float(rank)
    return result


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {"queries": 0}
    keys = [key for key in rows[0] if key != "rank"]
    output = {"queries": len(rows)}
    for key in keys:
        output[key] = sum(row[key] for row in rows) / len(rows)
    ranks = [row["rank"] for row in rows]
    output["mean_rank"] = sum(ranks) / len(ranks)
    output["median_rank"] = float(median(ranks))
    return output


def print_summary_table(report: dict[str, Any]) -> None:
    print("")
    print(f"mode={report['evaluation_mode']} queries={report['queries']} memories={report['memories']}")
    print("| baseline | R@1 | R@5 | R@10 | MRR | median rank |")
    print("|---|---:|---:|---:|---:|---:|")
    for item in report["baselines"]:
        metrics = item["overall"]
        print(
            "| {name} | {r1:.3f} | {r5:.3f} | {r10:.3f} | {mrr:.3f} | {median:.1f} |".format(
                name=item["name"],
                r1=metrics.get("recall_at_1", 0.0),
                r5=metrics.get("recall_at_5", 0.0),
                r10=metrics.get("recall_at_10", 0.0),
                mrr=metrics.get("mrr", 0.0),
                median=metrics.get("median_rank", 0.0),
            )
        )


def evaluate_score_function(
    name: str,
    rows: list[dict[str, Any]],
    memories: list[Any],
    memories_by_id: dict[str, Any],
    memories_by_date: dict[str, Any],
    use_oracle_window: bool,
    score_fn: Callable[[dict[str, Any], Any, list[Any]], list[float]],
) -> dict[str, Any]:
    all_metrics = []
    by_intent: dict[str, list[dict[str, float]]] = defaultdict(list)

    for row in rows:
        positive_ids = positive_ids_for(row, memories_by_id, memories_by_date)
        if not positive_ids:
            continue
        scores = score_fn(row, query_for_row(row, use_oracle_window), memories)
        ranked_indices = sorted(range(len(memories)), key=lambda index: scores[index], reverse=True)
        ranked_ids = [memories[index].entry_id for index in ranked_indices]
        metrics = rank_metrics(ranked_ids, positive_ids)
        if not metrics:
            continue
        all_metrics.append(metrics)
        by_intent[str(row.get("intent") or "unknown")].append(metrics)

    return {
        "name": name,
        "overall": summarize(all_metrics),
        "by_intent": {intent: summarize(values) for intent, values in sorted(by_intent.items())},
    }


def neural_score_function(
    args: argparse.Namespace,
    memories: list[Any],
) -> Callable[[dict[str, Any], Any, list[Any]], list[float]] | None:
    if not args.include_neural or not args.neural_model.exists() or not args.embeddings.exists():
        return None

    device = choose_device(args.device)
    checkpoint = torch_load(args.neural_model)
    cache = torch_load(args.embeddings)
    if [memory.entry_id for memory in memories] != list(cache["memory_ids"]):
        raise SystemExit("Diary memories no longer match neural embedding cache. Rebuild embeddings first.")

    config = checkpoint["config"]
    model = make_model(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    stats = dict(config.get("stats") or cache["stats"])
    memory_embeddings = cache["memory_embeddings"].to(torch.float32)
    memory_features = torch.tensor([memory_feature_values(memory, stats) for memory in memories], dtype=torch.float32)
    query_embeddings = cache["query_embeddings"].to(torch.float32)
    query_index = {text: index for index, text in enumerate(cache["query_texts"])}

    def score(row: dict[str, Any], query, _: list[Any]) -> list[float]:
        query_text = str(row.get("query") or "").strip()
        if query_text not in query_index:
            raise KeyError(f"Query missing from embedding cache: {query_text[:100]}")
        q_embedding = query_embeddings[query_index[query_text]].unsqueeze(0).to(device)
        q_features = torch.tensor([query_feature_values(query, stats)], dtype=torch.float32, device=device)
        anchors = anchor_ordinals(query, memories)
        pair_features = torch.tensor(
            [[pair_feature_values(memory, query, anchors) for memory in memories]],
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            scores, _ = model(
                q_embedding,
                q_features,
                memory_embeddings.unsqueeze(0).to(device),
                memory_features.unsqueeze(0).to(device),
                pair_features,
            )
        return [float(value) for value in scores.squeeze(0).detach().cpu().tolist()]

    return score


def neural_prior_score_function(
    args: argparse.Namespace,
    memories: list[Any],
    prior_weights: dict[str, float],
) -> Callable[[dict[str, Any], Any, list[Any]], list[float]] | None:
    base = neural_score_function(args, memories)
    if base is None:
        return None

    def score(row: dict[str, Any], query, memories_arg: list[Any]) -> list[float]:
        neural_scores = base(row, query, memories_arg)
        anchors = anchor_ordinals(query, memories_arg)
        priors = [scalar_prior_score(feature_vector(memory, query, anchors), prior_weights) for memory in memories_arg]
        return [neural + 0.85 * prior for neural, prior in zip(neural_scores, priors)]

    return score


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.training)
    if not rows:
        raise SystemExit(f"No rows found at {args.training}")
    memories = parse_diary_memories()
    memories_by_id = {memory.entry_id: memory for memory in memories}
    memories_by_date = {memory.date: memory for memory in memories}
    scalar_weights = load_scalar_weights(args.scalar_model)
    bm25 = BM25([memory.tokens for memory in memories])

    def scalar_htema_scores(_row: dict[str, Any], query, memories_arg: list[Any]) -> list[float]:
        anchors = anchor_ordinals(query, memories_arg)
        return [dot(scalar_weights, feature_vector(memory, query, anchors)) for memory in memories_arg]

    baseline_specs: list[tuple[str, Callable[[dict[str, Any], Any, list[Any]], list[float]]]] = [
        ("bm25_text", lambda _row, query, _memories: bm25.scores(query.tokens)),
        ("lexical_cosine", lambda _row, query, memories_arg: [cosine(query.token_vector, memory.token_vector) for memory in memories_arg]),
        ("temporal_only", lambda _row, query, memories_arg: [temporal_score(memory, query.time_window) for memory in memories_arg]),
        ("emotion_only", lambda _row, query, memories_arg: [emotion_score(memory, query.emotion) for memory in memories_arg]),
        ("entity_only", lambda _row, query, memories_arg: [entity_score(memory, query) for memory in memories_arg]),
        ("scalar_htema", scalar_htema_scores),
    ]

    semantic_scores = None
    if args.embeddings.exists():
        cache = torch_load(args.embeddings)
        query_index = {text: index for index, text in enumerate(cache["query_texts"])}
        memory_embeddings = cache["memory_embeddings"].to(torch.float32)
        query_embeddings = cache["query_embeddings"].to(torch.float32)

        def semantic_vector(row: dict[str, Any], _query, _memories) -> list[float]:
            query_text = str(row.get("query") or "").strip()
            if query_text not in query_index:
                return [0.0 for _ in memories]
            q = query_embeddings[query_index[query_text]]
            return [float(value) for value in torch.mv(memory_embeddings, q).tolist()]

        semantic_scores = semantic_vector
    if semantic_scores:
        baseline_specs.insert(2, ("semantic_embedding", semantic_scores))

    neural_raw = neural_score_function(args, memories)
    if neural_raw:
        baseline_specs.append(("neural_qkv_raw", neural_raw))
    neural_with_prior = neural_prior_score_function(args, memories, scalar_weights)
    if neural_with_prior:
        baseline_specs.append(("neural_qkv_plus_prior", neural_with_prior))

    report = {
        "training": str(args.training),
        "queries": len(rows),
        "memories": len(memories),
        "evaluation_mode": "oracle_window" if args.use_oracle_window else "query_text_only",
        "baselines": [],
    }
    for name, score_fn in baseline_specs:
        print(f"evaluating={name}", flush=True)
        report["baselines"].append(
            evaluate_score_function(name, rows, memories, memories_by_id, memories_by_date, args.use_oracle_window, score_fn)
        )

    if args.details:
        print(json.dumps(report, indent=2))
    else:
        print_summary_table(report)
    if not args.no_write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
