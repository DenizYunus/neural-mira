#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from htema_core import FEATURE_NAMES, anchor_ordinals, build_query_spec, compact_text, parse_diary_memories
from neural_htema_common import (
    DEFAULT_EMBEDDINGS,
    DEFAULT_MODEL,
    DEFAULT_PRIOR_WEIGHTS,
    DEFAULT_SCALAR_MODEL,
    choose_device,
    make_model,
    memory_feature_values,
    pair_feature_values,
    query_feature_values,
    scalar_prior_score,
)
from train_neural_htema import torch_load


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search diary memories with neural HTEMA Q/K/V adapters.")
    parser.add_argument("query")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--scalar-model", type=Path, default=DEFAULT_SCALAR_MODEL)
    parser.add_argument("--prior-weight", type=float, default=0.85)
    parser.add_argument("--temporal-floor", type=float, default=0.35)
    parser.add_argument("--include-out-of-window", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--show-text", action="store_true")
    return parser.parse_args()


def load_prior_weights(path: Path) -> dict[str, float]:
    if not path.exists():
        return DEFAULT_PRIOR_WEIGHTS
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        weights = payload.get("weights")
        if isinstance(weights, dict):
            return {name: float(weights.get(name, DEFAULT_PRIOR_WEIGHTS.get(name, 0.0))) for name in DEFAULT_PRIOR_WEIGHTS}
    except (OSError, ValueError, TypeError):
        pass
    return DEFAULT_PRIOR_WEIGHTS


def encode_query_text(query: str, cache: dict, device: torch.device) -> torch.Tensor:
    query_index = {text: index for index, text in enumerate(cache["query_texts"])}
    if query in query_index:
        return cache["query_embeddings"][query_index[query]].to(torch.float32)
    embedder = SentenceTransformer(str(cache["embedding_model"]), device=str(device))
    return embedder.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_tensor=True,
    )[0].to(torch.float32).cpu()


def main() -> int:
    args = parse_args()
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
    model.eval()

    stats = dict(config.get("stats") or cache["stats"])
    query = build_query_spec(args.query)
    anchors = anchor_ordinals(query, memories)
    memory_embeddings = cache["memory_embeddings"].to(torch.float32)
    memory_features = torch.tensor([memory_feature_values(memory, stats) for memory in memories], dtype=torch.float32)
    pair_features = torch.tensor(
        [[pair_feature_values(memory, query, anchors) for memory in memories]],
        dtype=torch.float32,
        device=device,
    )
    query_embedding = encode_query_text(args.query, cache, device).unsqueeze(0).to(device)
    query_features = torch.tensor([query_feature_values(query, stats)], dtype=torch.float32, device=device)

    with torch.no_grad():
        scores, _ = model(
            query_embedding,
            query_features,
            memory_embeddings.unsqueeze(0).to(device),
            memory_features.unsqueeze(0).to(device),
            pair_features,
        )
    neural_score_values = scores.squeeze(0).detach().cpu()
    prior_weights = load_prior_weights(args.scalar_model)
    pair_feature_rows = pair_features.squeeze(0).detach().cpu().tolist()
    prior_score_values = torch.tensor(
        [scalar_prior_score(row, prior_weights) for row in pair_feature_rows],
        dtype=torch.float32,
    )
    score_values = neural_score_values + args.prior_weight * prior_score_values
    candidate_indices = list(range(len(memories)))
    if query.time_window and not args.include_out_of_window:
        temporal_index = FEATURE_NAMES.index("temporal")
        filtered = [index for index in candidate_indices if pair_feature_rows[index][temporal_index] >= args.temporal_floor]
        if filtered:
            candidate_indices = filtered
    ranked_indices = sorted(candidate_indices, key=lambda index: float(score_values[index]), reverse=True)[: args.limit]

    results = []
    for rank, index in enumerate(ranked_indices, start=1):
        memory = memories[index]
        features = dict(zip(FEATURE_NAMES, [float(value) for value in pair_features[0, index].detach().cpu().tolist()]))
        results.append(
            {
                "rank": rank,
                "score": float(score_values[index]),
                "neural_score": float(neural_score_values[index]),
                "prior_score": float(prior_score_values[index]),
                "date": memory.date,
                "mood": memory.mood,
                "entry_id": memory.entry_id,
                "source_path": memory.source_path,
                "features": features,
                "preview": compact_text(memory.text, 900 if args.show_text else 320),
            }
        )

    if args.json:
        print(json.dumps({"query": args.query, "results": results}, ensure_ascii=False, indent=2))
        return 0

    print(f"query: {args.query}")
    for result in results:
        print(
            "\n#{rank} score={score:.4f} neural={neural:.4f} prior={prior:.4f} date={date} mood={mood} source={source}".format(
                rank=result["rank"],
                score=result["score"],
                neural=result["neural_score"],
                prior=result["prior_score"],
                date=result["date"],
                mood=result["mood"],
                source=result["source_path"],
            )
        )
        feature_text = ", ".join(f"{name}={value:.2f}" for name, value in result["features"].items())
        print(f"features: {feature_text}")
        print(result["preview"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
