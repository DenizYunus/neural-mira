#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from honest_mira import (
    BM25Index,
    DEFAULT_MODEL,
    DenseSemanticIndex,
    EvalExample,
    NeuralMIRARanker,
    SemanticLsaIndex,
    align_feature_matrix,
    build_pair_features,
    normalize_component,
)
from htema_core import FEATURE_NAMES, build_query_spec, compact_text, load_reflections, parse_all_memories


def torch_load(path: Path, device: str | torch.device = "cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.no_grad()
def score_query(
    query: str,
    model_path: Path,
    device: torch.device,
    *,
    include_rollups: bool = True,
    include_atoms: bool = True,
    include_reflections: bool = True,
) -> tuple[list, np.ndarray, dict[str, np.ndarray], dict, np.ndarray]:
    if not model_path.exists():
        raise SystemExit(f"No neural MIRA model found at {model_path}. Run scripts/honest_mira.py first.")

    memories = parse_all_memories(include_rollups=include_rollups, include_atoms=include_atoms)
    if include_reflections:
        reflections = load_reflections()
        if reflections:
            memories = sorted(memories + reflections, key=lambda m: m.ordinal)
    checkpoint = torch_load(model_path, device)
    model = NeuralMIRARanker(len(checkpoint["feature_names"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    bm25 = BM25Index([memory.tokens for memory in memories])
    semantic = SemanticLsaIndex([memory.text for memory in memories])
    try:
        dense: DenseSemanticIndex | None = DenseSemanticIndex(memories)
    except Exception as exc:
        print(f"warning: dense head unavailable ({exc}); falling back to LSA only.")
        dense = None

    example = EvalExample(
        query=query,
        positive_ids=(memories[0].entry_id,),
        intent="ad_hoc_search",
        style="ad_hoc",
        target_month="",
    )
    features, components, feature_names = build_pair_features([example], memories, bm25, semantic, dense=dense)
    features = align_feature_matrix(features, feature_names, checkpoint["feature_names"])
    mean = checkpoint["mean"]
    std = checkpoint["std"]
    x = torch.tensor((features - mean) / std, dtype=torch.float32, device=device)
    neural = model(x).detach().cpu().numpy()
    normalized = {
        "neural": normalize_component(neural),
        "bm25": normalize_component(components["bm25"]),
        "semantic": normalize_component(components["semantic_embed"]),
        "dense": normalize_component(components.get("dense_semantic", components["semantic_embed"])),
        "scalar": normalize_component(components["scalar_htema"]),
    }
    calibration = checkpoint["calibration"]
    final = sum(float(calibration.get(name, 0.0)) * normalized[name] for name in normalized)
    return memories, final[0], normalized, checkpoint, features[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search diary and WhatsApp memories with the trained neural MIRA ranker.")
    parser.add_argument("query", nargs="+")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--temporal-floor", type=float, default=0.35)
    parser.add_argument("--include-out-of-window", action="store_true")
    parser.add_argument("--no-rollups", action="store_true", help="Exclude week/month rollup tokens from the corpus.")
    parser.add_argument("--no-atoms", action="store_true", help="Exclude sub-day memory atoms.")
    parser.add_argument("--no-reflections", action="store_true", help="Exclude Level-5 reflection tokens.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--show-text", action="store_true")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> int:
    args = parse_args()
    query = " ".join(args.query)
    device = choose_device(args.device)
    memories, scores, components, checkpoint, features = score_query(
        query,
        args.model,
        device,
        include_rollups=not args.no_rollups,
        include_atoms=not args.no_atoms,
        include_reflections=not args.no_reflections,
    )
    candidate_indices = np.arange(len(memories))
    query_spec = build_query_spec(query)
    if query_spec.time_window and not args.include_out_of_window:
        temporal_index = checkpoint["feature_names"].index("temporal") if "temporal" in checkpoint["feature_names"] else FEATURE_NAMES.index("temporal")
        filtered = candidate_indices[features[:, temporal_index] >= args.temporal_floor]
        candidate_indices = filtered
    ranked = candidate_indices[np.argsort(-scores[candidate_indices])][: max(1, args.limit)]

    results = []
    for rank, index in enumerate(ranked, start=1):
        memory = memories[int(index)]
        results.append(
            {
                "rank": rank,
                "date": memory.date,
                "entry_id": memory.entry_id,
                "mood": memory.mood,
                "score": float(scores[index]),
                "neural": float(components["neural"][0, index]),
                "bm25": float(components["bm25"][0, index]),
                "semantic": float(components["semantic"][0, index]),
                "dense": float(components["dense"][0, index]),
                "scalar": float(components["scalar"][0, index]),
                "icons": memory.icons,
                "source_type": getattr(memory, "source_type", "diary"),
                "participants": list(getattr(memory, "participants", ())),
                "trust": memory.trust_payload(),
                "preview": compact_text(memory.text, 900 if args.show_text else 360),
            }
        )

    if args.json:
        print(
            json.dumps(
                {
                    "query": query,
                    "model": str(args.model),
                    "calibration": checkpoint["calibration"],
                    "time_window": query_spec.time_window,
                    "temporal_filter": None if args.include_out_of_window else args.temporal_floor,
                    "candidate_count": int(len(candidate_indices)),
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    print(f"query: {query}")
    print(f"model: {args.model}")
    if query_spec.time_window and not args.include_out_of_window:
        print(f"time_window: {query_spec.time_window[0]} to {query_spec.time_window[1]} (temporal_floor={args.temporal_floor})")
    print("calibration:", ", ".join(f"{key}={value:.2f}" for key, value in checkpoint["calibration"].items()))
    for result in results:
        print(
            "\n#{rank} score={score:.3f} date={date} mood={mood} src={source_type} trust={trust_level:.2f} evidence={evidence_type} neural={neural:.2f} bm25={bm25:.2f} semantic={semantic:.2f} dense={dense:.2f} scalar={scalar:.2f}".format(
                trust_level=float(result["trust"]["trust_level"] or 0.0),
                evidence_type=result["trust"]["evidence_type"],
                **result
            )
        )
        print(result["entry_id"])
        if result["source_type"] == "whatsapp" and result["participants"]:
            print("participants:", ", ".join(result["participants"][:6]))
        if result["trust"].get("provenance_excerpts"):
            print("provenance:", " | ".join(result["trust"]["provenance_excerpts"][:2]))
        print(result["preview"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
