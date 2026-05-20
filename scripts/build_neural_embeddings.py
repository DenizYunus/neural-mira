#!/usr/bin/env python3
"""Build pretrained embedding cache for the unified diary + WhatsApp memory corpus.

The cache is consumed by:
  - `train_neural_htema.py` (Q/K/V adapter training)
  - `honest_mira.py` (dense semantic head)
  - `search_mira.py` / `mira_service.py` (live retrieval)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from htema_core import (
    compact_text,
    load_jsonl,
    load_reflections,
    parse_all_memories,
    parse_diary_memories,
)
from neural_htema_common import DEFAULT_EMBEDDINGS, DEFAULT_TRAINING, normalizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the pretrained embedding cache for neural HTEMA.")
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--output", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model used for memories and generated queries.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None,
                        help="Optional SentenceTransformer device override (cpu, cuda, cuda:0, mps).")
    parser.add_argument("--diary-only", action="store_true",
                        help="Restrict the cache to diary memories (legacy behavior).")
    parser.add_argument("--max-chars", type=int, default=2000,
                        help="Truncate memory text to this many chars before encoding (helps with long WhatsApp windows).")
    parser.add_argument("--include-rollups", action="store_true",
                        help="Include week/month rollup tokens in the cache.")
    parser.add_argument("--include-atoms", action="store_true",
                        help="Include sub-day memory atoms in the cache.")
    parser.add_argument("--include-reflections", action="store_true",
                        help="Include Level-5 reflection tokens in the cache.")
    return parser.parse_args()


def _encode(model: SentenceTransformer, texts: list[str], batch_size: int, label: str) -> torch.Tensor:
    print(f"encoding_{label}={len(texts)}")
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_tensor=True,
    ).to(torch.float32).cpu()


def main() -> int:
    args = parse_args()

    if args.diary_only:
        memories = parse_diary_memories()
    else:
        memories = parse_all_memories(
            include_rollups=args.include_rollups,
            include_atoms=args.include_atoms,
        )
        if args.include_reflections:
            reflections = load_reflections()
            if reflections:
                memories = sorted(memories + reflections, key=lambda m: m.ordinal)
                print(f"included_reflections={len(reflections)}")
    if not memories:
        raise SystemExit("No memories found.")

    diary_count = sum(1 for memory in memories if getattr(memory, "source_type", "diary") == "diary")
    whatsapp_count = len(memories) - diary_count
    print(f"loaded_memories={len(memories)} (diary={diary_count}, whatsapp={whatsapp_count})")

    # Optional generated-query cache (only if the training file exists locally).
    query_texts: list[str] = []
    if args.training and args.training.exists():
        rows = load_jsonl(args.training)
        query_texts = sorted({
            str(row.get("query") or "").strip()
            for row in rows
            if str(row.get("query") or "").strip()
        })
    else:
        print(f"note: training file {args.training} not found; building cache without query embeddings.")

    memory_texts = [compact_text(memory.text, args.max_chars) for memory in memories]
    print(f"loading_embedding_model={args.embedding_model}")
    model = SentenceTransformer(args.embedding_model, device=args.device)

    started = time.time()
    memory_embeddings = _encode(model, memory_texts, args.batch_size, "memories")
    query_embeddings = (
        _encode(model, query_texts, args.batch_size, "queries")
        if query_texts
        else torch.empty((0, int(memory_embeddings.shape[-1])), dtype=torch.float32)
    )

    payload = {
        "embedding_model": args.embedding_model,
        "embedding_dim": int(memory_embeddings.shape[-1]),
        "stats": normalizer(memories),
        "memory_ids": [memory.entry_id for memory in memories],
        "memory_dates": [memory.date for memory in memories],
        "memory_source_paths": [memory.source_path for memory in memories],
        "memory_source_types": [getattr(memory, "source_type", "diary") for memory in memories],
        "memory_moods": [memory.mood for memory in memories],
        "memory_previews": [compact_text(memory.text, 700) for memory in memories],
        "query_texts": query_texts,
        "memory_embeddings": memory_embeddings,
        "query_embeddings": query_embeddings,
        "diary_count": diary_count,
        "whatsapp_count": whatsapp_count,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "build_seconds": round(time.time() - started, 2),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"wrote={args.output}")
    print(f"embedding_dim={payload['embedding_dim']} memories={len(memories)} queries={len(query_texts)}"
          f" duration={payload['build_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
