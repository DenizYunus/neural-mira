#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from htema_core import compact_text, load_jsonl, parse_diary_memories
from neural_htema_common import DEFAULT_EMBEDDINGS, DEFAULT_TRAINING, normalizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the pretrained embedding cache for neural HTEMA.")
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--output", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model used for diary memories and generated queries.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None, help="Optional SentenceTransformer device, for example cpu, cuda, or mps.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.training)
    if not rows:
        raise SystemExit(f"No training rows found at {args.training}")

    memories = parse_diary_memories()
    if not memories:
        raise SystemExit("No diary memories found.")

    query_texts = sorted({str(row.get("query") or "").strip() for row in rows if str(row.get("query") or "").strip()})
    memory_texts = [memory.text for memory in memories]
    print(f"loading_embedding_model={args.embedding_model}")
    model = SentenceTransformer(args.embedding_model, device=args.device)

    print(f"encoding_memories={len(memory_texts)}")
    memory_embeddings = model.encode(
        memory_texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_tensor=True,
    ).to(torch.float32).cpu()

    print(f"encoding_queries={len(query_texts)}")
    query_embeddings = model.encode(
        query_texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_tensor=True,
    ).to(torch.float32).cpu()

    payload = {
        "embedding_model": args.embedding_model,
        "embedding_dim": int(memory_embeddings.shape[-1]),
        "stats": normalizer(memories),
        "memory_ids": [memory.entry_id for memory in memories],
        "memory_dates": [memory.date for memory in memories],
        "memory_source_paths": [memory.source_path for memory in memories],
        "memory_moods": [memory.mood for memory in memories],
        "memory_previews": [compact_text(memory.text, 700) for memory in memories],
        "query_texts": query_texts,
        "memory_embeddings": memory_embeddings,
        "query_embeddings": query_embeddings,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"wrote={args.output}")
    print(f"embedding_dim={payload['embedding_dim']} memories={len(memories)} queries={len(query_texts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
