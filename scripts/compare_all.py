#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sentence_transformers import SentenceTransformer

from honest_mira import (
    BM25Index,
    EvalExample,
    NeuralMIRARanker,
    SemanticLsaIndex,
    build_pair_features,
    calibrate_scores,
    choose_device,
    evaluate_score_map,
    generate_benchmark,
    minmax,
    neural_scores,
    parse_diary_memories,
    rank_metrics,
    run_split,
    split_examples,
    target_matrix,
    train_neural_mira,
)
from htema_core import (
    FEATURE_NAMES,
    DiaryMemory,
    anchor_ordinals,
    build_query_spec,
    dot,
    feature_vector,
    ordinal,
    tokenize,
)
from neural_htema_common import (
    MEMORY_FEATURE_NAMES,
    QUERY_FEATURE_NAMES,
    NeuralHTEMA,
    make_model,
    memory_feature_values,
    normalizer,
    pair_feature_values,
    query_feature_values,
)

SCALAR_PRIOR_WEIGHTS = {
    "bias": 0.0,
    "semantic": 0.75,
    "temporal": 1.55,
    "emotion": 1.15,
    "entity": 0.95,
    "hierarchy": 0.65,
    "continuity": 0.55,
    "importance": 0.20,
    "unresolved": 0.10,
    "contradiction": 0.15,
}


class NeuralQKVRetriever:
    def __init__(
        self,
        memories: list[DiaryMemory],
        embedder_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: torch.device = torch.device("cpu"),
        adapter_dim: int = 160,
        hidden_dim: int = 320,
        dropout: float = 0.08,
    ) -> None:
        self.device = device
        self.memories = memories
        self.stats = normalizer(memories)
        self.embedder = SentenceTransformer(embedder_name, device=str(device))

        print(f"  Encoding {len(memories)} memories with {embedder_name}...")
        memory_texts = [m.text for m in memories]
        self.memory_embeddings = (
            self.embedder.encode(
                memory_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_tensor=True,
            )
            .to(torch.float32)
            .cpu()
        )
        self.memory_features = torch.tensor(
            [memory_feature_values(m, self.stats) for m in memories],
            dtype=torch.float32,
        )
        self.model: NeuralHTEMA | None = None

    def encode_query(self, text: str) -> torch.Tensor:
        emb = self.embedder.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_tensor=True,
        )
        return emb[0].to(torch.float32).cpu()

    def build_model(self, embedding_dim: int = 384) -> NeuralHTEMA:
        config = {
            "architecture": "HTEMA neural Q/K/V adapter",
            "version": 1,
            "embedding_model": str(self.embedder),
            "embedding_dim": embedding_dim,
            "adapter_dim": 160,
            "hidden_dim": 320,
            "dropout": 0.08,
            "memory_feature_names": list(MEMORY_FEATURE_NAMES),
            "query_feature_names": list(QUERY_FEATURE_NAMES),
            "pair_feature_names": list(FEATURE_NAMES),
            "stats": self.stats,
        }
        model = make_model(config).to(self.device)
        return model

    def train(
        self,
        train_examples: list[EvalExample],
        *,
        epochs: int = 60,
        batch_size: int = 24,
        negatives: int = 32,
        lr: float = 2e-4,
        weight_decay: float = 0.01,
        seed: int = 13,
    ) -> None:
        torch.manual_seed(seed)
        rng = random.Random(seed)
        memories = self.memories
        memories_by_id = {m.entry_id: m for m in memories}
        memories_by_date = {m.date: m for m in memories}
        memory_id_to_index = {m.entry_id: i for i, m in enumerate(memories)}
        me = self.memory_embeddings
        mf = self.memory_features

        model = self.build_model()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Build train/dev split
        indices = list(range(len(train_examples)))
        rng.shuffle(indices)
        dev_size = max(8, int(len(indices) * 0.15))
        dev_idx = indices[:dev_size]
        fit_idx = indices[dev_size:] or indices

        best_state = None
        best_dev_mrr = -1.0
        patience = 0

        for epoch in range(1, epochs + 1):
            model.train()
            rng.shuffle(fit_idx)
            losses = []
            for offset in range(0, len(fit_idx), batch_size):
                batch_i = fit_idx[offset : offset + batch_size]
                batch_ex = [train_examples[i] for i in batch_i]
                batch = self._make_batch(
                    batch_ex,
                    memories_by_id,
                    memories_by_date,
                    memory_id_to_index,
                    me,
                    mf,
                    rng,
                    negatives,
                )
                if batch is None:
                    continue
                q_emb, q_feat, m_embs, m_feats, p_feat, labels = [
                    b.to(self.device) for b in batch
                ]
                optimizer.zero_grad(set_to_none=True)
                scores, _ = model(q_emb, q_feat, m_embs, m_feats, p_feat)
                loss = F.cross_entropy(scores, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

            if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
                dev_scores = self._score_all(train_examples, dev_idx, model, me, mf)
                dev_metrics = rank_metrics(
                    [train_examples[i] for i in dev_idx], memories, dev_scores
                )
                if dev_metrics["mrr"] > best_dev_mrr:
                    best_dev_mrr = dev_metrics["mrr"]
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    }
                    patience = 0
                else:
                    patience += 1
            if patience >= 12:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        self.model = model

    def _make_batch(
        self,
        examples: list[EvalExample],
        memories_by_id: dict[str, DiaryMemory],
        memories_by_date: dict[str, DiaryMemory],
        memory_id_to_index: dict[str, int],
        memory_embeddings: torch.Tensor,
        memory_features: torch.Tensor,
        rng: random.Random,
        negatives: int,
    ) -> tuple | None:
        q_embs, q_feats, m_embs, m_feats, p_feats = [], [], [], [], []
        for ex in examples:
            pos_ids = [pid for pid in ex.positive_ids if pid in memories_by_id]
            if not pos_ids:
                continue
            pos_id = rng.choice(pos_ids)
            pos_mem = memories_by_id[pos_id]
            neg_candidates = [
                m for m in self.memories if m.entry_id not in ex.positive_ids
            ]
            if len(neg_candidates) < negatives:
                continue
            negs = rng.sample(neg_candidates, negatives)
            candidates = [pos_mem] + list(negs)
            qs = build_query_spec(ex.query)
            anchors = anchor_ordinals(qs, self.memories)

            q_emb = self.encode_query(ex.query)
            q_feat = query_feature_values(qs, self.stats)
            c_indices = [memory_id_to_index[m.entry_id] for m in candidates]
            m_emb = memory_embeddings[c_indices]
            m_feat = memory_features[c_indices]
            p_feat = [pair_feature_values(m, qs, anchors) for m in candidates]

            q_embs.append(q_emb)
            q_feats.append(q_feat)
            m_embs.append(m_emb)
            m_feats.append(m_feat)
            p_feats.append(p_feat)

        if not q_embs:
            return None
        return (
            torch.stack(q_embs),
            torch.tensor(q_feats, dtype=torch.float32),
            torch.stack(m_embs),
            torch.stack(m_feats),
            torch.tensor(p_feats, dtype=torch.float32),
            torch.zeros(len(q_embs), dtype=torch.long),
        )

    @torch.no_grad()
    def _score_all(
        self,
        examples: list[EvalExample],
        indices: list[int],
        model: NeuralHTEMA,
        memory_embeddings: torch.Tensor,
        memory_features: torch.Tensor,
    ) -> np.ndarray:
        model.eval()
        all_scores = np.zeros((len(indices), len(self.memories)), dtype=np.float32)
        me = memory_embeddings.unsqueeze(0).to(self.device)
        mf = memory_features.unsqueeze(0).to(self.device)
        for row, idx in enumerate(indices):
            ex = examples[idx]
            qs = build_query_spec(ex.query)
            anchors = anchor_ordinals(qs, self.memories)
            q_emb = self.encode_query(ex.query).unsqueeze(0).to(self.device)
            q_feat = torch.tensor(
                [query_feature_values(qs, self.stats)],
                dtype=torch.float32,
                device=self.device,
            )
            p_feat = torch.tensor(
                [[pair_feature_values(m, qs, anchors) for m in self.memories]],
                dtype=torch.float32,
                device=self.device,
            )
            scores, _ = model(q_emb, q_feat, me, mf, p_feat)
            all_scores[row] = scores.squeeze(0).detach().cpu().numpy()
        return all_scores

    def score_all(self, examples: list[EvalExample]) -> np.ndarray:
        indices = list(range(len(examples)))
        return self._score_all(
            examples, indices, self.model, self.memory_embeddings, self.memory_features
        )


def run_qkv_split(
    examples: list[EvalExample],
    memories: list[DiaryMemory],
    split: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    train_examples, test_examples = split_examples(
        examples, split, args.seed, args.test_ratio
    )
    print(f"  QKV train={len(train_examples)} test={len(test_examples)}")

    retriever = NeuralQKVRetriever(memories, device=device)
    retriever.train(
        train_examples,
        epochs=args.qkv_epochs,
        batch_size=args.batch_size,
        lr=args.qkv_lr,
        seed=args.seed,
    )

    train_scores = retriever.score_all(train_examples)
    test_scores = retriever.score_all(test_examples)
    train_metrics = rank_metrics(train_examples, memories, train_scores)
    test_metrics = rank_metrics(test_examples, memories, test_scores)

    print(
        f"  QKV train: R@1={train_metrics['recall_at_1']:.3f} R@5={train_metrics['recall_at_5']:.3f} MRR={train_metrics['mrr']:.3f}"
    )
    print(
        f"  QKV test:  R@1={test_metrics['recall_at_1']:.3f} R@5={test_metrics['recall_at_5']:.3f} MRR={test_metrics['mrr']:.3f}"
    )

    return {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare all 5 retrieval methods on the honest MIRA benchmark."
    )
    parser.add_argument(
        "--split",
        default="all",
        choices=["all", "random", "month_holdout", "style_holdout"],
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--test-ratio", type=float, default=0.22)
    parser.add_argument("--epochs", type=int, default=90, help="Neural MIRA epochs")
    parser.add_argument(
        "--qkv-epochs", type=int, default=60, help="Neural Q/K/V HTEMA epochs"
    )
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument(
        "--lr", type=float, default=8e-4, help="Neural MIRA learning rate"
    )
    parser.add_argument(
        "--qkv-lr", type=float, default=2e-4, help="Neural Q/K/V HTEMA learning rate"
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.time()
    device = choose_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    memories = parse_diary_memories()
    examples = generate_benchmark(memories, args.seed)
    print(f"memories={len(memories)} benchmark_queries={len(examples)}")

    splits = (
        ["random", "month_holdout", "style_holdout"]
        if args.split == "all"
        else [args.split]
    )
    all_results = []

    for split in splits:
        print(f"\n{'=' * 60}")
        print(f"SPLIT: {split}")
        print(f"{'=' * 60}")

        train_ex, test_ex = split_examples(examples, split, args.seed, args.test_ratio)
        print(f"train={len(train_ex)} test={len(test_ex)}")

        # --- BM25 + Semantic baselines ---
        bm25 = BM25Index([m.tokens for m in memories])
        semantic = SemanticLsaIndex([m.text for m in memories])
        train_features, train_comp, _ = build_pair_features(
            train_ex, memories, bm25, semantic
        )
        test_features, test_comp, _ = build_pair_features(
            test_ex, memories, bm25, semantic
        )

        train_baselines = evaluate_score_map(train_ex, memories, train_comp)
        test_baselines = evaluate_score_map(test_ex, memories, test_comp)

        # --- Neural MIRA ---
        print(f"\n--- Neural MIRA ---")
        train_neural, test_neural, model_payload = train_neural_mira(
            train_ex,
            test_ex,
            memories,
            train_features,
            test_features,
            train_comp,
            test_comp,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            device=device,
        )
        train_mira = rank_metrics(train_ex, memories, train_neural)
        test_mira = rank_metrics(test_ex, memories, test_neural)

        # --- Neural Q/K/V HTEMA ---
        print(f"\n--- Neural Q/K/V HTEMA ---")
        qkv_result = run_qkv_split(examples, memories, split, args, device)

        # --- Collect all metrics ---
        row = {"split": split, "train_count": len(train_ex), "test_count": len(test_ex)}
        for name, metrics in test_baselines.items():
            row[f"test_{name}_r1"] = metrics["recall_at_1"]
            row[f"test_{name}_r5"] = metrics["recall_at_5"]
            row[f"test_{name}_mrr"] = metrics["mrr"]
        row["test_neural_mira_r1"] = test_mira["recall_at_1"]
        row["test_neural_mira_r5"] = test_mira["recall_at_5"]
        row["test_neural_mira_mrr"] = test_mira["mrr"]
        row["test_neural_qkv_r1"] = qkv_result["test_metrics"]["recall_at_1"]
        row["test_neural_qkv_r5"] = qkv_result["test_metrics"]["recall_at_5"]
        row["test_neural_qkv_mrr"] = qkv_result["test_metrics"]["mrr"]
        all_results.append(row)

        # Print per-split comparison
        print(f"\n{'=' * 60}")
        print(f"RESULTS - {split}")
        print(f"{'=' * 60}")
        print(f"{'Method':20s} {'R@1':>8s} {'R@5':>8s} {'MRR':>8s}")
        print("-" * 48)
        methods = [
            ("BM25", test_baselines["bm25"]),
            ("Semantic Embed", test_baselines["semantic_embed"]),
            ("Scalar HTEMA", test_baselines["scalar_htema"]),
            ("Neural MIRA", test_mira),
            ("Neural Q/K/V", qkv_result["test_metrics"]),
        ]
        for name, m in methods:
            print(
                f"{name:20s} {m['recall_at_1']:>7.3f}  {m['recall_at_5']:>7.3f}  {m['mrr']:>7.3f}"
            )

    # Print final summary
    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY ACROSS ALL SPLITS")
    print(f"{'=' * 60}")
    header = f"{'Method':20s}"
    for r in all_results:
        header += f" | {r['split'][:10]:>10s}"
        header += f" {r['split'][:10]:>10s}"
        header += f" {r['split'][:10]:>10s}"
    print(header)

    for method_name in [
        "BM25",
        "Semantic Embed",
        "Scalar HTEMA",
        "Neural MIRA",
        "Neural Q/K/V",
    ]:
        key_prefix = method_name.lower().replace(" ", "_").replace("/", "_")
        if method_name == "Neural Q/K/V":
            key_prefix = "neural_qkv"
        line = f"{method_name:20s}"
        for r in all_results:
            line += f" | R@1={r[f'test_{key_prefix}_r1']:.3f} R@5={r[f'test_{key_prefix}_r5']:.3f} MRR={r[f'test_{key_prefix}_mrr']:.3f}"
        print(line)

    print(f"\nTotal time: {time.time() - start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
