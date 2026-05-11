#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from torch import nn
from torch.nn import functional as F

from htema_core import (
    FEATURE_NAMES,
    DIARY_FEATURES,
    DiaryMemory,
    anchor_ordinals,
    build_query_spec,
    dot,
    feature_vector,
    parse_diary_memories,
    tokenize,
)


LAB_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = LAB_ROOT / "data" / "neural_mira_model.pt"
DEFAULT_METRICS = LAB_ROOT / "data" / "neural_mira_metrics.json"
DEFAULT_REPORT = LAB_ROOT / "docs" / "honest_evaluation_report.md"

MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}

GENERIC_TERMS = {
    "mood",
    "icons",
    "step",
    "count",
    "dailybean",
    "complete",
    "diary",
    "entry",
    "today",
    "tomorrow",
    "yesterday",
    "n/a",
    "n",
    "and",
    "ile",
    "bir",
    "çok",
    "cok",
    "gibi",
    "ama",
    "sonra",
    "daha",
    "olan",
    "için",
    "icin",
    "şey",
    "sey",
}

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


@dataclass(frozen=True)
class EvalExample:
    query: str
    positive_ids: tuple[str, ...]
    intent: str
    style: str
    target_month: str
    target_date: str | None = None
    note: str = ""


class BM25Index:
    def __init__(self, docs: list[tuple[str, ...]], k1: float = 1.45, b: float = 0.72) -> None:
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.lengths = np.array([len(doc) for doc in docs], dtype=np.float32)
        self.avg_len = float(np.mean(self.lengths)) if len(self.lengths) else 1.0
        self.term_freqs = [Counter(doc) for doc in docs]
        doc_freq = Counter()
        for doc in docs:
            doc_freq.update(set(doc))
        total = len(docs)
        self.idf = {
            term: math.log(1 + (total - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def scores(self, query: str) -> np.ndarray:
        query_terms = tokenize(query)
        values = np.zeros(len(self.docs), dtype=np.float32)
        if not query_terms:
            return values
        for index, freqs in enumerate(self.term_freqs):
            score = 0.0
            doc_len = self.lengths[index] or 1.0
            norm = self.k1 * (1 - self.b + self.b * doc_len / self.avg_len)
            for term in query_terms:
                tf = freqs.get(term, 0)
                if not tf:
                    continue
                score += self.idf.get(term, 0.0) * (tf * (self.k1 + 1)) / (tf + norm)
            values[index] = score
        return values


class SemanticLsaIndex:
    def __init__(self, texts: list[str], components: int = 96) -> None:
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents=None,
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b[\w'-]+\b",
            max_features=9000,
        )
        sparse = self.vectorizer.fit_transform(texts)
        max_components = min(max(2, sparse.shape[0] - 1), max(2, sparse.shape[1] - 1), components)
        self.svd = TruncatedSVD(n_components=max_components, random_state=13)
        dense = self.svd.fit_transform(sparse)
        self.embeddings = normalize(dense)

    def scores(self, query: str) -> np.ndarray:
        sparse = self.vectorizer.transform([query])
        dense = normalize(self.svd.transform(sparse))
        return np.asarray(self.embeddings @ dense.T, dtype=np.float32).reshape(-1)


class NeuralMIRARanker(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 96, dropout: float = 0.08) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.residual = nn.Linear(input_dim, 1, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return (self.net(features) + 0.35 * self.residual(features)).squeeze(-1)


def month_key(memory: DiaryMemory) -> str:
    return f"{memory.year}-{memory.month:02d}"


def month_label(memory: DiaryMemory) -> str:
    return f"{MONTH_NAMES[memory.month]} {memory.year}"


def mood_word(mood: int | None) -> str:
    if mood is None:
        return "unclear"
    if mood >= 5:
        return "happy"
    if mood == 4:
        return "good"
    if mood == 3:
        return "mixed"
    return "sad"


def phase_word(day: int) -> str:
    if day <= 10:
        return "early"
    if day <= 20:
        return "middle"
    return "late"


def feature_label(memory: DiaryMemory) -> str | None:
    candidates = [(name, value) for name, value in memory.diary_features.items() if value >= 0.2]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[1])[0]


def clean_query_term(term: str) -> str | None:
    term = term.strip().lower()
    if len(term) < 3:
        return None
    if term in GENERIC_TERMS:
        return None
    if term.isdigit():
        return None
    if any(char.isdigit() for char in term) and len(term) < 5:
        return None
    return term


def rare_terms_for_memories(memories: list[DiaryMemory]) -> dict[str, list[str]]:
    doc_freq: Counter[str] = Counter()
    for memory in memories:
        doc_freq.update({term for term in memory.tokens if clean_query_term(term)})

    rare_by_id: dict[str, list[str]] = {}
    for memory in memories:
        candidates = []
        for term in set(memory.tokens):
            clean = clean_query_term(term)
            if not clean:
                continue
            freq = doc_freq[clean]
            if freq <= max(4, len(memories) * 0.035):
                candidates.append((freq, -len(clean), clean))
        candidates.sort()
        rare_by_id[memory.entry_id] = [term for _, _, term in candidates[:5]]
    return rare_by_id


def add_example(
    examples: list[EvalExample],
    seen: set[tuple[str, tuple[str, ...]]],
    *,
    query: str,
    positive_ids: list[str] | tuple[str, ...],
    intent: str,
    style: str,
    target_month: str,
    target_date: str | None = None,
    note: str = "",
) -> None:
    positives = tuple(sorted(set(positive_ids)))
    if not positives:
        return
    key = (query.lower(), positives)
    if key in seen:
        return
    seen.add(key)
    examples.append(
        EvalExample(
            query=query,
            positive_ids=positives,
            intent=intent,
            style=style,
            target_month=target_month,
            target_date=target_date,
            note=note,
        )
    )


def generate_benchmark(memories: list[DiaryMemory], seed: int = 13) -> list[EvalExample]:
    rng = random.Random(seed)
    by_month: dict[str, list[DiaryMemory]] = defaultdict(list)
    for memory in memories:
        by_month[month_key(memory)].append(memory)

    rare_by_id = rare_terms_for_memories(memories)
    examples: list[EvalExample] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()

    for key, month_memories in sorted(by_month.items()):
        label = month_label(month_memories[0])
        mooded = [memory for memory in month_memories if memory.mood is not None]
        if mooded:
            max_mood = max(memory.mood or 0 for memory in mooded)
            min_mood = min(memory.mood or 5 for memory in mooded)
            if max_mood >= 4:
                positives = [memory.entry_id for memory in mooded if memory.mood == max_mood]
                add_example(
                    examples,
                    seen,
                    query=f"happiest diary days in {label}",
                    positive_ids=positives,
                    intent="emotion_temporal_recall",
                    style="emotion_month",
                    target_month=key,
                    note="group query: any top-mood day in the month is relevant",
                )
            if min_mood <= 3:
                positives = [memory.entry_id for memory in mooded if memory.mood == min_mood]
                add_example(
                    examples,
                    seen,
                    query=f"lowest mood diary days in {label}",
                    positive_ids=positives,
                    intent="emotion_temporal_recall",
                    style="emotion_month",
                    target_month=key,
                    note="group query: any lowest-mood day in the month is relevant",
                )

        for feature in DIARY_FEATURES:
            positives = [memory.entry_id for memory in month_memories if memory.diary_features.get(feature, 0.0) >= 0.2]
            if positives:
                add_example(
                    examples,
                    seen,
                    query=f"{feature} memories in {label}",
                    positive_ids=positives,
                    intent="feature_temporal_recall",
                    style="feature_month",
                    target_month=key,
                )

    for memory in memories:
        key = month_key(memory)
        label = month_label(memory)
        terms = rare_by_id.get(memory.entry_id, [])
        feature = feature_label(memory)
        icon_terms = [clean_query_term(icon.replace("-", " ")) for icon in memory.icons[:4]]
        icon_terms = [item for item in icon_terms if item]

        if memory.mood is not None:
            signal = feature or (icon_terms[0] if icon_terms else None)
            if signal:
                add_example(
                    examples,
                    seen,
                    query=f"which {mood_word(memory.mood)} {signal} day happened in {label}",
                    positive_ids=[memory.entry_id],
                    intent="emotion_feature_recall",
                    style="emotion_feature",
                    target_month=key,
                    target_date=memory.date,
                )
            if icon_terms:
                add_example(
                    examples,
                    seen,
                    query=f"find the {label} entry with mood {memory.mood}/5 and {icon_terms[0]} energy",
                    positive_ids=[memory.entry_id],
                    intent="diary_signal_recall",
                    style="mood_signal",
                    target_month=key,
                    target_date=memory.date,
                )

        if len(terms) >= 2:
            phrase = " ".join(terms[:2])
            add_example(
                examples,
                seen,
                query=f"when did I write about {phrase}",
                positive_ids=[memory.entry_id],
                intent="semantic_recall",
                style="rare_terms",
                target_month=key,
                target_date=memory.date,
            )
            neighbor_ids = [
                item.entry_id
                for item in memories
                if abs(item.ordinal - memory.ordinal) <= 1
            ]
            add_example(
                examples,
                seen,
                query=f"what was happening around the {phrase} moment",
                positive_ids=neighbor_ids,
                intent="continuity_recall",
                style="around_rare_terms",
                target_month=key,
                target_date=memory.date,
                note="local window query: target day and adjacent diary days are relevant",
            )

        if memory.mood is not None and (feature or terms):
            detail = feature or terms[0]
            day = int(memory.date[-2:])
            add_example(
                examples,
                seen,
                query=f"the {phase_word(day)} {label} {mood_word(memory.mood)} {detail} memory",
                positive_ids=[memory.entry_id],
                intent="temporal_emotional_detail",
                style="relative_temporal",
                target_month=key,
                target_date=memory.date,
            )

    rng.shuffle(examples)
    return examples


def split_examples(
    examples: list[EvalExample],
    split: str,
    seed: int,
    test_ratio: float,
) -> tuple[list[EvalExample], list[EvalExample]]:
    rng = random.Random(seed)
    if split == "random":
        shuffled = examples[:]
        rng.shuffle(shuffled)
        test_size = max(1, int(len(shuffled) * test_ratio))
        return shuffled[test_size:], shuffled[:test_size]

    if split in {"month_holdout", "date_holdout"}:
        months = sorted({example.target_month for example in examples})
        rng.shuffle(months)
        held = set(months[: max(1, int(len(months) * test_ratio))])
        train = [example for example in examples if example.target_month not in held]
        test = [example for example in examples if example.target_month in held]
        return train, test

    if split == "style_holdout":
        held_styles = {"emotion_month", "mood_signal", "relative_temporal"}
        train = [example for example in examples if example.style not in held_styles]
        test = [example for example in examples if example.style in held_styles]
        return train, test

    raise ValueError(f"Unknown split: {split}")


def reciprocal_rank(ranked_ids: list[str], positive_ids: set[str]) -> float:
    for index, entry_id in enumerate(ranked_ids, start=1):
        if entry_id in positive_ids:
            return 1.0 / index
    return 0.0


def rank_metrics(examples: list[EvalExample], memories: list[DiaryMemory], score_matrix: np.ndarray) -> dict[str, float]:
    if not examples:
        return {"queries": 0, "recall_at_1": 0.0, "recall_at_5": 0.0, "mrr": 0.0}
    memory_ids = [memory.entry_id for memory in memories]
    recall_1 = 0
    recall_5 = 0
    reciprocal = 0.0
    by_style: dict[str, list[float]] = defaultdict(list)

    for row_index, example in enumerate(examples):
        positive_ids = set(example.positive_ids)
        ranked_indices = np.argsort(-score_matrix[row_index]).tolist()
        ranked_ids = [memory_ids[index] for index in ranked_indices]
        top_5 = set(ranked_ids[:5])
        rr = reciprocal_rank(ranked_ids, positive_ids)
        recall_1 += int(ranked_ids[0] in positive_ids)
        recall_5 += int(bool(top_5.intersection(positive_ids)))
        reciprocal += rr
        by_style[example.style].append(rr)

    payload: dict[str, float] = {
        "queries": float(len(examples)),
        "recall_at_1": recall_1 / len(examples),
        "recall_at_5": recall_5 / len(examples),
        "mrr": reciprocal / len(examples),
    }
    for style, values in sorted(by_style.items()):
        payload[f"mrr_style/{style}"] = float(sum(values) / len(values))
    return payload


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    low = float(values.min())
    high = float(values.max())
    if high <= low:
        return np.zeros_like(values, dtype=np.float32)
    return (values - low) / (high - low)


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    std = float(values.std())
    if std < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return (values - float(values.mean())) / std


def build_pair_features(
    examples: list[EvalExample],
    memories: list[DiaryMemory],
    bm25: BM25Index,
    semantic: SemanticLsaIndex,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str]]:
    all_rows = []
    component_scores = {
        "bm25": [],
        "semantic_embed": [],
        "scalar_htema": [],
    }
    feature_names = [
        *FEATURE_NAMES,
        "bm25",
        "bm25_z",
        "bm25_rank",
        "semantic_embed",
        "semantic_z",
        "semantic_rank",
        "scalar_htema",
        "semantic_x_temporal",
        "bm25_x_temporal",
        "emotion_x_temporal",
        "entity_x_temporal",
        "semantic_x_entity",
    ]

    for example in examples:
        query = build_query_spec(example.query)
        anchors = anchor_ordinals(query, memories)
        bm25_scores = bm25.scores(example.query)
        semantic_scores = semantic.scores(example.query)
        bm25_norm = minmax(bm25_scores)
        semantic_norm = minmax(semantic_scores)
        bm25_z = zscore(bm25_scores)
        semantic_z = zscore(semantic_scores)
        bm25_order = np.argsort(-bm25_scores)
        semantic_order = np.argsort(-semantic_scores)
        bm25_rank = np.zeros(len(memories), dtype=np.float32)
        semantic_rank = np.zeros(len(memories), dtype=np.float32)
        for rank, index in enumerate(bm25_order, start=1):
            bm25_rank[index] = 1.0 / math.log2(rank + 1)
        for rank, index in enumerate(semantic_order, start=1):
            semantic_rank[index] = 1.0 / math.log2(rank + 1)

        rows = []
        scalar_scores = []
        for index, memory in enumerate(memories):
            base = feature_vector(memory, query, anchors)
            scalar_score = dot(SCALAR_PRIOR_WEIGHTS, base)
            scalar_scores.append(scalar_score)
            semantic_value = float(semantic_norm[index])
            bm25_value = float(bm25_norm[index])
            temporal = float(base["temporal"])
            emotion = float(base["emotion"])
            entity = float(base["entity"])
            row = [
                *[float(base[name]) for name in FEATURE_NAMES],
                bm25_value,
                float(bm25_z[index]),
                float(bm25_rank[index]),
                semantic_value,
                float(semantic_z[index]),
                float(semantic_rank[index]),
                float(scalar_score),
                semantic_value * temporal,
                bm25_value * temporal,
                emotion * temporal,
                entity * temporal,
                semantic_value * entity,
            ]
            rows.append(row)

        all_rows.append(rows)
        component_scores["bm25"].append(bm25_scores)
        component_scores["semantic_embed"].append(semantic_scores)
        component_scores["scalar_htema"].append(np.asarray(scalar_scores, dtype=np.float32))

    features = np.asarray(all_rows, dtype=np.float32)
    components = {name: np.asarray(values, dtype=np.float32) for name, values in component_scores.items()}
    return features, components, feature_names


def target_matrix(examples: list[EvalExample], memories: list[DiaryMemory]) -> np.ndarray:
    memory_index = {memory.entry_id: index for index, memory in enumerate(memories)}
    targets = np.zeros((len(examples), len(memories)), dtype=np.float32)
    for row_index, example in enumerate(examples):
        indices = [memory_index[entry_id] for entry_id in example.positive_ids if entry_id in memory_index]
        if not indices:
            continue
        value = 1.0 / len(indices)
        for index in indices:
            targets[row_index, index] = value
    return targets


@torch.no_grad()
def neural_scores(
    model: NeuralMIRARanker,
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    model.eval()
    outputs = []
    for offset in range(0, len(features), batch_size):
        batch = (features[offset : offset + batch_size] - mean) / std
        batch_tensor = torch.tensor(batch, dtype=torch.float32, device=device)
        outputs.append(model(batch_tensor).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, features.shape[1]), dtype=np.float32)


def evaluate_score_map(
    examples: list[EvalExample],
    memories: list[DiaryMemory],
    score_map: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    return {name: rank_metrics(examples, memories, scores) for name, scores in score_map.items()}


def train_neural_mira(
    train_examples: list[EvalExample],
    test_examples: list[EvalExample],
    memories: list[DiaryMemory],
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_components: dict[str, np.ndarray],
    test_components: dict[str, np.ndarray],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    train_indices = list(range(len(train_examples)))
    rng.shuffle(train_indices)
    dev_size = max(16, int(len(train_indices) * 0.15)) if len(train_indices) >= 80 else max(1, len(train_indices) // 5)
    dev_indices = train_indices[:dev_size]
    fit_indices = train_indices[dev_size:] or train_indices

    mean = train_features[fit_indices].reshape(-1, train_features.shape[-1]).mean(axis=0, keepdims=True)
    std = train_features[fit_indices].reshape(-1, train_features.shape[-1]).std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    mean = mean.astype(np.float32)

    train_targets = target_matrix(train_examples, memories)
    model = NeuralMIRARanker(train_features.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.015)

    best_state = None
    best_dev_mrr = -1.0
    history = []
    patience = 0

    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(fit_indices)
        losses = []
        for offset in range(0, len(fit_indices), batch_size):
            batch_indices = fit_indices[offset : offset + batch_size]
            batch_x = (train_features[batch_indices] - mean) / std
            batch_y = train_targets[batch_indices]
            x = torch.tensor(batch_x, dtype=torch.float32, device=device)
            y = torch.tensor(batch_y, dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            scores = model(x)
            log_probs = F.log_softmax(scores, dim=-1)
            loss = -(y * log_probs).sum(dim=-1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        should_eval = epoch == 1 or epoch % 5 == 0 or epoch == epochs
        record = {"epoch": epoch, "loss": float(sum(losses) / max(len(losses), 1))}
        if should_eval:
            dev_scores = neural_scores(model, train_features[dev_indices], mean, std, device)
            dev_metrics = rank_metrics([train_examples[index] for index in dev_indices], memories, dev_scores)
            record["dev_mrr"] = dev_metrics["mrr"]
            if dev_metrics["mrr"] > best_dev_mrr:
                best_dev_mrr = dev_metrics["mrr"]
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
        history.append(record)
        if patience >= 14:
            break

    if best_state:
        model.load_state_dict(best_state)

    train_neural = neural_scores(model, train_features, mean, std, device)
    test_neural = neural_scores(model, test_features, mean, std, device)

    calibration, train_calibrated, test_calibrated = calibrate_scores(
        [train_examples[index] for index in dev_indices],
        train_examples,
        test_examples,
        memories,
        train_neural[dev_indices],
        train_neural,
        test_neural,
        {name: values[dev_indices] for name, values in train_components.items()},
        train_components,
        test_components,
    )

    payload = {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "mean": mean,
        "std": std,
        "calibration": calibration,
        "history": history,
        "best_dev_mrr": best_dev_mrr,
    }
    return train_calibrated, test_calibrated, payload


def normalize_component(values: np.ndarray) -> np.ndarray:
    rows = []
    for row in values:
        rows.append(minmax(row))
    return np.asarray(rows, dtype=np.float32)


def calibrate_scores(
    calibration_examples: list[EvalExample],
    train_examples: list[EvalExample],
    test_examples: list[EvalExample],
    memories: list[DiaryMemory],
    calibration_neural: np.ndarray,
    train_neural: np.ndarray,
    test_neural: np.ndarray,
    calibration_components: dict[str, np.ndarray],
    train_components: dict[str, np.ndarray],
    test_components: dict[str, np.ndarray],
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    components_calibration = {
        "neural": normalize_component(calibration_neural),
        "bm25": normalize_component(calibration_components["bm25"]),
        "semantic": normalize_component(calibration_components["semantic_embed"]),
        "scalar": normalize_component(calibration_components["scalar_htema"]),
    }
    components_train = {
        "neural": normalize_component(train_neural),
        "bm25": normalize_component(train_components["bm25"]),
        "semantic": normalize_component(train_components["semantic_embed"]),
        "scalar": normalize_component(train_components["scalar_htema"]),
    }
    components_test = {
        "neural": normalize_component(test_neural),
        "bm25": normalize_component(test_components["bm25"]),
        "semantic": normalize_component(test_components["semantic_embed"]),
        "scalar": normalize_component(test_components["scalar_htema"]),
    }

    best_weights = {"neural": 1.0, "bm25": 0.0, "semantic": 0.0, "scalar": 0.0}
    best_mrr = -1.0
    grid = [0.0, 0.15, 0.3, 0.5, 0.75, 1.0, 1.35]
    for neural_w in [0.6, 0.85, 1.0, 1.25, 1.6, 2.0]:
        for bm25_w in grid:
            for semantic_w in [0.0, 0.15, 0.3, 0.5]:
                for scalar_w in grid:
                    scores = (
                        neural_w * components_calibration["neural"]
                        + bm25_w * components_calibration["bm25"]
                        + semantic_w * components_calibration["semantic"]
                        + scalar_w * components_calibration["scalar"]
                    )
                    metrics = rank_metrics(calibration_examples, memories, scores)
                    if metrics["mrr"] > best_mrr:
                        best_mrr = metrics["mrr"]
                        best_weights = {
                            "neural": neural_w,
                            "bm25": bm25_w,
                            "semantic": semantic_w,
                            "scalar": scalar_w,
                        }

    train_scores = sum(best_weights[name] * components_train[name] for name in best_weights)
    test_scores = sum(best_weights[name] * components_test[name] for name in best_weights)
    best_weights["dev_calibration_mrr"] = best_mrr
    return best_weights, train_scores, test_scores


def run_split(
    memories: list[DiaryMemory],
    examples: list[EvalExample],
    split: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    train_examples, test_examples = split_examples(examples, split, args.seed, args.test_ratio)
    if not train_examples or not test_examples:
        raise SystemExit(f"Split {split} produced train={len(train_examples)} test={len(test_examples)}")

    bm25 = BM25Index([memory.tokens for memory in memories])
    semantic = SemanticLsaIndex([memory.text for memory in memories])
    train_features, train_components, feature_names = build_pair_features(train_examples, memories, bm25, semantic)
    test_features, test_components, _ = build_pair_features(test_examples, memories, bm25, semantic)

    train_baselines = evaluate_score_map(train_examples, memories, train_components)
    test_baselines = evaluate_score_map(test_examples, memories, test_components)
    train_neural, test_neural, model_payload = train_neural_mira(
        train_examples,
        test_examples,
        memories,
        train_features,
        test_features,
        train_components,
        test_components,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=device,
    )
    train_metrics = {
        **train_baselines,
        "neural_mira": rank_metrics(train_examples, memories, train_neural),
    }
    test_metrics = {
        **test_baselines,
        "neural_mira": rank_metrics(test_examples, memories, test_neural),
    }

    return {
        "split": split,
        "train_examples": len(train_examples),
        "test_examples": len(test_examples),
        "feature_names": feature_names,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "model": model_payload,
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Honest MIRA / Neural HTEMA Evaluation",
        "",
        "This report is generated by `scripts/honest_mira.py`.",
        "",
        "Rules:",
        "- Query text is the only input at retrieval time.",
        "- `positive_window` is not used.",
        "- Positive target dates are used only as labels for training/evaluation.",
        "- BM25 and dense semantic embedding baselines run over the same diary day tokens.",
        "",
        f"Generated at: {payload['created_at']}",
        f"Diary day tokens: {payload['memory_count']}",
        f"Deterministic benchmark queries: {payload['example_count']}",
        "",
    ]

    for split_result in payload["splits"]:
        lines.append(f"## {split_result['split']}")
        lines.append("")
        lines.append(f"Train examples: {split_result['train_examples']}; test examples: {split_result['test_examples']}")
        lines.append("")
        lines.append("| Model | R@1 | R@5 | MRR |")
        lines.append("| --- | ---: | ---: | ---: |")
        for name, metrics in split_result["test_metrics"].items():
            lines.append(
                "| {name} | {r1:.3f} | {r5:.3f} | {mrr:.3f} |".format(
                    name=name,
                    r1=metrics["recall_at_1"],
                    r5=metrics["recall_at_5"],
                    mrr=metrics["mrr"],
                )
            )
        lines.append("")
        calibration = split_result["model"]["calibration"]
        lines.append(
            "Calibration: "
            + ", ".join(f"{key}={value:.2f}" for key, value in calibration.items() if isinstance(value, float))
        )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-free BM25/Semantic/Neural MIRA evaluation.")
    parser.add_argument("--split", default="all", choices=["all", "random", "month_holdout", "date_holdout", "style_holdout"])
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--test-ratio", type=float, default=0.22)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-save-model", action="store_true")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> int:
    args = parse_args()
    start = time.time()
    device = choose_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    memories = parse_diary_memories()
    if not memories:
        raise SystemExit("No diary memories found.")
    examples = generate_benchmark(memories, args.seed)
    if not examples:
        raise SystemExit("No benchmark examples could be generated.")

    splits = ["random", "month_holdout", "style_holdout"] if args.split == "all" else [args.split]
    split_results = []
    saved_model_payload = None
    for split in splits:
        print(f"\n=== split={split} device={device} ===")
        result = run_split(memories, examples, split, args, device)
        split_results.append(result)
        saved_model_payload = result if split == "random" else saved_model_payload
        for name, metrics in result["test_metrics"].items():
            print(
                "{name:16s} R@1={r1:.3f} R@5={r5:.3f} MRR={mrr:.3f}".format(
                    name=name,
                    r1=metrics["recall_at_1"],
                    r5=metrics["recall_at_5"],
                    mrr=metrics["mrr"],
                )
            )

    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_seconds": round(time.time() - start, 2),
        "memory_count": len(memories),
        "example_count": len(examples),
        "seed": args.seed,
        "splits": [
            {
                key: value
                for key, value in result.items()
                if key != "model"
            }
            | {
                "model": {
                    key: value
                    for key, value in result["model"].items()
                    if key not in {"state_dict", "mean", "std"}
                }
            }
            for result in split_results
        ],
    }

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(args.report, payload)

    if not args.no_save_model and saved_model_payload:
        model_payload = saved_model_payload["model"]
        args.model.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "created_at": payload["created_at"],
                "split": saved_model_payload["split"],
                "feature_names": saved_model_payload["feature_names"],
                "state_dict": model_payload["state_dict"],
                "mean": model_payload["mean"],
                "std": model_payload["std"],
                "calibration": model_payload["calibration"],
                "metrics": saved_model_payload["test_metrics"],
            },
            args.model,
        )

    print(f"\nwrote_metrics={args.metrics}")
    print(f"wrote_report={args.report}")
    if not args.no_save_model:
        print(f"wrote_model={args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
