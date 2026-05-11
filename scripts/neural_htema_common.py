from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from htema_core import (
    DIARY_FEATURES,
    FEATURE_NAMES,
    DiaryMemory,
    QuerySpec,
    build_query_spec,
    feature_vector,
    ordinal,
)


LAB_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING = LAB_ROOT / "data" / "generated_queries.jsonl"
DEFAULT_EMBEDDINGS = LAB_ROOT / "data" / "neural_embeddings.pt"
DEFAULT_MODEL = LAB_ROOT / "data" / "neural_htema_model.pt"
DEFAULT_METRICS = LAB_ROOT / "data" / "neural_htema_metrics.json"
DEFAULT_SCALAR_MODEL = LAB_ROOT / "data" / "htema_model.json"

DEFAULT_PRIOR_WEIGHTS = {
    "bias": 0.0,
    "semantic": 1.0,
    "temporal": 1.15,
    "emotion": 1.0,
    "entity": 0.9,
    "hierarchy": 0.7,
    "continuity": 0.8,
    "importance": 0.35,
    "unresolved": 0.2,
    "contradiction": 0.25,
}

DIARY_FEATURE_NAMES = tuple(DIARY_FEATURES.keys())
MEMORY_FEATURE_NAMES = (
    "year_norm",
    "month_sin",
    "month_cos",
    "ordinal_norm",
    "mood_norm",
    "valence",
    "arousal",
    "importance",
    "unresolved",
    "icon_count_norm",
    *DIARY_FEATURE_NAMES,
)
QUERY_FEATURE_NAMES = (
    "has_window",
    "window_start_norm",
    "window_end_norm",
    "window_mid_norm",
    "window_span_norm",
    "emotion_present",
    "target_mood_norm",
    "target_valence",
    "target_arousal",
    "around",
    "contradiction",
    *DIARY_FEATURE_NAMES,
)


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalizer(memories: list[DiaryMemory]) -> dict[str, float]:
    ordinals = [memory.ordinal for memory in memories]
    years = [memory.year for memory in memories]
    return {
        "min_ordinal": float(min(ordinals)),
        "max_ordinal": float(max(ordinals)),
        "min_year": float(min(years)),
        "max_year": float(max(years)),
    }


def normalized(value: float, minimum: float, maximum: float) -> float:
    if maximum <= minimum:
        return 0.0
    return max(0.0, min(1.0, (value - minimum) / (maximum - minimum)))


def memory_feature_values(memory: DiaryMemory, stats: dict[str, float]) -> list[float]:
    mood = memory.mood if memory.mood is not None else 3
    values = [
        normalized(memory.year, stats["min_year"], stats["max_year"]),
        math.sin(2 * math.pi * memory.month / 12),
        math.cos(2 * math.pi * memory.month / 12),
        normalized(memory.ordinal, stats["min_ordinal"], stats["max_ordinal"]),
        (mood - 1) / 4,
        float(memory.emotion.get("valence") or 0.0),
        float(memory.emotion.get("arousal") or 0.0),
        float(memory.importance),
        float(memory.unresolved),
        min(len(memory.icons), 16) / 16,
    ]
    values.extend(float(memory.diary_features.get(name, 0.0)) for name in DIARY_FEATURE_NAMES)
    return values


def query_feature_values(query: QuerySpec, stats: dict[str, float]) -> list[float]:
    has_window = 1.0 if query.time_window else 0.0
    if query.time_window:
        start, end = (ordinal(value) for value in query.time_window)
        mid = (start + end) / 2
        span = max(0, end - start)
        full_span = max(1.0, stats["max_ordinal"] - stats["min_ordinal"])
        start_norm = normalized(start, stats["min_ordinal"], stats["max_ordinal"])
        end_norm = normalized(end, stats["min_ordinal"], stats["max_ordinal"])
        mid_norm = normalized(mid, stats["min_ordinal"], stats["max_ordinal"])
        span_norm = min(1.0, span / full_span)
    else:
        start_norm = end_norm = mid_norm = span_norm = 0.0

    emotion = query.emotion or {}
    mood = emotion.get("mood")
    values = [
        has_window,
        start_norm,
        end_norm,
        mid_norm,
        span_norm,
        1.0 if query.emotion else 0.0,
        ((int(mood) - 1) / 4) if mood else 0.5,
        float(emotion.get("valence") or 0.0),
        float(emotion.get("arousal") or 0.0),
        1.0 if query.around else 0.0,
        1.0 if query.contradiction else 0.0,
    ]
    values.extend(float(query.diary_features.get(name, 0.0)) for name in DIARY_FEATURE_NAMES)
    return values


def pair_feature_values(memory: DiaryMemory, query: QuerySpec, anchors: list[int] | None = None) -> list[float]:
    features = feature_vector(memory, query, anchors or [])
    return [float(features.get(name, 0.0)) for name in FEATURE_NAMES]


def scalar_prior_score(features: dict[str, float] | list[float] | tuple[float, ...], weights: dict[str, float] | None = None) -> float:
    weights = weights or DEFAULT_PRIOR_WEIGHTS
    if isinstance(features, dict):
        return sum(float(weights.get(name, 0.0)) * float(features.get(name, 0.0)) for name in FEATURE_NAMES)
    return sum(float(weights.get(name, 0.0)) * float(value) for name, value in zip(FEATURE_NAMES, features))


def build_query_from_row(row: dict[str, Any]) -> QuerySpec:
    override_window = None
    window = row.get("positive_window")
    if isinstance(window, list) and len(window) == 2 and all(isinstance(item, str) for item in window):
        override_window = (window[0], window[1])
    return build_query_spec(str(row.get("query") or ""), override_window)


class NeuralHTEMA(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        query_feature_dim: int,
        memory_feature_dim: int,
        pair_feature_dim: int,
        adapter_dim: int = 160,
        hidden_dim: int = 320,
        dropout: float = 0.08,
    ) -> None:
        super().__init__()
        self.adapter_dim = adapter_dim
        self.query_adapter = self._adapter(embedding_dim + query_feature_dim, hidden_dim, adapter_dim, dropout)
        self.key_adapter = self._adapter(embedding_dim + memory_feature_dim, hidden_dim, adapter_dim, dropout)
        self.value_adapter = self._adapter(embedding_dim + memory_feature_dim, hidden_dim, adapter_dim, dropout)
        self.pair_bias = nn.Sequential(
            nn.LayerNorm(pair_feature_dim),
            nn.Linear(pair_feature_dim, max(32, adapter_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(32, adapter_dim // 2), 1),
        )
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.value_score_scale = nn.Parameter(torch.tensor(0.25))

    @staticmethod
    def _adapter(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        query_embedding: torch.Tensor,
        query_features: torch.Tensor,
        memory_embeddings: torch.Tensor,
        memory_features: torch.Tensor,
        pair_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_input = torch.cat([query_embedding, query_features], dim=-1)
        memory_input = torch.cat([memory_embeddings, memory_features], dim=-1)

        q = F.normalize(self.query_adapter(query_input), dim=-1)
        k = F.normalize(self.key_adapter(memory_input), dim=-1)
        v = self.value_adapter(memory_input)
        value_relevance = (q.unsqueeze(1) * F.normalize(v, dim=-1)).sum(dim=-1)

        attention_logits = (q.unsqueeze(1) * k).sum(dim=-1) * self.logit_scale.exp().clamp(max=100.0)
        scores = attention_logits + self.value_score_scale * value_relevance + self.pair_bias(pair_features).squeeze(-1)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        context = (weights * v).sum(dim=1)
        return scores, context


def checkpoint_config(
    embedding_model: str,
    embedding_dim: int,
    adapter_dim: int,
    hidden_dim: int,
    dropout: float,
    stats: dict[str, float],
) -> dict[str, Any]:
    return {
        "architecture": "HTEMA neural Q/K/V adapter",
        "version": 1,
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
        "adapter_dim": adapter_dim,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "memory_feature_names": list(MEMORY_FEATURE_NAMES),
        "query_feature_names": list(QUERY_FEATURE_NAMES),
        "pair_feature_names": list(FEATURE_NAMES),
        "stats": stats,
    }


def make_model(config: dict[str, Any]) -> NeuralHTEMA:
    return NeuralHTEMA(
        embedding_dim=int(config["embedding_dim"]),
        query_feature_dim=len(config["query_feature_names"]),
        memory_feature_dim=len(config["memory_feature_names"]),
        pair_feature_dim=len(config["pair_feature_names"]),
        adapter_dim=int(config["adapter_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config.get("dropout", 0.0)),
    )
