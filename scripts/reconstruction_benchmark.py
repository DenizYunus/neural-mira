#!/usr/bin/env python3
"""Missing-day reconstruction benchmark for Neural MIRA.

The benchmark hides direct diary memories, removes any derived memories whose
provenance points at the hidden day, and asks the ranker to gather indirect
evidence. It does not call an LLM: the first paper-grade scaffold should be
deterministic, reproducible, and safe to run on private corpora.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from htema_core import (
    LAB_ROOT,
    STOPWORDS,
    DiaryMemory,
    build_query_spec,
    compact_text,
    cosine,
    load_reflections,
    parse_all_memories,
    parse_diary_memories,
    score_memories,
    source_weight,
    tokenize,
)


DEFAULT_REPORT = LAB_ROOT / "data" / "reconstruction_report.md"
DEFAULT_METRICS = LAB_ROOT / "data" / "reconstruction_metrics.json"

DEFAULT_WEIGHTS = {
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

GENERIC_RECONSTRUCTION_TERMS = {
    "diary",
    "entry",
    "mood",
    "icons",
    "today",
    "memory",
    "memories",
    "dailybean",
    "complete",
}


@dataclass(frozen=True)
class TargetSpec:
    target_date: str
    event_id: str | None = None
    public_hint: str = ""
    sparse_user_hint: str = ""
    query: str = ""
    confuser_event_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class EventTruth:
    event_id: str
    date: str
    title: str
    truth: str
    facts: dict[str, str]
    people: tuple[str, ...] = ()
    location: str = ""
    contradiction: str = ""
    correction: str = ""
    confuser_for: tuple[str, ...] = ()

    def fact_text(self) -> str:
        parts = [
            self.title,
            self.truth,
            " ".join(self.facts.values()),
            f"People: {', '.join(self.people)}." if self.people else "",
            f"Location: {self.location}." if self.location else "",
            self.contradiction,
            self.correction,
        ]
        return " ".join(part for part in parts if part)


@dataclass(frozen=True)
class RankingResult:
    target: DiaryMemory
    target_spec: TargetSpec
    event_truth: EventTruth | None
    query_mode: str
    query: str
    model: str
    rows: list[dict[str, Any]]
    metrics: dict[str, float | None]
    reconstruction_text: str


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    kind: str
    source_types: tuple[str, ...] = ()
    trust_strategy: str = "none"
    confidence_mode: str = "raw"


BASELINES = (
    BaselineSpec("mira_trust", "mira", trust_strategy="rank"),
    BaselineSpec("mira_trust_v2", "mira", trust_strategy="confidence", confidence_mode="date_calibrated"),
    BaselineSpec("mira_trust_v3", "mira", trust_strategy="diverse_confidence", confidence_mode="date_calibrated"),
    BaselineSpec("mira_no_trust", "mira"),
    BaselineSpec("simple_rag_all", "simple_rag"),
    BaselineSpec("diary_only", "mira", source_types=("diary",)),
    BaselineSpec("chat_only", "mira", source_types=("whatsapp", "whatsapp_synthetic")),
    BaselineSpec("chronological_neighbors", "chronological"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-targets", type=int, default=40, help="Maximum hidden diary days to evaluate. Use 0 for all.")
    parser.add_argument("--top-k", type=int, default=8, help="Evidence memories retrieved per hidden day.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--diary-root", type=Path, help="Directory or file containing dated markdown diary labels.")
    parser.add_argument("--whatsapp-root", type=Path, help="Directory containing WhatsApp-style chat export folders.")
    parser.add_argument("--photo-metadata", type=Path, help="JSONL file containing dated photo metadata evidence.")
    parser.add_argument("--ground-truth-events", type=Path, help="JSONL file containing event-level gold facts.")
    parser.add_argument("--target-dates-file", type=Path, help="JSONL/TXT file listing target dates to hide/evaluate.")
    parser.add_argument(
        "--query-mode",
        choices=("date_only", "date_plus_public_metadata", "sparse_user_hint", "legacy_hidden_label"),
        default="date_only",
        help="How much non-diary information the reconstruction query may include.",
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--no-rollups", action="store_true", help="Exclude week/month rollup memories.")
    parser.add_argument("--no-atoms", action="store_true", help="Exclude sub-day atom memories.")
    parser.add_argument("--no-reflections", action="store_true", help="Exclude reflection memories.")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def diary_files_from_root(path: Path | None) -> list[Path] | None:
    if path is None:
        return None
    if path.is_file():
        return [path]
    if not path.exists():
        raise SystemExit(f"diary root not found: {path}")
    return sorted(file_path for file_path in path.glob("*.md") if file_path.is_file())


def load_target_specs(path: Path | None) -> dict[str, TargetSpec]:
    if path is None:
        return {}
    if not path.exists():
        raise SystemExit(f"target dates file not found: {path}")
    specs: dict[str, TargetSpec] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            row = json.loads(line)
            value = row.get("target_date") or row.get("date")
            if not isinstance(value, str) or not value:
                continue
            target_date = value[:10]
            confusers = row.get("confuser_event_ids") or ()
            specs[target_date] = TargetSpec(
                target_date=target_date,
                event_id=str(row.get("event_id") or row.get("gold_event_id") or "") or None,
                public_hint=str(row.get("public_hint") or ""),
                sparse_user_hint=str(row.get("sparse_user_hint") or ""),
                query=str(row.get("query") or ""),
                confuser_event_ids=tuple(str(item) for item in confusers),
            )
        else:
            specs[line[:10]] = TargetSpec(target_date=line[:10])
    return specs


def load_event_truths(path: Path | None) -> tuple[dict[str, EventTruth], dict[str, EventTruth]]:
    if path is None:
        return {}, {}
    if not path.exists():
        raise SystemExit(f"ground truth events file not found: {path}")
    by_id: dict[str, EventTruth] = {}
    by_date: dict[str, EventTruth] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        event_id = str(row.get("event_id") or "")
        event_date = normalize_json_date(row.get("date"))
        if not event_id or not event_date:
            continue
        truth = EventTruth(
            event_id=event_id,
            date=event_date,
            title=str(row.get("title") or ""),
            truth=str(row.get("truth") or ""),
            facts={
                str(key): str(value)
                for key, value in (row.get("facts") or {}).items()
                if str(key).strip() and str(value).strip()
            },
            people=tuple(str(item) for item in row.get("people") or []),
            location=str(row.get("location") or ""),
            contradiction=str(row.get("contradiction") or ""),
            correction=str(row.get("correction") or ""),
            confuser_for=tuple(str(item)[:10] for item in row.get("confuser_for") or [] if str(item)),
        )
        by_id[event_id] = truth
        by_date[event_date] = truth
    return by_id, by_date


def normalize_json_date(value: Any) -> str | None:
    if not isinstance(value, str) or len(value) < 10:
        return None
    return value[:10]


def salient_terms(memory: DiaryMemory, *, limit: int = 36) -> tuple[str, ...]:
    return salient_terms_from_text(memory.text, limit=limit)


def salient_terms_from_text(text: str, *, limit: int = 36) -> tuple[str, ...]:
    terms = []
    for token in tokenize(text):
        if token in STOPWORDS or token in GENERIC_RECONSTRUCTION_TERMS:
            continue
        if len(token) < 4 or token.isdigit():
            continue
        terms.append(token)
    counts = Counter(terms)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return tuple(term for term, _ in ranked[:limit])


def fact_slot_support(facts: dict[str, str], evidence_terms: set[str]) -> dict[str, dict[str, float]]:
    slots: dict[str, dict[str, float]] = {}
    for slot, value in facts.items():
        terms = set(salient_terms_from_text(value, limit=24))
        if not terms:
            continue
        matched = terms.intersection(evidence_terms)
        recall = len(matched) / max(len(terms), 1)
        if len(terms) <= 2:
            supported = recall >= 0.5
        else:
            supported = recall >= 0.34
        slots[slot] = {
            "recall": recall,
            "supported": 1.0 if supported else 0.0,
            "term_count": float(len(terms)),
            "matched_count": float(len(matched)),
        }
    return slots


def query_for_target(memory: DiaryMemory, target_spec: TargetSpec, query_mode: str) -> str:
    date_query = target_spec.query or f"Reconstruct what happened around {memory.date}."
    if query_mode == "date_only":
        return f"Reconstruct what happened around {memory.date}."
    if query_mode == "date_plus_public_metadata":
        hint = target_spec.public_hint.strip()
        return f"{date_query} {hint}".strip()
    if query_mode == "sparse_user_hint":
        hint = target_spec.sparse_user_hint.strip() or target_spec.public_hint.strip()
        return f"{date_query} {hint}".strip()

    hints = [icon.replace("_", " ") for icon in memory.icons[:5]]
    mood_hint = f"mood {memory.mood}" if memory.mood else ""
    hint_text = " ".join(item for item in [mood_hint, *hints] if item)
    return f"reconstruct what happened around {memory.date} {hint_text}".strip()


def target_refs(memory: DiaryMemory) -> set[str]:
    return {memory.entry_id, *memory.provenance_ids}


def leaks_hidden_target(memory: DiaryMemory, target: DiaryMemory) -> bool:
    if memory.entry_id == target.entry_id:
        return True
    if memory.source_type == "diary" and memory.date == target.date:
        return True
    refs = set(memory.provenance_ids)
    return bool(refs.intersection(target_refs(target)))


def allowed_evidence(memories: list[DiaryMemory], target: DiaryMemory) -> list[DiaryMemory]:
    return [memory for memory in memories if not leaks_hidden_target(memory, target)]


def filter_sources(memories: list[DiaryMemory], source_types: tuple[str, ...]) -> list[DiaryMemory]:
    if not source_types:
        return memories
    allowed = set(source_types)
    return [memory for memory in memories if memory.source_type in allowed]


def row_for_memory(memory: DiaryMemory, *, score: float, ranking_score: float, features: dict[str, float] | None = None) -> dict[str, Any]:
    return {
        "memory": memory,
        "score": score,
        "ranking_score": ranking_score,
        "confidence_score": ranking_score,
        "features": features or {},
    }


def trust_v3_rerank(target: DiaryMemory, rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    remaining = rows[:]
    selected_sources: set[str] = set()
    selected_non_target_dates: set[str] = set()

    while remaining and len(selected) < top_k:
        best_index = 0
        best_score = float("-inf")
        for index, row in enumerate(remaining):
            memory: DiaryMemory = row["memory"]
            base_score = float(row["score"])
            day_distance = abs(memory.ordinal - target.ordinal)
            same_date = memory.date == target.date
            date_multiplier = 1.24 if same_date else (0.88 if day_distance <= 4 else 0.96)
            source_multiplier = 1.08 if memory.source_type not in selected_sources else 0.94
            repeat_neighbor_multiplier = 0.72 if (not same_date and memory.date in selected_non_target_dates) else 1.0
            metadata_anchor = 1.05 if same_date and memory.source_type in {"whatsapp", "photo_metadata"} else 1.0
            ranking_score = base_score * date_multiplier * source_multiplier * repeat_neighbor_multiplier * metadata_anchor
            if ranking_score > best_score:
                best_score = ranking_score
                best_index = index

        chosen = remaining.pop(best_index)
        memory = chosen["memory"]
        trust = source_weight(memory.source_type)
        chosen["ranking_score"] = best_score
        chosen["confidence_score"] = float(chosen["score"]) * trust
        features = dict(chosen.get("features") or {})
        features["trust_v3_same_date"] = 1.0 if memory.date == target.date else 0.0
        features["trust_v3_day_distance"] = float(abs(memory.ordinal - target.ordinal))
        chosen["features"] = features
        selected.append(chosen)
        selected_sources.add(memory.source_type)
        if memory.date != target.date:
            selected_non_target_dates.add(memory.date)

    return selected


def rank_mira_evidence(
    *,
    target: DiaryMemory,
    target_spec: TargetSpec,
    query_mode: str,
    memories: list[DiaryMemory],
    top_k: int,
    trust_strategy: str,
) -> tuple[str, list[dict[str, Any]]]:
    query_text = query_for_target(target, target_spec, query_mode)
    query = build_query_spec(query_text, override_window=(target.date, target.date))
    ranked = score_memories(query, memories, DEFAULT_WEIGHTS)
    for row in ranked:
        memory = row["memory"]
        base_score = float(row["score"])
        trust = source_weight(memory.source_type)
        row["ranking_score"] = base_score * trust if trust_strategy == "rank" else base_score
        row["confidence_score"] = base_score * trust if trust_strategy in {"rank", "confidence"} else base_score
    if trust_strategy == "diverse_confidence":
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return query_text, trust_v3_rerank(target, ranked, top_k)
    ranked.sort(key=lambda item: item["ranking_score"], reverse=True)
    return query_text, ranked[:top_k]


def rank_simple_rag_evidence(
    *,
    target: DiaryMemory,
    target_spec: TargetSpec,
    query_mode: str,
    memories: list[DiaryMemory],
    top_k: int,
) -> tuple[str, list[dict[str, Any]]]:
    query_text = query_for_target(target, target_spec, query_mode)
    query = build_query_spec(query_text, override_window=(target.date, target.date))
    rows = []
    for memory in memories:
        score = cosine(query.token_vector, memory.token_vector)
        rows.append(row_for_memory(memory, score=score, ranking_score=score, features={"rag": score}))
    rows.sort(key=lambda item: item["ranking_score"], reverse=True)
    return query_text, rows[:top_k]


def rank_chronological_evidence(
    *,
    target: DiaryMemory,
    target_spec: TargetSpec,
    query_mode: str,
    memories: list[DiaryMemory],
    top_k: int,
) -> tuple[str, list[dict[str, Any]]]:
    query_text = f"nearest memories around {target.date}"
    rows = []
    for memory in memories:
        distance = abs(memory.ordinal - target.ordinal)
        proximity = 1.0 / (1.0 + distance)
        trust_adjusted = proximity * source_weight(memory.source_type)
        rows.append(
            row_for_memory(
                memory,
                score=proximity,
                ranking_score=trust_adjusted,
                features={"day_distance": float(distance), "chronological_proximity": proximity},
            )
        )
    rows.sort(key=lambda item: item["ranking_score"], reverse=True)
    return query_text, rows[:top_k]


def rank_baseline(
    *,
    spec: BaselineSpec,
    target: DiaryMemory,
    target_spec: TargetSpec,
    query_mode: str,
    memories: list[DiaryMemory],
    top_k: int,
) -> tuple[str, list[dict[str, Any]]]:
    scoped = filter_sources(memories, spec.source_types)
    if spec.kind == "mira":
        return rank_mira_evidence(
            target=target,
            target_spec=target_spec,
            query_mode=query_mode,
            memories=scoped,
            top_k=top_k,
            trust_strategy=spec.trust_strategy,
        )
    if spec.kind == "simple_rag":
        return rank_simple_rag_evidence(target=target, target_spec=target_spec, query_mode=query_mode, memories=scoped, top_k=top_k)
    if spec.kind == "chronological":
        return rank_chronological_evidence(target=target, target_spec=target_spec, query_mode=query_mode, memories=scoped, top_k=top_k)
    raise ValueError(f"Unknown baseline kind: {spec.kind}")


def evaluate_rows(
    target: DiaryMemory,
    rows: list[dict[str, Any]],
    top_k: int,
    *,
    event_truth: EventTruth | None,
    confuser_dates: set[str],
    confidence_mode: str,
) -> dict[str, float | None]:
    target_terms = set(salient_terms(target))
    event_terms = set(salient_terms_from_text(event_truth.fact_text())) if event_truth else set()
    evidence_terms: set[str] = set()
    audit_ready = 0
    supported = 0
    trust_violations = 0
    direct_leaks = 0
    inferred_rows = 0
    contradiction_needed = bool(event_truth and event_truth.contradiction) or any(
        term in target_terms for term in {"contradict", "conflict", "argument", "never", "always"}
    )
    contradiction_evidence = 0
    trust_sum = 0.0
    target_date_hits = 0
    confuser_hits = 0

    for row in rows:
        memory: DiaryMemory = row["memory"]
        evidence_terms.update(tokenize(memory.text))
        target_date_hits += 1 if memory.date == target.date else 0
        confuser_hits += 1 if memory.date in confuser_dates else 0
        trust_payload = memory.trust_payload()
        has_provenance = bool(trust_payload["provenance_ids"] or trust_payload["provenance_steps"])
        audit_ready += 1 if has_provenance else 0
        leaked = leaks_hidden_target(memory, target)
        direct_leaks += 1 if leaked else 0
        supported += 1 if has_provenance and not leaked else 0
        inference_status = str(memory.inference_status or "")
        if inference_status in {"inferred", "speculative"}:
            inferred_rows += 1
            if not (memory.provenance_excerpts or memory.provenance_ids):
                trust_violations += 1
        if "contradict" in memory.entry_id or "contradiction" in memory.text.lower() or "conflict" in memory.text.lower():
            contradiction_evidence += 1
        trust_sum += float(memory.trust_level or 0.5)

    diary_term_recall = len(target_terms.intersection(evidence_terms)) / max(len(target_terms), 1)
    fact_support = fact_slot_support(event_truth.facts, evidence_terms) if event_truth else {}
    if fact_support:
        structured_fact_recall = sum(slot["recall"] for slot in fact_support.values()) / len(fact_support)
        fact_slot_accuracy = sum(slot["supported"] for slot in fact_support.values()) / len(fact_support)
    else:
        structured_fact_recall = None
        fact_slot_accuracy = None
    if event_terms:
        event_hits = len(event_terms.intersection(evidence_terms))
        event_recall = event_hits / max(len(event_terms), 1)
        event_precision = event_hits / max(len(evidence_terms), 1)
        event_f1 = (2 * event_precision * event_recall / (event_precision + event_recall)) if event_precision + event_recall else 0.0
        reconstruction_recall = structured_fact_recall if structured_fact_recall is not None else event_recall
    else:
        event_recall = None
        event_precision = None
        event_f1 = None
        reconstruction_recall = diary_term_recall
    source_precision = supported / max(len(rows), 1)
    audit_success_rate = audit_ready / max(len(rows), 1)
    avg_trust = trust_sum / max(len(rows), 1)
    coverage = min(1.0, len(rows) / max(top_k, 1))
    target_date_precision = target_date_hits / max(len(rows), 1)
    raw_trust_confidence = avg_trust * source_precision * coverage
    date_calibration = 0.5 + 0.5 * target_date_precision
    date_calibrated_confidence = raw_trust_confidence * date_calibration
    evidence_confidence = date_calibrated_confidence if confidence_mode == "date_calibrated" else raw_trust_confidence
    confidence_error = abs(evidence_confidence - reconstruction_recall)
    trust_violation_rate = trust_violations / max(inferred_rows, 1) if inferred_rows else 0.0
    contradiction_terms = set(salient_terms_from_text(event_truth.contradiction)) if event_truth and event_truth.contradiction else set()
    contradiction_preserved = bool(contradiction_evidence or contradiction_terms.intersection(evidence_terms))

    return {
        "reconstruction_recall": reconstruction_recall,
        "diary_term_recall": diary_term_recall,
        "event_recall": event_recall,
        "event_precision": event_precision,
        "event_f1": event_f1,
        "structured_fact_recall": structured_fact_recall,
        "fact_slot_accuracy": fact_slot_accuracy,
        "fact_slot_count": float(len(fact_support)) if fact_support else None,
        "source_precision": source_precision,
        "evidence_confidence": evidence_confidence,
        "raw_trust_confidence": raw_trust_confidence,
        "date_calibrated_confidence": date_calibrated_confidence,
        "confidence_error": confidence_error,
        "trust_violation_rate": trust_violation_rate,
        "direct_override_error": 1.0 if direct_leaks else 0.0,
        "target_date_precision": target_date_precision,
        "confuser_intrusion_rate": confuser_hits / max(len(rows), 1),
        "contradiction_preservation": (
            1.0 if contradiction_preserved else 0.0
        ) if contradiction_needed else None,
        "audit_success_rate": audit_success_rate,
        "avg_trust_level": avg_trust,
    }


def serialize_row(row: dict[str, Any]) -> dict[str, Any]:
    memory: DiaryMemory = row["memory"]
    return {
        "entry_id": memory.entry_id,
        "date": memory.date,
        "source_type": memory.source_type,
        "ranking_score": round(float(row["ranking_score"]), 6),
        "base_score": round(float(row["score"]), 6),
        "confidence_score": round(float(row.get("confidence_score", row["ranking_score"])), 6),
        "trust": memory.trust_payload(),
        "preview": compact_text(memory.text, 180),
    }


def evidence_terms_for_rows(rows: list[dict[str, Any]]) -> set[str]:
    terms: set[str] = set()
    for row in rows:
        memory: DiaryMemory = row["memory"]
        terms.update(tokenize(memory.text))
    return terms


def build_reconstruction_text(target: DiaryMemory, rows: list[dict[str, Any]], metrics: dict[str, float | None]) -> str:
    confidence = metric_value(metrics, "evidence_confidence")
    if confidence >= 0.70:
        confidence_label = "medium-high"
    elif confidence >= 0.45:
        confidence_label = "medium"
    else:
        confidence_label = "low"

    source_labels = sorted({row["memory"].source_type for row in rows})
    evidence_lines = []
    for row in rows:
        memory: DiaryMemory = row["memory"]
        evidence_lines.append(
            f"- {memory.source_type} {memory.date} ({memory.entry_id}): {compact_text(memory.text, 140)}"
        )
    return "\n".join([
        f"Inferred reconstruction for {target.date}.",
        f"Confidence: {confidence_label} ({confidence:.3f}).",
        "This is not a direct diary memory; it is reconstructed from indirect and neighboring evidence.",
        f"Evidence sources: {', '.join(source_labels) if source_labels else 'none'}.",
        "Audit trail:",
        *evidence_lines,
    ])


def reconstruction_audit_metrics(reconstruction_text: str, rows: list[dict[str, Any]]) -> dict[str, float]:
    lowered = reconstruction_text.lower()
    source_types = {row["memory"].source_type for row in rows}
    source_mentions = sum(1 for source_type in source_types if source_type.lower() in lowered)
    provenance_mentions = sum(1 for row in rows if str(row["memory"].entry_id).lower() in lowered)
    return {
        "uncertainty_label_success": 1.0 if "confidence:" in lowered else 0.0,
        "inferred_label_success": 1.0 if "inferred reconstruction" in lowered else 0.0,
        "not_direct_label_success": 1.0 if "not a direct diary memory" in lowered else 0.0,
        "source_label_coverage": source_mentions / max(len(source_types), 1),
        "audit_trail_coverage": provenance_mentions / max(len(rows), 1),
    }


def summarize(results: list[RankingResult]) -> dict[str, dict[str, float]]:
    by_model: dict[str, list[dict[str, float | None]]] = {}
    for result in results:
        by_model.setdefault(result.model, []).append(result.metrics)
    out: dict[str, dict[str, float]] = {}
    for model, rows in by_model.items():
        metric_names = sorted({name for row in rows for name in row})
        out[model] = {}
        for name in metric_names:
            values = [float(row[name]) for row in rows if row.get(name) is not None]
            if values:
                out[model][name] = sum(values) / len(values)
    return out


def summarize_slots(results: list[RankingResult]) -> dict[str, dict[str, dict[str, float]]]:
    buckets: dict[str, dict[str, list[dict[str, float]]]] = {}
    for result in results:
        if not result.event_truth:
            continue
        support = fact_slot_support(result.event_truth.facts, evidence_terms_for_rows(result.rows))
        for slot, metrics in support.items():
            buckets.setdefault(result.model, {}).setdefault(slot, []).append(metrics)

    out: dict[str, dict[str, dict[str, float]]] = {}
    for model, slots in buckets.items():
        out[model] = {}
        for slot, rows in slots.items():
            out[model][slot] = {
                "recall": sum(row["recall"] for row in rows) / len(rows),
                "accuracy": sum(row["supported"] for row in rows) / len(rows),
                "n": float(len(rows)),
            }
    return out


def metric_value(metrics: dict[str, Any], name: str, default: float = 0.0) -> float:
    value = metrics.get(name)
    return float(value) if value is not None else default


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Missing-Day Reconstruction Benchmark",
        "",
        f"Generated at: {payload['created_at']}",
        f"Hidden diary targets: {payload['target_count']}",
        f"Memory pool size: {payload['memory_count']}",
        f"Top-k evidence: {payload['top_k']}",
        f"Query mode: {payload['query_mode']}",
        "",
        "## Aggregate Metrics",
        "",
        "| Model | Fact Recall | Fact Accuracy | Event F1 | Confuser Intrusion | Confidence Error | Not-Direct Label | Audit Trail | Source Labels |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model, metrics in payload["summary"].items():
        lines.append(
            "| {model} | {recall:.3f} | {fact_accuracy:.3f} | {event_f1:.3f} | {confuser:.3f} | {confidence_error:.3f} | {not_direct:.3f} | {audit_trail:.3f} | {source_labels:.3f} |".format(
                model=model,
                recall=metric_value(metrics, "reconstruction_recall"),
                fact_accuracy=metric_value(metrics, "fact_slot_accuracy"),
                event_f1=metric_value(metrics, "event_f1"),
                confuser=metric_value(metrics, "confuser_intrusion_rate"),
                confidence_error=metric_value(metrics, "confidence_error"),
                not_direct=metric_value(metrics, "not_direct_label_success"),
                audit_trail=metric_value(metrics, "audit_trail_coverage"),
                source_labels=metric_value(metrics, "source_label_coverage"),
            )
        )
    if payload.get("slot_summary"):
        lines.extend([
            "",
            "## Per-Slot Fact Recovery",
            "",
            "| Model | Slot | Recall | Accuracy | N |",
            "| --- | --- | ---: | ---: | ---: |",
        ])
        for model, slots in payload["slot_summary"].items():
            for slot, metrics in sorted(slots.items()):
                lines.append(
                    "| {model} | {slot} | {recall:.3f} | {accuracy:.3f} | {n:.0f} |".format(
                        model=model,
                        slot=slot,
                        recall=metric_value(metrics, "recall"),
                        accuracy=metric_value(metrics, "accuracy"),
                        n=metric_value(metrics, "n"),
                    )
                )
    lines.extend([
        "",
        "## Lowest-Recall Cases",
        "",
        "These rows include IDs and short previews for debugging. Keep reports under `data/` for private corpora.",
        "",
    ])
    trust_cases = [case for case in payload["cases"] if case["model"] == "mira_trust"]
    trust_cases.sort(key=lambda item: item["metrics"]["reconstruction_recall"])
    for case in trust_cases[:8]:
        lines.append(f"### {case['target_id']} ({case['target_date']})")
        lines.append("")
        lines.append(f"- query: `{case['query']}`")
        lines.append(f"- fact recall: `{case['metrics']['reconstruction_recall']:.3f}`")
        if case["metrics"].get("fact_slot_accuracy") is not None:
            lines.append(f"- fact slot accuracy: `{case['metrics']['fact_slot_accuracy']:.3f}`")
        if case["metrics"].get("event_f1") is not None:
            lines.append(f"- event f1: `{case['metrics']['event_f1']:.3f}`")
        lines.append("- evidence:")
        for row in case["evidence"][:3]:
            lines.append(
                f"  - `{row['source_type']}` `{row['date']}` trust={row['trust']['trust_level']:.2f} `{row['entry_id']}`"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    diary_files = diary_files_from_root(args.diary_root)
    target_specs = load_target_specs(args.target_dates_file)
    events_by_id, events_by_date = load_event_truths(args.ground_truth_events)
    diary_targets = parse_diary_memories(files=diary_files)
    if not diary_targets:
        raise SystemExit("No diary targets found. Add dated diary files under data/diaries or set DIARY_ROOT/DIARY_FILES.")
    if target_specs:
        diary_targets = [memory for memory in diary_targets if memory.date in target_specs]
        if not diary_targets:
            raise SystemExit(f"No diary targets matched dates from {args.target_dates_file}")

    memories = parse_all_memories(
        diary_files=diary_files,
        whatsapp_root=args.whatsapp_root,
        photo_metadata_path=args.photo_metadata,
        include_photo_metadata=bool(args.photo_metadata),
        include_rollups=not args.no_rollups,
        include_atoms=not args.no_atoms,
    )
    if not args.no_reflections:
        memories = sorted(memories + load_reflections(), key=lambda item: item.ordinal)

    rng = random.Random(args.seed)
    targets = diary_targets[:]
    rng.shuffle(targets)
    if args.max_targets > 0:
        targets = targets[: args.max_targets]
    targets.sort(key=lambda item: item.ordinal)

    results: list[RankingResult] = []
    for target in targets:
        target_spec = target_specs.get(target.date, TargetSpec(target_date=target.date))
        event_truth = events_by_id.get(target_spec.event_id or "") or events_by_date.get(target.date)
        confuser_dates = {
            truth.date
            for truth in events_by_id.values()
            if target.date in truth.confuser_for or truth.event_id in target_spec.confuser_event_ids
        }
        evidence_pool = allowed_evidence(memories, target)
        if not evidence_pool:
            continue
        for spec in BASELINES:
            query, rows = rank_baseline(
                spec=spec,
                target=target,
                target_spec=target_spec,
                query_mode=args.query_mode,
                memories=evidence_pool,
                top_k=args.top_k,
            )
            metrics = evaluate_rows(
                target,
                rows,
                args.top_k,
                event_truth=event_truth,
                confuser_dates=confuser_dates,
                confidence_mode=spec.confidence_mode,
            )
            reconstruction_text = build_reconstruction_text(target, rows, metrics)
            metrics.update(reconstruction_audit_metrics(reconstruction_text, rows))
            results.append(
                RankingResult(
                    target=target,
                    target_spec=target_spec,
                    event_truth=event_truth,
                    query_mode=args.query_mode,
                    query=query,
                    model=spec.name,
                    rows=rows,
                    metrics=metrics,
                    reconstruction_text=reconstruction_text,
                )
            )

    payload = {
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "target_count": len({result.target.entry_id for result in results}),
        "memory_count": len(memories),
        "top_k": args.top_k,
        "query_mode": args.query_mode,
        "ground_truth_event_count": len(events_by_id),
        "summary": summarize(results),
        "slot_summary": summarize_slots(results),
        "cases": [
            {
                "model": result.model,
                "target_id": result.target.entry_id,
                "target_date": result.target.date,
                "event_id": result.target_spec.event_id,
                "query_mode": result.query_mode,
                "query": result.query,
                "reconstruction_text": result.reconstruction_text,
                "metrics": result.metrics,
                "fact_support": (
                    fact_slot_support(result.event_truth.facts, evidence_terms_for_rows(result.rows))
                    if result.event_truth else {}
                ),
                "evidence": [serialize_row(row) for row in result.rows],
            }
            for result in results
        ],
    }

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(args.report, payload)

    if args.json:
        print(json.dumps({
            "summary": payload["summary"],
            "slot_summary": payload["slot_summary"],
            "target_count": payload["target_count"],
            "query_mode": args.query_mode,
        }, indent=2))
    else:
        print(f"targets={payload['target_count']} memories={payload['memory_count']} top_k={payload['top_k']} query_mode={args.query_mode}")
        for model, metrics in payload["summary"].items():
            print(
                f"{model}: fact_recall={metric_value(metrics, 'reconstruction_recall'):.3f} "
                f"fact_accuracy={metric_value(metrics, 'fact_slot_accuracy'):.3f} "
                f"event_f1={metric_value(metrics, 'event_f1'):.3f} "
                f"date_precision={metric_value(metrics, 'target_date_precision'):.3f} "
                f"confuser={metric_value(metrics, 'confuser_intrusion_rate'):.3f} "
                f"confidence_error={metric_value(metrics, 'confidence_error'):.3f} "
                f"audit={metric_value(metrics, 'audit_success_rate'):.3f}"
            )
        print(f"wrote {args.report}")
        print(f"wrote {args.metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
