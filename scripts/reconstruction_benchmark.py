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
class RankingResult:
    target: DiaryMemory
    query: str
    model: str
    rows: list[dict[str, Any]]
    metrics: dict[str, float | None]


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    kind: str
    source_types: tuple[str, ...] = ()
    use_trust_weighting: bool = False


BASELINES = (
    BaselineSpec("mira_trust", "mira", use_trust_weighting=True),
    BaselineSpec("mira_no_trust", "mira", use_trust_weighting=False),
    BaselineSpec("simple_rag_all", "simple_rag"),
    BaselineSpec("diary_only", "mira", source_types=("diary",), use_trust_weighting=False),
    BaselineSpec("chat_only", "mira", source_types=("whatsapp", "whatsapp_synthetic"), use_trust_weighting=False),
    BaselineSpec("chronological_neighbors", "chronological"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-targets", type=int, default=40, help="Maximum hidden diary days to evaluate. Use 0 for all.")
    parser.add_argument("--top-k", type=int, default=8, help="Evidence memories retrieved per hidden day.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--no-rollups", action="store_true", help="Exclude week/month rollup memories.")
    parser.add_argument("--no-atoms", action="store_true", help="Exclude sub-day atom memories.")
    parser.add_argument("--no-reflections", action="store_true", help="Exclude reflection memories.")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def salient_terms(memory: DiaryMemory, *, limit: int = 36) -> tuple[str, ...]:
    terms = []
    for token in tokenize(memory.text):
        if token in STOPWORDS or token in GENERIC_RECONSTRUCTION_TERMS:
            continue
        if len(token) < 4 or token.isdigit():
            continue
        terms.append(token)
    counts = Counter(terms)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return tuple(term for term, _ in ranked[:limit])


def query_for_target(memory: DiaryMemory) -> str:
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
        "features": features or {},
    }


def rank_mira_evidence(
    *,
    target: DiaryMemory,
    memories: list[DiaryMemory],
    top_k: int,
    use_trust_weighting: bool,
) -> tuple[str, list[dict[str, Any]]]:
    query_text = query_for_target(target)
    query = build_query_spec(query_text, override_window=(target.date, target.date))
    ranked = score_memories(query, memories, DEFAULT_WEIGHTS)
    for row in ranked:
        memory = row["memory"]
        base_score = float(row["score"])
        trust = source_weight(memory.source_type) if use_trust_weighting else 1.0
        row["ranking_score"] = base_score * trust
    ranked.sort(key=lambda item: item["ranking_score"], reverse=True)
    return query_text, ranked[:top_k]


def rank_simple_rag_evidence(
    *,
    target: DiaryMemory,
    memories: list[DiaryMemory],
    top_k: int,
) -> tuple[str, list[dict[str, Any]]]:
    query_text = query_for_target(target)
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
    memories: list[DiaryMemory],
    top_k: int,
) -> tuple[str, list[dict[str, Any]]]:
    scoped = filter_sources(memories, spec.source_types)
    if spec.kind == "mira":
        return rank_mira_evidence(
            target=target,
            memories=scoped,
            top_k=top_k,
            use_trust_weighting=spec.use_trust_weighting,
        )
    if spec.kind == "simple_rag":
        return rank_simple_rag_evidence(target=target, memories=scoped, top_k=top_k)
    if spec.kind == "chronological":
        return rank_chronological_evidence(target=target, memories=scoped, top_k=top_k)
    raise ValueError(f"Unknown baseline kind: {spec.kind}")


def evaluate_rows(target: DiaryMemory, rows: list[dict[str, Any]], top_k: int) -> dict[str, float | None]:
    target_terms = set(salient_terms(target))
    evidence_terms: set[str] = set()
    audit_ready = 0
    supported = 0
    trust_violations = 0
    direct_leaks = 0
    inferred_rows = 0
    contradiction_needed = any(term in target_terms for term in {"contradict", "conflict", "argument", "never", "always"})
    contradiction_evidence = 0
    trust_sum = 0.0

    for row in rows:
        memory: DiaryMemory = row["memory"]
        evidence_terms.update(tokenize(memory.text))
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

    reconstruction_recall = len(target_terms.intersection(evidence_terms)) / max(len(target_terms), 1)
    source_precision = supported / max(len(rows), 1)
    audit_success_rate = audit_ready / max(len(rows), 1)
    avg_trust = trust_sum / max(len(rows), 1)
    evidence_confidence = avg_trust * source_precision * min(1.0, len(rows) / max(top_k, 1))
    confidence_error = abs(evidence_confidence - reconstruction_recall)
    trust_violation_rate = trust_violations / max(inferred_rows, 1) if inferred_rows else 0.0

    return {
        "reconstruction_recall": reconstruction_recall,
        "source_precision": source_precision,
        "evidence_confidence": evidence_confidence,
        "confidence_error": confidence_error,
        "trust_violation_rate": trust_violation_rate,
        "direct_override_error": 1.0 if direct_leaks else 0.0,
        "contradiction_preservation": (
            1.0 if contradiction_evidence else 0.0
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
        "trust": memory.trust_payload(),
        "preview": compact_text(memory.text, 180),
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


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Missing-Day Reconstruction Benchmark",
        "",
        f"Generated at: {payload['created_at']}",
        f"Hidden diary targets: {payload['target_count']}",
        f"Memory pool size: {payload['memory_count']}",
        f"Top-k evidence: {payload['top_k']}",
        "",
        "## Aggregate Metrics",
        "",
        "| Model | Recall | Source Precision | Confidence Error | Trust Violations | Direct Override | Audit Success | Avg Trust |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model, metrics in payload["summary"].items():
        lines.append(
            "| {model} | {recall:.3f} | {source_precision:.3f} | {confidence_error:.3f} | {trust_violation:.3f} | {direct_override:.3f} | {audit:.3f} | {avg_trust:.3f} |".format(
                model=model,
                recall=metrics.get("reconstruction_recall", 0.0),
                source_precision=metrics.get("source_precision", 0.0),
                confidence_error=metrics.get("confidence_error", 0.0),
                trust_violation=metrics.get("trust_violation_rate", 0.0),
                direct_override=metrics.get("direct_override_error", 0.0),
                audit=metrics.get("audit_success_rate", 0.0),
                avg_trust=metrics.get("avg_trust_level", 0.0),
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
        lines.append(f"- recall: `{case['metrics']['reconstruction_recall']:.3f}`")
        lines.append("- evidence:")
        for row in case["evidence"][:3]:
            lines.append(
                f"  - `{row['source_type']}` `{row['date']}` trust={row['trust']['trust_level']:.2f} `{row['entry_id']}`"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    diary_targets = parse_diary_memories()
    if not diary_targets:
        raise SystemExit("No diary targets found. Add dated diary files under data/diaries or set DIARY_ROOT/DIARY_FILES.")

    memories = parse_all_memories(
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
        evidence_pool = allowed_evidence(memories, target)
        if not evidence_pool:
            continue
        for spec in BASELINES:
            query, rows = rank_baseline(
                spec=spec,
                target=target,
                memories=evidence_pool,
                top_k=args.top_k,
            )
            results.append(
                RankingResult(
                    target=target,
                    query=query,
                    model=spec.name,
                    rows=rows,
                    metrics=evaluate_rows(target, rows, args.top_k),
                )
            )

    payload = {
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "target_count": len({result.target.entry_id for result in results}),
        "memory_count": len(memories),
        "top_k": args.top_k,
        "summary": summarize(results),
        "cases": [
            {
                "model": result.model,
                "target_id": result.target.entry_id,
                "target_date": result.target.date,
                "query": result.query,
                "metrics": result.metrics,
                "evidence": [serialize_row(row) for row in result.rows],
            }
            for result in results
        ],
    }

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(args.report, payload)

    if args.json:
        print(json.dumps({"summary": payload["summary"], "target_count": payload["target_count"]}, indent=2))
    else:
        print(f"targets={payload['target_count']} memories={payload['memory_count']} top_k={payload['top_k']}")
        for model, metrics in payload["summary"].items():
            print(
                f"{model}: recall={metrics.get('reconstruction_recall', 0):.3f} "
                f"source_precision={metrics.get('source_precision', 0):.3f} "
                f"confidence_error={metrics.get('confidence_error', 0):.3f} "
                f"audit={metrics.get('audit_success_rate', 0):.3f}"
            )
        print(f"wrote {args.report}")
        print(f"wrote {args.metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
