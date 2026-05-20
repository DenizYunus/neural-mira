#!/usr/bin/env python3
"""Reflection compiler — Phase 4 of the MIRA design.

Reads diary day tokens and synthesizes Level-5 reflection tokens covering
mood streaks (peaks and dips), recurring themes, contradictions inside a
window, and persistent icon clusters. Output is JSONL at
`data/reflections.jsonl`, consumable by `honest_mira.py --include-reflections`
or `htema_core.load_reflections()` for direct retrieval.

Reflections are *generated* from diary structure rather than written by an
LLM. That keeps them auditable and reproducible. An LLM polish pass can be
layered on later by reading the JSONL and rewriting the `text` field.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from htema_core import (
    DIARY_FEATURES,
    DiaryMemory,
    LAB_ROOT,
    parse_diary_memories,
)


REFLECTIONS_DEFAULT = LAB_ROOT / "data" / "reflections.jsonl"


@dataclass
class Reflection:
    id: str
    kind: str
    label: str
    start: str
    end: str
    text: str
    mood: int | None
    icons: list[str]
    importance: float
    participants: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)


def _mean_mood(memories: list[DiaryMemory]) -> float | None:
    moods = [m.mood for m in memories if m.mood is not None]
    return sum(moods) / len(moods) if moods else None


def _dominant_features(memories: list[DiaryMemory], threshold: float = 0.18) -> list[tuple[str, float]]:
    aggregate: dict[str, float] = {}
    for memory in memories:
        for name, value in memory.diary_features.items():
            aggregate[name] = aggregate.get(name, 0.0) + value
    items = []
    n = max(len(memories), 1)
    for name, total in aggregate.items():
        mean = total / n
        if mean >= threshold:
            items.append((name, mean))
    items.sort(key=lambda pair: -pair[1])
    return items


def _icon_counter(memories: list[DiaryMemory]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for memory in memories:
        counter.update(memory.icons)
    return counter


def find_mood_runs(memories: list[DiaryMemory], min_length: int = 3) -> list[Reflection]:
    """Detect runs of consecutive days with similar mood signal.

    Peak: 3+ days with mood >= 4 (or strong positive valence proxy).
    Dip: 3+ days with mood <= 2.
    """
    if not memories:
        return []
    reflections: list[Reflection] = []

    def _flush(run: list[DiaryMemory], kind: str) -> None:
        if len(run) < min_length:
            return
        start = run[0].date
        end = run[-1].date
        mood_avg = _mean_mood(run)
        icons = [name for name, _ in _icon_counter(run).most_common(8)]
        moods_string = ", ".join(f"{m.date}=mood{m.mood}" for m in run)
        body_lines = [
            f"Run of {len(run)} consecutive {'high-mood' if kind == 'peak' else 'low-mood'} days.",
            f"Average mood: {mood_avg:.2f}." if mood_avg is not None else "Average mood: n/a.",
            f"Days: {moods_string}",
        ]
        if icons:
            body_lines.append(f"Recurring icons: {', '.join(icons)}.")
        reflections.append(Reflection(
            id=f"reflection:{kind}:{start}:{end}",
            kind=f"mood_{kind}",
            label=("Mood peak streak" if kind == "peak" else "Mood dip streak"),
            start=start,
            end=end,
            text="\n".join(body_lines),
            mood=int(round(mood_avg)) if mood_avg is not None else None,
            icons=icons,
            importance=min(1.0, 0.45 + 0.07 * len(run)),
            source_ids=[m.entry_id for m in run],
        ))

    current_kind: str | None = None
    current_run: list[DiaryMemory] = []
    for memory in memories:
        if memory.mood is None:
            if current_kind is not None:
                _flush(current_run, current_kind)
            current_kind, current_run = None, []
            continue
        kind = "peak" if memory.mood >= 4 else ("dip" if memory.mood <= 2 else None)
        if kind is None:
            if current_kind is not None:
                _flush(current_run, current_kind)
            current_kind, current_run = None, []
            continue
        if kind != current_kind:
            if current_kind is not None:
                _flush(current_run, current_kind)
            current_kind, current_run = kind, [memory]
        else:
            current_run.append(memory)
    if current_kind is not None:
        _flush(current_run, current_kind)
    return reflections


def find_theme_arcs(memories: list[DiaryMemory], min_days: int = 4) -> list[Reflection]:
    """Detect monthly windows where a diary feature dominates."""
    reflections: list[Reflection] = []
    by_month: dict[tuple[int, int], list[DiaryMemory]] = defaultdict(list)
    for memory in memories:
        by_month[(memory.year, memory.month)].append(memory)

    for (year, month), days in sorted(by_month.items()):
        if len(days) < min_days:
            continue
        dominant = _dominant_features(days, threshold=0.22)
        if not dominant:
            continue
        feature, score = dominant[0]
        if feature not in DIARY_FEATURES:
            continue
        start = min(days, key=lambda m: m.ordinal).date
        end = max(days, key=lambda m: m.ordinal).date
        sample_lines: list[str] = []
        for memory in sorted(days, key=lambda m: -memory_signal(m, feature))[:4]:
            first_line = memory.text.strip().split("\n", 1)[0].lstrip("# ")
            sample_lines.append(f"- {memory.date}: {first_line}")
        text = (
            f"In {year}-{month:02d}, the dominant theme was \"{feature}\" "
            f"(intensity {score:.2f} averaged over {len(days)} diary days).\n"
            + "\n".join(sample_lines)
        )
        icons = [name for name, _ in _icon_counter(days).most_common(8)]
        mood_avg = _mean_mood(days)
        reflections.append(Reflection(
            id=f"reflection:theme:{feature}:{year}-{month:02d}",
            kind=f"theme_{feature}",
            label=f"Theme: {feature} in {year}-{month:02d}",
            start=start,
            end=end,
            text=text,
            mood=int(round(mood_avg)) if mood_avg is not None else None,
            icons=icons,
            importance=min(1.0, 0.5 + score * 0.4),
            source_ids=[m.entry_id for m in days],
        ))
    return reflections


def memory_signal(memory: DiaryMemory, feature: str) -> float:
    return float(memory.diary_features.get(feature, 0.0))


def find_contradictions(memories: list[DiaryMemory]) -> list[Reflection]:
    """Find weeks containing both high (>=4) and low (<=2) mood days."""
    reflections: list[Reflection] = []
    by_week: dict[tuple[int, int], list[DiaryMemory]] = defaultdict(list)
    for memory in memories:
        iso = date.fromisoformat(memory.date).isocalendar()
        by_week[(iso.year, iso.week)].append(memory)

    for (iso_year, iso_week), days in sorted(by_week.items()):
        moods = [m for m in days if m.mood is not None]
        if not moods:
            continue
        highs = [m for m in moods if m.mood >= 4]
        lows = [m for m in moods if m.mood <= 2]
        if not highs or not lows:
            continue
        sample_high = max(highs, key=lambda m: m.mood)
        sample_low = min(lows, key=lambda m: m.mood)
        start = min(days, key=lambda m: m.ordinal).date
        end = max(days, key=lambda m: m.ordinal).date
        text = (
            f"Within week {iso_year}-W{iso_week:02d}, mood swung between "
            f"{sample_high.date} (mood {sample_high.mood}) and "
            f"{sample_low.date} (mood {sample_low.mood}).\n"
            f"High point: {sample_high.text.strip().split(chr(10), 1)[0]}\n"
            f"Low point:  {sample_low.text.strip().split(chr(10), 1)[0]}"
        )
        icons = [name for name, _ in _icon_counter(days).most_common(6)]
        reflections.append(Reflection(
            id=f"reflection:contradiction:{iso_year}-W{iso_week:02d}",
            kind="contradiction_week",
            label=f"Mood swing in week {iso_year}-W{iso_week:02d}",
            start=start,
            end=end,
            text=text,
            mood=None,
            icons=icons,
            importance=0.65,
            source_ids=[m.entry_id for m in days],
        ))
    return reflections


def find_identity_patterns(memories: list[DiaryMemory], min_occurrences: int = 12) -> list[Reflection]:
    """Aggregate-level identity patterns: which features recur most across the whole corpus.

    These are the closest thing to "Level 5: identity patterns" that we can produce
    deterministically. An LLM polish pass can later rewrite them as first-person
    observations.
    """
    if not memories:
        return []
    reflections: list[Reflection] = []
    counter: dict[str, list[DiaryMemory]] = defaultdict(list)
    for memory in memories:
        for name, value in memory.diary_features.items():
            if value >= 0.25:
                counter[name].append(memory)
    for name, days in counter.items():
        if len(days) < min_occurrences:
            continue
        days.sort(key=lambda m: m.ordinal)
        start = days[0].date
        end = days[-1].date
        years = sorted({m.year for m in days})
        mood_avg = _mean_mood(days)
        text = (
            f"{name.title()} appears as a recurring life theme across {len(days)} diary days "
            f"spanning {', '.join(str(y) for y in years)}.\n"
            f"Average mood on these days: {mood_avg:.2f}.\n"
            if mood_avg is not None
            else f"{name.title()} is a recurring theme across {len(days)} diary days "
                 f"spanning {', '.join(str(y) for y in years)}.\n"
        )
        text += "Sample days:\n" + "\n".join(
            f"- {m.date}: {m.text.strip().split(chr(10), 1)[0].lstrip('# ')}"
            for m in days[:: max(1, len(days) // 5)][:5]
        )
        icons = [name for name, _ in _icon_counter(days).most_common(8)]
        reflections.append(Reflection(
            id=f"reflection:identity:{name}",
            kind=f"identity_{name}",
            label=f"Recurring theme: {name}",
            start=start,
            end=end,
            text=text,
            mood=int(round(mood_avg)) if mood_avg is not None else None,
            icons=icons,
            importance=min(1.0, 0.55 + len(days) * 0.01),
            source_ids=[m.entry_id for m in days],
        ))
    return reflections


def write_reflections(path: Path, reflections: list[Reflection]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for reflection in reflections:
            handle.write(json.dumps(asdict(reflection), ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile Level-5 reflection tokens from the diary.")
    parser.add_argument("--output", type=Path, default=REFLECTIONS_DEFAULT)
    parser.add_argument("--min-mood-run", type=int, default=3, help="Minimum consecutive days for a mood streak.")
    parser.add_argument("--min-theme-days", type=int, default=4, help="Minimum days in a month for a theme arc.")
    parser.add_argument("--min-identity-occurrences", type=int, default=12, help="Minimum feature occurrences for identity patterns.")
    parser.add_argument("--skip-mood-runs", action="store_true")
    parser.add_argument("--skip-themes", action="store_true")
    parser.add_argument("--skip-contradictions", action="store_true")
    parser.add_argument("--skip-identity", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    memories = parse_diary_memories()
    if not memories:
        raise SystemExit("No diary memories found.")
    print(f"loaded_diary_memories={len(memories)}")

    reflections: list[Reflection] = []
    if not args.skip_mood_runs:
        reflections.extend(find_mood_runs(memories, args.min_mood_run))
    if not args.skip_themes:
        reflections.extend(find_theme_arcs(memories, args.min_theme_days))
    if not args.skip_contradictions:
        reflections.extend(find_contradictions(memories))
    if not args.skip_identity:
        reflections.extend(find_identity_patterns(memories, args.min_identity_occurrences))

    write_reflections(args.output, reflections)
    by_kind: Counter[str] = Counter(r.kind for r in reflections)
    duration = round(time.time() - started, 2)
    print(f"reflections_written={len(reflections)} duration={duration}s -> {args.output}")
    for kind, count in by_kind.most_common():
        print(f"  {kind}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
