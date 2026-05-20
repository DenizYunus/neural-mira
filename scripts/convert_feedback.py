#!/usr/bin/env python3
"""Convert Jarvis feedback into MIRA eval examples.

Pipeline:
  jarvis-platform/data/feedback.jsonl
      ↓ (this script)
  memory-attention-lab/data/feedback_eval_examples.jsonl
      ↓ (--extra-examples)
  scripts/honest_mira.py

Rules — preserve learning signal without amplifying retrieval bugs:

- rating=useful
    Treat the displayed source ids/paths as positives. Keep the query text and
    record the original retrieval mode as `note`.

- rating=wrong | needs_work
    The displayed sources are NOT positives — they're what the system already
    surfaced and the user rejected. We emit them as hard negatives instead,
    via `negative_ids`. If the feedback `note` contains a YYYY-MM-DD date or
    references a known memory id, we use that as the positive. Otherwise we
    skip the row (a wrong example with no recoverable target adds noise).

This implements the protocol the README has been asking for and that the
existing `feedbackStore.mjs` already supports on the capture side.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from htema_core import LAB_ROOT, parse_all_memories


REPO_ROOT = LAB_ROOT.parent
DEFAULT_FEEDBACK = REPO_ROOT / "jarvis-platform" / "data" / "feedback.jsonl"
DEFAULT_OUTPUT = LAB_ROOT / "data" / "feedback_eval_examples.jsonl"

DATE_RE = re.compile(r"(20\d{2})[./-](\d{1,2})[./-](\d{1,2})")
ID_RE = re.compile(r"(whatsapp:[^\s,]+|atom:[^\s,]+|rollup:[^\s,]+|reflection:[^\s,]+|[a-zA-Z0-9_/\\.-]+:20\d{2}-\d{2}-\d{2})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-empty", action="store_true",
                        help="Include rows without recoverable positives (useful for inspection only).")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _normalize_date(value: str) -> str | None:
    match = DATE_RE.search(value)
    if not match:
        return None
    return f"{match.group(1)}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"


def _ids_from_sources(sources: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        sid = source.get("id") or source.get("entry_id")
        if sid:
            out.append(str(sid))
            continue
        path = source.get("path")
        locator = source.get("locator")
        if path and locator:
            out.append(f"{path}:{locator}")
    return list(dict.fromkeys(out))  # de-dup keeping order


def _positives_from_note(note: str, memories_by_id: dict[str, Any], memories_by_date: dict[str, list[Any]]) -> list[str]:
    if not note:
        return []
    found: list[str] = []
    for match in ID_RE.finditer(note):
        candidate = match.group(0)
        if candidate in memories_by_id:
            found.append(candidate)
    date_value = _normalize_date(note)
    if date_value and date_value in memories_by_date:
        # Prefer the diary entry for that date if there is one, else fall back to all matches.
        diary_match = [m.entry_id for m in memories_by_date[date_value] if getattr(m, "source_type", "diary") == "diary"]
        found.extend(diary_match or [m.entry_id for m in memories_by_date[date_value]])
    return list(dict.fromkeys(found))


def convert(rows: list[dict[str, Any]], memories) -> tuple[list[dict[str, Any]], dict[str, int]]:
    memories_by_id = {m.entry_id: m for m in memories}
    memories_by_date: dict[str, list[Any]] = {}
    for memory in memories:
        memories_by_date.setdefault(memory.date, []).append(memory)

    examples: list[dict[str, Any]] = []
    stats = {"total": 0, "kept": 0, "skipped_no_query": 0, "skipped_no_target": 0, "useful": 0, "wrong": 0, "needs_work": 0}

    for row in rows:
        stats["total"] += 1
        rating = str(row.get("rating") or "").strip()
        query = str(row.get("query") or "").strip()
        if not query:
            stats["skipped_no_query"] += 1
            continue

        source_ids = _ids_from_sources(row.get("sources") or [])
        note = str(row.get("note") or "").strip()

        if rating == "useful":
            stats["useful"] += 1
            positives = [sid for sid in source_ids if sid in memories_by_id]
            if not positives:
                # Try note recovery as a last resort.
                positives = _positives_from_note(note, memories_by_id, memories_by_date)
            if not positives:
                stats["skipped_no_target"] += 1
                continue
            examples.append({
                "query": query,
                "positive_ids": positives,
                "intent": "real_feedback_useful",
                "style": "real_feedback",
                "note": note or "user marked answer useful; sources kept as positives",
                "source": "jarvis_feedback",
                "rating": rating,
            })
            stats["kept"] += 1
            continue

        if rating in {"wrong", "needs_work"}:
            stats[rating] += 1
            recovered = _positives_from_note(note, memories_by_id, memories_by_date)
            if not recovered:
                # Per protocol: don't fabricate positives. Skip unless explicitly opted in.
                stats["skipped_no_target"] += 1
                continue
            negatives = [sid for sid in source_ids if sid in memories_by_id and sid not in recovered]
            examples.append({
                "query": query,
                "positive_ids": recovered,
                "negative_ids": negatives,
                "intent": f"real_feedback_{rating}",
                "style": "real_feedback",
                "note": note,
                "source": "jarvis_feedback",
                "rating": rating,
            })
            stats["kept"] += 1
            continue

        # Unknown rating; ignore.

    return examples, stats


def main() -> int:
    args = parse_args()
    if not args.feedback.exists():
        print(f"No feedback file at {args.feedback}; nothing to convert.")
        return 0

    rows = []
    with args.feedback.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                print(f"warning: skipped invalid JSON at {args.feedback}:{line_number}: {exc}")

    if not rows:
        print("Feedback file is empty.")
        return 0

    memories = parse_all_memories()
    examples, stats = convert(rows, memories)

    if args.dry_run:
        print(json.dumps({"stats": stats, "preview": examples[:3]}, indent=2, ensure_ascii=False))
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"wrote={args.output} examples={len(examples)} stats={stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
