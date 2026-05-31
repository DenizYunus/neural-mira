#!/usr/bin/env python3
"""
Generate weak-supervision query examples for MIRA / HTEMA.

Reads diary path + LLM credentials from `.env` via `nm_config`.
Intentionally never prints API keys or full diary entries.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nm_config import (
    DIARY_ROOT,
    LAB_ROOT,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_API_KEY,
    diary_files,
)


DEFAULT_OUTPUT = LAB_ROOT / "data" / "generated_queries.jsonl"
DIARY_FILES = diary_files()


@dataclass
class DiaryEntry:
    entry_id: str
    date: str
    year: int
    source_path: str
    mood: int | None
    icons: list[str]
    text: str


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        values[key.strip()] = value
    return values


def normalize_date(value: str) -> str | None:
    match = re.search(r"(\d{4})[./-](\d{1,2})[./-](\d{1,2})", value)
    if not match:
        return None
    return f"{match.group(1)}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"


def parse_mood(text: str) -> int | None:
    match = re.search(
        r"\*\*Mood\*\*:\s*([1-5])|mood[^0-9]{0,20}([1-5])|(?:^|\s)([1-5])/5(?:\s|$)",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    return int(next(group for group in match.groups() if group))


def parse_icons(text: str) -> list[str]:
    match = re.search(r"\*\*Icons\*\*:\s*([^\n]+)", text, re.IGNORECASE)
    if not match:
        return []
    return [item.strip().lower() for item in match.group(1).split(",") if item.strip()]


def compact_entry(text: str, max_chars: int) -> str:
    cleaned = re.sub(r"\n{3,}", "\n\n", text.strip())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + "\n[truncated]"


def parse_diary_entries(files: list[Path] = DIARY_FILES) -> list[DiaryEntry]:
    entries: list[DiaryEntry] = []
    for file_path in files:
        if not file_path.exists():
            continue
        raw = file_path.read_text(encoding="utf-8")
        parts = re.split(r"(?=^###\s+\d{4}[./-]\d{1,2}[./-]\d{1,2})", raw, flags=re.MULTILINE)
        for part in parts:
            date = normalize_date(part[:120])
            if not date:
                continue
            rel_path = str(file_path.relative_to(LAB_ROOT)) if file_path.is_relative_to(LAB_ROOT) else str(file_path)
            entries.append(
                DiaryEntry(
                    entry_id=f"{rel_path}:{date}",
                    date=date,
                    year=int(date[:4]),
                    source_path=rel_path,
                    mood=parse_mood(part),
                    icons=parse_icons(part),
                    text=part.strip(),
                )
            )
    return sorted(entries, key=lambda item: item.date)


def build_prompt(entry: DiaryEntry, neighbors: list[DiaryEntry], examples_per_entry: int) -> str:
    neighbor_lines = [
        {
            "date": item.date,
            "mood": item.mood,
            "icons": item.icons[:12],
            "short_text": compact_entry(item.text, 600),
        }
        for item in neighbors
    ]

    payload = {
        "target_entry": {
            "entry_id": entry.entry_id,
            "date": entry.date,
            "mood": entry.mood,
            "icons": entry.icons[:16],
            "text": compact_entry(entry.text, 1800),
        },
        "neighbor_entries": neighbor_lines,
        "examples_per_entry": examples_per_entry,
    }

    return f"""
You generate weak-supervision training queries for a diary-native memory retriever called MIRA / HTEMA.

The retriever must learn temporal, emotional, social, and narrative attention over diary memories.

Given one target diary entry and nearby entries, create {examples_per_entry} natural user queries that should retrieve the target entry or its local event window.

Return strict JSON only with this shape:

{{
  "examples": [
    {{
      "query": "natural user question",
      "intent": "one of: factual_recall, emotional_recall, temporal_recall, relationship_context, event_arc, pattern_detection, contradiction_or_counterevidence, unresolved_loop",
      "positive_ids": ["source_path:YYYY-MM-DD"],
      "positive_dates": ["YYYY-MM-DD"],
      "positive_window": ["YYYY-MM-DD", "YYYY-MM-DD"],
      "required_heads": ["semantic", "temporal", "emotion", "entity", "hierarchy", "continuity", "contradiction"],
      "hard_negative_strategy": "short description of confusing negatives",
      "why": "brief reason this query trains HTEMA"
    }}
  ]
}}

Rules:
- Do not mention this prompt, labels, or training data.
- Queries should sound like the diary owner asking their own assistant.
- Include a mix of direct recall and reflective questions.
- Use names, places, emotions, dates, and event arcs only when supported by the entry.
- If the entry is emotionally important, include emotion-focused queries.
- If the neighbor entries form a sequence, include at least one continuity/event-arc query.
- Do not invent unsupported facts.

Diary payload:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()


def build_batch_prompt(batch: list[DiaryEntry], entries: list[DiaryEntry], neighbor_radius: int, examples_per_entry: int) -> str:
    targets = []
    for entry in batch:
        targets.append(
            {
                "entry_id": entry.entry_id,
                "date": entry.date,
                "mood": entry.mood,
                "icons": entry.icons[:14],
                "text": compact_entry(entry.text, 1200),
                "neighbor_entries": [
                    {
                        "entry_id": item.entry_id,
                        "date": item.date,
                        "mood": item.mood,
                        "icons": item.icons[:10],
                        "short_text": compact_entry(item.text, 360),
                    }
                    for item in neighbors_for(entry, entries, neighbor_radius)
                ],
            }
        )

    payload = {
        "targets": targets,
        "examples_per_target": examples_per_entry,
    }

    return f"""
You generate weak-supervision training queries for a diary-native memory retriever called MIRA / HTEMA.

The retriever must learn temporal, emotional, social, and narrative attention over diary memories.

Given multiple target diary entries and nearby entries, create exactly {examples_per_entry} natural user queries for EACH target entry.
Each query should retrieve its target entry or local event window.

Return strict JSON only with this shape:

{{
  "examples": [
    {{
      "target_entry_id": "source_path:YYYY-MM-DD",
      "query": "natural user question",
      "intent": "one of: factual_recall, emotional_recall, temporal_recall, relationship_context, event_arc, pattern_detection, contradiction_or_counterevidence, unresolved_loop",
      "positive_ids": ["source_path:YYYY-MM-DD"],
      "positive_dates": ["YYYY-MM-DD"],
      "positive_window": ["YYYY-MM-DD", "YYYY-MM-DD"],
      "required_heads": ["semantic", "temporal", "emotion", "entity", "hierarchy", "continuity", "contradiction"],
      "hard_negative_strategy": "short description of confusing negatives",
      "why": "brief reason this query trains HTEMA"
    }}
  ]
}}

Rules:
- Return one flat "examples" array.
- Generate exactly {examples_per_entry} examples per target entry.
- Do not mention this prompt, labels, or training data.
- Queries should sound like the diary owner asking their own assistant.
- Include a mix of direct recall and reflective questions.
- Use names, places, emotions, dates, and event arcs only when supported by the target or neighbors.
- If the entry is emotionally important, include emotion-focused queries.
- If the neighbor entries form a sequence, include at least one continuity/event-arc query.
- Do not invent unsupported facts.

Diary payload:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", stripped)
        if not match:
            raise
        return json.loads(match.group(0))


def call_openai_compatible(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    timeout: int,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a careful data generator for personal diary retrieval. "
                    "Return only valid JSON. Do not include markdown."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "top_p": 0.9,
        "max_tokens": max_tokens,
        "stream": False,
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))

    content = data["choices"][0]["message"]["content"]
    return extract_json_object(content)


def select_entries(
    entries: list[DiaryEntry],
    limit: int,
    seed: int,
    year: int | None,
    start_date: str | None,
    end_date: str | None,
) -> list[DiaryEntry]:
    filtered = [
        entry
        for entry in entries
        if (year is None or entry.year == year)
        and (start_date is None or entry.date >= start_date)
        and (end_date is None or entry.date <= end_date)
    ]
    # Prefer entries with stronger metadata, but keep deterministic variety.
    weighted = sorted(
        filtered,
        key=lambda entry: (
            0 if entry.mood is None else 1,
            len(entry.icons),
            len(entry.text),
            entry.date,
        ),
        reverse=True,
    )
    rng = random.Random(seed)
    head = weighted[: max(limit * 4, limit)]
    rng.shuffle(head)
    return sorted(head[:limit], key=lambda entry: entry.date)


def neighbors_for(entry: DiaryEntry, entries: list[DiaryEntry], radius: int) -> list[DiaryEntry]:
    try:
        index = entries.index(entry)
    except ValueError:
        return []
    start = max(0, index - radius)
    end = min(len(entries), index + radius + 1)
    return [item for item in entries[start:end] if item.entry_id != entry.entry_id]


def normalize_example(example: dict[str, Any], entry: DiaryEntry) -> dict[str, Any]:
    positive_ids = example.get("positive_ids") or [entry.entry_id]
    positive_dates = example.get("positive_dates") or [entry.date]
    return {
        "query": str(example.get("query", "")).strip(),
        "intent": str(example.get("intent", "temporal_recall")).strip(),
        "positive_ids": positive_ids,
        "positive_dates": positive_dates,
        "positive_window": example.get("positive_window") or [min(positive_dates), max(positive_dates)],
        "required_heads": example.get("required_heads") or ["semantic", "temporal"],
        "hard_negative_strategy": str(example.get("hard_negative_strategy", "")).strip(),
        "why": str(example.get("why", "")).strip(),
        "target_entry_id": entry.entry_id,
        "target_date": entry.date,
        "target_mood": entry.mood,
        "target_icons": entry.icons,
        "generator": "llm_openai_compatible",
        "created_at": int(time.time()),
    }


def normalize_batch_example(example: dict[str, Any], entries_by_id: dict[str, DiaryEntry]) -> dict[str, Any] | None:
    target_id = str(example.get("target_entry_id") or "").strip()
    entry = entries_by_id.get(target_id)
    if not entry:
        positive_ids = [str(item) for item in example.get("positive_ids") or []]
        entry = next((entries_by_id[item] for item in positive_ids if item in entries_by_id), None)
    if not entry:
        return None
    return normalize_example(example, entry)


def batches(items: list[DiaryEntry], size: int) -> list[list[DiaryEntry]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MIRA / HTEMA query training examples with an OpenAI-compatible LLM.")
    parser.add_argument("--limit", type=int, default=5, help="Number of diary entries to process.")
    parser.add_argument("--examples-per-entry", type=int, default=3, help="Generated query examples per diary entry.")
    parser.add_argument("--year", type=int, default=None, help="Optional diary year filter.")
    parser.add_argument("--start-date", type=str, default=None, help="Optional inclusive date filter, YYYY-MM-DD.")
    parser.add_argument("--end-date", type=str, default=None, help="Optional inclusive date filter, YYYY-MM-DD.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sampling seed.")
    parser.add_argument("--neighbor-radius", type=int, default=2, help="Nearby entries included in the prompt.")
    parser.add_argument("--batch-size", type=int, default=1, help="Diary entries per API call. Use 4-6 for full corpus generation.")
    parser.add_argument("--max-tokens", type=int, default=4800, help="Max completion tokens per API call.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSONL output path.")
    parser.add_argument("--append", action="store_true", help="Append to output instead of overwriting.")
    parser.add_argument("--dry-run", action="store_true", help="Do not call the API; write prompt previews as JSONL.")
    parser.add_argument("--timeout", type=int, default=90, help="API timeout in seconds.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature.")
    parser.add_argument("--env", type=Path, default=None, help="(Deprecated) Path to a .env file. Defaults to neural-mira/.env via nm_config.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # nm_config has already loaded neural-mira/.env at import time; a --env
    # override is honored by re-overlaying its values on top.
    base_url = LLM_BASE_URL
    model = LLM_MODEL
    api_key = LLM_API_KEY
    if args.env:
        overlay = load_env_file(args.env)
        base_url = overlay.get("LLM_BASE_URL", base_url)
        model = overlay.get("LLM_MODEL", model)
        api_key = (
            overlay.get("LLM_API_KEY")
            or overlay.get("DEEPSEEK_API_KEY")
            or overlay.get("NVIDIA_API_KEY")
            or api_key
        )

    entries = parse_diary_entries()
    selected = select_entries(entries, args.limit, args.seed, args.year, args.start_date, args.end_date)

    if not selected:
        print("No diary entries found for the requested filters.", file=sys.stderr)
        return 1

    if not args.dry_run and not api_key:
        print("No LLM_API_KEY, DEEPSEEK_API_KEY, or NVIDIA_API_KEY found in env.", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    entries_by_id = {entry.entry_id: entry for entry in selected}
    work_batches = batches(selected, max(1, args.batch_size))

    mode = "a" if args.append else "w"
    with args.output.open(mode, encoding="utf-8") as handle:
        for index, batch in enumerate(work_batches, start=1):
            if len(batch) == 1:
                entry = batch[0]
                prompt = build_prompt(
                    entry,
                    neighbors_for(entry, entries, args.neighbor_radius),
                    args.examples_per_entry,
                )
            else:
                prompt = build_batch_prompt(batch, entries, args.neighbor_radius, args.examples_per_entry)

            if args.dry_run:
                handle.write(
                    json.dumps(
                        {
                            "target_entry_ids": [entry.entry_id for entry in batch],
                            "target_dates": [entry.date for entry in batch],
                            "prompt_preview": prompt[:2400],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1
                continue

            try:
                result = call_openai_compatible(
                    base_url=base_url,
                    api_key=api_key or "",
                    model=model,
                    prompt=prompt,
                    timeout=args.timeout,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, KeyError) as error:
                print(
                    f"[{index}/{len(work_batches)}] failed {batch[0].date}..{batch[-1].date}: {error}",
                    file=sys.stderr,
                )
                continue

            examples = result.get("examples") or []
            for example in examples:
                normalized = (
                    normalize_example(example, batch[0])
                    if len(batch) == 1
                    else normalize_batch_example(example, entries_by_id)
                )
                if not normalized:
                    continue
                if not normalized["query"]:
                    continue
                handle.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                written += 1

            print(
                f"[{index}/{len(work_batches)}] {batch[0].date}..{batch[-1].date}: wrote {len(examples)} examples"
            )

    print(f"Wrote {written} records to {args.output}")
    print(f"Model: {model}; Base URL: {base_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
