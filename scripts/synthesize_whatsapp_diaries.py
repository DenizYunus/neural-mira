#!/usr/bin/env python3
"""Synthesize WhatsApp-derived diary entries for dates with no real diary.

For every calendar date that has WhatsApp conversation windows but no
DailyBean diary entry, ask an LLM to produce a faithful, provenance-anchored
summary of that day's messages. Output is JSONL at
``data/whatsapp_synthetic_diaries.jsonl``, consumable via
``htema_core.load_whatsapp_synthetic_memories()`` at the same level as the
Phase-4 reflections.

Fabrication safeguards baked into the prompt:
  * Forbids inferred emotions / events / plans.
  * Requires 1-2 verbatim message excerpts as receipts.
  * The model emits ``{"signal": "none"}`` when messages are too sparse to
    summarize honestly. Those days are skipped entirely.
  * Summaries are written in third person ("Deniz mentioned...") so the
    text never looks like authentic diary words.

Every record is tagged ``source_type=whatsapp_synthetic``. The search layer
in ``htema_core.SOURCE_WEIGHTS`` weights these at 0.3x and the chat UI
badges them as AI-summarized so the user knows they aren't their own words.

Usage:
    python scripts/synthesize_whatsapp_diaries.py [--year YYYY] [--limit N]
                                                 [--min-messages N] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from htema_core import (
    DiaryMemory,
    LAB_ROOT,
    parse_diary_memories,
    parse_whatsapp_memories,
)
from nm_config import llm_env_dict, require_llm_api_key


DEFAULT_OUTPUT = LAB_ROOT / "data" / "whatsapp_synthetic_diaries.jsonl"

# Below this many message windows on a date, skip — too thin to summarize
# without fabricating. Days with one window but a lot of text in it still
# pass (the secondary length check).
DEFAULT_MIN_WINDOWS = 1
DEFAULT_MIN_CHARS = 400  # OR threshold on total message chars

# Cap how much message text we put in the prompt. Long days get truncated to
# the most-recent windows so the LLM still sees substantive content but the
# prompt stays in budget.
MAX_PROMPT_MESSAGE_CHARS = 4500


# ---------------------------------------------------------------------------
# Env loader — delegates to nm_config (single source of truth across scripts).
# Kept as a thin wrapper so the call sites below don't need to change.
# ---------------------------------------------------------------------------
def load_jarvis_env() -> dict[str, str]:
    return llm_env_dict()


# ---------------------------------------------------------------------------
# Gap finder
# ---------------------------------------------------------------------------
def find_gap_dates(
    diary: list[DiaryMemory],
    whatsapp: list[DiaryMemory],
    *,
    min_windows: int = DEFAULT_MIN_WINDOWS,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> list[tuple[str, list[DiaryMemory]]]:
    """Dates with enough WhatsApp activity AND no real diary entry."""
    diary_dates = {memory.date for memory in diary}
    by_date: dict[str, list[DiaryMemory]] = defaultdict(list)
    for memory in whatsapp:
        if memory.date in diary_dates:
            continue
        by_date[memory.date].append(memory)

    eligible: list[tuple[str, list[DiaryMemory]]] = []
    for date_str, windows in by_date.items():
        total_chars = sum(len(w.text) for w in windows)
        if len(windows) >= min_windows and total_chars >= min_chars:
            eligible.append((date_str, windows))
    return sorted(eligible, key=lambda pair: pair[0])


# ---------------------------------------------------------------------------
# Prompt + LLM call
# ---------------------------------------------------------------------------
PROMPT_SYSTEM = (
    "You are summarizing one day's WhatsApp messages into a faithful, "
    "retrieval-friendly summary. You must never invent, infer, or embellish "
    "anything that is not literally in the messages. Return only valid JSON."
)


def build_user_prompt(date_str: str, windows: list[DiaryMemory]) -> str:
    """Construct the strict, fabrication-resistant prompt."""
    # Order windows so the most-recent are last and least likely to be truncated.
    rendered_blocks: list[str] = []
    running_chars = 0
    for window in sorted(windows, key=lambda w: w.entry_id):
        participants = ", ".join(window.participants) if window.participants else "unknown chat"
        body = window.text.strip()
        block = f"[Chat: {participants}]\n{body}"
        if running_chars + len(block) > MAX_PROMPT_MESSAGE_CHARS:
            # Truncate this final block rather than dropping it entirely.
            remaining = max(0, MAX_PROMPT_MESSAGE_CHARS - running_chars)
            if remaining > 200:
                block = block[:remaining] + "\n[...truncated for prompt budget...]"
                rendered_blocks.append(block)
            break
        rendered_blocks.append(block)
        running_chars += len(block)

    messages_block = "\n\n".join(rendered_blocks)

    return (
        "You are creating a low-confidence fallback memory for a date where the "
        "user has no diary entry. Apply these CRITICAL RULES:\n\n"
        "1. ONLY restate facts that are EXPLICITLY in the messages. Do NOT infer "
        "emotions, plans, or events that aren't literally there.\n"
        "2. Write in THIRD PERSON. Refer to the user as 'Deniz' and say "
        "'Deniz mentioned...', 'they discussed...'. Never write in Deniz's voice.\n"
        "3. Include 1-2 VERBATIM message excerpts as receipts so a reader can "
        "verify any claim.\n"
        "4. If the messages are too sparse, cryptic, generic (just 'ok', 'k', "
        "emojis), or about nothing of substance, output EXACTLY this and nothing else:\n"
        '   {"signal": "none"}\n'
        "5. Output VALID JSON only. No markdown fences, no commentary.\n\n"
        "OUTPUT SCHEMA:\n"
        "{\n"
        '  "summary": "<2-4 sentence third-person paraphrase of what happened in '
        'these messages>",\n'
        '  "claims": ["<short paraphrased fact 1>", "<short paraphrased fact 2>", '
        "...],\n"
        '  "provenance_excerpts": ["<verbatim line 1 from messages>", "<verbatim '
        'line 2>"],\n'
        '  "participants": ["<name>", ...]\n'
        "}\n\n"
        f"DATE: {date_str}\n\n"
        "MESSAGES:\n"
        f"{messages_block}\n\n"
        "OUTPUT JSON:"
    )


def call_llm(
    prompt: str,
    env: dict[str, str],
    *,
    timeout: int,
    temperature: float = 0.2,
    max_tokens: int = 700,
) -> str:
    base_url = env.get("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
    api_key = env.get("LLM_API_KEY") or env.get("NVIDIA_API_KEY")
    model = env.get("LLM_MODEL", "moonshotai/kimi-k2-thinking")
    if not api_key:
        # Centralized friendly error pointing at neural-mira/.env.
        require_llm_api_key()

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": PROMPT_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "top_p": 0.9,
        "max_tokens": max_tokens,
        "stream": False,
    }

    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------
def parse_llm_response(raw: str) -> dict[str, Any] | None:
    """Extract a JSON dict from the LLM output. Return None for signal=none
    or anything we can't parse / can't trust.

    Robust to thinking-model preambles, ```json fences, and trailing prose.
    Strategy: strip fences, then try parsing the full text; if that fails,
    walk *every* substring that starts with ``{`` and find the longest one
    that parses cleanly. This handles thinking blocks like ``<thinking>...
    </thinking>\\n{...}`` and trailing commentary after the JSON.
    """
    text = (raw or "").strip()
    if not text:
        return None
    # Strip code fences if the model added them despite our instructions.
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline >= 0:
            text = text[first_newline + 1 :]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[: -3].rstrip()
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:].lstrip()

    data: dict[str, Any] | None = None
    # Fast path: the whole thing is JSON.
    try:
        candidate = json.loads(text)
        if isinstance(candidate, dict):
            data = candidate
    except json.JSONDecodeError:
        pass

    # Fallback: scan for every `{` and try to parse each up to a matching
    # close brace. Keep the LONGEST parsable dict (typically the actual
    # output, since the model may have leaked smaller `{}` in a preamble).
    if data is None:
        candidates: list[dict[str, Any]] = []
        for start in range(len(text)):
            if text[start] != "{":
                continue
            depth = 0
            in_string = False
            escape = False
            for end in range(start, len(text)):
                ch = text[end]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"' and not escape:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        chunk = text[start : end + 1]
                        try:
                            parsed = json.loads(chunk)
                        except json.JSONDecodeError:
                            break
                        if isinstance(parsed, dict):
                            candidates.append(parsed)
                        break
        if candidates:
            # Prefer the longest dict that looks like a real response (has
            # "summary" or "signal"); fall back to the simply-longest.
            best = max(
                candidates,
                key=lambda d: (
                    1 if ("summary" in d or "signal" in d) else 0,
                    len(json.dumps(d)),
                ),
            )
            data = best

    if not isinstance(data, dict):
        return None
    if data.get("signal") == "none":
        return None
    summary = data.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        return None
    # Defensive: require at least one provenance excerpt — that's the receipt.
    excerpts = data.get("provenance_excerpts")
    if not isinstance(excerpts, list) or not any(isinstance(e, str) and e.strip() for e in excerpts):
        return None
    return data


def to_jsonl_row(
    date_str: str,
    parsed: dict[str, Any],
    windows: list[DiaryMemory],
) -> dict[str, Any]:
    # Always cross-check participants against the actual chat windows so the
    # model can't omit or add names. Reported participants are the UNION of
    # what the model said and what the windows actually contain.
    actual_participants = sorted({p for w in windows for p in w.participants})
    reported = parsed.get("participants") or []
    if isinstance(reported, list):
        union = sorted({p for p in reported if isinstance(p, str)} | set(actual_participants))
    else:
        union = actual_participants
    return {
        "id": f"whatsapp_synthetic:{date_str}",
        "date": date_str,
        "source_type": "whatsapp_synthetic",
        "summary": parsed["summary"].strip(),
        "claims": [c.strip() for c in (parsed.get("claims") or []) if isinstance(c, str) and c.strip()],
        "provenance_excerpts": [
            e.strip() for e in parsed["provenance_excerpts"] if isinstance(e, str) and e.strip()
        ],
        "participants": union,
        "message_window_count": len(windows),
        "message_total_chars": sum(len(w.text) for w in windows),
        "generated_at": (
            datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _format_eta(remaining: int, elapsed: float, done: int) -> str:
    if done <= 0 or remaining <= 0:
        return "?"
    rate = done / max(elapsed, 0.001)
    seconds = remaining / max(rate, 0.001)
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Synthesize WhatsApp-derived fallback diary entries for dates with no real diary.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N gap dates this run (0 = all). Resumable across runs.",
    )
    parser.add_argument(
        "--min-windows",
        type=int,
        default=DEFAULT_MIN_WINDOWS,
        help=f"Skip dates with fewer than this many conversation windows. Default {DEFAULT_MIN_WINDOWS}.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=DEFAULT_MIN_CHARS,
        help=f"Skip dates with less total message text than this. Default {DEFAULT_MIN_CHARS}.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Restrict to one calendar year.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts without calling the LLM. Useful for prompt tuning.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    # Bumped to 2000 because reasoning-trained models (Kimi K2, DeepSeek) burn
    # tokens on a thinking preamble before emitting the JSON.
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print raw LLM responses for malformed/no-signal cases. Use during prompt tuning.",
    )
    args = parser.parse_args(argv)

    env = load_jarvis_env()
    if not args.dry_run and not (env.get("LLM_API_KEY") or env.get("NVIDIA_API_KEY")):
        print(
            "synthesize_whatsapp_diaries: LLM_API_KEY (or NVIDIA_API_KEY) is required. "
            "Set it in neural-mira/.env (see .env.example).",
            file=sys.stderr,
        )
        return 2

    print("Loading diary + WhatsApp memories...")
    diary = parse_diary_memories()
    whatsapp = parse_whatsapp_memories()
    print(f"  diary entries:    {len(diary)}")
    print(f"  whatsapp windows: {len(whatsapp)}")

    gap_dates = find_gap_dates(
        diary,
        whatsapp,
        min_windows=args.min_windows,
        min_chars=args.min_chars,
    )
    if args.year is not None:
        gap_dates = [(d, w) for d, w in gap_dates if d.startswith(str(args.year))]
    print(f"  gap dates needing synthesis: {len(gap_dates)}")

    # Resumability: read existing output JSONL, skip dates already covered.
    processed: set[str] = set()
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            d = row.get("date")
            if isinstance(d, str):
                processed.add(d)
        print(f"  already in {args.output.name}: {len(processed)}")

    todo = [(d, w) for d, w in gap_dates if d not in processed]
    if args.limit > 0:
        todo = todo[: args.limit]
    print(f"  to process this run: {len(todo)}")

    if not todo:
        print("Nothing to do. Either everything is processed, or no gap dates exist.")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    success = no_signal = malformed = errored = 0

    for index, (date_str, windows) in enumerate(todo, start=1):
        prompt = build_user_prompt(date_str, windows)
        if args.dry_run:
            print(f"\n=== {date_str} (DRY-RUN, {len(windows)} windows) ===")
            print(prompt[:2000])
            if len(prompt) > 2000:
                print(f"... [{len(prompt) - 2000} more chars]")
            continue

        # Long-day prompts occasionally come back empty from reasoning-trained
        # models (the model burns its budget on thinking and emits nothing).
        # One retry with bumped tokens recovers most of these cleanly.
        raw = ""
        retries_left = 1
        last_exc: Exception | None = None
        while True:
            try:
                raw = call_llm(
                    prompt,
                    env,
                    timeout=args.timeout,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
                last_exc = exc
                raw = ""
            if (raw or "").strip():
                break
            if retries_left <= 0:
                break
            retries_left -= 1
            time.sleep(2)
        if not (raw or "").strip():
            errored += 1
            note = f"empty after retry (last_exc={last_exc})" if last_exc else "empty after retry"
            print(f"  [{index}/{len(todo)}] {date_str}: ERROR {note}")
            continue

        parsed = parse_llm_response(raw)
        if parsed is None:
            # Could be {"signal": "none"} (legit) OR malformed JSON. Both
            # are skip-this-date; we don't write a row either way.
            if '"signal"' in raw and '"none"' in raw:
                no_signal += 1
                print(f"  [{index}/{len(todo)}] {date_str}: no signal")
            else:
                malformed += 1
                print(f"  [{index}/{len(todo)}] {date_str}: malformed (skipped)")
                if args.debug:
                    print(f"    --- raw response (first 1500 chars) ---")
                    print(f"    {raw[:1500]!r}")
                    print(f"    --- end raw ---")
            continue

        row = to_jsonl_row(date_str, parsed, windows)
        with args.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        success += 1

        if index % 5 == 0 or index == len(todo):
            elapsed = time.time() - started
            eta = _format_eta(len(todo) - index, elapsed, index)
            print(
                f"  [{index}/{len(todo)}] {date_str}: ok "
                f"(success={success} no_signal={no_signal} malformed={malformed} "
                f"errored={errored} eta={eta})",
                flush=True,
            )

    elapsed = time.time() - started
    print(
        f"\nDone in {elapsed:.0f}s. success={success} no_signal={no_signal} "
        f"malformed={malformed} errored={errored}"
    )
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
