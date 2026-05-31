#!/usr/bin/env python3
"""Bootstrap the Anne Frank demo dataset for Neural MIRA.

Pipeline:
    1. Download Anne Frank's diary plain text from Internet Archive (cached).
    2. Parse entries by their original date headers ("Sunday, 14 June, 1942").
    3. For each entry, call the configured LLM (.env: LLM_*) to infer:
         - mood (1-5)
         - icons (lowercase string tags for themes, people, places, activities)
    4. Write to data/diaries/anne_frank.md in the format scripts/htema_core.py
       parses (### YYYY-MM-DD headers + **Mood**/**Icons** metadata lines).

Usage:
    python scripts/fetch_demo_data.py                  # full diary (~270 entries)
    python scripts/fetch_demo_data.py --max 20         # quick sanity test
    python scripts/fetch_demo_data.py --no-llm         # date + body only, no mood/tags
    python scripts/fetch_demo_data.py --no-resume      # overwrite instead of resuming

The script is resumable by default: if data/diaries/anne_frank.md already
contains N entries, only the remaining (entries.length - N) entries are
processed. Output is flushed per entry, so Ctrl-C is safe.

Source attribution:
    Anne Frank, 'The Diary of a Young Girl' — Internet Archive OCR text.
    Public domain in the EU since 2016 (70 years after Anne Frank's death
    in 1945) and in Australia since 1995. US users should verify their
    local copyright status; the script accepts any plain-text URL via
    --source-url, so you can point it at a different source you have rights to.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from nm_config import (
    DIARY_ROOT,
    LAB_ROOT,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_TIMEOUT_MS,
    require_llm_api_key,
)


DEFAULT_SOURCE_URL = (
    "https://archive.org/stream/in.ernet.dli.2015.201940/"
    "2015.201940.Anne-Frank_djvu.txt"
)
CACHE_PATH = LAB_ROOT / "data" / ".cache" / "anne_frank_raw.txt"
OUTPUT_PATH = DIARY_ROOT / "anne_frank.md"

MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

# Anne Frank's date headers look like "Sunday, 14 June, 1942" or sometimes
# "Saturday, 20 June, 1942." with trailing punctuation. The day-of-week
# prefix is sometimes missing in OCR'd copies, so we make it optional.
ENTRY_HEADER_RE = re.compile(
    r"(?:^|\n)\s*"
    r"(?:Sunday|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday)?\s*,?\s*"
    r"(\d{1,2})\s+"
    r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s*,?\s*"
    r"(\d{4})",
    re.IGNORECASE,
)


# --- Step 1: download (with on-disk cache) -----------------------------
def download_diary(url: str, cache: Path) -> str:
    """Fetch once, reuse forever. Cache lives outside data/diaries so the
    parser doesn't see it as a diary file."""
    if cache.exists():
        print(f"Using cached diary text at {cache}", file=sys.stderr)
        return cache.read_text(encoding="utf-8", errors="replace")

    print(f"Downloading diary text from {url}", file=sys.stderr)
    cache.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url, headers={"User-Agent": "neural-mira/1.0 fetch_demo_data"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw_bytes = resp.read()
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Failed to download diary from {url}: {error}\n"
            f"Try --source-url <different_url> if the source has moved."
        ) from error

    text = raw_bytes.decode("utf-8", errors="replace")
    cache.write_text(text, encoding="utf-8")
    print(f"Cached {len(text):,} bytes to {cache}", file=sys.stderr)
    return text


# --- Step 2: parse into (date, body) pairs -----------------------------
def parse_entries(raw: str) -> list[tuple[str, str]]:
    """Split the raw OCR text into (ISO date, body) pairs.

    Stops looking for headers once we hit the matching position list —
    body[i] runs from the end of header[i] to the start of header[i+1].
    Entries shorter than ~30 chars are dropped (usually OCR noise that
    matched the header regex by accident).
    """
    matches = list(ENTRY_HEADER_RE.finditer(raw))
    entries: list[tuple[str, str]] = []
    for i, match in enumerate(matches):
        day = int(match.group(1))
        month = MONTH_NAMES[match.group(2).lower()]
        year = int(match.group(3))
        # Sanity check: Anne Frank's diary spans 1942-1944. Anything outside
        # that range is almost certainly an OCR misread of another date
        # mentioned in the body text.
        if year < 1942 or year > 1944:
            continue
        date_iso = f"{year:04d}-{month:02d}-{day:02d}"

        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        body = raw[start:end].strip()

        # Strip the "Dear Kitty," opener that prefaces almost every entry —
        # it's noise for retrieval and just adds 12 tokens per entry.
        body = re.sub(
            r"^\s*Dear\s+Kitty\s*,?\s*\n?", "", body, count=1, flags=re.IGNORECASE
        )
        # Collapse multiple blank lines so the markdown output stays readable.
        body = re.sub(r"\n{3,}", "\n\n", body).strip()

        if len(body) < 30:
            continue
        entries.append((date_iso, body))
    return entries


# --- Step 3: LLM enrichment (one call per entry) -----------------------
LLM_PROMPT_TEMPLATE = """\
You are tagging one entry from Anne Frank's diary 'The Diary of a Young Girl' \
to enable later semantic retrieval.

Entry date: {date}

Entry text:
{body}

Return ONLY valid JSON (no prose, no markdown fences) with these fields:

- "mood": integer 1 to 5, where:
    1 = deeply sad, scared, hopeless
    2 = anxious, lonely, melancholic
    3 = neutral, reflective, mixed
    4 = positive, content, hopeful
    5 = very happy, joyful, full of hope

- "icons": list of 4 to 8 lowercase short string tags capturing what this \
entry is ABOUT. Mix:
    - PEOPLE mentioned by name (e.g. "kitty", "peter", "margot", "mother", \
"father", "mrs_van_daan", "mr_dussel")
    - PLACES (e.g. "secret_annex", "amsterdam", "school", "outdoors")
    - THEMES (e.g. "fear", "hope", "boredom", "love", "loneliness", \
"war_news", "growing_up", "writing", "argument")
    - ACTIVITIES (e.g. "reading", "studying", "cooking", "listening_radio", \
"birthday", "celebration")

Use underscores for multi-word tags. Keep them lowercase. Be specific where \
possible.

Output JSON shape:
{{"mood": 4, "icons": ["birthday", "kitty", "presents", "family", "hopeful", "school"]}}\
"""


def call_llm(
    prompt: str,
    *,
    base_url: str,
    model: str,
    api_key: str,
    timeout_s: int,
) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a diary-tagging assistant. Return only valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 400,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    content = payload["choices"][0]["message"]["content"].strip()
    # Some models wrap JSON in ```json ... ``` fences despite the system prompt.
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*\n?", "", content)
        content = re.sub(r"\n?\s*```\s*$", "", content)
    return json.loads(content)


def enrich_entry(
    date: str,
    body: str,
    *,
    base_url: str,
    model: str,
    api_key: str,
    timeout_s: int,
) -> dict[str, Any]:
    """LLM-extract mood + icons. Retries 3x with exponential backoff."""
    # Anne Frank's later entries get long (>5000 chars). Truncate the prompt
    # body to keep the call cheap + fast — the tag set is robust to truncation
    # because the entry's primary themes are usually in the opening paragraphs.
    body_clip = body[:3000] + ("\n\n[... entry continues ...]" if len(body) > 3000 else "")
    prompt = LLM_PROMPT_TEMPLATE.format(date=date, body=body_clip)

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            raw = call_llm(
                prompt,
                base_url=base_url,
                model=model,
                api_key=api_key,
                timeout_s=timeout_s,
            )
            mood = int(raw.get("mood", 3))
            mood = max(1, min(5, mood))
            icons_raw = raw.get("icons", [])
            if isinstance(icons_raw, str):
                icons_raw = [s.strip() for s in icons_raw.split(",")]
            icons = [
                str(item).strip().lower().replace(" ", "_")
                for item in icons_raw
                if item
            ][:10]
            return {"mood": mood, "icons": icons}
        except Exception as error:
            last_error = error
            wait_s = 2**attempt
            print(
                f"    LLM call failed (attempt {attempt + 1}/3): {error}. "
                f"Retrying in {wait_s}s...",
                file=sys.stderr,
            )
            time.sleep(wait_s)
    raise RuntimeError(f"LLM enrichment failed after 3 retries: {last_error}")


# --- Step 4: write entry in the format htema_core.py parses ------------
def format_entry(date: str, body: str, mood: int | None, icons: list[str]) -> str:
    lines = [f"### {date}"]
    if mood is not None:
        lines.append(f"**Mood**: {mood}")
    if icons:
        lines.append(f"**Icons**: {', '.join(icons)}")
    lines.append("")
    lines.append(body)
    lines.append("")
    lines.append("")  # trailing blank for readability
    return "\n".join(lines)


def load_done_dates(output_path: Path) -> set[str]:
    """Read existing output to support resume — match the ### YYYY-MM-DD header."""
    if not output_path.exists():
        return set()
    found: set[str] = set()
    pattern = re.compile(r"^###\s+(\d{4}-\d{2}-\d{2})", re.MULTILINE)
    for match in pattern.finditer(output_path.read_text(encoding="utf-8")):
        found.add(match.group(1))
    return found


def write_header(out, source_url: str) -> None:
    out.write("<!-- Anne Frank, 'The Diary of a Young Girl'.\n")
    out.write(f"     Source: {source_url}\n")
    out.write("     Generated by scripts/fetch_demo_data.py — do not edit by hand.\n")
    out.write("     Mood and Icons inferred by LLM per entry. -->\n\n")


# --- main ---------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch Anne Frank's diary and convert to the DailyBean-style "
            "format Neural MIRA expects. Default behavior: LLM-enrich mood "
            "and icons per entry. Resumable; safe to Ctrl-C."
        )
    )
    parser.add_argument(
        "--source-url",
        default=DEFAULT_SOURCE_URL,
        help=f"Plain-text diary URL (default: {DEFAULT_SOURCE_URL})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Output markdown path (default: {OUTPUT_PATH})",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=0,
        help="Process at most N entries (0 = all). Useful for sanity tests.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM enrichment — write date + body only, no mood/icons.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Overwrite output instead of resuming from existing entries.",
    )
    args = parser.parse_args(argv)

    # Hard fail early if LLM is required and key is missing — better than
    # downloading 280 entries and dying on the first API call.
    if not args.no_llm:
        require_llm_api_key()

    # Pipeline ---------------------------------------------------------
    raw = download_diary(args.source_url, CACHE_PATH)
    entries = parse_entries(raw)
    print(f"Parsed {len(entries)} entries from raw text.", file=sys.stderr)
    if not entries:
        print(
            "ERROR: zero entries parsed. The source URL may not be the expected "
            "format, or the date-header regex needs adjustment. Inspect the "
            f"cached file at {CACHE_PATH} to debug.",
            file=sys.stderr,
        )
        return 2

    if args.max > 0:
        entries = entries[: args.max]
        print(f"Limited to first {len(entries)} entries (--max).", file=sys.stderr)

    done = set() if args.no_resume else load_done_dates(args.output)
    if done:
        print(
            f"Resuming: {len(done)} entries already in {args.output}.",
            file=sys.stderr,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    is_fresh = args.no_resume or not args.output.exists() or args.output.stat().st_size == 0
    mode = "w" if args.no_resume else "a"
    timeout_s = max(15, LLM_TIMEOUT_MS // 1000)

    processed = 0
    skipped = 0
    failed = 0
    with args.output.open(mode, encoding="utf-8") as out:
        if is_fresh:
            write_header(out, args.source_url)

        for i, (date, body) in enumerate(entries, start=1):
            if date in done:
                skipped += 1
                continue
            try:
                if args.no_llm:
                    mood = None
                    icons: list[str] = []
                    print(
                        f"[{i}/{len(entries)}] {date}  (no-llm mode)",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"[{i}/{len(entries)}] {date}  enriching with LLM…",
                        file=sys.stderr,
                    )
                    meta = enrich_entry(
                        date,
                        body,
                        base_url=LLM_BASE_URL,
                        model=LLM_MODEL,
                        api_key=LLM_API_KEY,
                        timeout_s=timeout_s,
                    )
                    mood = meta["mood"]
                    icons = meta["icons"]

                out.write(format_entry(date, body, mood, icons))
                out.flush()
                processed += 1
            except Exception as error:
                failed += 1
                print(
                    f"    ! Skipped {date} after error: {error}", file=sys.stderr
                )

    print(
        f"\nDone. processed={processed} skipped={skipped} failed={failed}\n"
        f"Output: {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
