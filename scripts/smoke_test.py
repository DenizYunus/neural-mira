#!/usr/bin/env python3
"""Hermetic smoke test — no network, no API key, no heavy deps.

Exercises the core pure-Python paths so CI can prove the repo isn't broken
on every push:

  1. fetch_demo_data.parse_entries  — date-header parsing + TOC filtering
  2. htema_core.parse_diary_memories — markdown -> DiaryMemory objects
  3. htema_core.build_query_spec + score_memories — scalar HTEMA ranking

All inputs are tiny in-memory fixtures. Runs in well under a second with
only the Python standard library (htema_core has no third-party imports;
the scalar search path needs neither numpy nor torch).

Exit code 0 = all checks passed, 1 = a check failed.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))


FIXTURE_DIARY = """\
### 2024-01-05
**Mood**: 4
**Icons**: writing, focus, train, coffee

Sharp focus all morning — got the whole chapter outline done in one sitting,
which never happens. Coffee on the train, watched the rain.

### 2024-01-06
**Mood**: 2
**Icons**: argument, mother, tired

Slow, heavy day. Argued with Mum about the move again. Couldn't shake the
feeling I'd let everyone down. Went to bed early.

### 2024-02-14
**Mood**: 5
**Icons**: love, celebration, dinner

Best evening in months. Dinner out, candles, felt genuinely happy for the
first time in a while.
"""

# Raw text mimicking the PDF-extracted diary shape: a table-of-contents block
# (dot leaders) followed by real dated prose. parse_entries must drop the TOC
# rows and keep only the prose entries.
FIXTURE_RAW = """\
Contents:
Saturday, 13 June 1942 ....................................................... 7
Sunday, 21 June 1942 ......................................................... 9

Saturday, 13 June 1942
On Friday it was my birthday and I woke up early. I got lots of presents and
the nicest one of all was my diary. I am so happy to finally have somewhere to
write down everything I am thinking and feeling each day.

Sunday, 21 June 1942
Everyone at school is nervous about who will move up a class. My teachers
mostly like me, though Mr Keesing is cross because I talk too much in lessons.
"""


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    line = f"  [{status}] {label}"
    if detail:
        line += f" — {detail}"
    print(line)
    if not condition:
        check.failed = True  # type: ignore[attr-defined]


check.failed = False  # type: ignore[attr-defined]


def test_fetch_parse_entries() -> None:
    print("fetch_demo_data.parse_entries (TOC filtering + date parse)")
    from fetch_demo_data import parse_entries

    entries = parse_entries(FIXTURE_RAW)
    dates = [d for d, _ in entries]
    check("parsed exactly the 2 real entries (TOC rows dropped)", len(entries) == 2,
          f"got {len(entries)}: {dates}")
    check("dates normalized to ISO", dates == ["1942-06-13", "1942-06-21"],
          f"got {dates}")
    if entries:
        first_body = entries[0][1]
        check("body is prose, not dot-leaders", "birthday" in first_body.lower(),
              first_body[:40])


def test_parse_diary_memories() -> None:
    print("htema_core.parse_diary_memories (markdown -> DiaryMemory)")
    from htema_core import parse_diary_memories

    with tempfile.TemporaryDirectory() as tmp:
        fixture = Path(tmp) / "fixture.md"
        fixture.write_text(FIXTURE_DIARY, encoding="utf-8")
        memories = parse_diary_memories(files=[fixture])

    check("parsed all 3 entries", len(memories) == 3, f"got {len(memories)}")
    if len(memories) == 3:
        m0 = memories[0]
        check("first entry date correct", m0.date == "2024-01-05", m0.date)
        check("mood parsed", m0.mood == 4, str(m0.mood))
        check("icons parsed", "writing" in m0.icons, str(m0.icons))
        check("text body captured", len(m0.text) > 20, f"{len(m0.text)} chars")
        check("direct diary trust profile attached", m0.evidence_type == "direct", str(m0.trust_payload()))
        check("trust provenance points at source", bool(m0.provenance_ids), str(m0.provenance_ids))
        check("entries sorted by date", [m.date for m in memories] ==
              sorted(m.date for m in memories))


def test_scalar_scoring() -> None:
    print("htema_core.build_query_spec + score_memories (scalar HTEMA)")
    from htema_core import build_query_spec, parse_diary_memories, score_memories

    default_weights = {
        "bias": 0.0, "semantic": 1.0, "temporal": 1.0, "emotion": 1.0,
        "entity": 1.0, "hierarchy": 0.5, "continuity": 0.8,
        "importance": 0.35, "unresolved": 0.25, "contradiction": 0.25,
    }
    with tempfile.TemporaryDirectory() as tmp:
        fixture = Path(tmp) / "fixture.md"
        fixture.write_text(FIXTURE_DIARY, encoding="utf-8")
        memories = parse_diary_memories(files=[fixture])

    query = build_query_spec("a happy celebration with dinner")
    ranked = score_memories(query, memories, default_weights)
    check("scorer returned a ranking", len(ranked) == 3, f"got {len(ranked)}")
    if ranked:
        top = ranked[0]
        check("ranked item has a score", "score" in top, str(list(top.keys()))[:60])
        check("ranked item has per-head features", "features" in top)
        distinct_scores = len({round(r["score"], 6) for r in ranked})
        check("scoring produced distinct scores", distinct_scores > 1,
              f"{distinct_scores} distinct scores across {len(ranked)} results")


def test_parse_photo_metadata_memories() -> None:
    print("htema_core.parse_photo_metadata_memories (JSONL -> metadata evidence)")
    from htema_core import parse_photo_metadata_memories

    with tempfile.TemporaryDirectory() as tmp:
        fixture = Path(tmp) / "photos.jsonl"
        row = {
            "photo_id": "photo:2024-01-07",
            "date": "2024-01-07",
            "caption": "Notebook on a cafe table beside two coffee cups.",
            "people": ["Maya", "Sara"],
            "location": "Karga Cafe",
        }
        fixture.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
        memories = parse_photo_metadata_memories(fixture)

    check("parsed one photo metadata row", len(memories) == 1, f"got {len(memories)}")
    if memories:
        memory = memories[0]
        check("photo source type attached", memory.source_type == "photo_metadata", memory.source_type)
        check("photo evidence has provenance", bool(memory.provenance_ids), str(memory.provenance_ids))
        check("photo participants parsed", "Maya" in memory.participants, str(memory.participants))


def main() -> int:
    print("=" * 60)
    print("Neural MIRA smoke test (hermetic, stdlib-only)")
    print("=" * 60)
    test_fetch_parse_entries()
    test_parse_diary_memories()
    test_parse_photo_metadata_memories()
    test_scalar_scoring()
    print("=" * 60)
    if check.failed:  # type: ignore[attr-defined]
        print("RESULT: FAILED")
        return 1
    print("RESULT: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
