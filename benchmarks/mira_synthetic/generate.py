#!/usr/bin/env python3
"""Generate a public synthetic MIRA benchmark corpus."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "generated"
CREATED_AT = "2026-01-01T00:00:00Z"


@dataclass(frozen=True)
class Event:
    date: str
    title: str
    mood: int
    icons: tuple[str, ...]
    people: tuple[str, ...]
    location: str
    truth: str
    diary: str
    chat: tuple[tuple[str, str, str], ...]
    photo_caption: str
    contradiction: str = ""
    correction: str = ""
    target: bool = False


EVENTS: tuple[Event, ...] = (
    Event(
        date="2026-01-01",
        title="New year reset",
        mood=4,
        icons=("planning", "family", "hope"),
        people=("Maya", "Nora"),
        location="home",
        truth="Maya decided to reset her project schedule after a calm family breakfast.",
        diary="I started the year quietly at home with Nora. We made tea, cleaned the kitchen, and I wrote a new project schedule on yellow paper. I felt hopeful, but I also promised myself not to turn January into another punishment month.",
        chat=(
            ("09:12:00", "Nora", "Happy new year. The yellow plan on your fridge looks intense but kind of beautiful."),
            ("09:16:00", "Maya", "It is intense, but today it feels hopeful instead of scary. I want structure without punishing myself."),
            ("09:22:00", "Nora", "Then keep the breakfast rule. No planning before tea."),
        ),
        photo_caption="Yellow project schedule on a fridge next to two tea glasses.",
    ),
    Event(
        date="2026-01-02",
        title="Missed rehearsal",
        mood=2,
        icons=("music", "guilt", "rain"),
        people=("Maya", "Leo"),
        location="metro station",
        truth="Maya missed a music rehearsal because the metro stopped during heavy rain.",
        diary="The metro froze between stations and I missed Leo's rehearsal. I felt embarrassed even though it was not my fault. The rain made everything slower and my apology sounded smaller than I wanted.",
        chat=(
            ("18:04:00", "Maya", "I am still stuck near Kadikoy. The rehearsal is basically gone for me."),
            ("18:06:00", "Leo", "Don't spiral. The rain broke half the city. We can redo vocals tomorrow."),
            ("18:12:00", "Maya", "I hate being the unreliable one. I had the harmony ready."),
        ),
        photo_caption="Blurry metro platform sign, wet coat sleeve, guitar case strap.",
        target=True,
    ),
    Event(
        date="2026-01-03",
        title="Coffee shop focus",
        mood=5,
        icons=("writing", "coffee", "focus"),
        people=("Maya", "Sara"),
        location="Karga Cafe",
        truth="Maya finished the first draft of her paper outline at Karga Cafe with Sara nearby.",
        diary="I worked at Karga Cafe and finally finished the first real outline of the paper. Sara sat across from me reading quietly. For once my concentration felt clean, not desperate.",
        chat=(
            ("13:35:00", "Sara", "You looked locked in for two hours. Did the outline finally click?"),
            ("13:38:00", "Maya", "Yes. The trust hierarchy section makes sense now: diary first, chat as evidence, summaries as lower-confidence."),
            ("13:41:00", "Sara", "That is the sentence. Keep it."),
        ),
        photo_caption="Laptop with outline headings: trust hierarchy, provenance, reconstruction.",
        target=True,
    ),
    Event(
        date="2026-01-04",
        title="Argument about Berlin",
        mood=1,
        icons=("argument", "travel", "stress"),
        people=("Maya", "Nora"),
        location="home",
        truth="Maya argued with Nora about whether she should accept a Berlin workshop invitation.",
        diary="Nora and I argued about Berlin. She thought I was refusing the workshop because I was afraid, and that hurt because it was partly true. I said money was the reason, but fear was louder.",
        chat=(
            ("21:20:00", "Nora", "I pushed too hard about Berlin. Sorry. I don't think you are weak."),
            ("21:24:00", "Maya", "I know. I was defensive because I am scared of going alone and calling it budget logic."),
            ("21:30:00", "Nora", "Then call it both: budget and fear. Not a failure."),
        ),
        photo_caption="Workshop invitation email open on a phone, suitcase visible in the corner.",
        contradiction="Maya first told Leo the decision was only about money, but later admitted fear was part of it.",
        target=True,
    ),
    Event(
        date="2026-01-05",
        title="Doctor appointment",
        mood=3,
        icons=("health", "waiting", "relief"),
        people=("Maya", "Dr. Irem"),
        location="clinic",
        truth="Maya's wrist pain was diagnosed as strain, not a fracture.",
        diary="The clinic was slow, but the X-ray was fine. Dr. Irem called it a strain and told me to rest my wrist for a week. I felt silly for worrying and relieved at the same time.",
        chat=(
            ("11:07:00", "Maya", "Good news, no fracture. Just strain and a dramatic wrist."),
            ("11:10:00", "Leo", "Your wrist has main character energy. Rest it before vocals tomorrow."),
            ("11:14:00", "Maya", "I will try to be medically boring."),
        ),
        photo_caption="Clinic waiting number 42 and a wrist brace on Maya's bag.",
    ),
    Event(
        date="2026-01-06",
        title="Studio recovery",
        mood=4,
        icons=("music", "recovery", "friendship"),
        people=("Maya", "Leo"),
        location="small studio",
        truth="Maya and Leo successfully rerecorded the missed vocal harmony.",
        diary="We redid the vocals today and they sounded better than the version I missed. Leo made a joke about the metro accidentally improving the song. I felt forgiven.",
        chat=(
            ("16:02:00", "Leo", "The second harmony is cleaner now. Metro disaster redeemed."),
            ("16:05:00", "Maya", "Please don't give public transport production credit."),
            ("16:09:00", "Leo", "Too late. Track title: Signal Failure in B Minor."),
        ),
        photo_caption="Studio headphones, lyric sheet, wrist brace beside the microphone.",
        target=True,
    ),
    Event(
        date="2026-01-07",
        title="Quiet admin day",
        mood=3,
        icons=("admin", "tired", "bills"),
        people=("Maya",),
        location="home",
        truth="Maya spent the day paying bills and replying to delayed emails.",
        diary="Nothing dramatic happened. I paid electricity, answered four delayed emails, and cleaned my desk. A flat day, but maybe flat days are part of staying alive.",
        chat=(
            ("19:44:00", "Sara", "Did you send the workshop reply?"),
            ("19:47:00", "Maya", "Not yet. I did bills and inbox archaeology. Berlin reply tomorrow."),
            ("19:49:00", "Sara", "Tomorrow then. No shame spiral tonight."),
        ),
        photo_caption="Desk with paid bill receipts and a half-empty water glass.",
    ),
    Event(
        date="2026-01-08",
        title="Workshop acceptance",
        mood=5,
        icons=("travel", "courage", "berlin"),
        people=("Maya", "Sara", "Nora"),
        location="home",
        truth="Maya accepted the Berlin workshop after discussing money and fear with Sara and Nora.",
        diary="I accepted Berlin. I still feel nervous, but the fear is no longer pretending to be the whole truth. Sara helped me budget, Nora helped me breathe, and I pressed send.",
        chat=(
            ("10:11:00", "Maya", "I accepted Berlin."),
            ("10:12:00", "Sara", "YES. Budget spreadsheet did its job."),
            ("10:13:00", "Nora", "Proud of you. Scared and going is still going."),
        ),
        photo_caption="Sent email confirmation for Berlin workshop and a handwritten budget.",
        correction="Later correction: Maya clarified that Sara helped with budget, while Nora helped with the emotional decision.",
        target=True,
    ),
    Event(
        date="2026-01-09",
        title="Contradictory calm",
        mood=2,
        icons=("stress", "family", "contradiction"),
        people=("Maya", "Mother"),
        location="phone call",
        truth="Maya told her mother she was calm about Berlin, but privately felt stressed.",
        diary="I told Mom I was calm about Berlin, which was not honest. I wanted her not to worry, so I sounded brave. After the call I sat on the floor and felt my chest tighten.",
        chat=(
            ("20:01:00", "Mother", "You sounded calm about Berlin. I am relieved."),
            ("20:05:00", "Maya", "I wanted you not to worry. I am managing, but not exactly calm."),
            ("20:08:00", "Mother", "Thank you for saying the real version."),
        ),
        photo_caption="Phone call screen with Mom, notebook page saying 'not exactly calm'.",
        contradiction="Direct diary says Maya was stressed after presenting calmness to her mother.",
        target=True,
    ),
    Event(
        date="2026-01-10",
        title="Park walk",
        mood=4,
        icons=("walk", "friendship", "sun"),
        people=("Maya", "Sara"),
        location="Moda Park",
        truth="Maya and Sara took a long walk and talked about making the memory project public.",
        diary="Sara and I walked in Moda Park until sunset. We talked about making the memory project public, but only if the privacy section is as strong as the model section.",
        chat=(
            ("17:33:00", "Sara", "Your paper idea is strongest when it admits the scary parts."),
            ("17:36:00", "Maya", "Trust-weighted reconstruction, not life surveillance. That has to be the line."),
            ("17:40:00", "Sara", "Exactly. Put that in the intro."),
        ),
        photo_caption="Sunset over Moda Park, two coffee cups on a bench.",
    ),
    Event(
        date="2026-01-11",
        title="Lost notebook",
        mood=1,
        icons=("loss", "panic", "notebook"),
        people=("Maya", "Leo"),
        location="bus",
        truth="Maya thought she lost her notebook on the bus, then Leo found it in the studio bag.",
        diary="I panicked because I thought the black notebook was gone. For two hours I imagined every private sentence in a stranger's hands. Leo found it later in the studio bag.",
        chat=(
            ("14:02:00", "Maya", "I think I lost the black notebook on the bus. I feel sick."),
            ("14:45:00", "Leo", "Wait. Is this it? Black cover, silver sticker, in the studio tote?"),
            ("14:47:00", "Maya", "YES. I owe the universe an apology and you coffee."),
        ),
        photo_caption="Black notebook with silver sticker inside a canvas studio tote.",
        target=True,
    ),
    Event(
        date="2026-01-12",
        title="Boundary message",
        mood=3,
        icons=("boundary", "relationship", "relief"),
        people=("Maya", "Nora"),
        location="home",
        truth="Maya sent a careful boundary message to an old collaborator.",
        diary="I sent the boundary message. It was short, kind, and final. Nora read it first and said I did not over-explain. The relief came late.",
        chat=(
            ("12:16:00", "Nora", "The message is clear. You don't need the paragraph defending why you deserve a boundary."),
            ("12:20:00", "Maya", "Deleting the paragraph felt harder than writing it."),
            ("12:24:00", "Nora", "That is usually the sign."),
        ),
        photo_caption="Draft message with a deleted paragraph highlighted.",
        target=True,
    ),
)


def reset_output() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    for relative in ("diaries", "whatsapp/Maya-Leo", "whatsapp/Maya-Sara", "whatsapp/Maya-Nora", "photos", "queries"):
        (OUT / relative).mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_diary() -> None:
    chunks = []
    for event in EVENTS:
        chunks.append(
            "\n".join(
                [
                    f"### {event.date}",
                    f"**Mood**: {event.mood}",
                    f"**Icons**: {', '.join(event.icons)}",
                    "",
                    event.diary,
                    "",
                ]
            )
        )
    (OUT / "diaries" / "maya_aydin_2026.md").write_text("\n".join(chunks), encoding="utf-8")


def chat_folder(sender: str) -> Path:
    if sender == "Leo":
        return OUT / "whatsapp" / "Maya-Leo"
    if sender == "Sara":
        return OUT / "whatsapp" / "Maya-Sara"
    if sender == "Nora":
        return OUT / "whatsapp" / "Maya-Nora"
    return OUT / "whatsapp" / "Maya-Nora"


def write_chats() -> None:
    grouped: dict[Path, list[str]] = {}
    for event in EVENTS:
        for time_value, sender, body in event.chat:
            folder = chat_folder(sender)
            grouped.setdefault(folder, []).append(f"[{event.date.replace('-', '.')}., {time_value}] {sender}: {body}")
    for folder, lines in grouped.items():
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "chat.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metadata() -> None:
    events = [
        {
            "event_id": f"event:{event.date}",
            "date": event.date,
            "title": event.title,
            "truth": event.truth,
            "mood": event.mood,
            "icons": list(event.icons),
            "people": list(event.people),
            "location": event.location,
            "target": event.target,
            "contradiction": event.contradiction,
            "correction": event.correction,
        }
        for event in EVENTS
    ]
    photos = [
        {
            "photo_id": f"photo:{event.date}",
            "date": event.date,
            "caption": event.photo_caption,
            "people": list(event.people),
            "location": event.location,
        }
        for event in EVENTS
    ]
    targets = [
        {
            "target_date": event.date,
            "event_id": f"event:{event.date}",
            "reason": "hidden diary day with indirect chat/photo evidence",
        }
        for event in EVENTS
        if event.target
    ]
    queries = [
        {
            "query": f"Reconstruct what happened around {event.date}.",
            "target_date": event.date,
            "gold_event_id": f"event:{event.date}",
            "must_not_present_as_direct": True,
            "expected_sources": ["diary_hidden_label", "whatsapp", "photo_metadata"],
        }
        for event in EVENTS
        if event.target
    ]
    corrections = [
        {
            "date": event.date,
            "event_id": f"event:{event.date}",
            "correction": event.correction,
        }
        for event in EVENTS
        if event.correction
    ]
    write_jsonl(OUT / "ground_truth" / "events.jsonl", events)
    write_jsonl(OUT / "photos" / "photo_metadata.jsonl", photos)
    write_jsonl(OUT / "queries" / "reconstruction_targets.jsonl", targets)
    write_jsonl(OUT / "queries" / "reconstruction_queries.jsonl", queries)
    write_jsonl(OUT / "corrections.jsonl", corrections)


def write_readme() -> None:
    readme = f"""# Generated MIRA Synthetic Corpus

Generated at: {CREATED_AT}

This is a deterministic public benchmark corpus. It contains no private user
data and no LLM-generated text.

Contents:

- `diaries/maya_aydin_2026.md`: first-person diary labels
- `whatsapp/*/chat.txt`: WhatsApp-style indirect evidence
- `ground_truth/events.jsonl`: hidden event truth table
- `queries/reconstruction_targets.jsonl`: dates to hide/evaluate
- `queries/reconstruction_queries.jsonl`: query metadata for future human/LLM eval
- `photos/photo_metadata.jsonl`: metadata-only photo evidence
- `corrections.jsonl`: explicit user corrections

Use:

```bash
npm run synthetic:reconstruction
```
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")


def main() -> int:
    reset_output()
    write_diary()
    write_chats()
    write_metadata()
    write_readme()
    manifest = {
        "created_at": CREATED_AT,
        "event_count": len(EVENTS),
        "target_count": sum(1 for event in EVENTS if event.target),
        "diary_file": "diaries/maya_aydin_2026.md",
        "whatsapp_root": "whatsapp",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"events={manifest['event_count']} targets={manifest['target_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
