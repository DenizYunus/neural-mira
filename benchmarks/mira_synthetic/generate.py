#!/usr/bin/env python3
"""Generate a public synthetic MIRA benchmark corpus."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import urllib.request
from dataclasses import dataclass
from datetime import date, timedelta
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
    sparse_hint: str = ""
    confuser_for: tuple[str, ...] = ()
    target: bool = False
    persona: str = "maya_aydin"


MAYA_EVENTS: tuple[Event, ...] = (
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
    Event(
        date="2026-01-13",
        title="Notebook archive",
        mood=3,
        icons=("notebook", "archive", "coffee"),
        people=("Maya", "Sara"),
        location="Karga Cafe",
        truth="Maya organized old notebook pages at Karga Cafe without losing anything.",
        diary="I sorted old notebook pages at Karga Cafe with Sara. The black notebook stayed on the table the whole time. It was admin, not panic.",
        chat=(
            ("15:05:00", "Sara", "You brought the black notebook again. This time it is definitely on the table."),
            ("15:08:00", "Maya", "No bus drama today. Just archiving old pages and drinking coffee."),
            ("15:12:00", "Sara", "Good. Let the notebook have a boring day."),
        ),
        photo_caption="Black notebook open beside archived pages and coffee.",
        confuser_for=("2026-01-11",),
    ),
    Event(
        date="2026-01-14",
        title="Berlin spreadsheet cleanup",
        mood=3,
        icons=("berlin", "budget", "admin"),
        people=("Maya", "Sara"),
        location="home",
        truth="Maya revised the Berlin budget spreadsheet after already accepting the workshop.",
        diary="Sara and I cleaned up the Berlin budget spreadsheet. This was not the decision day anymore, just making the accepted workshop feel practical.",
        chat=(
            ("10:02:00", "Sara", "Budget cleanup looks less scary now."),
            ("10:06:00", "Maya", "Yes. Important detail: this is after accepting, not me deciding again."),
            ("10:09:00", "Sara", "Exactly. Implementation day, not courage day."),
        ),
        photo_caption="Berlin budget spreadsheet with travel rows highlighted.",
        confuser_for=("2026-01-04", "2026-01-08", "2026-01-09"),
    ),
    Event(
        date="2026-01-15",
        title="Dry vocal practice",
        mood=4,
        icons=("music", "practice", "studio"),
        people=("Maya", "Leo"),
        location="small studio",
        truth="Maya and Leo practiced vocals normally with no metro delay or recovery session.",
        diary="Leo and I had a normal dry vocal practice. No rain, no missed rehearsal, no dramatic redemption arc. Just repetition until the second harmony settled.",
        chat=(
            ("17:18:00", "Leo", "Normal practice day. Weirdly peaceful."),
            ("17:21:00", "Maya", "No rain, no metro, no guilt. I support this genre."),
            ("17:24:00", "Leo", "Track title: Nothing Went Wrong in B Minor."),
        ),
        photo_caption="Studio lyric sheet from a normal vocal practice.",
        confuser_for=("2026-01-02", "2026-01-06"),
    ),
)


@dataclass(frozen=True)
class PersonaProfile:
    slug: str
    name: str
    start_date: str
    friend: str
    collaborator: str
    family: str
    mentor: str
    project: str
    venue: str
    trip_place: str
    object_name: str


PERSONAS: tuple[PersonaProfile, ...] = (
    PersonaProfile(
        slug="atlas_kaya",
        name="Atlas",
        start_date="2026-02-01",
        friend="Mert",
        collaborator="Ece",
        family="Selin",
        mentor="Prof. Hale",
        project="urban garden sensor",
        venue="Halic Lab",
        trip_place="Izmir workshop",
        object_name="blue field notebook",
    ),
    PersonaProfile(
        slug="lina_demir",
        name="Lina",
        start_date="2026-03-01",
        friend="Omar",
        collaborator="Aylin",
        family="Cem",
        mentor="Dr. Rafi",
        project="documentary edit",
        venue="Pera Studio",
        trip_place="Ankara archive visit",
        object_name="silver audio recorder",
    ),
)


def iso_day(start: str, offset: int) -> str:
    return (date.fromisoformat(start) + timedelta(days=offset)).isoformat()


def generated_persona_events(profile: PersonaProfile) -> tuple[Event, ...]:
    name = profile.name
    friend = profile.friend
    collaborator = profile.collaborator
    family = profile.family
    mentor = profile.mentor
    project = profile.project
    venue = profile.venue
    trip_place = profile.trip_place
    object_name = profile.object_name
    d = lambda offset: iso_day(profile.start_date, offset)
    return (
        Event(
            date=d(0),
            title="Project reset",
            mood=4,
            icons=("planning", "focus", "project"),
            people=(name, collaborator),
            location="home",
            truth=f"{name} reset the schedule for the {project} after a calm planning call with {collaborator}.",
            diary=f"I reset the {project} schedule today after talking with {collaborator}. The plan finally felt like a tool instead of a threat.",
            chat=(
                ("09:10:00", collaborator, f"The new {project} schedule looks realistic."),
                ("09:13:00", name, "Realistic is the miracle word. I want structure without panic."),
                ("09:16:00", collaborator, "Then keep the small checkpoints and stop moving the finish line."),
            ),
            photo_caption=f"Printed schedule for the {project} beside a mug and pencil.",
            persona=profile.slug,
        ),
        Event(
            date=d(1),
            title="Transit delay",
            mood=2,
            icons=("delay", "guilt", "rain"),
            people=(name, friend),
            location="tram stop",
            truth=f"{name} missed a planning session with {friend} because transit stopped during heavy rain.",
            diary=f"The tram stopped in the rain and I missed the planning session with {friend}. I knew it was not my fault, but I still felt unreliable.",
            chat=(
                ("18:04:00", name, "Still stuck at the tram stop. I am going to miss the whole session."),
                ("18:07:00", friend, "Do not turn weather into a moral failure. We can redo it tomorrow."),
                ("18:12:00", name, "I had the notes ready and still feel awful."),
            ),
            photo_caption="Rain on a tram shelter glass panel and a soaked backpack strap.",
            target=True,
            persona=profile.slug,
        ),
        Event(
            date=d(2),
            title="Cafe breakthrough",
            mood=5,
            icons=("writing", "coffee", "focus"),
            people=(name, collaborator),
            location=venue,
            truth=f"{name} finished the first usable outline for the {project} at {venue} with {collaborator} nearby.",
            diary=f"I worked at {venue} and finished the first usable outline for the {project}. {collaborator} sat nearby, and the focus felt clean.",
            chat=(
                ("13:35:00", collaborator, "You were quiet for two hours. Did the outline finally unlock?"),
                ("13:39:00", name, f"Yes. The {project} now has a spine instead of a pile of tabs."),
                ("13:42:00", collaborator, "Keep that sentence for the summary."),
            ),
            photo_caption=f"Laptop outline for the {project} beside coffee at {venue}.",
            target=True,
            persona=profile.slug,
        ),
        Event(
            date=d(3),
            title="Trip argument",
            mood=1,
            icons=("argument", "travel", "fear"),
            people=(name, family),
            location="home",
            truth=f"{name} argued with {family} about whether to accept the {trip_place} invitation.",
            diary=f"{family} and I argued about the {trip_place}. I said money was the only issue, but fear was louder than I wanted to admit.",
            chat=(
                ("21:20:00", family, f"I pushed too hard about the {trip_place}. Sorry."),
                ("21:24:00", name, "I was defensive because I am scared and calling it logistics."),
                ("21:30:00", family, "Then call it both. Budget and fear can both be true."),
            ),
            photo_caption=f"Invitation for the {trip_place} open next to a half-packed tote.",
            contradiction=f"{name} first described the trip decision as only financial, then admitted fear was part of it.",
            target=True,
            persona=profile.slug,
        ),
        Event(
            date=d(4),
            title="Health check",
            mood=3,
            icons=("health", "waiting", "relief"),
            people=(name, mentor),
            location="clinic",
            truth=f"{name} learned that persistent wrist pain was strain, not a fracture.",
            diary=f"The clinic was slow, but the X-ray was clear. {mentor} reminded me that rest is also part of finishing the {project}.",
            chat=(
                ("11:07:00", name, "No fracture. Just strain and a dramatic wrist."),
                ("11:10:00", friend, "Rest it before you try to become a keyboard hero again."),
                ("11:14:00", name, "I will attempt medical boringness."),
            ),
            photo_caption="Clinic waiting number and a wrist brace on a canvas bag.",
            target=True,
            persona=profile.slug,
        ),
        Event(
            date=d(5),
            title="Recovered session",
            mood=4,
            icons=("recovery", "friendship", "project"),
            people=(name, friend),
            location=venue,
            truth=f"{name} and {friend} successfully redid the planning session that was missed after the transit delay.",
            diary=f"We redid the missed planning session at {venue}. It was better than the version I missed, and I felt forgiven.",
            chat=(
                ("16:02:00", friend, "The redo was cleaner than the original plan would have been."),
                ("16:05:00", name, "Please do not give the tram creative credit."),
                ("16:09:00", friend, "Too late. Transit delay: executive producer."),
            ),
            photo_caption=f"Whiteboard notes for the {project} at {venue}.",
            target=True,
            persona=profile.slug,
        ),
        Event(
            date=d(6),
            title="Admin day",
            mood=3,
            icons=("admin", "tired", "bills"),
            people=(name,),
            location="home",
            truth=f"{name} spent the day paying bills and answering delayed messages.",
            diary="Nothing dramatic happened. I paid bills, answered delayed messages, and cleaned my desk. Flat days still count.",
            chat=(
                ("19:44:00", collaborator, f"Did you send the {trip_place} reply?"),
                ("19:47:00", name, "Not yet. Bills and inbox archaeology today. Reply tomorrow."),
                ("19:49:00", collaborator, "Tomorrow then. No guilt spiral tonight."),
            ),
            photo_caption="Paid bill receipts and a cleaned desk.",
            persona=profile.slug,
        ),
        Event(
            date=d(7),
            title="Trip acceptance",
            mood=5,
            icons=("travel", "courage", "decision"),
            people=(name, collaborator, family),
            location="home",
            truth=f"{name} accepted the {trip_place} after separating budget concerns from fear.",
            diary=f"I accepted the {trip_place}. {collaborator} helped with the budget, {family} helped me breathe, and I pressed send.",
            chat=(
                ("10:11:00", name, f"I accepted the {trip_place}."),
                ("10:12:00", collaborator, "Budget spreadsheet did its job."),
                ("10:13:00", family, "Proud of you. Scared and going is still going."),
            ),
            photo_caption=f"Sent acceptance email for the {trip_place} beside a handwritten budget.",
            correction=f"Later correction: {name} clarified that {collaborator} helped with budget while {family} helped with the emotional decision.",
            target=True,
            persona=profile.slug,
        ),
        Event(
            date=d(8),
            title="Contradictory calm",
            mood=2,
            icons=("stress", "family", "contradiction"),
            people=(name, family),
            location="phone call",
            truth=f"{name} told {family} they were calm about the {trip_place}, but privately felt stressed.",
            diary=f"I told {family} I was calm about the {trip_place}, which was not honest. After the call my chest tightened.",
            chat=(
                ("20:01:00", family, "You sounded calm earlier. I felt relieved."),
                ("20:05:00", name, "I wanted you not to worry. I am managing, but not exactly calm."),
                ("20:08:00", family, "Thank you for the real version."),
            ),
            photo_caption="Phone call screen and notebook page saying 'not exactly calm'.",
            contradiction=f"Direct diary says {name} felt stressed after sounding calm to {family}.",
            target=True,
            persona=profile.slug,
        ),
        Event(
            date=d(9),
            title="Public project talk",
            mood=4,
            icons=("walk", "ethics", "public"),
            people=(name, collaborator),
            location="park",
            truth=f"{name} and {collaborator} discussed making the {project} public only with a strong privacy section.",
            diary=f"{collaborator} and I walked until sunset and talked about making the {project} public. The privacy section has to be as strong as the model section.",
            chat=(
                ("17:33:00", collaborator, "The idea is strongest when it admits the scary parts."),
                ("17:36:00", name, "Reconstruction, not surveillance. That has to be the line."),
                ("17:40:00", collaborator, "Exactly. Put that in the intro."),
            ),
            photo_caption="Sunset over a park bench with two coffee cups.",
            target=True,
            persona=profile.slug,
        ),
        Event(
            date=d(10),
            title="Lost object",
            mood=1,
            icons=("loss", "panic", "object"),
            people=(name, friend),
            location="bus",
            truth=f"{name} thought the {object_name} was lost on the bus, but {friend} found it in the studio bag.",
            diary=f"I panicked because I thought the {object_name} was gone. {friend} found it later in the studio bag.",
            chat=(
                ("14:02:00", name, f"I think I lost the {object_name} on the bus. I feel sick."),
                ("14:45:00", friend, f"Wait. Is this it? {object_name}, in the studio tote?"),
                ("14:47:00", name, "YES. I owe you coffee and the universe an apology."),
            ),
            photo_caption=f"{object_name} inside a canvas studio tote.",
            target=True,
            persona=profile.slug,
        ),
        Event(
            date=d(11),
            title="Boundary message",
            mood=3,
            icons=("boundary", "relationship", "relief"),
            people=(name, family),
            location="home",
            truth=f"{name} sent a careful boundary message to an old collaborator after {family} reviewed it.",
            diary=f"I sent the boundary message. It was short, kind, and final. {family} said I did not need the defensive paragraph.",
            chat=(
                ("12:16:00", family, "The message is clear. You do not need the paragraph defending the boundary."),
                ("12:20:00", name, "Deleting the paragraph was harder than writing it."),
                ("12:24:00", family, "That is usually the sign."),
            ),
            photo_caption="Draft boundary message with one paragraph deleted.",
            target=True,
            persona=profile.slug,
        ),
        Event(
            date=d(12),
            title="Object archive",
            mood=3,
            icons=("object", "archive", "coffee"),
            people=(name, collaborator),
            location=venue,
            truth=f"{name} organized old notes at {venue} with the {object_name} safely on the table.",
            diary=f"I sorted old notes at {venue}. The {object_name} stayed on the table the whole time. It was admin, not panic.",
            chat=(
                ("15:05:00", collaborator, f"You brought the {object_name} again. It is definitely on the table."),
                ("15:08:00", name, "No bus drama today. Just archiving."),
                ("15:12:00", collaborator, "Good. Let the object have a boring day."),
            ),
            photo_caption=f"{object_name} open beside archived pages and coffee.",
            confuser_for=(d(10),),
            persona=profile.slug,
        ),
        Event(
            date=d(13),
            title="Trip spreadsheet cleanup",
            mood=3,
            icons=("travel", "budget", "admin"),
            people=(name, collaborator),
            location="home",
            truth=f"{name} revised the {trip_place} budget after already accepting the invitation.",
            diary=f"{collaborator} and I cleaned up the {trip_place} budget. This was implementation day, not decision day.",
            chat=(
                ("10:02:00", collaborator, "Budget cleanup looks less scary now."),
                ("10:06:00", name, "Important detail: this is after accepting, not me deciding again."),
                ("10:09:00", collaborator, "Exactly. Implementation day, not courage day."),
            ),
            photo_caption=f"Budget spreadsheet for the {trip_place} with travel rows highlighted.",
            confuser_for=(d(3), d(7), d(8)),
            persona=profile.slug,
        ),
        Event(
            date=d(14),
            title="Normal practice",
            mood=4,
            icons=("practice", "project", "normal"),
            people=(name, friend),
            location=venue,
            truth=f"{name} and {friend} practiced normally with no transit delay or recovery session.",
            diary=f"{friend} and I had a normal practice session. No rain, no missed meeting, no redemption arc. Just repetition.",
            chat=(
                ("17:18:00", friend, "Normal practice day. Weirdly peaceful."),
                ("17:21:00", name, "No rain, no tram, no guilt. I support this genre."),
                ("17:24:00", friend, "Title: Nothing Went Wrong."),
            ),
            photo_caption=f"Routine {project} notes from a normal practice session.",
            confuser_for=(d(1), d(5)),
            persona=profile.slug,
        ),
    )


EVENTS: tuple[Event, ...] = MAYA_EVENTS + tuple(
    event for profile in PERSONAS for event in generated_persona_events(profile)
)


def reset_output() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    for relative in ("diaries", "whatsapp", "photos", "queries"):
        (OUT / relative).mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_diary() -> None:
    by_persona: dict[str, list[str]] = {}
    for event in EVENTS:
        by_persona.setdefault(event.persona, []).append(
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
    for persona, chunks in sorted(by_persona.items()):
        (OUT / "diaries" / f"{persona}_2026.md").write_text("\n".join(chunks), encoding="utf-8")


def safe_chat_name(value: str) -> str:
    return value.replace(" ", "_").replace(".", "")


def chat_folder(event: Event) -> Path:
    primary = event.people[0]
    counterparts: list[str] = []
    for _, sender, _ in event.chat:
        if sender != primary and sender not in counterparts:
            counterparts.append(sender)
    if not counterparts:
        counterparts = [person for person in event.people[1:] if person != primary]
    if not counterparts:
        counterparts = ["Notes"]
    safe_primary = safe_chat_name(primary)
    safe_other = "_".join(safe_chat_name(person) for person in counterparts)
    return OUT / "whatsapp" / f"{safe_primary}-{safe_other}"


def write_chats() -> None:
    grouped: dict[Path, list[str]] = {}
    for event in EVENTS:
        folder = chat_folder(event)
        for time_value, sender, body in event.chat:
            grouped.setdefault(folder, []).append(f"[{event.date.replace('-', '.')}., {time_value}] {sender}: {body}")
    for folder, lines in grouped.items():
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "chat.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def public_hint(event: Event) -> str:
    people = ", ".join(event.people)
    return f"Public metadata: people={people}; location={event.location}."


def sparse_user_hint(event: Event) -> str:
    if event.sparse_hint:
        return f"User hint: {event.sparse_hint}."
    keywords = ", ".join(event.icons[:2])
    return f"User hint: {keywords}."


def confuser_event_ids(target_date: str) -> list[str]:
    return [f"event:{event.date}" for event in EVENTS if target_date in event.confuser_for]


def structured_facts(event: Event) -> dict[str, str]:
    facts = {
        "who": ", ".join(event.people),
        "what_happened": event.truth,
        "where": event.location,
        "emotional_context": f"mood={event.mood}; themes={', '.join(event.icons)}",
        "visual_evidence": event.photo_caption,
    }
    if event.contradiction:
        facts["contradiction"] = event.contradiction
    if event.correction:
        facts["correction"] = event.correction
    return facts


def write_metadata() -> None:
    events = [
        {
            "event_id": f"event:{event.date}",
            "date": event.date,
            "persona": event.persona,
            "title": event.title,
            "truth": event.truth,
            "mood": event.mood,
            "icons": list(event.icons),
            "people": list(event.people),
            "location": event.location,
            "target": event.target,
            "contradiction": event.contradiction,
            "correction": event.correction,
            "confuser_for": list(event.confuser_for),
            "facts": structured_facts(event),
        }
        for event in EVENTS
    ]
    photos = [
        {
            "photo_id": f"photo:{event.date}",
            "date": event.date,
            "persona": event.persona,
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
            "persona": event.persona,
            "public_hint": public_hint(event),
            "sparse_user_hint": sparse_user_hint(event),
            "confuser_event_ids": confuser_event_ids(event.date),
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
            "persona": event.persona,
            "query_modes": {
                "date_only": f"Reconstruct what happened around {event.date}.",
                "date_plus_public_metadata": f"Reconstruct what happened around {event.date}. {public_hint(event)}",
                "sparse_user_hint": f"Reconstruct what happened around {event.date}. {sparse_user_hint(event)}",
            },
            "public_hint": public_hint(event),
            "sparse_user_hint": sparse_user_hint(event),
            "confuser_event_ids": confuser_event_ids(event.date),
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
- `queries/reconstruction_targets.jsonl`: dates, query hints, and confuser IDs
- `queries/reconstruction_queries.jsonl`: query-mode metadata for future human/LLM eval
- `photos/photo_metadata.jsonl`: metadata-only photo evidence
- `corrections.jsonl`: explicit user corrections

The target rows support date-only, public-metadata, and sparse-hint query modes.
Confuser events are included as adversarial distractors for intrusion scoring.
Ground-truth event rows include structured fact slots for paper-grade scoring.

Use:

```bash
npm run synthetic:reconstruction
```
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-provider", choices=("none", "deepseek"), default="none")
    parser.add_argument("--deepseek-api-key", help="DeepSeek API key. Prefer --deepseek-api-key-env for local use.")
    parser.add_argument("--deepseek-api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--deepseek-model", default="deepseek-chat")
    parser.add_argument("--deepseek-workers", type=int, default=5)
    parser.add_argument("--deepseek-drafts", type=int, default=0, help="Number of synthetic event-pack drafts to request.")
    parser.add_argument("--deepseek-output", type=Path, default=OUT / "llm_drafts" / "deepseek_event_packs.jsonl")
    return parser.parse_args()


def deepseek_prompt(index: int) -> str:
    return (
        "Create one synthetic autobiographical memory benchmark event pack. "
        "Return strict JSON with keys persona, events. events must be a list of 5 objects, "
        "each with date_offset, title, truth, diary, chat_messages, photo_caption, facts, "
        "target, contradiction, correction, confuser_for_offsets. "
        "No real private people, no copyrighted diary text, no explanations. "
        f"Draft index: {index}."
    )


def call_deepseek(prompt: str, *, api_key: str, model: str) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You generate safe synthetic benchmark data as strict JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.8,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = json.loads(response.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {"raw": content}
    return {"provider": "deepseek", "model": model, "draft": parsed}


async def write_deepseek_drafts(args: argparse.Namespace) -> None:
    if args.llm_provider != "deepseek" or args.deepseek_drafts <= 0:
        return
    api_key = args.deepseek_api_key or os.environ.get(args.deepseek_api_key_env)
    if not api_key:
        raise SystemExit(f"DeepSeek requested but no API key was provided. Set {args.deepseek_api_key_env} or pass --deepseek-api-key.")

    semaphore = asyncio.Semaphore(max(1, args.deepseek_workers))

    async def one(index: int) -> dict[str, Any]:
        async with semaphore:
            return await asyncio.to_thread(call_deepseek, deepseek_prompt(index), api_key=api_key, model=args.deepseek_model)

    drafts = await asyncio.gather(*(one(index) for index in range(args.deepseek_drafts)))
    args.deepseek_output.parent.mkdir(parents=True, exist_ok=True)
    args.deepseek_output.write_text(
        "".join(json.dumps(draft, ensure_ascii=False, sort_keys=True) + "\n" for draft in drafts),
        encoding="utf-8",
    )
    print(f"deepseek_drafts={len(drafts)} wrote {args.deepseek_output}")


def main() -> int:
    args = parse_args()
    reset_output()
    write_diary()
    write_chats()
    write_metadata()
    write_readme()
    manifest = {
        "created_at": CREATED_AT,
        "event_count": len(EVENTS),
        "persona_count": len({event.persona for event in EVENTS}),
        "personas": sorted({event.persona for event in EVENTS}),
        "target_count": sum(1 for event in EVENTS if event.target),
        "diary_root": "diaries",
        "whatsapp_root": "whatsapp",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    asyncio.run(write_deepseek_drafts(args))
    print(f"wrote {OUT}")
    print(f"events={manifest['event_count']} targets={manifest['target_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
