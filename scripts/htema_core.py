from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any


LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent
DIARY_ROOT = REPO_ROOT / "knowledge_base" / "Deniz" / "diary"
WHATSAPP_ROOT = REPO_ROOT / "knowledge_base" / "chunks-manager" / "whatsapp-chunks"

DIARY_FILES = [
    DIARY_ROOT / "dailybean_2023_complete.md",
    DIARY_ROOT / "dailybean_2024_complete.md",
    DIARY_ROOT / "dailybean_2025_complete.md",
]

MONTHS = {
    # English
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
    # Turkish
    "ocak": 1,
    "şubat": 2, "subat": 2,
    "mart": 3,
    "nisan": 4,
    "mayıs": 5, "mayis": 5,
    "haziran": 6,
    "temmuz": 7,
    "ağustos": 8, "agustos": 8,
    "eylül": 9, "eylul": 9,
    "ekim": 10,
    "kasım": 11, "kasim": 11,
    "aralık": 12, "aralik": 12,
}

# Seasons (Northern Hemisphere) and Turkish equivalents.
# Each entry: (start_month, end_month).
SEASONS = {
    "spring": (3, 5), "ilkbahar": (3, 5), "bahar": (3, 5),
    "summer": (6, 8), "yaz": (6, 8),
    "autumn": (9, 11), "fall": (9, 11), "sonbahar": (9, 11), "güz": (9, 11), "guz": (9, 11),
    "winter": (12, 2), "kış": (12, 2), "kis": (12, 2),
}

# Relative phrases — resolved against parse_time_window's `now` reference.
RELATIVE_PHRASES_EN = [
    "today", "yesterday", "tomorrow",
    "this week", "last week", "next week",
    "this month", "last month", "next month",
    "this year", "last year", "next year",
    "this summer", "last summer", "next summer",
    "this winter", "last winter", "next winter",
    "this spring", "last spring", "next spring",
    "this autumn", "last autumn", "this fall", "last fall",
]
RELATIVE_PHRASES_TR = [
    "bugün", "bugun", "dün", "dun", "yarın", "yarin",
    "bu hafta", "geçen hafta", "gecen hafta",
    "bu ay", "geçen ay", "gecen ay",
    "bu yıl", "bu yil", "bu sene", "geçen yıl", "gecen yil", "geçen sene", "gecen sene",
    "bu yaz", "geçen yaz", "gecen yaz",
    "bu kış", "bu kis", "geçen kış", "gecen kis",
]

STOPWORDS = {
    "a",
    "about",
    "and",
    "are",
    "at",
    "be",
    "by",
    "day",
    "days",
    "did",
    "for",
    "from",
    "how",
    "i",
    "in",
    "into",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "which",
    "who",
    "why",
    "with",
}

TOKEN_ALIASES = {
    "cappadocia": ["kapadokya", "nevşehir", "nevsehir", "peri", "baca"],
    "kapadokya": ["cappadocia", "nevşehir", "nevsehir"],
    "nevsehir": ["nevşehir", "kapadokya", "cappadocia"],
    "nevşehir": ["nevsehir", "kapadokya", "cappadocia"],
    "newyear": ["yilbasi", "yılbaşı", "00.00"],
    "yilbasi": ["yılbaşı", "newyear"],
    "yılbaşı": ["yilbasi", "newyear"],
}

EMOTION_TERMS = {
    "happy": {"valence": 1.0, "arousal": 0.45, "mood": 5},
    "happiest": {"valence": 1.0, "arousal": 0.5, "mood": 5},
    "excited": {"valence": 0.8, "arousal": 0.9},
    "enthusiastic": {"valence": 0.8, "arousal": 0.8},
    "hopeful": {"valence": 0.75, "arousal": 0.35},
    "proud": {"valence": 0.75, "arousal": 0.45},
    "relaxed": {"valence": 0.65, "arousal": -0.35},
    "calm": {"valence": 0.55, "arousal": -0.55},
    "refreshed": {"valence": 0.65, "arousal": -0.1},
    "sad": {"valence": -0.8, "arousal": -0.1, "mood": 1},
    "saddest": {"valence": -1.0, "arousal": 0.0, "mood": 1},
    "depressed": {"valence": -0.95, "arousal": -0.25, "mood": 1},
    "anxious": {"valence": -0.75, "arousal": 0.85},
    "angry": {"valence": -0.8, "arousal": 0.85},
    "tired": {"valence": -0.45, "arousal": -0.75},
    "stressed": {"valence": -0.7, "arousal": 0.75},
    "peaceful": {"valence": 0.7, "arousal": -0.65},
    "romantic": {"valence": 0.85, "arousal": 0.45},
}

DIARY_FEATURES = {
    "travel": [
        "travel",
        "trip",
        "hotel",
        "airport",
        "flight",
        "cappadocia",
        "kapadokya",
        "nevsehir",
        "nevşehir",
        "bali",
        "vietnam",
        "thailand",
        "uludağ",
        "uludag",
    ],
    "relationship": ["sena", "partner", "romantic", "relationship", "date", "ring", "sex", "pit-a-pat"],
    "music": ["music", "vocal", "song", "guitar", "concert", "stage", "recording", "karaoke"],
    "work": ["work", "office", "company", "project", "startup", "zephio", "client", "corexas"],
    "health": ["health", "doctor", "pain", "sleep", "tired", "sick", "hospital", "hastane"],
    "social": ["friend", "friends", "family", "ugur", "uğur", "tansu", "busra", "büşra", "conversation"],
    "productivity": ["productive", "study", "learn", "focus", "task", "plan", "exam", "sınav"],
}

FEATURE_NAMES = [
    "bias",
    "semantic",
    "temporal",
    "emotion",
    "entity",
    "hierarchy",
    "continuity",
    "importance",
    "unresolved",
    "contradiction",
]


@dataclass(frozen=True)
class DiaryMemory:
    entry_id: str
    date: str
    ordinal: int
    year: int
    month: int
    source_path: str
    mood: int | None
    icons: tuple[str, ...]
    text: str
    tokens: tuple[str, ...]
    token_vector: dict[str, float]
    emotion: dict[str, float | int | None]
    diary_features: dict[str, float]
    importance: float
    unresolved: float
    source_type: str = "diary"
    participants: tuple[str, ...] = ()


@dataclass
class QuerySpec:
    raw: str
    tokens: tuple[str, ...]
    token_vector: dict[str, float]
    time_window: tuple[str, str] | None
    emotion: dict[str, float | int | str] | None
    diary_features: dict[str, float]
    around: bool
    contradiction: bool


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def normalize_date(value: str) -> str | None:
    match = re.search(r"(\d{4})[./-](\d{1,2})[./-](\d{1,2})", value)
    if not match:
        return None
    return f"{match.group(1)}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"


def ordinal(value: str) -> int:
    return date.fromisoformat(value).toordinal()


def days_in_month(year: int, month: int) -> int:
    if month == 12:
        return (date(year + 1, 1, 1) - date(year, 12, 1)).days
    return (date(year, month + 1, 1) - date(year, month, 1)).days


def tokenize(value: str) -> list[str]:
    tokens = [
        token
        for token in re.split(r"[^a-z0-9ğüşöçıİĞÜŞÖÇ]+", value.lower())
        if len(token) > 1 and token not in STOPWORDS
    ]
    expanded: list[str] = []
    for token in tokens:
        expanded.append(token)
        expanded.extend(TOKEN_ALIASES.get(token, []))
    return expanded


def vectorize(tokens: list[str] | tuple[str, ...]) -> dict[str, float]:
    vector: dict[str, float] = {}
    for token in tokens:
        vector[token] = vector.get(token, 0.0) + 1.0
    return vector


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(value * b.get(key, 0.0) for key, value in a.items())
    a_norm = math.sqrt(sum(value * value for value in a.values()))
    b_norm = math.sqrt(sum(value * value for value in b.values()))
    if not a_norm or not b_norm:
        return 0.0
    return dot / (a_norm * b_norm)


def parse_mood(text: str) -> int | None:
    match = re.search(
        r"\*\*Mood\*\*:\s*([1-5])|mood[^0-9]{0,20}([1-5])|(?:^|\s)([1-5])/5(?:\s|$)",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    return int(next(group for group in match.groups() if group))


def parse_icons(text: str) -> tuple[str, ...]:
    match = re.search(r"\*\*Icons\*\*:\s*([^\n]+)", text, re.IGNORECASE)
    if not match:
        return tuple()
    return tuple(item.strip().lower() for item in match.group(1).split(",") if item.strip())


def emotion_vector(mood: int | None, icons: tuple[str, ...], text: str) -> dict[str, float | int | None]:
    terms = set(tokenize(text))
    for icon in icons:
        terms.update(tokenize(icon))
    valence = (mood - 3) / 2 if mood else 0.0
    arousal = 0.0
    hits = 1 if mood else 0
    for term in terms:
        emotion = EMOTION_TERMS.get(term)
        if not emotion:
            continue
        valence += float(emotion.get("valence", 0.0))
        arousal += float(emotion.get("arousal", 0.0))
        hits += 1
    return {
        "valence": max(-1.0, min(1.0, valence / max(hits, 1))),
        "arousal": max(-1.0, min(1.0, arousal / max(hits, 1))),
        "mood": mood,
    }


def diary_feature_vector(text: str, icons: tuple[str, ...] = tuple()) -> dict[str, float]:
    haystack = f"{text} {' '.join(icons)}".lower()
    features: dict[str, float] = {}
    for name, terms in DIARY_FEATURES.items():
        hits = sum(1 for term in terms if term in haystack)
        features[name] = clamp(hits / max(len(terms) / 2, 1))
    return features


def memory_importance(mood: int | None, icons: tuple[str, ...], text: str) -> float:
    mood_extreme = abs((mood or 3) - 3) / 2
    icon_signal = min(len(icons), 12) / 12
    length_signal = min(len(text), 2200) / 2200
    return clamp(0.45 * mood_extreme + 0.3 * icon_signal + 0.25 * length_signal)


def unresolved_score(text: str, icons: tuple[str, ...]) -> float:
    haystack = f"{text} {' '.join(icons)}".lower()
    terms = ["conflict", "anxious", "pressured", "question", "uncertain", "kavga", "problem", "need", "should", "?"]
    return clamp(sum(1 for term in terms if term in haystack) / 4)


def parse_diary_memories(files: list[Path] | None = None) -> list[DiaryMemory]:
    files = files or DIARY_FILES
    memories: list[DiaryMemory] = []
    for file_path in files:
        if not file_path.exists():
            continue
        raw = file_path.read_text(encoding="utf-8")
        parts = re.split(r"(?=^###\s+\d{4}[./-]\d{1,2}[./-]\d{1,2})", raw, flags=re.MULTILINE)
        for part in parts:
            normalized = normalize_date(part[:120])
            if not normalized:
                continue
            mood = parse_mood(part)
            icons = parse_icons(part)
            tokens = tuple(tokenize(part))
            rel_path = str(file_path.relative_to(REPO_ROOT))
            year, month, _ = (int(value) for value in normalized.split("-"))
            memories.append(
                DiaryMemory(
                    entry_id=f"{rel_path}:{normalized}",
                    date=normalized,
                    ordinal=ordinal(normalized),
                    year=year,
                    month=month,
                    source_path=rel_path,
                    mood=mood,
                    icons=icons,
                    text=part.strip(),
                    tokens=tokens,
                    token_vector=vectorize(tokens),
                    emotion=emotion_vector(mood, icons, part),
                    diary_features=diary_feature_vector(part, icons),
                    importance=memory_importance(mood, icons, part),
                    unresolved=unresolved_score(part, icons),
                    source_type="diary",
                    participants=(),
                )
            )
    return sorted(memories, key=lambda item: item.ordinal)


# ---------------------------------------------------------------------------
# WhatsApp conversation memory parsing
# ---------------------------------------------------------------------------

_WHATSAPP_MSG_RE = re.compile(
    r"^\[?(?:\u200e|\u200f)*(\d{4})\.\s*(\d{2})\.\s*(\d{2})\.,\s*"
    r"(\d{1,2}):(\d{2}):(\d{2})\]\s+([^:]+):\s?(.*)",
    re.UNICODE,
)


@dataclass
class _WhatsAppMessage:
    date: str
    timestamp_ms: int
    sender: str
    raw: str


def _parse_whatsapp_file(text: str) -> list[_WhatsAppMessage]:
    """Parse raw WhatsApp export text into individual messages."""
    messages: list[_WhatsAppMessage] = []
    current: _WhatsAppMessage | None = None

    def finish() -> None:
        nonlocal current
        if current and current.raw.strip():
            messages.append(current)
        current = None

    for line in text.split("\n"):
        cleaned = line.lstrip("\u200e\u200f")
        match = _WHATSAPP_MSG_RE.match(cleaned)
        if match:
            finish()
            year, month, day = match.group(1), match.group(2), match.group(3)
            hour, minute, second = match.group(4), match.group(5), match.group(6)
            sender = match.group(7).strip()
            body = match.group(8)
            ts = int(datetime(
                int(year), int(month), int(day),
                int(hour), int(minute), int(second),
            ).timestamp() * 1000)
            current = _WhatsAppMessage(
                date=f"{year}-{month}-{day}",
                timestamp_ms=ts,
                sender=sender,
                raw=f"[{year}. {month}. {day}., {hour.zfill(2)}:{minute}:{second}] {sender}: {body}",
            )
        elif current:
            current.raw += f"\n{line}"

    finish()
    return messages


def _group_conversation_windows(
    messages: list[_WhatsAppMessage],
    max_gap_ms: int = 2 * 60 * 60 * 1000,
    max_chars: int = 2200,
) -> list[list[_WhatsAppMessage]]:
    """Group messages into conversation windows by date change, time gap, or size."""
    if not messages:
        return []
    windows: list[list[_WhatsAppMessage]] = []
    current: list[_WhatsAppMessage] = []

    def current_length(extra: str = "") -> int:
        return sum(len(m.raw) + 1 for m in current) + len(extra)

    def flush() -> None:
        nonlocal current
        if current:
            windows.append(current)
        current = []

    for msg in messages:
        prev = current[-1] if current else None
        if prev:
            gap = msg.timestamp_ms - prev.timestamp_ms
            date_changed = msg.date != prev.date
            too_large = current_length(msg.raw) > max_chars
            if date_changed or gap > max_gap_ms or too_large:
                flush()
        current.append(msg)

    flush()
    return windows


def _chat_name_from_path(folder_name: str) -> str:
    """Extract a clean chat identifier from the folder name."""
    # Folder names look like: "Deniz Yunus Göğüş ✨-Notlarım" or "Böbrek (Hüso)-Deniz Yunus Göğüş ✨"
    parts = folder_name.split("-")
    # Remove Deniz's own name to get the other participant
    other = [p.strip() for p in parts if "deniz" not in p.strip().lower() and "göğüş" not in p.strip().lower()]
    return other[0] if other else folder_name


def _infer_whatsapp_mood(text: str) -> int | None:
    """Rough mood inference from WhatsApp text using emotion terms."""
    lowered = text.lower()
    positive = sum(1 for t in ["haha", "😂", "😁", "❤", "güzel", "süper", "harika", "iyi", "mutlu", "eğlen", "keyif"]
                   if t in lowered)
    negative = sum(1 for t in ["üzgün", "kötü", "sinir", "kavga", "problem", "stres", "yorgun", "ağla"]
                   if t in lowered)
    if positive >= 3 and negative == 0:
        return 4
    if negative >= 2 and positive == 0:
        return 2
    return None  # Unknown mood — this is fine, MIRA handles None mood


def parse_whatsapp_memories(root: Path | None = None, min_chars: int = 200) -> list[DiaryMemory]:
    """Parse all WhatsApp chunk files into DiaryMemory objects (conversation windows).

    Args:
        root: Path to the whatsapp-chunks directory.
        min_chars: Minimum combined text length for a conversation window.
                   Windows shorter than this are filtered as noise.
    """
    root = root or WHATSAPP_ROOT
    if not root.exists():
        return []

    memories: list[DiaryMemory] = []
    seen_ids: set[str] = set()

    for chat_dir in sorted(root.iterdir()):
        if not chat_dir.is_dir() or chat_dir.name.startswith("."):
            continue

        chat_name = _chat_name_from_path(chat_dir.name)
        txt_files = sorted(chat_dir.glob("*.txt"))

        for txt_file in txt_files:
            try:
                raw = txt_file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            messages = _parse_whatsapp_file(raw)
            if len(messages) < 2:
                continue

            windows = _group_conversation_windows(messages)
            for window in windows:
                first = window[0]
                last = window[-1]
                combined_text = "\n".join(m.raw for m in window)
                if len(combined_text) < min_chars:
                    continue
                window_date = normalize_date(first.date)
                if not window_date:
                    continue

                participants_set = {m.sender for m in window}
                content_hash = hashlib.sha1(combined_text.encode("utf-8")).hexdigest()[:12]
                entry_id = f"whatsapp:{chat_name}:{window_date}:{content_hash}"
                dedup_key = f"{chat_name}:{window_date}:{content_hash}"
                if dedup_key in seen_ids:
                    continue
                seen_ids.add(dedup_key)

                tokens = tuple(tokenize(combined_text))
                year, month, _ = (int(v) for v in window_date.split("-"))
                mood = _infer_whatsapp_mood(combined_text)
                icons_tuple: tuple[str, ...] = ()
                rel_path = str(txt_file.relative_to(REPO_ROOT))

                memories.append(
                    DiaryMemory(
                        entry_id=entry_id,
                        date=window_date,
                        ordinal=ordinal(window_date),
                        year=year,
                        month=month,
                        source_path=rel_path,
                        mood=mood,
                        icons=icons_tuple,
                        text=combined_text,
                        tokens=tokens,
                        token_vector=vectorize(tokens),
                        emotion=emotion_vector(mood, icons_tuple, combined_text),
                        diary_features=diary_feature_vector(combined_text),
                        importance=memory_importance(mood, icons_tuple, combined_text),
                        unresolved=unresolved_score(combined_text, icons_tuple),
                        source_type="whatsapp",
                        participants=tuple(sorted(participants_set)),
                    )
                )

    return sorted(memories, key=lambda item: item.ordinal)


def parse_all_memories(
    diary_files: list[Path] | None = None,
    whatsapp_root: Path | None = None,
    *,
    include_rollups: bool = False,
    include_atoms: bool = False,
    rollup_max_text: int = 1800,
    atom_min_chars: int = 80,
    atom_max_chars: int = 700,
) -> list[DiaryMemory]:
    """Load diary AND WhatsApp memories into a unified sorted list.

    When `include_rollups` is True, append synthetic week/month rollup tokens
    that aggregate child diary days. When `include_atoms` is True, append
    paragraph-level memory atoms split out of long diary days. Both kinds carry
    distinct `source_type` values so the source-feature head can attend to them.
    """
    diary = parse_diary_memories(diary_files)
    whatsapp = parse_whatsapp_memories(whatsapp_root)
    combined = diary + whatsapp
    extras: list[DiaryMemory] = []
    if include_atoms:
        extras.extend(build_memory_atoms(diary, min_chars=atom_min_chars, max_chars=atom_max_chars))
    if include_rollups:
        extras.extend(build_rollup_memories(diary, max_text=rollup_max_text))
    return sorted(combined + extras, key=lambda item: item.ordinal)


# ---------------------------------------------------------------------------
# Hierarchy: week / month rollups
# ---------------------------------------------------------------------------


def _iso_week_key(memory: DiaryMemory) -> tuple[int, int]:
    iso = date.fromisoformat(memory.date).isocalendar()
    return iso.year, iso.week


def _summarize_icons(memories: list[DiaryMemory], top_n: int = 12) -> tuple[str, ...]:
    counter: dict[str, int] = {}
    for memory in memories:
        for icon in memory.icons:
            counter[icon] = counter.get(icon, 0) + 1
    ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return tuple(name for name, _ in ranked[:top_n])


def _summarize_text(memories: list[DiaryMemory], max_chars: int) -> str:
    lines: list[str] = []
    for memory in memories:
        excerpt = re.sub(r"^###\s+", "", memory.text.strip().split("\n", 1)[0])
        lines.append(f"- {memory.date}: {excerpt}")
        body = " ".join(memory.text.split())
        if len(body) > 220:
            body = body[:220].rstrip() + "..."
        lines.append(f"  {body}")
    joined = "\n".join(lines)
    if len(joined) <= max_chars:
        return joined
    return joined[: max_chars - 16].rstrip() + "\n[truncated...]"


def _aggregate_emotion(memories: list[DiaryMemory]) -> dict[str, float | int | None]:
    valences = [float(m.emotion.get("valence") or 0.0) for m in memories]
    arousals = [float(m.emotion.get("arousal") or 0.0) for m in memories]
    moods = [m.mood for m in memories if m.mood is not None]
    return {
        "valence": float(sum(valences) / max(len(valences), 1)),
        "arousal": float(sum(arousals) / max(len(arousals), 1)),
        "mood": int(round(sum(moods) / len(moods))) if moods else None,
    }


def _aggregate_diary_features(memories: list[DiaryMemory]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in DIARY_FEATURES:
        values = [memory.diary_features.get(name, 0.0) for memory in memories]
        out[name] = float(sum(values) / max(len(values), 1))
    return out


def _aggregate_importance(memories: list[DiaryMemory]) -> float:
    if not memories:
        return 0.0
    return clamp(max(m.importance for m in memories) * 0.7 + (sum(m.importance for m in memories) / len(memories)) * 0.3)


def _build_rollup(
    *,
    entry_id: str,
    label: str,
    start_iso: str,
    end_iso: str,
    children: list[DiaryMemory],
    source_type: str,
    max_text: int,
) -> DiaryMemory:
    text_block = f"### {label} ({start_iso} to {end_iso})\n" + _summarize_text(children, max_text)
    tokens = tuple(tokenize(text_block))
    icons = _summarize_icons(children)
    emotion = _aggregate_emotion(children)
    mood = emotion["mood"] if isinstance(emotion["mood"], int) else None
    importance = _aggregate_importance(children)
    middle_ordinal = ordinal(start_iso) + (ordinal(end_iso) - ordinal(start_iso)) // 2
    middle_iso = date.fromordinal(middle_ordinal).isoformat()
    middle_year, middle_month, _ = (int(part) for part in middle_iso.split("-"))
    return DiaryMemory(
        entry_id=entry_id,
        date=middle_iso,
        ordinal=middle_ordinal,
        year=middle_year,
        month=middle_month,
        source_path=children[0].source_path if children else "rollup",
        mood=mood,
        icons=icons,
        text=text_block,
        tokens=tokens,
        token_vector=vectorize(tokens),
        emotion=emotion,
        diary_features=_aggregate_diary_features(children),
        importance=float(importance),
        unresolved=float(clamp(sum(m.unresolved for m in children) / max(len(children), 1))),
        source_type=source_type,
        participants=(),
    )


def build_rollup_memories(
    diary: list[DiaryMemory],
    max_text: int = 1800,
    min_children: int = 2,
) -> list[DiaryMemory]:
    """Group diary days into week and month rollup tokens.

    WhatsApp windows are intentionally excluded: WhatsApp days don't represent
    Deniz's first-person account, and rolling them up would dilute the diary
    narrative signal that makes these tokens useful.
    """
    weekly: dict[tuple[int, int], list[DiaryMemory]] = {}
    monthly: dict[tuple[int, int], list[DiaryMemory]] = {}
    for memory in diary:
        weekly.setdefault(_iso_week_key(memory), []).append(memory)
        monthly.setdefault((memory.year, memory.month), []).append(memory)

    rollups: list[DiaryMemory] = []
    for (iso_year, iso_week), children in sorted(weekly.items()):
        if len(children) < min_children:
            continue
        children.sort(key=lambda m: m.ordinal)
        start_iso = children[0].date
        end_iso = children[-1].date
        rollups.append(_build_rollup(
            entry_id=f"rollup:week:{iso_year}-W{iso_week:02d}",
            label=f"Week {iso_year}-W{iso_week:02d}",
            start_iso=start_iso,
            end_iso=end_iso,
            children=children,
            source_type="rollup_week",
            max_text=max_text,
        ))

    for (year, month), children in sorted(monthly.items()):
        if len(children) < min_children:
            continue
        children.sort(key=lambda m: m.ordinal)
        start_iso = _date_str(year, month, 1)
        end_iso = _date_str(year, month, days_in_month(year, month))
        rollups.append(_build_rollup(
            entry_id=f"rollup:month:{year}-{month:02d}",
            label=f"{year}-{month:02d}",
            start_iso=start_iso,
            end_iso=end_iso,
            children=children,
            source_type="rollup_month",
            max_text=max_text,
        ))
    return rollups


# ---------------------------------------------------------------------------
# Memory atoms: paragraph-level slices of long diary days
# ---------------------------------------------------------------------------


def _split_into_atoms(body: str, min_chars: int, max_chars: int) -> list[str]:
    if not body or not body.strip():
        return []
    # First try paragraph splits; fall back to sentence splits inside oversized chunks.
    paragraphs = [block.strip() for block in re.split(r"\n\s*\n+", body) if block.strip()]
    chunks: list[str] = []
    for block in paragraphs:
        if len(block) <= max_chars:
            chunks.append(block)
            continue
        sentences = re.split(r"(?<=[\.\!\?...])\s+(?=[A-ZÇŞĞÜÖİ\"„'])", block)
        bucket = ""
        for sentence in sentences:
            if not sentence:
                continue
            if len(bucket) + len(sentence) + 1 > max_chars and bucket:
                chunks.append(bucket.strip())
                bucket = sentence
            else:
                bucket = f"{bucket} {sentence}".strip()
        if bucket:
            chunks.append(bucket.strip())
    return [chunk for chunk in chunks if len(chunk) >= min_chars]


def build_memory_atoms(
    diary: list[DiaryMemory],
    min_chars: int = 80,
    max_chars: int = 700,
    parents_only_above: int = 800,
) -> list[DiaryMemory]:
    """Split long diary days into smaller atoms.

    The parent day stays in the corpus; atoms are appended for retrieval
    granularity. We only fragment days whose body is longer than
    `parents_only_above` to avoid creating noise from short entries.
    """
    atoms: list[DiaryMemory] = []
    for parent in diary:
        body = parent.text
        # Strip the date header line so atoms don't all share the same prefix.
        body_stripped = re.sub(r"^###\s+\S+.*?\n", "", body, count=1).strip()
        if len(body_stripped) < parents_only_above:
            continue
        slices = _split_into_atoms(body_stripped, min_chars, max_chars)
        if len(slices) < 2:
            continue
        for idx, slice_text in enumerate(slices):
            slice_full = f"### {parent.date} atom {idx + 1}/{len(slices)}\n{slice_text}"
            tokens = tuple(tokenize(slice_full))
            atoms.append(DiaryMemory(
                entry_id=f"atom:{parent.entry_id}:{idx:02d}",
                date=parent.date,
                ordinal=parent.ordinal,
                year=parent.year,
                month=parent.month,
                source_path=parent.source_path,
                mood=parent.mood,
                icons=parent.icons,
                text=slice_full,
                tokens=tokens,
                token_vector=vectorize(tokens),
                emotion=emotion_vector(parent.mood, parent.icons, slice_full),
                diary_features=diary_feature_vector(slice_full, parent.icons),
                importance=memory_importance(parent.mood, parent.icons, slice_full),
                unresolved=unresolved_score(slice_full, parent.icons),
                source_type="atom",
                participants=parent.participants,
            ))
    return atoms


# ---------------------------------------------------------------------------
# Reflections: optionally load Level-5 identity patterns from disk
# ---------------------------------------------------------------------------


REFLECTIONS_DEFAULT = LAB_ROOT / "data" / "reflections.jsonl"


def load_reflections(path: Path | None = None) -> list[DiaryMemory]:
    path = path or REFLECTIONS_DEFAULT
    if not path.exists():
        return []
    out: list[DiaryMemory] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        try:
            start_iso = str(row["start"])
            end_iso = str(row["end"])
            label = str(row.get("label") or row.get("title") or "Reflection")
            kind = str(row.get("kind") or "pattern")
            text_body = str(row.get("text") or row.get("body") or label)
            mood = row.get("mood")
            icons_tuple = tuple(str(item) for item in row.get("icons") or [])
        except KeyError:
            continue
        text_block = f"### Reflection: {label} ({start_iso} to {end_iso})\n{text_body}"
        tokens = tuple(tokenize(text_block))
        middle_ordinal = (ordinal(start_iso) + ordinal(end_iso)) // 2
        middle_iso = date.fromordinal(middle_ordinal).isoformat()
        middle_year, middle_month, _ = (int(part) for part in middle_iso.split("-"))
        out.append(DiaryMemory(
            entry_id=str(row.get("id") or f"reflection:{kind}:{start_iso}:{end_iso}"),
            date=middle_iso,
            ordinal=middle_ordinal,
            year=middle_year,
            month=middle_month,
            source_path="reflections",
            mood=int(mood) if isinstance(mood, (int, float)) else None,
            icons=icons_tuple,
            text=text_block,
            tokens=tokens,
            token_vector=vectorize(tokens),
            emotion=emotion_vector(int(mood) if isinstance(mood, (int, float)) else None, icons_tuple, text_block),
            diary_features=diary_feature_vector(text_block, icons_tuple),
            importance=float(row.get("importance") or 0.7),
            unresolved=float(row.get("unresolved") or 0.0),
            source_type="reflection",
            participants=tuple(sorted(str(item) for item in row.get("participants") or [])),
        ))
    return out


def _date_str(year: int, month: int, day: int) -> str:
    return f"{year}-{month:02d}-{day:02d}"


def _month_window(year: int, month: int) -> tuple[str, str]:
    return _date_str(year, month, 1), _date_str(year, month, days_in_month(year, month))


def _season_window(year: int, start_month: int, end_month: int) -> tuple[str, str]:
    if start_month <= end_month:
        return _date_str(year, start_month, 1), _date_str(year, end_month, days_in_month(year, end_month))
    # Winter wraps: Dec start_year -> Feb start_year+1
    return _date_str(year, start_month, 1), _date_str(year + 1, end_month, days_in_month(year + 1, end_month))


def parse_time_window(query: str, now: date | None = None) -> tuple[str, str] | None:
    """Parse a natural-language time window from `query`.

    Handles:
      - Explicit YYYY-MM-DD, YYYY/MM/DD, YYYY.MM.DD dates (single or range)
      - "between A and B", "from A to B" with bare year/month parts
      - English + Turkish month names with a year
      - Seasons (spring/summer/autumn/winter, ilkbahar/yaz/sonbahar/kış) with a year
      - Relative phrases: today, yesterday, this/last/next week|month|year|season (EN + TR)
      - Phase words inside a year: early/late/start/end/beginning + erken/geç/başı/sonu/ortası
      - Bare year fallback
    """
    now = now or date.today()
    lowered = query.lower()

    # 1) Explicit dates (any count): take min..max if 2+ otherwise single day window.
    explicit = []
    for match in re.finditer(r"(20\d{2})[./-](\d{1,2})[./-](\d{1,2})", lowered):
        normalized = normalize_date(match.group(0))
        if normalized:
            explicit.append(normalized)
    if explicit:
        if len(explicit) >= 2:
            return min(explicit), max(explicit)
        # Single explicit date: treat as that exact day.
        return explicit[0], explicit[0]

    # 2) Relative phrases (resolved against `now`).
    rel = _parse_relative_window(lowered, now)
    if rel:
        return rel

    # 3) Year-anchored parses.
    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", lowered)
    year = int(year_match.group(1)) if year_match else None

    # 3a) Year + season.
    if year:
        for name, (start_month, end_month) in SEASONS.items():
            if re.search(rf"\b{re.escape(name)}\b", lowered):
                return _season_window(year, start_month, end_month)

    # 3b) Year + month name.
    if year:
        month_match = _find_month(lowered)
        if month_match is not None:
            return _month_window(year, month_match)

    # 3c) Year + phase word (end/last/late/early/start/beginning + Turkish equivalents).
    if year and re.search(r"\b(end|last|sonu|sonunda)\b", lowered):
        return _date_str(year, 12, 1), _date_str(year, 12, 31)
    if year and re.search(r"\b(late|geç|gec)\b", lowered):
        return _date_str(year, 10, 1), _date_str(year, 12, 31)
    if year and re.search(r"\b(early|start|beginning|başı|basi|başında|basinda)\b", lowered):
        return _date_str(year, 1, 1), _date_str(year, 3, 31)
    if year and re.search(r"\b(mid|middle|ortası|ortasi|ortasında|ortasinda)\b", lowered):
        return _date_str(year, 5, 1), _date_str(year, 8, 31)
    if year and re.search(r"\b(first half|ilk yarı|ilk yari)\b", lowered):
        return _date_str(year, 1, 1), _date_str(year, 6, 30)
    if year and re.search(r"\b(second half|ikinci yarı|ikinci yari)\b", lowered):
        return _date_str(year, 7, 1), _date_str(year, 12, 31)

    # 3d) Bare year.
    if year:
        return _date_str(year, 1, 1), _date_str(year, 12, 31)

    # 4) Month name without year — assume the most recent such month relative to `now`.
    month_match = _find_month(lowered)
    if month_match is not None:
        recent_year = now.year if month_match <= now.month else now.year - 1
        return _month_window(recent_year, month_match)

    # 5) Season without year — most recent occurrence relative to `now`.
    for name, (start_month, end_month) in SEASONS.items():
        if re.search(rf"\b{re.escape(name)}\b", lowered):
            anchor_year = now.year
            # If we haven't reached this season yet, fall back to last year.
            if start_month > now.month:
                anchor_year -= 1
            return _season_window(anchor_year, start_month, end_month)

    return None


def _find_month(lowered: str) -> int | None:
    for name, month in MONTHS.items():
        if re.search(rf"\b{re.escape(name)}\b", lowered):
            return month
    return None


def _parse_relative_window(lowered: str, now: date) -> tuple[str, str] | None:
    # Single-day relatives.
    if re.search(r"\btoday\b|\bbugün\b|\bbugun\b", lowered):
        return now.isoformat(), now.isoformat()
    if re.search(r"\byesterday\b|\bdün\b|\bdun\b", lowered):
        d = date.fromordinal(now.toordinal() - 1)
        return d.isoformat(), d.isoformat()
    if re.search(r"\btomorrow\b|\byarın\b|\byarin\b", lowered):
        d = date.fromordinal(now.toordinal() + 1)
        return d.isoformat(), d.isoformat()

    # Week.
    if re.search(r"\bthis week\b|\bbu hafta\b", lowered):
        start = date.fromordinal(now.toordinal() - now.weekday())
        end = date.fromordinal(start.toordinal() + 6)
        return start.isoformat(), end.isoformat()
    if re.search(r"\blast week\b|\bgeçen hafta\b|\bgecen hafta\b", lowered):
        start = date.fromordinal(now.toordinal() - now.weekday() - 7)
        end = date.fromordinal(start.toordinal() + 6)
        return start.isoformat(), end.isoformat()

    # Month.
    if re.search(r"\bthis month\b|\bbu ay\b", lowered):
        return _month_window(now.year, now.month)
    if re.search(r"\blast month\b|\bgeçen ay\b|\bgecen ay\b", lowered):
        first_of_this = date(now.year, now.month, 1)
        last_month_end = date.fromordinal(first_of_this.toordinal() - 1)
        return _month_window(last_month_end.year, last_month_end.month)

    # Year.
    if re.search(r"\bthis year\b|\bbu yıl\b|\bbu yil\b|\bbu sene\b", lowered):
        return _date_str(now.year, 1, 1), _date_str(now.year, 12, 31)
    if re.search(r"\blast year\b|\bgeçen yıl\b|\bgecen yil\b|\bgeçen sene\b|\bgecen sene\b", lowered):
        y = now.year - 1
        return _date_str(y, 1, 1), _date_str(y, 12, 31)

    # Seasons with "this" / "last".
    for name, (start_month, end_month) in SEASONS.items():
        if re.search(rf"\bthis\s+{re.escape(name)}\b|\bbu\s+{re.escape(name)}\b", lowered):
            anchor_year = now.year if start_month <= now.month else now.year - 1
            return _season_window(anchor_year, start_month, end_month)
        if re.search(rf"\blast\s+{re.escape(name)}\b|\bgeçen\s+{re.escape(name)}\b|\bgecen\s+{re.escape(name)}\b", lowered):
            anchor_year = (now.year - 1) if start_month <= now.month else (now.year - 2)
            return _season_window(anchor_year, start_month, end_month)

    return None


def parse_query_emotion(query: str) -> dict[str, float | int | str] | None:
    lowered = query.lower()
    tokens = tokenize(lowered)
    direct = re.search(r"\bmood\s*([1-5])\b|([1-5])/5", lowered)
    mood = int(next(group for group in direct.groups() if group)) if direct else None
    valence = 0.0
    arousal = 0.0
    hits = 0
    mode = "mixed"
    for token in tokens:
        emotion = EMOTION_TERMS.get(token)
        if not emotion:
            continue
        mood = mood or int(emotion["mood"]) if "mood" in emotion else mood
        valence += float(emotion.get("valence", 0.0))
        arousal += float(emotion.get("arousal", 0.0))
        hits += 1

    if re.search(r"happiest|happy|best|great|good", lowered):
        mood = mood or 5
        valence += 1
        arousal += 0.25
        hits += 1
        mode = "happy"
    elif re.search(r"saddest|sad|worst|bad|depressed", lowered):
        mood = mood or 1
        valence -= 1
        hits += 1
        mode = "sad"
    elif re.search(r"anxious|stress|stressed|nervous", lowered):
        valence -= 0.7
        arousal += 0.9
        hits += 1
        mode = "anxious"
    elif re.search(r"calm|peaceful|relaxed", lowered):
        valence += 0.6
        arousal -= 0.7
        hits += 1
        mode = "calm"

    if not mood and not hits:
        return None
    return {
        "mood": mood,
        "valence": max(-1.0, min(1.0, valence / max(hits, 1))),
        "arousal": max(-1.0, min(1.0, arousal / max(hits, 1))),
        "mode": mode,
    }


def build_query_spec(query: str, override_window: tuple[str, str] | None = None) -> QuerySpec:
    tokens = tuple(tokenize(query))
    lowered = query.lower()
    return QuerySpec(
        raw=query,
        tokens=tokens,
        token_vector=vectorize(tokens),
        time_window=override_window or parse_time_window(query),
        emotion=parse_query_emotion(query),
        diary_features=diary_feature_vector(query),
        around=bool(re.search(r"\b(around|after|before|changed|timeline|led|during|streak|arc)\b", lowered)),
        contradiction=bool(re.search(r"\b(contradict|counter|evidence|actually|always|never)\b|not true", lowered)),
    )


def temporal_score(memory: DiaryMemory, window: tuple[str, str] | None) -> float:
    if not window:
        return 0.5
    start, end = (ordinal(item) for item in window)
    if start <= memory.ordinal <= end:
        return 1.0
    distance = start - memory.ordinal if memory.ordinal < start else memory.ordinal - end
    decay = 12 if end - start <= 35 else 45
    return math.exp(-distance / decay)


def emotion_score(memory: DiaryMemory, target: dict[str, float | int | str] | None) -> float:
    if not target:
        return 0.5
    target_mood = target.get("mood")
    mood_score = 0.5
    if target_mood and memory.mood:
        mood_score = 1 - abs(memory.mood - int(target_mood)) / 4
    valence = float(memory.emotion.get("valence") or 0.0)
    arousal = float(memory.emotion.get("arousal") or 0.0)
    valence_score = 1 - abs(valence - float(target.get("valence") or 0.0)) / 2
    arousal_score = 1 - abs(arousal - float(target.get("arousal") or 0.0)) / 2
    return clamp(0.55 * mood_score + 0.3 * valence_score + 0.15 * arousal_score)


def entity_score(memory: DiaryMemory, query: QuerySpec) -> float:
    if not query.tokens:
        return 0.0
    haystack = set(memory.tokens)
    direct = sum(1 for token in query.tokens if token in haystack) / max(len(query.tokens), 1)
    desired = [(name, value) for name, value in query.diary_features.items() if value > 0]
    if not desired:
        return clamp(direct)
    feature = sum(memory.diary_features.get(name, 0.0) for name, _ in desired) / len(desired)
    return clamp(0.65 * feature + 0.35 * direct)


def hierarchy_score(memory: DiaryMemory, query: QuerySpec) -> float:
    if not query.time_window:
        year_terms = [int(token) for token in query.tokens if re.fullmatch(r"20\d{2}", token)]
        if year_terms and memory.year in year_terms:
            return 0.7
        return 0.25
    start, end = query.time_window
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    if start_date.year == end_date.year and memory.year == start_date.year:
        if start_date.month == end_date.month and memory.month == start_date.month:
            return 1.0
        if start_date.month == end_date.month:
            return 0.35
        if start_date.month <= memory.month <= end_date.month:
            return 0.9
        return 0.55
    return 0.4


def anchor_ordinals(query: QuerySpec, memories: list[DiaryMemory]) -> list[int]:
    if not query.around:
        return []
    candidates = []
    for memory in memories:
        semantic = cosine(query.token_vector, memory.token_vector)
        entity = entity_score(memory, query)
        if semantic >= 0.1:
            candidates.append((semantic + 0.35 * entity, memory))

    if not candidates:
        for memory in memories:
            semantic = cosine(query.token_vector, memory.token_vector)
            entity = entity_score(memory, query)
            if entity >= 0.55:
                candidates.append((semantic + 0.35 * entity, memory))

    anchors = [memory for _, memory in sorted(candidates, key=lambda item: item[0], reverse=True)]
    return [item.ordinal for item in anchors[:5]]


def continuity_score(memory: DiaryMemory, query: QuerySpec, anchors: list[int]) -> float:
    if not anchors:
        return 0.0
    return max(math.exp(-abs(memory.ordinal - anchor) / 1.8) for anchor in anchors)


def contradiction_score(memory: DiaryMemory, query: QuerySpec) -> float:
    if not query.contradiction:
        return 0.0
    # Counter-evidence often comes from high-importance, opposite-valence memories.
    if query.emotion:
        query_valence = float(query.emotion.get("valence") or 0.0)
        memory_valence = float(memory.emotion.get("valence") or 0.0)
        opposite = max(0.0, -query_valence * memory_valence)
    else:
        opposite = 0.25
    return clamp(0.55 * opposite + 0.45 * memory.importance)


def feature_vector(memory: DiaryMemory, query: QuerySpec, anchors: list[int] | None = None) -> dict[str, float]:
    anchors = anchors or []
    return {
        "bias": 1.0,
        "semantic": cosine(query.token_vector, memory.token_vector),
        "temporal": temporal_score(memory, query.time_window),
        "emotion": emotion_score(memory, query.emotion),
        "entity": entity_score(memory, query),
        "hierarchy": hierarchy_score(memory, query),
        "continuity": continuity_score(memory, query, anchors),
        "importance": memory.importance,
        "unresolved": memory.unresolved,
        "contradiction": contradiction_score(memory, query),
    }


def dot(weights: dict[str, float], features: dict[str, float]) -> float:
    return sum(weights.get(name, 0.0) * features.get(name, 0.0) for name in FEATURE_NAMES)


def score_memories(query: QuerySpec, memories: list[DiaryMemory], weights: dict[str, float]) -> list[dict[str, Any]]:
    anchors = anchor_ordinals(query, memories)
    scored = []
    for memory in memories:
        features = feature_vector(memory, query, anchors)
        score = dot(weights, features)
        scored.append({"memory": memory, "features": features, "score": score})
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def compact_text(text: str, max_chars: int = 900) -> str:
    cleaned = re.sub(r"\n{3,}", "\n\n", text.strip())
    return cleaned if len(cleaned) <= max_chars else cleaned[:max_chars].rstrip() + "\n[truncated]"


def split_train_test(rows: list[dict[str, Any]], test_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    test_size = max(1, int(len(shuffled) * test_ratio)) if len(shuffled) > 5 else 0
    return shuffled[test_size:], shuffled[:test_size]
