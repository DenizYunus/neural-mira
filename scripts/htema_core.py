from __future__ import annotations

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

DIARY_FILES = [
    DIARY_ROOT / "dailybean_2023_complete.md",
    DIARY_ROOT / "dailybean_2024_complete.md",
    DIARY_ROOT / "dailybean_2025_complete.md",
]

MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

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
                )
            )
    return sorted(memories, key=lambda item: item.ordinal)


def parse_time_window(query: str) -> tuple[str, str] | None:
    lowered = query.lower()
    explicit = [normalize_date(match.group(0)) for match in re.finditer(r"20\d{2}[./-]\d{1,2}[./-]\d{1,2}", lowered)]
    explicit = [item for item in explicit if item]
    if explicit:
        return min(explicit), max(explicit)

    year_match = re.search(r"\b(20\d{2})\b", lowered)
    year = int(year_match.group(1)) if year_match else None
    month_name = next((name for name in MONTHS if re.search(rf"\b{name}\b", lowered)), None)
    if year and month_name:
        month = MONTHS[month_name]
        return f"{year}-{month:02d}-01", f"{year}-{month:02d}-{days_in_month(year, month):02d}"

    if year and re.search(r"\b(end|last)\b", lowered):
        return f"{year}-12-01", f"{year}-12-31"
    if year and re.search(r"\blate\b", lowered):
        return f"{year}-10-01", f"{year}-12-31"
    if year and re.search(r"\b(start|early|beginning)\b", lowered):
        return f"{year}-01-01", f"{year}-03-31"
    if year:
        return f"{year}-01-01", f"{year}-12-31"
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
