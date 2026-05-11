# Diary Temporal-Emotional Attention Design

## Why Not Train a Transformer Yet

A real Transformer learns query, key, and value projections from a lot of examples. The diary corpus is personally rich but too small for training those matrices from scratch.

So the first practical version is a Transformer-like retrieval mechanism:

- use a ready semantic model for text embeddings when available
- use deterministic feature projections for diary-specific structure
- combine several attention heads with transparent weights
- leave the projection interface compatible with learned weights later

This gives us useful behavior now and a path to a learned reranker later.

## Memory Token

Each day is one memory token.

```ts
type DayMemory = {
  id: string
  date: string
  year: number
  ordinal: number
  mood: number | null
  icons: string[]
  text: string
  key: {
    semantic: Vector
    temporal: TemporalFeatures
    emotion: EmotionFeatures
    diary: DiaryFeatures
    continuity: SequenceFeatures
  }
  value: {
    summaryText: string
    fullText: string
    sourcePath: string
    chunkLocator: string
  }
}
```

## Query Token

The user question is converted into a query.

```ts
type MemoryQuery = {
  raw: string
  semantic: Vector
  timeWindow?: {
    startOrdinal: number
    endOrdinal: number
    centerOrdinal: number
    granularity: "day" | "month" | "year" | "range"
  }
  emotionTarget?: {
    mood?: number
    valence?: number
    arousal?: number
    mode: "happy" | "sad" | "anxious" | "calm" | "mixed"
  }
  diaryHints: {
    people: string[]
    places: string[]
    activities: string[]
    exactTerms: string[]
  }
}
```

## Attention Heads

### 1. Semantic Head

Compares the user question against the text of each day.

```text
semantic_score = cosine(Q_semantic, K_semantic)
```

In the prototype this is lexical cosine. In production it should use an existing embedding model, for example:

- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- OpenAI `text-embedding-3-small`
- BGE-M3 for multilingual dense retrieval

### 2. Temporal Head

Answers whether the day belongs to the requested time window.

```text
if day inside requested range:
  temporal_score = 1
else:
  temporal_score = exp(-distance_days / decay)
```

This is what prevents "happy March entry" from beating "December entry" when the question says "end of 2024".

### 3. Emotion Head

Uses mood score, icons, and emotion words.

```text
emotion_score =
  mood_alignment +
  valence_alignment +
  arousal_alignment +
  emotional_tag_overlap
```

Example:

```text
"happiest days" -> target mood 5, high valence
"anxious weeks" -> anxiety tag, high arousal, low or mixed valence
"peaceful days" -> calm/relaxed tags, lower arousal
```

### 4. Diary Feature Head

Diary entries have structured hints that normal embeddings underuse:

- icons
- people
- places
- trips
- work/project language
- music/performance language
- health/sleep/productivity markers

This head compares query diary hints to extracted day features.

### 5. Continuity Head

Human memory is not isolated by chunk. Days form sequences.

```text
continuity_score =
  neighbor_similarity +
  same_trip_or_arc_bonus +
  mood_streak_bonus +
  before_after_relevance
```

This is important for questions such as:

- "what led up to that?"
- "how did I feel after the trip?"
- "what changed around New Year's Eve?"

For "around" questions, the prototype first finds strong anchor days, then expands attention to neighboring days with an exponential distance decay. This mimics one useful part of self-attention: a strong token can pull related nearby tokens into the context window even if they do not match the query text directly.

```text
anchor_continuity(day) = max(exp(-abs(day - anchor_day) / 1.8))
```

That is why a query such as "what changed around the Cappadocia trip" can return the anchor day and nearby days in the same trip arc.

## Alias Layer

Personal memory is messy: names, nicknames, English spellings, Turkish spellings, and app-specific labels all refer to the same thing.

Before vectorization, the prototype expands known aliases:

```text
cappadocia -> kapadokya, nevsehir, nevşehir
newyear -> yilbasi, yılbaşı, 00.00
```

Later this should be replaced or extended with a user-owned alias table, extracted from the Knowledge Base and editable in the dashboard.

## Final Score

```text
score(day) =
  w_semantic   * semantic_score +
  w_time       * temporal_score +
  w_emotion    * emotion_score +
  w_diary      * diary_feature_score +
  w_continuity * continuity_score
```

Default weights are query-dependent:

```text
date-heavy question:
  time up, semantic down slightly

emotion-heavy question:
  emotion up

project/person question:
  diary-feature up

timeline question:
  continuity up
```

## What Gets Stored in Qdrant Later

Payload fields:

```json
{
  "source_type": "diary",
  "date_start": "2024-12-31",
  "date_end": "2024-12-31",
  "granularity": "day",
  "mood": 5,
  "icons": ["partner", "happy", "travel"],
  "people": ["Sena"],
  "activities": ["travel", "shopping"],
  "sequence_key": "dailybean_2024",
  "sequence_index": 89
}
```

Vector fields:

```text
semantic_vector: pretrained embedding of full entry or summary
emotion_vector: small deterministic or learned emotion vector
diary_feature_vector: tag/person/activity projection
```

Qdrant can do initial filtering by payload, then Jarvis can rerank with this attention mechanism.

## Next Research Steps

1. Replace lexical semantic head with the local embedding service.
2. Extract people and places more carefully.
3. Add explicit before/after expansion.
4. Add an evaluator with example questions and expected days.
5. Add preference feedback, such as "this result was useful", to learn head weights.
