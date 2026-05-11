# MIRA / HTEMA Design

MIRA is the product-facing architecture:

```text
MIRA = Memory, Introspection, Reflection Architecture
```

HTEMA is the retrieval/modeling core inside MIRA:

```text
HTEMA = Hierarchical Temporal-Emotional Memory Attention
```

The intent is not to claim a new general-purpose Transformer. The serious claim is narrower and more useful:

> A diary-native memory architecture for longitudinal self-dialogue, combining episodic memory atoms, temporal-emotional embeddings, narrative graph retrieval, and reflection-based consolidation.

## Why Diary Memory Is Different

Diary entries are not normal documents. They are:

- time-indexed
- emotionally dense
- repetitive
- contradictory
- person-centered
- unresolved
- identity-forming

Normal RAG retrieves similar chunks. MIRA retrieves moments, emotional states, people, repeated loops, contradictions, and life arcs.

## Core Stack

```text
Raw diary entries
  -> day parser
  -> memory atom extractor
  -> temporal-emotional token builder
  -> episodic graph memory
  -> HTEMA attention/reranking
  -> reflection compiler
  -> narrative self model
  -> conversation engine
```

## Hierarchy

MIRA stores memory at multiple levels:

```text
Level 0: raw entries
Level 1: memory atoms
Level 2: day tokens
Level 3: week summaries
Level 4: month / era summaries
Level 5: identity and recurring-pattern summaries
```

Each level can attend to lower and higher levels.

Example:

```text
Raw: "I felt useless today."
Atom: shame, self-doubt, AI studies
Day: May 10 was dominated by academic insecurity
Week: Deniz felt behind before exams
Month: impostor syndrome recurred despite visible progress
Identity: Deniz is ambitious and tends to move the success bar immediately after wins
```

## Memory Atoms

Every diary day is decomposed into memory atoms:

```json
{
  "atom_id": "2024-12-31:a2",
  "date": "2024-12-31",
  "type": "social_event",
  "content": "Entered New Year's Eve with Sena after wine and deep conversation",
  "people": ["Sena"],
  "places": ["hotel", "Cappadocia"],
  "emotion": {
    "labels": ["happy", "hopeful", "romantic"],
    "valence": 0.91,
    "arousal": 0.52,
    "mood": 5
  },
  "importance": 0.86,
  "unresolved": false,
  "threads": ["relationship", "travel", "new year"]
}
```

Atoms are better training units than full diary chunks because a single day can contain multiple separate memories.

## Token Composition

Each atom/day/week/month becomes a token with compositional subspaces:

```text
token =
  semantic_embedding(text)
  + temporal_embedding(date_or_range)
  + emotion_embedding(mood/icons/labels)
  + entity_embedding(people/places/projects)
  + hierarchy_embedding(level + parent period)
  + sequence_embedding(position in diary)
```

Keep the subspaces inspectable at first:

```json
{
  "semantic": "384d or 768d pretrained vector",
  "temporal": "64d positional/date vector",
  "emotion": "32d emotion state vector",
  "entity": "64d people/place/activity vector",
  "hierarchy": "32d level/month/year/event vector",
  "sequence": "32d neighbor/arc vector"
}
```

## Temporal Embeddings

Use Transformer-style sinusoidal encodings for absolute day position:

```text
TE(date)[2i]   = sin(day_index / 10000^(2i/d))
TE(date)[2i+1] = cos(day_index / 10000^(2i/d))
```

Add cyclic calendar signals:

```text
day_of_week
day_of_month
month_of_year
season
year
```

This makes nearby days naturally close while preserving monthly and yearly rhythms.

## Hierarchical Tokens

Create parent tokens:

```text
day -> week -> month -> year -> era/event arc -> identity pattern
```

Parent tokens are not just summaries. They act as context anchors.

Example:

```text
Dec 29, 2024
Dec 30, 2024
Dec 31, 2024
Jan 01, 2025
  -> Cappadocia trip event token
  -> late 2024 relationship/travel period token
```

## HTEMA Attention

For a query token `q` and memory token `m_i`:

```text
Q = Wq(q)
K_i = Wk(m_i)
V_i = Wv(m_i)
```

Base attention:

```text
a_i = softmax((Q . K_i) / sqrt(d))
context = sum_i a_i V_i
```

HTEMA adds diary-native biases:

```text
score_i =
  (Q . K_i) / sqrt(d)
  + temporal_bias(q, m_i)
  + emotion_bias(q, m_i)
  + hierarchy_bias(q, m_i)
  + graph_bias(q, m_i)
  + unresolved_loop_bias(q, m_i)
```

So the final attention is:

```text
a_i = softmax(score_i)
```

This is much closer to a real attention model than the deterministic prototype, while still being practical.

## Heads

HTEMA can be implemented as multi-head cross-attention:

```text
Head 1: semantic memory
Head 2: temporal proximity and ranges
Head 3: emotional state
Head 4: people and relationship context
Head 5: goals, projects, and unresolved loops
Head 6: contradiction / counter-evidence
Head 7: narrative continuity
Head 8: hierarchy parent-child context
```

The model can start with fixed biases, then learn head weights from synthetic/self-supervised examples.

## No Hand Training Data

We avoid manual labels. Training signals come from:

1. Diary structure:
   - date
   - mood
   - icons
   - title/entry text

2. Temporal self-supervision:
   - adjacent days are positives
   - same month days are weak positives
   - random far days are negatives

3. Emotion self-supervision:
   - same mood/icon days are positives
   - opposite mood days are negatives

4. Event arc mining:
   - consecutive days with shared themes form event tokens

5. LLM query generation:
   - generate natural questions that should retrieve known days/atoms
   - use those pairs as weak supervision

## LLM-Generated Query Training

For each diary day or atom, generate queries such as:

```json
{
  "query": "Which days around New Year's Eve with Sena felt happiest?",
  "intent": "emotional_temporal_recall",
  "positive_ids": ["dailybean_2024_complete.md:2024-12-31"],
  "positive_window": ["2024-12-29", "2025-01-01"],
  "required_heads": ["temporal", "emotion", "relationship", "continuity"],
  "hard_negative_strategy": "same emotion outside time window"
}
```

This is not perfect truth, but it is very useful weak supervision.

## Training Plan

Phase 0: deterministic prototype.

```text
Current diary_attention.mjs
```

Phase 1: synthetic query dataset.

```text
diary entries -> LLM query generator -> JSONL training pairs
```

Phase 2: trainable linear adapters.

```text
freeze pretrained text embeddings
train Wq/Wk/Wv adapters and bias weights
```

The implemented v1 is a lightweight version of this phase. It trains pairwise ranking weights over the HTEMA heads:

```text
score(q, memory) =
  w_semantic      * semantic(q, memory)
  + w_temporal    * temporal(q, memory)
  + w_emotion     * emotion(q, memory)
  + w_entity      * entity(q, memory)
  + w_hierarchy   * hierarchy(q, memory)
  + w_continuity  * continuity(q, memory)
  + w_importance  * importance(memory)
  + w_unresolved  * unresolved(memory)
  + w_contradict  * contradiction(q, memory)
```

Training examples are generated by an LLM from diary entries. Positives come from the target entry/window, and negatives are sampled from confusing memories such as same-emotion wrong-time entries, same-window wrong-entry entries, and random far entries.

This is not yet a neural Q/K/V adapter, but it is a real learned attention-style reranker over diary-native heads. The next step is to replace or augment the scalar head weights with small trainable projection matrices over pretrained embedding subspaces.

Phase 3: graph-aware reranker.

```text
retrieve candidates with Qdrant
expand through graph
rerank with HTEMA
```

Phase 4: reflection compiler.

```text
weekly/monthly/identity tokens
contradiction and unresolved-loop summaries
```

## Evaluation

Compare:

```text
Naive RAG
Long-context LLM
Deterministic MIRA
Trainable HTEMA
```

Metrics:

- temporal accuracy
- emotion recall
- person recall
- event arc recall
- contradiction retrieval
- hallucination rate
- user-rated helpfulness

## One-Sentence Pitch

> MIRA is a diary-native AI memory architecture that converts personal entries into episodic memory atoms, links them through a narrative graph, and uses hierarchical temporal-emotional attention to support long-term self-reflective conversation.
