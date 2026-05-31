# Components

A map of what's in this repo and how the pieces connect.

## Three retrieval layers

Run side-by-side in the honest benchmark so each can be ablated:

1. **Scalar HTEMA** — lightweight deterministic ranker over per-pair feature vectors. Transparent, no training needed. Defined in `scripts/htema_core.py`.
2. **Leakage-free evaluation harness** — `scripts/honest_mira.py` compares BM25, sparse semantic (TF-IDF + LSA), dense semantic (multilingual MiniLM), scalar HTEMA, and Neural MIRA, all from query text only.
3. **Neural MIRA reranker** — PyTorch model over a 38-dim per-memory-pair feature (semantic, temporal, emotional, diary-feature, dense cosine, source/participant, continuity, BM25, and pairwise interactions). Trained checkpoint: `data/neural_mira_model.pt`.

## Five-level memory hierarchy

All generated deterministically — **no LLM in the retrieval pipeline**, just in optional training-data synthesis:

| Level | Source type   | Builder                                                                       |
| ----- | ------------- | ----------------------------------------------------------------------------- |
| 1     | `atom`        | `htema_core.build_memory_atoms` (paragraph splits of long diary days)         |
| 2     | `diary`       | `parse_diary_memories`                                                        |
| 2     | `whatsapp`    | `parse_whatsapp_memories`                                                     |
| 3     | `rollup_week` | `htema_core.build_rollup_memories`                                            |
| 4     | `rollup_month`| `htema_core.build_rollup_memories`                                            |
| 5     | `reflection`  | `scripts/reflect.py` (mood streaks, theme arcs, contradictions, identity patterns) |

Re-generate Level-5 reflections after major corpus changes:

```bash
python scripts/reflect.py     # writes data/reflections.jsonl
```

Reflections are intentionally **auditable** — every reflection lists its `source_ids` so a human can trace why it was emitted.

## Continuous learning loop

`scripts/convert_feedback.py` converts user feedback (thumbs-up / thumbs-down on retrieved memories) into eval examples:

- `useful` rows → displayed sources become `positive_ids`
- `wrong` / `needs_work` rows → displayed sources become `negative_ids` (hard negatives — the system surfaced these and the user rejected them). If the user's note contains an explicit date or memory id, that becomes the recovered positive; otherwise the row is skipped (no-self-reinforcement protocol).

See [`data/README.md`](../data/README.md) for the feedback log schema.

## Idea — memory tokens + query tokens

Each diary day or WhatsApp conversation window becomes a memory token:

```text
token(memory) = {
  key:     semantic + time + emotion + diary-feature projections,
  value:   memory text + extracted metadata + neighboring sequence context,
  source:  diary | whatsapp | rollup | atom | reflection,
  participants: people attached to the memory window,
}
```

Each user question becomes a query token:

```text
query(question) = {
  semantic intent,
  requested time window,
  emotion target,
  diary-specific hints (mood, people, trip, work, music, health),
}
```

Then the retriever runs multiple attention heads:

- **semantic head** — text similarity
- **temporal head** — date range and proximity
- **emotion head** — mood, valence, arousal, emotional words
- **diary-feature head** — people, places, icons, activities, tags
- **continuity head** — neighboring days, streaks

Output: a ranked set of memory values plus per-head scores so you can inspect why each result was selected.

A small **alias layer** powers cross-lingual matching — for example, `Cappadocia` expands toward `Kapadokya` and `Nevsehir`, so English queries still find Turkish diary text. See [`docs/architecture.md`](architecture.md) for the full design.

## Date parser

`htema_core.parse_time_window` resolves natural-language windows in both English and Turkish:

- **Explicit dates** — `2024.06.01`, `between 2024-06-01 and 2024-06-15`
- **Year + month name** — `March 2025`, `Mart 2025`, `Mayıs 2025`
- **Seasons** — `spring 2024`, `ilkbahar 2024`, `last summer`, `geçen yaz`
- **Year + phase** — `late 2024`, `2024 sonu`, `first half of 2025`, `early 2023`, `2023 başı`
- **Relative** — `today`, `yesterday`, `bugün`, `dün`, `this week`, `geçen ay`, `last year`, `bu sene`
- **Bare year** — `2024`

Pass a reference date via `parse_time_window(query, now=...)` for deterministic relative-date tests.

## File layout

```
neural-mira/
├── src/
│   ├── diary_attention.mjs    # Node CLI: query → ranked memories (scalar HTEMA)
│   └── nm_config.mjs          # config loader (paths + .env)
├── scripts/
│   ├── nm_config.py           # config loader (paths + LLM creds + .env)
│   ├── htema_core.py          # memory parsing + scalar HTEMA + hierarchy builders
│   ├── honest_mira.py         # honest benchmark (the eval that matters)
│   ├── search_mira.py         # CLI: query → ranked memories (neural MIRA)
│   ├── reflect.py             # Level-5 reflection generator
│   ├── convert_feedback.py    # feedback log → eval examples
│   ├── generate_training_queries.py    # LLM-synthesized training data
│   ├── generate_style_augmentations.py # LLM-synthesized query style variants
│   ├── synthesize_whatsapp_diaries.py  # LLM-summarized WhatsApp days
│   ├── build_neural_embeddings.py      # MiniLM embedding cache builder
│   ├── train_neural_htema.py           # neural reranker training (legacy)
│   ├── search_neural_htema.py          # query CLI (legacy neural)
│   ├── evaluate_neural_htema.py        # legacy neural eval
│   ├── evaluate_baselines.py           # baseline sweep
│   └── train_htema.py / search_htema.py / evaluate_htema.py  # scalar variant
├── docs/
│   ├── quickstart.md
│   ├── components.md          # this file
│   ├── architecture.md        # full design notes
│   ├── training.md
│   ├── evaluation.md
│   └── mira_htema_design.md
└── data/                      # gitignored — your own diaries, embeddings, checkpoints
    └── README.md              # expected file formats
```
