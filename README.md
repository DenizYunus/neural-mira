# Memory Attention Lab

A personal-memory retrieval lab over diary entries, WhatsApp conversation windows,
hierarchical rollups, sub-day atoms, and reflective patterns. The retrieval core
is **HTEMA** (Hierarchical Temporal-Emotional Memory Attention) inside the wider
**MIRA** architecture.

Active retrieval layers:

1. A lightweight deterministic / scalar HTEMA ranker that runs immediately on the local DailyBean diary files.
2. A leakage-free evaluation harness that compares BM25, sparse semantic (TF-IDF+LSA), **dense semantic (multilingual MiniLM)**, scalar HTEMA, and neural MIRA from query text only.
3. A neural MIRA reranker over a 38-dim per-memory pair feature including semantic, temporal, emotional, diary-feature, **dense semantic cosine**, source/participant, continuity, BM25, and pairwise cross interactions.

Active memory hierarchy:

- **Level 2 — day tokens** (diary entries, one per day).
- **Level 2 — WhatsApp conversation windows** (chunked by date+time gap).
- **Level 1 — memory atoms** (paragraph-level slices of long diary days; auto-generated via `--include-atoms`).
- **Level 3 — weekly rollups** and **Level 4 — monthly rollups** (auto-aggregated via `--include-rollups`).
- **Level 5 — reflection tokens** (mood-streak, theme-arc, contradiction, and identity-pattern summaries from `scripts/reflect.py`, included via `--include-reflections`).

Continuous learning:

- `scripts/convert_feedback.py` converts Jarvis `feedback.jsonl` ratings into eval examples, treating `useful` sources as positives and `wrong`/`needs_work` sources as hard negatives (per the no-self-reinforcement protocol).

## Idea

Each diary day or WhatsApp conversation window becomes a memory token:

```text
token(memory) = {
  key: semantic + time + emotion + diary-feature projections,
  value: memory text + extracted metadata + neighboring sequence context
  source: diary | whatsapp
  participants: people attached to the memory window
}
```

Each user question becomes a query token:

```text
query(question) = {
  semantic intent,
  requested time window,
  emotion target,
  diary-specific hints such as mood, people, trip, work, music, health
}
```

Then the retriever runs multiple attention heads:

- semantic head: text similarity
- temporal head: date range and proximity
- emotion head: mood, valence, arousal, and emotional words
- diary-feature head: people, places, icons, activities, and diary tags
- continuity head: neighboring days and streaks

The output is a ranked set of memory values, plus per-head scores so we can inspect why a day was selected.

The prototype also includes a small alias layer for personal and multilingual matching. For example, `Cappadocia` expands toward `Kapadokya` and `Nevsehir`, so English queries can still find Turkish diary text.

## Run

From the repository root:

```bash
cd memory-attention-lab
npm run demo -- "happiest days at the end of 2024"
npm run query -- "what changed around the cappadocia trip" -- --limit 8
npm run query -- "sad and anxious days in March 2025" -- --json
```

The script reads:

```text
../knowledge_base/Deniz/diary/dailybean_2023_complete.md
../knowledge_base/Deniz/diary/dailybean_2024_complete.md
../knowledge_base/Deniz/diary/dailybean_2025_complete.md
```

## Where This Fits Later

This can become a Jarvis memory layer:

1. Keep Qdrant for vector search.
2. Add temporal and emotional payload fields to each chunk.
3. Use this attention scorer as a reranker after initial retrieval.
4. Add a `temporal_memory_search` tool that returns ranked days with attention-head explanations.
5. Later replace deterministic projections with learned linear layers or a small cross-encoder reranker once enough preference data exists.

## MIRA / HTEMA

The broader architecture is documented in [docs/mira_htema_design.md](docs/mira_htema_design.md).

Short version:

```text
MIRA = Memory, Introspection, Reflection Architecture
HTEMA = Hierarchical Temporal-Emotional Memory Attention
```

MIRA is the diary-native memory system. HTEMA is the attention/reranking core that learns how to attend over day, atom, week, month, event, and identity-pattern tokens.

## Generate Synthetic Query Training Data

The generator reads the existing Jarvis `.env`:

```text
../jarvis-platform/.env
```

and uses `LLM_BASE_URL`, `LLM_MODEL`, and `LLM_API_KEY` as an OpenAI-compatible API.

```bash
python3 scripts/generate_training_queries.py --limit 5
python3 scripts/generate_training_queries.py --limit 25 --output data/generated_queries.jsonl
python3 scripts/generate_training_queries.py --limit 5 --dry-run
```

The output is JSONL: one generated query training example per line.

## Generate Query Style Augmentations

The style augmenter uses the unified diary + WhatsApp benchmark and asks DeepSeek/OpenAI-compatible chat completions to create more natural query phrasings: vague fragments, Turkish-English mixes, typo-heavy search queries, voice-assistant style requests, emotional reflection, relationship context, and cross-source diary/WhatsApp questions.

```bash
python scripts/generate_style_augmentations.py --dry-run --limit 8 --output data/style_aug_prompt_preview.jsonl
python scripts/generate_style_augmentations.py --limit 1000 --augmentations-per-example 8 --batch-size 1 --workers 8 --model deepseek-v4-flash --output data/style_augmented_queries.jsonl
```

The script resumes safely by default. Re-running the same command only fills base examples that still need more augmentations.

Use the generated augmentations in the honest benchmark:

```bash
python scripts/honest_mira.py --split all --epochs 35 --max-examples 2400 --candidate-top-k 768 --batch-size 16 --extra-examples data/style_augmented_queries.jsonl
```

To focus on the current weak spot, generate more style-holdout-like examples:

```bash
python scripts/generate_style_augmentations.py --limit 1200 --augmentations-per-example 8 --style-filter emotion_month,mood_signal,relative_temporal --batch-size 1 --workers 8 --model deepseek-v4-flash --output data/style_augmented_queries.jsonl
```

Full-corpus generation can be parallelized by year:

```bash
python3 scripts/generate_training_queries.py --year 2023 --limit 999 --examples-per-entry 3 --batch-size 10 --output data/generated_queries_2023.jsonl --timeout 240 --max-tokens 12000
python3 scripts/generate_training_queries.py --year 2024 --limit 999 --examples-per-entry 3 --batch-size 10 --output data/generated_queries_2024.jsonl --timeout 240 --max-tokens 12000
python3 scripts/generate_training_queries.py --year 2025 --limit 999 --examples-per-entry 3 --batch-size 10 --output data/generated_queries_2025.jsonl --timeout 240 --max-tokens 12000
```

## Train HTEMA

Train the lightweight learned attention/reranking model:

```bash
python3 scripts/train_htema.py --training data/generated_queries.jsonl --model data/htema_model.json --epochs 120 --negatives 24
```

Search with the trained model:

```bash
python3 scripts/search_htema.py "happiest days at the end of 2024" --limit 5
python3 scripts/search_htema.py "what changed around the cappadocia trip" --limit 5
python3 scripts/search_htema.py "why do I keep thinking I always fail?" --limit 8
```

Evaluate:

```bash
python3 scripts/evaluate_htema.py --training data/generated_queries.jsonl --model data/htema_model.json
```

Important: `train_htema.py`, `evaluate_htema.py`, and the neural adapter scripts now run in query-text-only mode by default. The old `positive_window` behavior is available only with `--oracle-window` and should be treated as a diagnostic, not a real retrieval metric.

Legacy oracle-window diagnostic:

```text
Diary day tokens: 350
Synthetic query examples: 1052
Model: HTEMA lightweight pairwise ranker
Full generated-set recall@1: 0.781
Full generated-set recall@5: 0.933
Full generated-set MRR: 0.851
```

Those numbers are useful only for proving that temporal/emotional labels contain signal. They are not honest query-only retrieval results.

The trained model is stored at:

```text
data/htema_model.json
```

Generated JSONL data is ignored by git because it contains private diary-derived examples.

## Honest Neural MIRA Evaluation

The main current evaluator is:

```bash
python scripts/honest_mira.py --split all --epochs 35 --max-examples 1800 \
    --candidate-top-k 768 --include-rollups --include-atoms --include-reflections
```

It generates a deterministic personal-memory benchmark from diary day tokens,
WhatsApp conversation windows, weekly/monthly rollups, sub-day atoms, and
Level-5 reflections, then evaluates:

- `bm25`: Okapi BM25 over memory text
- `semantic_embed`: sparse TF-IDF + LSA semantic embeddings (kept as a sanity baseline)
- `dense_semantic`: dense multilingual **MiniLM** (`paraphrase-multilingual-MiniLM-L12-v2`) cosine over the same memory pool — works for Turkish-English mixes
- `scalar_htema`: fixed transparent HTEMA prior over per-pair feature vectors
- `neural_mira`: PyTorch candidate reranker fusing all of the above plus interaction features (`semantic_x_temporal`, `dense_x_temporal`, `dense_x_entity`, `bm25_x_temporal`, …)

Rules:

- Query text is the only retrieval input.
- `positive_window` is not used.
- Target dates are labels only.
- Random, month-holdout, and query-style-holdout splits are all reported.
- WhatsApp-derived labels are evaluated separately from diary labels in the report.
- `--candidate-top-k` makes neural evaluation scalable by reranking a transparent candidate pool built from BM25, semantic LSA, and scalar HTEMA.
- Candidate reranking calibrates neural, BM25, semantic, and scalar scores on the dev split so the final model can fall back toward transparent HTEMA when a held-out query style needs it.

Latest run (2026-05-18) — diary + WhatsApp + rollups + atoms + reflections, dense MiniLM head on:

```text
Memory tokens: 7657 (350 diary, 7098 WhatsApp, 87 rollups, 67 atoms, 55 reflections)
Generated benchmark examples before cap: 28873
Default benchmark cap: 1800 stratified examples
Default neural candidate pool: 768 memories/query

Split: random
BM25              R@1 0.457   R@5 0.596   MRR 0.534
Semantic embed    R@1 0.020   R@5 0.056   MRR 0.051
Dense semantic    R@1 0.033   R@5 0.091   MRR 0.070
Scalar HTEMA      R@1 0.427   R@5 0.636   MRR 0.528
Neural MIRA       R@1 0.826   R@5 0.962   MRR 0.885

Split: month_holdout
BM25              R@1 0.391   R@5 0.498   MRR 0.457
Semantic embed    R@1 0.004   R@5 0.032   MRR 0.035
Dense semantic    R@1 0.016   R@5 0.067   MRR 0.051
Scalar HTEMA      R@1 0.435   R@5 0.680   MRR 0.551
Neural MIRA       R@1 0.767   R@5 0.945   MRR 0.844

Split: style_holdout
BM25              R@1 0.045   R@5 0.156   MRR 0.119
Semantic embed    R@1 0.010   R@5 0.038   MRR 0.040
Dense semantic    R@1 0.005   R@5 0.038   MRR 0.034
Scalar HTEMA      R@1 0.275   R@5 0.641   MRR 0.445
Neural MIRA       R@1 0.390   R@5 0.756   MRR 0.548
```

Notes:

- Calibration weights are learned per-split on the dev set. The dense head
  is one of the candidate channels and one of the calibration components,
  not a strict baseline beater on its own — it shines when fused with BM25,
  scalar HTEMA, and the source/temporal features in the neural reranker.
- Hierarchical co-positives: a month-level query like "happiest diary days
  in March 2025" counts the `rollup:month:2025-03` token and any `atom:` of
  the gold leaves as correct answers. This rewards the model for legitimate
  surfacing of aggregate memory rather than penalising it.

The generated aggregate report is stored at:

```text
docs/honest_evaluation_report.md
```

Gate the latest metrics against conservative hardening thresholds:

```bash
npm run eval:gate
```

Use this after retraining so random, month-holdout, and style-holdout regressions
are caught before a checkpoint becomes the active Jarvis model.

## Feedback-Derived Eval Cases

Jarvis stores answer feedback locally in:

```text
../jarvis-platform/data/feedback.jsonl
```

Convert ratings into MIRA eval examples:

```bash
python scripts/convert_feedback.py --output data/feedback_eval_examples.jsonl
```

Protocol — preserve learning signal without amplifying retrieval bugs:

- `useful` rows: displayed sources are emitted as `positive_ids`.
- `wrong` / `needs_work` rows: displayed sources are emitted as `negative_ids`.
  The script looks in the user's `note` for an explicit date or memory id; if
  one is found it becomes the positive, otherwise the row is skipped (the
  README's no-self-reinforcement rule).

Then mix the converted examples into the next training run:

```bash
python scripts/honest_mira.py --split all --epochs 35 --max-examples 1800 \
    --candidate-top-k 768 --extra-examples data/feedback_eval_examples.jsonl \
    --include-rollups --include-atoms --include-reflections
```

Private `.pt` artifacts are tracked in this private repo with Git LFS so the
model can continue training/generalizing from the latest checkpoint:

```text
data/neural_embeddings.pt
data/neural_htema_model.pt
data/neural_mira_model.pt
```

These files are not intended to contain raw diary or chat text, but they are
still private-derived model state. Keep them inside the private repo boundary.

Generated metric JSON remains local:

```text
data/neural_mira_metrics.json
```

Search with the trained neural MIRA ranker:

```bash
python scripts/search_mira.py "happiest days at the end of 2024" --limit 8
python scripts/search_mira.py "what did Kemal Piknik Budapest say about ahshahahahahahshs" --limit 5 --include-out-of-window
python scripts/search_mira.py "sad and anxious days in May 2025" --limit 8 --json
```

For date-bound questions, `search_mira.py` applies a temporal candidate filter by default. Pass `--include-out-of-window` when you explicitly want emotionally/semantically similar memories outside the parsed date window.

## Dense Semantic Head

Cached multilingual MiniLM embeddings live in `data/neural_embeddings.pt`.
Rebuild after expanding the corpus (e.g. new WhatsApp exports, new reflections,
new diary entries):

```bash
python scripts/build_neural_embeddings.py --batch-size 128 --device cuda:0 \
    --include-rollups --include-atoms --include-reflections
```

When `honest_mira.py` / `search_mira.py` / `mira_service.py` start, they detect
the cache, verify that `memory_ids` matches the current corpus, and reuse the
embeddings directly. When the corpus drifts (new memories, no rebuild yet),
they fall back to fresh encoding on first call.

Disable the dense head for ablation:

```bash
python scripts/honest_mira.py --no-dense
```

## Hierarchy: Rollups, Atoms, Reflections

The HTEMA design has five levels. The lab now generates all of them from
deterministic rules:

| Level | Source type | Builder |
| --- | --- | --- |
| 1 | `atom` | `htema_core.build_memory_atoms` (paragraph splits of long diary days) |
| 2 | `diary` | `parse_diary_memories` |
| 2 | `whatsapp` | `parse_whatsapp_memories` |
| 3 | `rollup_week` | `htema_core.build_rollup_memories` |
| 4 | `rollup_month` | `htema_core.build_rollup_memories` |
| 5 | `reflection` | `scripts/reflect.py` (mood streaks, theme arcs, contradictions, identity patterns) |

Generate Level-5 reflection tokens once after major corpus changes:

```bash
python scripts/reflect.py
```

That writes `data/reflections.jsonl`. The reflection schema is intentionally
auditable (no LLM is invoked) — every reflection lists its `source_ids` so a
human can trace why it was emitted.

## Date Parser

`htema_core.parse_time_window` resolves natural language windows in both
English and Turkish:

- Explicit dates: `2024.06.01`, `between 2024-06-01 and 2024-06-15`
- Year + month name: `March 2025`, `Mart 2025`, `Mayıs 2025`
- Seasons: `spring 2024`, `ilkbahar 2024`, `last summer`, `geçen yaz`
- Year + phase: `late 2024`, `2024 sonu`, `first half of 2025`, `early 2023`, `2023 başı`
- Relative: `today`, `yesterday`, `bugün`, `dün`, `this week`, `geçen ay`, `last year`, `bu sene`
- Bare year: `2024`

Pass a reference date via `parse_time_window(query, now=...)` for deterministic
relative-date tests.

## Neural Q/K/V HTEMA

The neural path keeps the diary-specific temporal, emotional, entity, continuity, contradiction, and importance signals, then learns query/key/value adapters over pretrained multilingual sentence embeddings.

At search time, it uses:

- neural Q/K/V adapter score
- scalar diary-attention prior from `data/htema_model.json`
- temporal filtering for date-bound questions, unless `--include-out-of-window` is passed

Create the local Python environment and install the neural dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-neural.txt
```

Build the embedding cache:

```bash
python scripts/build_neural_embeddings.py --training data/generated_queries.jsonl --embedding-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --batch-size 64
```

Train the neural adapter reranker:

```bash
python scripts/train_neural_htema.py --training data/generated_queries.jsonl --epochs 120 --negatives 64 --batch-size 32 --device auto
```

Search with the neural model:

```bash
python scripts/search_neural_htema.py "happiest days at the end of 2024" --limit 8 --device auto
python scripts/search_neural_htema.py "happiest days at the end of 2024" --limit 8 --device auto --include-out-of-window
```

Evaluate the final neural model:

```bash
python scripts/evaluate_neural_htema.py --training data/generated_queries.jsonl --device auto
```

Legacy neural oracle-window diagnostic:

```text
Diary day tokens: 350
Synthetic query examples: 1052
Model: HTEMA neural Q/K/V adapter
Full generated-set recall@1: 0.999
Full generated-set recall@5: 1.000
Full generated-set MRR: 0.999
```

Those numbers were inflated by generated labels and should not be used as the main claim. Use `scripts/honest_mira.py` for the current honest benchmark.

The trained neural model and metrics are stored at:

```text
data/neural_htema_model.pt
data/neural_htema_metrics.json
```

## Baseline Evaluation

Use baseline evaluation before claiming the architecture works. The default mode uses only the query text, so it does not leak generated label metadata into retrieval:

```bash
python scripts/evaluate_baselines.py --training data/generated_queries.jsonl --device auto
```

For a leakage / upper-bound diagnostic, rerun with oracle windows:

```bash
python scripts/evaluate_baselines.py --training data/generated_queries.jsonl --device auto --use-oracle-window
```

To include the trained neural Q/K/V model in the same sweep:

```bash
python scripts/evaluate_baselines.py --training data/generated_queries.jsonl --device auto --include-neural
```

First baseline readout:

```text
Query-text-only mode:
BM25 R@1 0.280, R@5 0.472, MRR 0.373
Scalar HTEMA R@1 0.220, R@5 0.388, MRR 0.305
Semantic embedding R@1 0.106, R@5 0.225, MRR 0.174

Oracle-window mode:
Temporal-only R@1 0.700, R@5 0.983, MRR 0.808
Scalar HTEMA R@1 0.781, R@5 0.934, MRR 0.851
```

Interpretation: the original high scores depended heavily on label-provided time windows. The next serious benchmark should use harder query-only splits and retrain/evaluate without oracle windows.
