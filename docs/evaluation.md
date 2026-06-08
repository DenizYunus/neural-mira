# Evaluation

How Neural MIRA is benchmarked, and why the legacy numbers (R@1 ≈ 0.99) were misleading.

## Honest benchmark (the one that matters)

```bash
python scripts/honest_mira.py --split all --epochs 35 --max-examples 1800 \
    --candidate-top-k 768 --include-rollups --include-atoms --include-reflections
```

Generates a deterministic personal-memory benchmark from diary day tokens, WhatsApp conversation windows, weekly/monthly rollups, sub-day atoms, and Level-5 reflections, then evaluates:

| Method | What it is |
|---|---|
| `bm25` | Okapi BM25 over memory text |
| `semantic_embed` | sparse TF-IDF + LSA (sanity baseline) |
| `dense_semantic` | dense multilingual MiniLM (`paraphrase-multilingual-MiniLM-L12-v2`) cosine over the memory pool — works for Turkish ↔ English mixes |
| `scalar_htema` | fixed transparent HTEMA prior over per-pair feature vectors |
| `neural_mira` | PyTorch candidate reranker fusing all of the above + interaction features (`semantic_x_temporal`, `dense_x_temporal`, `dense_x_entity`, `bm25_x_temporal`, …) |

## Evaluation rules

- **Query text is the only retrieval input.** No `positive_window` leakage.
- **Target dates are labels only.**
- Three splits are reported: **random**, **month_holdout**, **query-style_holdout**.
- WhatsApp-derived labels are evaluated separately from diary labels in the report.
- `--candidate-top-k` makes neural eval scalable by reranking a transparent candidate pool built from BM25 + semantic LSA + scalar HTEMA.
- Candidate reranking calibrates the four base scorers on the dev split so the final model can fall back toward transparent HTEMA when a held-out query style needs it.

## Latest run

Diary + WhatsApp + rollups + atoms + reflections; dense MiniLM head on.

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

## Reading these numbers

- Neural MIRA wins all three splits, but the **gap shrinks dramatically** on `style_holdout` — that's the hardest case (the model sees query phrasings it wasn't trained on). Style augmentations (`scripts/generate_style_augmentations.py`) are how we close it.
- The dense semantic baseline looks weak on its own — that's expected. It's a **candidate channel** and a calibration component, not a strict beater. It shines when fused with BM25, scalar HTEMA, and source/temporal features in the neural reranker.
- **Hierarchical co-positives**: a month-level query like *"happiest diary days in March 2025"* counts the `rollup:month:2025-03` token and any `atom:` of the gold leaves as correct answers. This rewards the model for legitimate surfacing of aggregate memory rather than penalising it.

## Regression gate

After retraining, gate the latest metrics against conservative thresholds:

```bash
npm run eval:gate
```

Catches random / month-holdout / style-holdout regressions before a checkpoint becomes the active model.

The aggregate report is written to:

```text
docs/honest_evaluation_report.md
```

## Missing-day reconstruction benchmark

The first paper-track benchmark hides direct diary days, removes any derived
memory whose provenance points at the hidden day, and retrieves indirect
evidence. It now reports the first required paper baselines:

- `mira_trust`: scalar MIRA with source-trust weighting
- `mira_no_trust`: same scorer with trust weighting disabled
- `simple_rag_all`: lexical cosine over all allowed memories
- `diary_only`: MIRA over direct diary memories only
- `chat_only`: MIRA over WhatsApp / WhatsApp-synthetic memories only
- `chronological_neighbors`: nearest allowed memories by date

```bash
npm run reconstruction:bench
```

Outputs stay under `data/` by default so private corpora do not leak into docs:

```text
data/reconstruction_report.md
data/reconstruction_metrics.json
```

This is the scaffold for the arXiv claim: the final paper version should add
public/synthetic splits, human-validated queries, dense retrieval, and
LLM/long-context baselines where feasible.

## Public synthetic benchmark

The Anne Frank demo is diary-only, so it cannot test chat-derived
reconstruction. Generate the public multi-source synthetic corpus instead:

```bash
npm run synthetic:generate
npm run synthetic:reconstruction
```

The corpus lives under `benchmarks/mira_synthetic/generated/` and includes
diary labels, WhatsApp-style chats, photo metadata, ground truth events,
corrections, and hidden target dates. The synthetic reconstruction command
passes diary labels, chats, and photo metadata into the benchmark, while keeping
local private reflections out of the public run.

## Baseline sweep

Before claiming the architecture works, you should sanity-check baselines. The default mode uses only query text — no generated-label leakage:

```bash
python scripts/evaluate_baselines.py \
    --training data/generated_queries.jsonl --device auto
```

For an upper-bound diagnostic with oracle windows:

```bash
python scripts/evaluate_baselines.py \
    --training data/generated_queries.jsonl --device auto --use-oracle-window
```

Include the trained neural Q/K/V model:

```bash
python scripts/evaluate_baselines.py \
    --training data/generated_queries.jsonl --device auto --include-neural
```

### Why the old "R@1 ≈ 0.99" numbers were misleading

Earlier evaluations passed the `positive_window` label into retrieval as an oracle. Removing that:

```text
Query-text-only mode:
  BM25                R@1 0.280  R@5 0.472  MRR 0.373
  Scalar HTEMA        R@1 0.220  R@5 0.388  MRR 0.305
  Semantic embedding  R@1 0.106  R@5 0.225  MRR 0.174

Oracle-window mode:
  Temporal-only       R@1 0.700  R@5 0.983  MRR 0.808
  Scalar HTEMA        R@1 0.781  R@5 0.934  MRR 0.851
```

The original "high scores" depended on label-provided time windows. The current `honest_mira.py` numbers above are the truth.
