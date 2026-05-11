# Memory Attention Lab

Separate experiment for a diary-focused, Transformer-like retrieval mechanism.

The lab now has two working layers:

1. A lightweight deterministic / scalar HTEMA ranker that runs immediately on the local DailyBean diary files.
2. A neural HTEMA ranker that learns Q/K/V adapters over pretrained multilingual sentence embeddings, then combines the neural score with a calibrated diary-attention prior for robust search.

## Idea

Each diary day becomes a memory token:

```text
token(day) = {
  key: semantic + time + emotion + diary-feature projections,
  value: day text + extracted metadata + neighboring sequence context
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

This can become a Jarvis v2 memory layer:

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

Current v1 run:

```text
Diary day tokens: 350
Synthetic query examples: 1052
Model: HTEMA lightweight pairwise ranker
Full generated-set recall@1: 0.781
Full generated-set recall@5: 0.933
Full generated-set MRR: 0.851
```

The trained model is stored at:

```text
data/htema_model.json
```

Generated JSONL data is ignored by git because it contains private diary-derived examples.

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

Current neural run:

```text
Diary day tokens: 350
Synthetic query examples: 1052
Model: HTEMA neural Q/K/V adapter
Full generated-set recall@1: 0.999
Full generated-set recall@5: 1.000
Full generated-set MRR: 0.999
```

The trained neural model and metrics are stored at:

```text
data/neural_htema_model.pt
data/neural_htema_metrics.json
```
