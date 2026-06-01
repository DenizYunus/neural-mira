# Neural MIRA

> Neural attention head for personal recall. Trains on diary entries to learn temporal, source-trust, and relevance-aware ranking.

+ ![Neural MIRA](docs/header.png)

## Why

Generic retrieval-augmented systems treat *"find my notes about X"* the same as *"find a Wikipedia article about X"* — a single vector-similarity sort over a flat document store. That works for facts. It fails for **autobiographical** memory, where the right answer depends on:

- **When** something happened ("last week", "last year", "around the move")
- **Where** the trace lives (a polished diary entry beats a WhatsApp message beats an LLM-summarized day)
- **Who** was involved (people, places, recurring themes)
- **What state** you were in (mood, the surrounding week, what came before)

Neural MIRA is a small attention head trained to fuse those signals — plus dense semantic match — into a single ranking. It rides on top of a transparent feature pipeline (BM25, scalar HTEMA, multilingual MiniLM, source-trust priors, temporal proximity) so you can see *why* a memory surfaced, not just *that* it did.

## Quick start

The fastest way to see Neural MIRA work is on someone else's diary. **Anne Frank's *The Diary of a Young Girl*** is short (~76 entries, June 1942 → August 1944), emotionally rich (mood, fear, hope, growing up, hiding from war), and structurally simple — every entry is dated. Public domain in the EU since 2016 and Australia since 1995.

Five commands from clone to working search:

```bash
git clone https://github.com/DenizYunus/neural-mira.git
cd neural-mira
pip install -r requirements-neural.txt
cp .env.example .env                  # then paste your LLM_API_KEY
python scripts/fetch_demo_data.py     # downloads diary, LLM-tags each entry (~$0.10-0.30)
python scripts/search_htema.py "days she felt hopeful" --limit 5
```

What happens under the hood:

1. Downloads a text-based PDF of the diary (cached locally — runs offline after the first fetch).
2. Extracts 76 dated entries with clean prose.
3. Calls your configured LLM (DeepSeek / OpenAI / any compatible) once per entry to extract:
   - **mood** 1–5 from sentiment
   - **icons** — lowercase tags for people (`kitty`, `peter`, `father`), places (`secret_annex`, `school`), themes (`fear`, `hope`, `war_news`, `growing_up`), and activities (`birthday`, `reading`, `argument_with_mother`)
4. Writes everything to `data/diaries/anne_frank.md` in the format the parser expects.

After that, try queries that exercise different signals:

```bash
python scripts/search_htema.py "her thoughts about Peter in early 1944"
python scripts/search_htema.py "moments of fear during air raids"
python scripts/search_htema.py "small joys inside the secret annex"
python scripts/search_htema.py "arguments with her mother"
```

The first hits the **temporal** head (early 1944), the second the **emotional** + **diary-feature** heads (fear + raids), the third combines **dense semantic** + **source-trust**, the fourth tests **people-entity** matching.

> **Want to use your own diaries instead?** Skip `fetch_demo_data.py` and drop `*.md` files into `data/diaries/` in the format documented at [`data/README.md`](data/README.md). The parser auto-discovers them.

Full walkthrough with sanity-check tips and bring-your-own-diary instructions: [`docs/quickstart.md`](docs/quickstart.md).

## What you get in 30 seconds

1. Parses dated markdown diary entries into **memory tokens**
2. Builds a **5-level hierarchy** — sub-day atoms, diary days, weekly rollups, monthly rollups, reflection patterns
3. Ranks candidates with a **neural attention head** trained on (query, positive, negatives) examples — fusing semantic, temporal, emotional, source-trust, and continuity signals
4. Returns ranked memories with per-head scores so you can inspect *why*

## Latest evaluation (honest benchmark, query-text-only)

```text
Split: random          Neural MIRA  R@1 0.826  R@5 0.962  MRR 0.885
Split: month_holdout   Neural MIRA  R@1 0.767  R@5 0.945  MRR 0.844
Split: style_holdout   Neural MIRA  R@1 0.390  R@5 0.756  MRR 0.548
```

Full numbers + baseline comparisons in [`docs/evaluation.md`](docs/evaluation.md).

## Deep dives

- [**`docs/quickstart.md`**](docs/quickstart.md) — Anne Frank demo walkthrough + bring-your-own-diary setup
- [**`docs/components.md`**](docs/components.md) — What's in the repo (3 retrieval layers, 5-level hierarchy, what each script does)
- [**`docs/training.md`**](docs/training.md) — End-to-end training pipeline: synthesizing query examples, style augmentations, neural training, dense embedding cache, feedback-derived eval cases
- [**`docs/evaluation.md`**](docs/evaluation.md) — Honest evaluation methodology, splits, metrics, baseline comparisons
- [**`docs/architecture.md`**](docs/architecture.md) — Detailed design: token/query structure, attention heads, why-not-transformer-yet, learning-rate path to learned weights
- [**`docs/mira_htema_design.md`**](docs/mira_htema_design.md) — Original design notes (MIRA = Memory, Introspection, Reflection Architecture; HTEMA = Hierarchical Temporal-Emotional Memory Attention)

## Origin

Originated as the research arm of a larger personal-memory system. References to `jarvis-platform` and `mira-platform` elsewhere in the docs point at that original parent project (not included here). The code in this repo is generic — point it at any directory of dated Markdown entries.

## License

MIT (see [LICENSE](LICENSE)).
