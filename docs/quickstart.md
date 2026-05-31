# Quick start

Two paths: try the **Anne Frank demo** (no diaries of your own needed) or **bring your own**.

## Path A — Anne Frank demo (~5 minutes + LLM time)

`scripts/fetch_demo_data.py` bootstraps the demo dataset from scratch.

```bash
# 1. Install + configure
pip install -r requirements-neural.txt
cp .env.example .env
# Open .env and set LLM_API_KEY (any OpenAI-compatible endpoint).
# DeepSeek works great and costs ~$0.30 for the full diary.

# 2. Fetch + tag the diary (LLM enriches each entry with mood + icons)
python scripts/fetch_demo_data.py

# Optional smoke test first — process only 5 entries to verify end-to-end:
python scripts/fetch_demo_data.py --max 5
```

What it does:

1. Downloads Anne Frank's *The Diary of a Young Girl* OCR plain text from Internet Archive (cached locally so re-runs don't re-download).
2. Parses entries by their original date headers ("Sunday, 14 June, 1942") and reformats into `### YYYY-MM-DD` blocks.
3. For each entry, calls the configured LLM to extract:
   - **mood** (1–5) from sentiment — 1 = hopeless, 5 = full of hope
   - **icons** — lowercase string tags for people (`kitty`, `peter`, `father`), places (`secret_annex`, `amsterdam`), themes (`fear`, `hope`, `growing_up`, `war_news`), and activities (`reading`, `writing`, `birthday`, `argument_with_mother`)
4. Writes everything to `data/diaries/anne_frank.md` in the exact format `parse_diary_memories` expects.

Resumable by default — Ctrl-C is safe, just re-run the same command. Output is flushed per entry, so you lose at most one entry to an interrupt.

The fetcher uses an LLM by default but `--no-llm` ships dates + bodies only if you'd rather not spend API tokens.

### After the fetch

Search with scalar HTEMA (no training needed — works immediately):

```bash
python scripts/search_htema.py "days when she felt hopeful" --limit 5
python scripts/search_htema.py "what changed after they heard about the camps"
python scripts/search_htema.py "her thoughts about Peter in early 1944"
```

Or train your own Neural MIRA reranker on the demo data (see [`docs/training.md`](training.md)):

```bash
python scripts/generate_training_queries.py --limit 50 --output data/generated_queries.jsonl
python scripts/honest_mira.py --split all --epochs 20 --max-examples 800
python scripts/search_mira.py "days when she felt hopeful" --limit 5
```

### Sourcing note

The diary is public domain in the EU since 2016 (70 years after Anne Frank's death in 1945) and in Australia since 1995. The default source URL is Internet Archive's OCR text. US users should verify their local copyright status — `--source-url` lets you point the fetcher at a different source if needed.

## Path B — Bring your own diary

### 1. Clone + install

```bash
git clone https://github.com/DenizYunus/neural-mira.git
cd neural-mira
python3 -m venv .venv
source .venv/bin/activate         # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements-neural.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env if you need LLM API access for training-data synthesis.
# (Defaults are fine if you only want to run the deterministic / scalar paths.)
```

### 3. Drop your diaries

Put one or more `*.md` files in `data/diaries/`. Two header styles are supported (parser auto-detects):

**Style A — DailyBean** (one entry per `## Month DD, YYYY` heading):

```markdown
## January 5, 2024
Felt sharp today. Got the chapter outline done in one sitting, which never
happens. Saw 🥲 emoji on the train and laughed for some reason.

## January 6, 2024
Slow morning, slower afternoon. Talked with M about the move — leaning yes.
```

**Style B — bullet** (`### YYYY-MM-DD` headers):

```markdown
### 2024-01-05
- Sharp focus all morning. Chapter outline done.
- Saw a 🥲 sticker on the train, made me laugh.

### 2024-01-06
- Slow day. Lunch with M, talked about the move.
```

Full format reference: [`data/README.md`](../data/README.md).

### 4. Build embeddings + search

```bash
# Cache MiniLM embeddings (one-time, runs again only when corpus drifts)
python scripts/build_neural_embeddings.py --batch-size 64 --device cpu

# Try scalar HTEMA — works immediately, no training needed
python scripts/search_htema.py "what changed around the move" --limit 5

# Or via Node
npm run query -- "happiest weeks of 2024" -- --limit 5
```

### 5. Train your own neural reranker (optional)

If you have at least a few hundred entries and want the full neural MIRA:

```bash
# Synthesize training data (needs LLM_API_KEY in .env)
python scripts/generate_training_queries.py --limit 100 --examples-per-entry 3 \
    --output data/generated_queries.jsonl

# Optional: style augmentations for robustness
python scripts/generate_style_augmentations.py --limit 200 \
    --output data/style_augmented_queries.jsonl

# Train
python scripts/honest_mira.py --split all --epochs 35 --max-examples 1800 \
    --candidate-top-k 768 \
    --extra-examples data/style_augmented_queries.jsonl

# Search with the trained checkpoint
python scripts/search_mira.py "your query here" --limit 5
```

Full training recipe + tuning tips: [`docs/training.md`](training.md).

## What to read next

- **You want to understand the design** → [`docs/architecture.md`](architecture.md) + [`docs/components.md`](components.md)
- **You want to train on your own data** → [`docs/training.md`](training.md)
- **You want to know how good it is** → [`docs/evaluation.md`](evaluation.md)
