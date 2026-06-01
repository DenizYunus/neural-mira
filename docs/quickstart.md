# Quick start

Two paths: try the **Anne Frank demo** (no diaries of your own needed) or **bring your own**.

## Path A — Anne Frank demo (~5 minutes + LLM time)

`scripts/fetch_demo_data.py` bootstraps the demo dataset from scratch.

### Why Anne Frank's diary as the demo?

It's a near-perfect test corpus for autobiographical-memory retrieval:

- **Real diary, real person, real emotional arc.** No synthetic-data fingerprint.
- **Short** — ~76 entries from June 1942 to August 1944. Full pipeline finishes in minutes.
- **Emotionally textured** — fear, boredom, hope, romance, grief, gallows humor. Exercises the mood + emotion heads properly.
- **Dense in named entities** — Kitty, Peter, Margot, Mrs. Van Daan, Mr. Dussel, Mummy, Daddy. Tests people-filter retrieval.
- **Naturally dated** — every entry has an explicit date header, so temporal queries work cleanly.
- **Public domain in the EU (since 2016) and Australia (since 1995).** The fetcher links to a third-party hosted copy; the bytes never enter this repo.

### Run it

```bash
# 1. Install + configure
pip install -r requirements-neural.txt
cp .env.example .env
# Open .env and set LLM_API_KEY. DeepSeek works great and costs $0.10-0.30
# for the full ~76-entry tagging pass. OpenAI / Anthropic-compatible /
# local vLLM endpoints all work — just set LLM_BASE_URL accordingly.

# 2. Smoke test first (5 entries, ~30 seconds, ~$0.01)
python scripts/fetch_demo_data.py --max 5

# 3. If that looks right, run the full thing
python scripts/fetch_demo_data.py
```

What it does:

1. **Downloads** a text-based PDF of *The Diary of a Young Girl* (cached locally — runs offline after first fetch).
2. **Extracts** 76 clean entries by parsing the original date headers ("Saturday, 13 June 1942"). The PDF's table of contents is detected and skipped automatically.
3. **Tags each entry via LLM** — one chat-completions call per entry, extracting:
   - **mood** (1–5) where 1 = hopeless, 5 = full of hope
   - **icons** — lowercase string tags mixing people (`kitty`, `peter`, `mother`, `father`, `mrs_van_daan`, `mr_dussel`), places (`secret_annex`, `school`, `amsterdam`), themes (`fear`, `hope`, `loneliness`, `growing_up`, `war_news`, `bombing`), and activities (`reading`, `writing`, `birthday`, `argument_with_mother`)
4. **Writes** to `data/diaries/anne_frank.md` in the exact `### YYYY-MM-DD` + `**Mood**` + `**Icons**` format `parse_diary_memories` expects.

Resumable by default — Ctrl-C is safe, just re-run the same command. Output is flushed per entry, so you lose at most one entry to an interrupt.

The fetcher uses an LLM by default but `--no-llm` ships dates + bodies only if you'd rather not spend API tokens.

### After the fetch — query examples

Search with scalar HTEMA (no training needed, works immediately):

```bash
# Mood + temporal
python scripts/search_htema.py "days she felt hopeful" --limit 5

# Person filter + temporal
python scripts/search_htema.py "her thoughts about Peter in early 1944"

# Emotional + diary-feature head
python scripts/search_htema.py "moments of fear during air raids"

# Place + mood
python scripts/search_htema.py "small joys inside the secret annex"

# Relationship + emotion
python scripts/search_htema.py "arguments with her mother"
```

Each query exercises a different combination of attention heads — the mood + temporal + emotion + diary-feature heads were designed to compose, and Anne Frank's diary has the structural richness to actually test them.

### Train your own Neural MIRA reranker (optional)

The demo works out-of-the-box with scalar HTEMA. To get the full Neural MIRA reranker, train on synthesized query examples (see [`docs/training.md`](training.md)):

```bash
# Generate ~150 training examples from the demo diary
python scripts/generate_training_queries.py --limit 50 --examples-per-entry 3 \
    --output data/generated_queries.jsonl

# Train
python scripts/honest_mira.py --split all --epochs 20 --max-examples 800

# Search with the trained checkpoint
python scripts/search_mira.py "days she felt hopeful" --limit 5
```

### Default source + fallback

The default source URL is a text-based PDF of the English Definitive Edition. If that URL ever stops working, the Internet Archive has a more stable (but noisier) OCR text fallback:

```bash
python scripts/fetch_demo_data.py --source-url \
    https://archive.org/download/in.ernet.dli.2015.201940/2015.201940.Anne-Frank_djvu.txt
```

Anyone in a jurisdiction where they prefer a different source — point `--source-url` at it. The script auto-detects format (`.pdf` → PDF extraction, anything else → plain text).

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
