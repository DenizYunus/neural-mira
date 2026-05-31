# Quick start

Two paths: try the **Anne Frank demo** (no diaries of your own needed) or **bring your own**.

## Path A — Anne Frank demo (3 minutes)

> ⚠️ The Anne Frank fetcher script (`scripts/fetch_demo_data.py`) is on the immediate roadmap. Once it lands, this section becomes the canonical "first thing you do after cloning." For now, follow Path B with any small set of dated markdown entries.

Anne Frank's diary is in the EU public domain since 2016 (70 years after her death in 1945). The fetcher will:

1. Download a stable PD edition of the diary text.
2. Parse each entry into the DailyBean markdown format Neural MIRA expects.
3. Infer mood (1–5) from sentiment over each entry.
4. Place the result at `data/diaries/anne_frank.md`.
5. Build the dense embedding cache.
6. Train a tiny reranker on a few hundred LLM-synthesized queries (or skip training and use scalar HTEMA out of the box).

After that you can search:

```bash
python scripts/search_mira.py "days when she felt hopeful" --limit 5
python scripts/search_mira.py "what changed after they heard about the camps"
python scripts/search_mira.py "her thoughts about Peter in early 1944"
```

The diary is short enough (~270 entries) that the full pipeline finishes in minutes on a laptop CPU.

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
