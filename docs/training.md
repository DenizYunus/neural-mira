# Training pipeline

End-to-end recipe to train Neural MIRA on your own diary corpus.

## Prerequisites

```bash
# Once
python3 -m venv .venv
source .venv/bin/activate    # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-neural.txt

# LLM credentials for synthesis scripts (not needed for training/eval themselves)
cp .env.example .env
# Edit .env: set LLM_API_KEY (any OpenAI-compatible endpoint via LLM_BASE_URL).
```

To reuse another private env file without copying secrets into this repo, set
`NM_ENV_FILE` for the command:

```bash
NM_ENV_FILE=/path/to/private/.env python scripts/generate_training_queries.py --limit 10
```

PowerShell:

```powershell
$env:NM_ENV_FILE="C:\path\to\private\.env"
python scripts\generate_training_queries.py --limit 10
```

## 0. Have a diary

Drop one or more `.md` files into `data/diaries/`. See [`data/README.md`](../data/README.md) for the expected DailyBean / bullet-style markdown formats.

## 1. Generate synthetic query training data

Reads `LLM_*` from `.env` (any OpenAI-compatible endpoint).

```bash
# Small smoke test (5 diary entries, ~15 generated examples)
python scripts/generate_training_queries.py --limit 5

# Real run — one year at a time, parallel-safe
python scripts/generate_training_queries.py --year 2024 --limit 999 \
    --examples-per-entry 3 --batch-size 10 \
    --output data/generated_queries_2024.jsonl --timeout 240 --max-tokens 12000

# Dry-run to inspect prompts without spending API tokens
python scripts/generate_training_queries.py --limit 5 --dry-run
```

Output is JSONL — one training example per line.

## 2. (Optional) WhatsApp synthesis for off-diary days

If you have WhatsApp chat exports and set `WHATSAPP_ROOT` in `.env`, fill in days where you didn't journal:

```bash
python scripts/synthesize_whatsapp_diaries.py \
    --output data/whatsapp_synthetic_diaries.jsonl
```

Summaries are written third-person ("Deniz mentioned…" — placeholder name in the prompt) and tagged `source_type="whatsapp_synthetic"` with provenance excerpts. They get a lower trust prior than your real diary words.

## 3. Style augmentations

Diversify query phrasings so the model doesn't overfit to one voice:

```bash
# Dry-run preview
python scripts/generate_style_augmentations.py --dry-run --limit 8 \
    --output data/style_aug_prompt_preview.jsonl

# Real run — 1000 base examples × 8 augmentations each
python scripts/generate_style_augmentations.py \
    --limit 1000 --augmentations-per-example 8 \
    --batch-size 1 --workers 8 \
    --model deepseek-v4-flash \
    --output data/style_augmented_queries.jsonl
```

Augmentations cover: vague fragments, Turkish-English mixes, typo-heavy search queries, voice-assistant style, emotional reflection, relationship context, cross-source diary/WhatsApp questions, multi-hop life context.

The script resumes safely by default — re-running fills base examples that still need augmentations.

To focus on a weak spot:

```bash
python scripts/generate_style_augmentations.py --limit 1200 \
    --augmentations-per-example 8 \
    --style-filter emotion_month,mood_signal,relative_temporal \
    --batch-size 1 --workers 8 \
    --output data/style_augmented_queries.jsonl
```

## 4. Build the dense embedding cache

Multilingual MiniLM embeddings cached to disk so honest_mira and search_mira can reuse them instead of re-encoding per query:

```bash
python scripts/build_neural_embeddings.py \
    --batch-size 128 --device cuda:0 \
    --include-rollups --include-atoms --include-reflections
```

Output: `data/neural_embeddings.pt`. Rebuild after every corpus expansion. The downstream scripts verify `memory_ids` match the current corpus on startup and fall back to fresh encoding if drifted.

## 5. Train the scalar HTEMA model

Lightweight learned reranker (legacy path, useful as a baseline + fallback):

```bash
python scripts/train_htema.py \
    --training data/generated_queries.jsonl \
    --model data/htema_model.json \
    --epochs 120 --negatives 24

# Evaluate
python scripts/evaluate_htema.py \
    --training data/generated_queries.jsonl \
    --model data/htema_model.json

# Query
python scripts/search_htema.py "happiest days at the end of 2024" --limit 5
```

> **Note**: `train_htema.py`, `evaluate_htema.py`, and the neural adapter scripts run in query-text-only mode by default. The old `positive_window` behavior is available only via `--oracle-window` and should be treated as diagnostic, not as a real retrieval metric.

## 6. Train Neural MIRA (the main path)

```bash
python scripts/honest_mira.py --split all --epochs 35 --max-examples 2400 \
    --candidate-top-k 768 --batch-size 16 \
    --extra-examples data/style_augmented_queries.jsonl \
    --include-rollups --include-atoms --include-reflections
```

Saves `data/neural_mira_model.pt` + `data/neural_mira_metrics.json`. Calibration weights are learned per-split on the dev set.

For ablation, disable the dense head:

```bash
python scripts/honest_mira.py --no-dense
```

## 7. Search with the trained model

```bash
python scripts/search_mira.py "happiest days at the end of 2024" --limit 8
python scripts/search_mira.py "sad and anxious days in May 2025" --limit 8 --json
python scripts/search_mira.py "what changed around the move" \
    --limit 5 --include-out-of-window
```

For date-bound questions, `search_mira.py` applies a temporal candidate filter by default. Pass `--include-out-of-window` when you explicitly want emotionally / semantically similar memories outside the parsed date window.

## 8. Feedback-derived eval cases

If your downstream UI captures thumbs-up / thumbs-down on retrieved memories, convert ratings into training examples:

```bash
python scripts/convert_feedback.py --output data/feedback_eval_examples.jsonl
```

Then mix into the next training run:

```bash
python scripts/honest_mira.py --split all --epochs 35 --max-examples 1800 \
    --candidate-top-k 768 --include-rollups --include-atoms --include-reflections \
    --extra-examples data/feedback_eval_examples.jsonl
```

Protocol (no self-reinforcement): `useful` rows emit displayed sources as positives; `wrong` / `needs_work` rows emit them as hard negatives. A wrong example needs an explicit date or memory id in the user's note to recover a positive — otherwise it's skipped to avoid amplifying retrieval bugs.

## 9. (Legacy) Neural Q/K/V HTEMA

Earlier variant — keeps diary-specific signals + learns Q/K/V adapters over MiniLM embeddings.

```bash
python scripts/train_neural_htema.py \
    --training data/generated_queries.jsonl \
    --epochs 120 --negatives 64 --batch-size 32 --device auto

python scripts/search_neural_htema.py "happiest days at the end of 2024" \
    --limit 8 --device auto

python scripts/evaluate_neural_htema.py \
    --training data/generated_queries.jsonl --device auto
```

> The legacy oracle-window numbers from this path (R@1 ≈ 0.99) are inflated by generated-label leakage. Use `honest_mira.py` for the current honest benchmark — see [`evaluation.md`](evaluation.md).
