# `data/` — your own corpus + training artifacts

This directory is the user-data side of Neural MIRA. Nothing personal ships
in the public repo (just this README + a `.gitkeep`). The expected layout
once you've populated it is:

```
data/
├── diaries/                              # YOUR diary entries — see format below
│   ├── 2023.md
│   ├── 2024.md
│   ├── 2025.md
│   └── ... (any number of .md files)
├── feedback.jsonl                        # Optional — search-feedback log
├── whatsapp/                             # Optional — WhatsApp chat exports
│   └── (your folder-per-chat structure)
└── generated/                            # Output dir for training scripts
    ├── generated_queries.jsonl           # from scripts/generate_training_queries.py
    ├── whatsapp_synthetic_diaries.jsonl  # from scripts/synthesize_whatsapp_diaries.py
    └── style_augmented_queries.jsonl     # from scripts/generate_style_augmentations.py
```

All of these paths are configurable via `.env` — see `.env.example` in the
repo root. Defaults assume the layout above.

## Diary file format

Each `.md` file in `data/diaries/` is a concatenated multi-month or multi-year
diary, with date headers separating entries. Two header styles are supported
(the parser auto-detects):

### Style A — DailyBean / one entry per `## Month DD, YYYY` heading

```markdown
## January 5, 2024
Felt sharp today. Got the chapter outline done in one sitting, which never
happens. Saw 🥲 emoji on the train and laughed for some reason.

## January 6, 2024
Slow morning, slower afternoon. Talked with M about the move — leaning yes.
```

### Style B — bullet-style with `### YYYY-MM-DD` headers

```markdown
### 2024-01-05
- Sharp focus all morning. Chapter outline done.
- Saw a 🥲 sticker on the train, made me laugh.

### 2024-01-06
- Slow day. Lunch with M, talked about the move.
```

**Required ingredients per entry**: a date in the heading. **Optional**:
emoji icons (parsed into `icons`), mood numbers if you use a 1-5 rating
convention, a trailing `*Mood: N*` line.

The parser in `scripts/htema_core.py` handles both styles and turns each
entry into a `DiaryMemory` with `date`, `text`, `icons`, `mood`, and
`source_path`.

## How to bootstrap from zero

If you have no diary content yet, the cheapest path is:

1. Start a `data/diaries/2025.md` file with one entry per day in Style A.
2. Run `python scripts/htema_core.py` (or import it from a notebook) to
   parse and inspect — confirms your format works.
3. Once you have ~30+ entries, use `python scripts/generate_training_queries.py
   --limit 30 --examples-per-entry 3` to generate weak-supervision examples
   (requires `LLM_API_KEY` in `.env`).
4. Then `python scripts/honest_mira.py --train ...` to train the attention
   head on your own data.

## Optional: WhatsApp chat exports

If you set `WHATSAPP_ROOT` in `.env` to a directory of WhatsApp exports
(one folder per chat, with the exported `.txt` files inside), the
`synthesize_whatsapp_diaries.py` script can fill in days where you didn't
journal by summarizing the chat traffic. The folder-name convention the
parser looks for is loose — names containing the participants work fine.

## Feedback log

If your downstream UI captures thumbs-up/down on retrieved memories, write
each annotation as one JSON line in `feedback.jsonl`:

```json
{"query": "the time I was anxious about the move",
 "displayed_ids": ["diary:2024-01-06", "whatsapp:2024-01-05:12"],
 "rating": "wrong",
 "note": "the real day was 2024-01-12, not 06",
 "ts": "2024-02-03T14:22:00Z"}
```

`scripts/convert_feedback.py` turns this into eval examples with hard
negatives (rejected sources) and positives recovered from the note.

## What NOT to commit

This whole `data/` directory is in `.gitignore` for a reason — diary
content is personal. If you fork this repo to share your version of the
trained model, **don't push your `data/`**. Publish trained weights via
GitHub Releases or Hugging Face Hub instead.
