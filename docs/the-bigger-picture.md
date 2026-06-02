# The bigger picture: a local-first personal-AI memory system

Neural MIRA didn't start as a standalone project. It's the retrieval brain of a larger personal-AI system I built to answer one deceptively hard question:

> *"What do I actually remember, and can a machine help me recall it the way I would?"*

This is the architecture writeup — what the whole system looks like, why each piece exists, and which engineering decisions actually mattered. Neural MIRA is the part I open-sourced; the rest stays private because it's wired directly to my own diaries, photos, videos, and chat history. But the **patterns** are general, and that's what this document is about.

## The problem with treating your life like a document store

The default way to "search your own data" is retrieval-augmented generation: embed everything into one vector space, cosine-sort against the query, feed the top-k to an LLM. This is great for *facts* and terrible for *autobiography*.

When you ask *"what was I anxious about around the move?"* the right answer depends on signals a flat cosine sort throws away:

- **Time** — "around the move" is a fuzzy window, not a keyword.
- **Source trust** — a sentence you wrote in your diary is higher-fidelity evidence than a throwaway WhatsApp line, which is higher than an LLM's summary of a day you didn't journal.
- **Emotion** — "anxious" is a mood signal, not just a token to match.
- **People & continuity** — who was around, what the surrounding week looked like.

So instead of one big embedding sort, the system is built as **specialized retrieval substrates**, each good at one modality, unified by an orchestrator and — for the hardest modality, personal text memory — a *learned* ranker.

## The shape of the system

```
                         ┌─────────────────────────┐
   you ──ask──▶          │   Orchestrator (Jarvis)  │
                         │  routes intent → tools,  │
                         │  composes the answer     │
                         └───────────┬─────────────┘
            ┌────────────────────────┼────────────────────────┐
            ▼                        ▼                        ▼
   ┌─────────────────┐   ┌────────────────────┐   ┌────────────────────┐
   │  Neural MIRA    │   │  Photo / Video      │   │  Face index        │
   │  (this repo)    │   │  semantic search    │   │  (who appears where)│
   │  diary + chat   │   │  CLIP + DINOv2 +    │   │  cluster + label    │
   │  learned rank   │   │  Whisper transcripts│   │                     │
   └────────┬────────┘   └─────────┬──────────┘   └─────────┬──────────┘
            └──────────────────────┴────────────────────────┘
                                    ▼
                          one vector DB (Qdrant),
                          one collection per modality
```

Each substrate is a small local Python service (an HTTP sidecar). The orchestrator is a Node app that exposes them to an LLM as tools and renders results. Everything runs **on my own machine** — no personal data leaves the box except embedding-time calls to a hosted LLM for *synthesis* tasks (and even those are optional).

## The substrates

### 1. Personal text memory → Neural MIRA (open-sourced)

The hardest and most interesting piece. Diary entries, chat windows, and synthesized summaries become **memory tokens** with multi-faceted keys (semantic, temporal, emotional, diary-feature). A query becomes a query token. Several attention heads score the match, and a small neural reranker — trained on weak-supervision examples generated from the corpus itself — fuses them.

The thing I care about most here is **honesty in evaluation**. Early versions scored R@1 ≈ 0.99 — because they leaked the answer's date into retrieval as an oracle window. Stripping that out dropped the real number by half and forced the actual work: learning to rank from query text alone. The [evaluation doc](evaluation.md) keeps both numbers visible on purpose.

A second principle: **source trust as a first-class signal**. Not all memories are equal evidence. A diary sentence I wrote that day outranks an LLM's third-person summary of a day I didn't journal. That prior is baked into the ranker, and synthetic memories carry provenance so the system can show its work.

### 2. Photos & videos → dual-vector semantic search

Photos and video frames are embedded with **two** vectors, not one:

- a **semantic** vector (CLIP ViT-L/14) that's text-aligned — so "a street at night" finds the right images and stops confusing a streetlight with a sunset, and
- a **perceptual** vector (DINOv2) that captures texture/shape/composition — so "find more photos that *look like* this one" works as image→image similarity.

They live as named vectors on the same point, so one query embedding can search either space. Videos add a third dimension: keyframes get the same CLIP treatment, and audio is transcribed (Whisper large-v3, strong on code-switching between languages) into a separate, time-stamped transcript collection. So you can search a video by *what it looks like* **or** *what was said in it*, and jump to the timestamp.

### 3. Faces → cluster, then label once

A face detector + embedder runs over photos and sampled video frames; embeddings are clustered (DBSCAN) into "this is probably the same person" groups. You label a cluster once and every appearance — across thousands of photos and video frames — inherits the name. The payoff: *"photos of X from last spring"* becomes a person-filter intersected with a time window intersected with a semantic query.

## The engineering decisions that actually mattered

A few choices did more for reliability and quality than any model upgrade:

**Local-first, CPU-tolerant.** The whole thing runs without a GPU. Embedding a few thousand photos on CPU takes ~an hour, a full video re-transcription overnight — slow, but it's a one-time cost and it means the data never leaves the machine. (A GPU would slot in for a 5–10× speedup; the code already supports it via a device flag.)

**Named vectors over separate collections.** Storing semantic + perceptual as named vectors on one point (rather than two parallel collections) keeps the two views of an image in lockstep and lets a single point ID resolve to both — which makes "find similar" a stored-vector lookup with zero re-embedding at query time.

**Orphan sweeps + delta ingest.** Re-running ingestion is cheap and idempotent: content-hash IDs mean re-embedding the same file overwrites the same point, a manifest tracks what's done so re-runs skip it, and every ingest first *sweeps* — files deleted from disk get their embeddings, derived faces, and cached thumbnails removed. The index tracks reality instead of growing monotonically.

**Retry the transient, fail fast on the permanent.** A multi-hour ingest once died at 565/822 on a single DNS blip to the cloud vector DB. The fix wasn't "retry everything" — it was a classifier that retries connection/timeout/5xx errors with exponential backoff but raises immediately on validation errors, so a genuine bug surfaces fast while a flaky network doesn't cost hours.

**A "settings" surface, not a pile of buttons.** Operational controls — start/stop services, run a delta embed, watch logs — live behind one gear icon. The heavy ML models lazy-load on first use, so the system boots instantly and only pays the model-load cost when you actually search.

## What's open vs. what's private

| Piece | Status | Why |
|-------|--------|-----|
| **Neural MIRA** (learned diary/chat ranker) | **Open** ([this repo](../README.md)) | The novel, generalizable part — point it at any dated-markdown corpus |
| Orchestrator, photo/video/face services | Private | Wired to my own personal data; the patterns are described here |
| The actual diaries / photos / chats | Never leaves my machine | Obviously |

The open piece ships with a fully runnable demo on **Anne Frank's diary** (public domain) so anyone can see the retrieval working in five commands — no need for a corpus of their own.

## Lessons

- **Specialized beats monolithic for personal data.** One embedding space can't serve "when," "who," "how did it look," and "what was said." Separate substrates, unified at the orchestrator, each beat a single mega-index on its own axis.
- **The honest metric is the only metric.** It's easy to build a personal-search demo that looks magical and is secretly cheating on held-out evaluation. Killing the oracle-window leak hurt the headline number and was the most important thing I did.
- **Reliability is a feature, not a footnote.** Idempotent ingest, orphan cleanup, and transient-retry did more for daily usability than any single model swap.

---

*Neural MIRA is the open core of this system. Start with the [README](../README.md), or jump to the [Anne Frank quick start](quickstart.md).*
