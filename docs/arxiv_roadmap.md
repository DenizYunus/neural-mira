# arXiv Roadmap

This repo is now the paper-focused Neural MIRA research artifact. The target is
not "diary + WhatsApp RAG"; the target is a defensible paper about
trust-weighted autobiographical memory reconstruction.

## Paper Claim

Working title:

```text
MIRA: Trust-Weighted Autobiographical Memory Reconstruction for Personal Archives
```

Research question:

```text
Can a personal memory system reconstruct missing autobiographical context from
indirect sources while preserving a trust hierarchy between first-person diary
records, conversational evidence, rollups, reflections, and LLM-inferred
memories?
```

The core safety rule:

```text
MIRA must never present inferred context as if it were a direct first-person
memory.
```

## What Exists Now

- Honest query-text-only evaluation in `scripts/honest_mira.py`.
- BM25, sparse semantic, dense semantic, scalar HTEMA, and Neural MIRA baselines.
- Diary, WhatsApp, rollup, atom, reflection, and WhatsApp-synthetic memory types.
- Source-trust priors and provenance excerpts for WhatsApp-synthetic memories.
- Public Anne Frank demo flow for a non-private corpus.

## What Must Be Built

### 1. Formal Trust Contract

Every `DiaryMemory` must carry:

- `evidence_type`
- `trust_level`
- `source_reliability`
- `inference_status`
- `visibility_label`
- `provenance_ids`
- `provenance_steps`
- `provenance_excerpts`
- `contradicts`
- `supersedes`

Status: implemented as the first pass. `scripts/htema_core.py` now attaches
trust defaults and an audit payload to each memory.

### 2. Missing-Memory Reconstruction Benchmark

Create a benchmark that hides direct diary entries and asks MIRA to reconstruct
the missing day from indirect evidence.

Metrics:

- `reconstruction_recall`
- `source_precision`
- `uncertainty_calibration`
- `trust_violation_rate`
- `direct_override_error`
- `contradiction_preservation`
- `audit_success_rate`

Required baselines:

- simple RAG over all memories
- chronological summarization
- diary-only
- chat-only
- MIRA without trust features
- MIRA without contradiction handling

Status: first baseline suite implemented. `scripts/reconstruction_benchmark.py`
hides direct diary days, retrieves indirect evidence, and reports
trust-weighted MIRA, no-trust MIRA, simple RAG, diary-only, chat-only, and
chronological-neighbor baselines.

### 3. Public Reproducible Dataset

Private diary/chat results are not enough for a serious paper. Build a
releasable benchmark with:

- synthetic personas
- synthetic diary timelines
- synthetic WhatsApp-style chats
- hidden diary days
- contradictions and corrections
- multilingual/noisy queries
- fixed train/dev/test splits

The Anne Frank demo can stay as a public qualitative demo, but the main public
benchmark should not depend on private or legally ambiguous data.

Status: started. `benchmarks/mira_synthetic/generate.py` creates a deterministic
public multi-source corpus with diary labels, WhatsApp-style chats, photo
metadata, corrections, ground truth events, adversarial confusers, and hidden
reconstruction targets. `scripts/reconstruction_benchmark.py` can now evaluate
date-only, public-metadata, sparse-hint, and legacy hidden-label query modes
against event-level gold facts.

### 4. Human-Validated Queries

Generated queries are useful but not sufficient. Add 200-500 human-written or
human-validated examples across:

- temporal recall
- emotional recall
- relationship/context recall
- missing-day reconstruction
- contradiction and counter-evidence
- audit/provenance questions
- Turkish-English mixed queries
- typo-heavy mobile-style queries

### 5. Ablations And Error Analysis

Report what each part contributes:

- no temporal features
- no emotion features
- no source/trust features
- no participant features
- no hierarchy
- no rollups
- no reflections
- no neural reranker
- no calibration

Error categories:

- wrong date
- right source, wrong interpretation
- unsupported inference
- false confidence
- contradiction collapsed into one story
- private detail surfaced unnecessarily

### 6. Privacy And Ethics Section

The paper needs a full threat model:

- raw diary/chat leakage
- co-participant consent
- private checkpoint leakage
- LLM inference of personal attributes
- false memory formation
- emotional distortion
- deletion/correction failure
- external LLM data flow

DeepSeek/OpenAI-compatible models may be used for query augmentation or answer
generation, but core retrieval/reconstruction metrics must be label-based and
not judged only by an LLM.

## arXiv Gate

Do not upload until all are true:

- The research question is clear.
- Trust/provenance fields are implemented and tested.
- Missing-memory reconstruction benchmark runs.
- Public synthetic/consented benchmark exists.
- Human-validated query set exists.
- Strong baselines and no-trust ablations are reported.
- Privacy threat model is written.
- DeepSeek/LLM usage is disclosed and non-essential to core metrics.
- Paper tables regenerate from scripts.
- Private case study results are aggregate/redacted only.

If any required gate fails, publish a build log instead of an arXiv paper.
