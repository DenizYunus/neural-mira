# Neural MIRA Paper Outline

## Working Title

Trust-Weighted Autobiographical Memory Reconstruction from Heterogeneous Personal Evidence

## Research Question

How can a personal AI memory system reconstruct missing autobiographical context from indirect sources while preserving a trust hierarchy between first-person memories, conversations, metadata, and inferred summaries?

## Core Claim

Autobiographical reconstruction should not optimize only for retrieval recall. It must also preserve provenance, distinguish direct memory from inferred memory, expose uncertainty, resist confusable neighboring events, and allow the user to audit why a memory exists.

## Hypotheses

1. A relevance-first, trust-labeled architecture can recover more structured event facts than blunt source-trust reranking.
2. Source trust reduces unsafe presentation errors and supports auditability even when it does not always maximize lexical retrieval metrics.
3. Query-mode difficulty matters: date-only, public metadata, sparse user hints, and hidden-label diagnostics expose different failure modes.

## System

Neural MIRA represents memory as timestamped evidence objects with source type, trust level, reliability, inference status, provenance IDs, and audit text. The current public benchmark uses:

- direct diary labels as hidden targets,
- WhatsApp-style chats as conversation evidence,
- photo metadata as indirect contextual evidence,
- corrections and contradictions as explicit trust tests,
- adversarial confuser events as retrieval distractors.

## Methods Compared

- `mira_trust`: scalar MIRA with trust-weighted ranking.
- `mira_trust_v2`: relevance-first ranking with trust-calibrated confidence.
- `mira_trust_v3`: relevance-first ranking with same-date/source-diversity reranking and trust-labeled presentation.
- `mira_no_trust`: scalar MIRA without trust reranking.
- `simple_rag_all`: lexical cosine baseline.
- `diary_only`, `chat_only`, and `chronological_neighbors` ablations.

## Evaluation

Primary metric: structured fact recall over gold slots in `ground_truth/events.jsonl`.

Secondary metrics:

- fact-slot accuracy,
- event-level lexical F1,
- confuser intrusion rate,
- direct diary override error,
- contradiction preservation,
- uncertainty label success,
- source label coverage,
- audit trail coverage,
- confidence error.

Query modes:

- `date_only`,
- `date_plus_public_metadata`,
- `sparse_user_hint`,
- `legacy_hidden_label` for diagnostic upper-bound behavior.

## Current Result Snapshot

Current regenerated run: 3 personas, 45 events, 28 hidden targets, top-k 5.

With `date_plus_public_metadata`, `mira_trust_v3` has the strongest structured fact recall and fact-slot accuracy among MIRA variants:

- `mira_trust_v3`: structured fact recall 0.751, fact-slot accuracy 0.870, confuser intrusion 0.057, audit trail 1.000.
- `mira_no_trust`: structured fact recall 0.708, fact-slot accuracy 0.763, confuser intrusion 0.043, audit trail 1.000.
- `simple_rag_all`: structured fact recall 0.687, fact-slot accuracy 0.715, confuser intrusion 0.071, audit trail 1.000.

Across the query-mode sweep, `mira_trust_v3` remains strongest on structured facts: `date_only` 0.811/0.936 recall/accuracy, `date_plus_public_metadata` 0.751/0.870, `sparse_user_hint` 0.781/0.921, and `legacy_hidden_label` 0.867/0.986.

`simple_rag_all` remains competitive on lexical event F1, which is useful: it shows why the paper should report structured facts, auditability, and confuser intrusion rather than only text overlap.

## Limitations

- The public corpus is synthetic and small.
- The current multi-persona corpus is still far smaller than a final benchmark.
- Structured fact scoring is deterministic and lexical; a later paper version should add human or LLM-judge validation.
- Generated reconstruction text is currently deterministic, not a full natural-language model output.
- Confidence calibration remains immature and should be evaluated separately from fact recovery.
- Privacy, consent, false-memory risk, and emotional distortion must be treated as first-class limitations.

## Next Paper Work

1. Add a larger synthetic split with more personas and longer timelines.
2. Add LLM reconstruction baselines with DeepSeek or another disclosed model.
3. Add human-written query sets and human audit judgments.
4. Add ablations for source reliability, contradiction handling, and correction handling.
5. Write the ethics section before polishing the results section.
