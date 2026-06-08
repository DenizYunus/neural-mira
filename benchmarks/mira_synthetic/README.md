# MIRA Synthetic Benchmark

This benchmark is a public, deterministic corpus for trust-weighted
autobiographical memory reconstruction. It exists because the Anne Frank demo is
useful for diary-only retrieval, but it cannot test chat-derived reconstruction
or source-trust behavior.

Generate the corpus:

```bash
npm run synthetic:generate
```

Run the missing-day reconstruction benchmark on it:

```bash
npm run synthetic:reconstruction
```

Generated files live under `benchmarks/mira_synthetic/generated/` and are
safe to publish. They contain synthetic diary entries, WhatsApp-style chats,
photo metadata, corrections, ground-truth event rows, and reconstruction target
dates. The reconstruction command consumes diary labels, chats, and photo
metadata as separate source types.

The generated target rows include:

- `public_hint` for date-plus-public-metadata queries
- `sparse_user_hint` for sparse user-hint queries
- `confuser_event_ids` for adversarial intrusion scoring

The benchmark's paper-facing score uses `ground_truth/events.jsonl`, not hidden
diary wording. Hidden diary term overlap is still emitted as a diagnostic.

Design rules:

- Ground truth is deterministic and script-owned.
- Diary entries are first-person labels used as hidden targets.
- Chat and photo records are indirect evidence.
- Contradictions and corrections are intentional, not data noise.
- Confuser events are intentional adversarial distractors.
- No private user data or API calls are required.
