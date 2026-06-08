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

Design rules:

- Ground truth is deterministic and script-owned.
- Diary entries are first-person labels used as hidden targets.
- Chat and photo records are indirect evidence.
- Contradictions and corrections are intentional, not data noise.
- No private user data or API calls are required.
