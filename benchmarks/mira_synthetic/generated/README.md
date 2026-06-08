# Generated MIRA Synthetic Corpus

Generated at: 2026-01-01T00:00:00Z

This is a deterministic public benchmark corpus. It contains no private user
data and no LLM-generated text.

Contents:

- `diaries/maya_aydin_2026.md`: first-person diary labels
- `whatsapp/*/chat.txt`: WhatsApp-style indirect evidence
- `ground_truth/events.jsonl`: hidden event truth table
- `queries/reconstruction_targets.jsonl`: dates to hide/evaluate
- `queries/reconstruction_queries.jsonl`: query metadata for future human/LLM eval
- `photos/photo_metadata.jsonl`: metadata-only photo evidence
- `corrections.jsonl`: explicit user corrections

Use:

```bash
npm run synthetic:reconstruction
```
