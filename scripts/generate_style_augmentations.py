#!/usr/bin/env python3
"""Generate creative query-style augmentations for honest MIRA training.

The script uses an OpenAI-compatible chat-completions endpoint, reading API
configuration from ../jarvis-platform/.env by default. It never prints API keys
or full private memory text.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from honest_mira import EvalExample, example_source, generate_benchmark, stratified_limit_examples
from htema_core import DiaryMemory, parse_all_memories


LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent
JARVIS_ENV = REPO_ROOT / "jarvis-platform" / ".env"
DEFAULT_OUTPUT = LAB_ROOT / "data" / "style_augmented_queries.jsonl"

DEFAULT_STYLES = [
    "vague_personal_memory",
    "half_remembered_fragment",
    "casual_turkish_english_mix",
    "voice_assistant_terse",
    "emotional_reflection",
    "relationship_context",
    "cross_source_diary_whatsapp",
    "temporal_approximation",
    "typo_noisy_search",
    "contradiction_counterevidence",
    "pattern_detection",
    "multi_hop_life_context",
]


@dataclass(frozen=True)
class BaseTask:
    base_index: int
    base_key: str
    query: str
    intent: str
    style: str
    source_mix: str
    positive_ids: tuple[str, ...]
    target_month: str
    target_date: str | None
    note: str


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def stable_key(query: str, positive_ids: tuple[str, ...]) -> str:
    payload = json.dumps(
        {"query": query.lower().strip(), "positive_ids": sorted(positive_ids)},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def compact_text(text: str, max_chars: int) -> str:
    cleaned = re.sub(r"\n{3,}", "\n\n", text.strip())
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + "\n[truncated]"


def memory_preview(memory: DiaryMemory, max_chars: int) -> dict[str, Any]:
    return {
        "id": memory.entry_id,
        "date": memory.date,
        "source_type": getattr(memory, "source_type", "diary"),
        "participants": list(getattr(memory, "participants", ()))[:8],
        "mood": memory.mood,
        "icons": list(memory.icons)[:12],
        "features": [
            name
            for name, value in sorted(memory.diary_features.items(), key=lambda item: item[1], reverse=True)
            if value >= 0.2
        ][:5],
        "text_preview": compact_text(memory.text, max_chars),
    }


def task_from_example(index: int, example: EvalExample) -> BaseTask:
    positives = tuple(sorted(set(example.positive_ids)))
    return BaseTask(
        base_index=index,
        base_key=stable_key(example.query, positives),
        query=example.query,
        intent=example.intent,
        style=example.style,
        source_mix=example_source(example),
        positive_ids=positives,
        target_month=example.target_month,
        target_date=example.target_date,
        note=example.note,
    )


def load_existing_counts(path: Path) -> tuple[dict[str, int], set[tuple[str, tuple[str, ...]]]]:
    counts: dict[str, int] = {}
    seen: set[tuple[str, tuple[str, ...]]] = set()
    if not path.exists():
        return counts, seen

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            base_key = str(row.get("base_key") or "")
            if base_key:
                counts[base_key] = counts.get(base_key, 0) + 1
            query = str(row.get("query") or "").strip().lower()
            positive_ids = tuple(sorted(str(item) for item in row.get("positive_ids") or []))
            if query and positive_ids:
                seen.add((query, positive_ids))
    return counts, seen


def strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    return stripped


def escape_control_chars_in_strings(text: str) -> str:
    output: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                output.append(char)
                escaped = False
            elif char == "\\":
                output.append(char)
                escaped = True
            elif char == '"':
                output.append(char)
                in_string = False
            elif char == "\n":
                output.append("\\n")
            elif char == "\r":
                output.append("\\r")
            elif ord(char) < 32:
                output.append(" ")
            else:
                output.append(char)
        else:
            output.append(char)
            if char == '"':
                in_string = True
    return "".join(output)


def extract_balanced_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise json.JSONDecodeError("No JSON object found", text, 0)

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return text[start:]


def repair_json_text(text: str) -> str:
    repaired = strip_markdown_fence(text)
    repaired = extract_balanced_json_object(repaired)
    repaired = escape_control_chars_in_strings(repaired)
    # Common LLM error: adjacent objects in an array with no comma between them.
    repaired = re.sub(r"}\s*(?=\{)", "},", repaired)
    # Another common error: trailing comma before array/object close.
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    return repaired


def iter_object_strings_from_array(text: str, key: str) -> list[str]:
    key_pos = text.find(f'"{key}"')
    if key_pos < 0:
        return []
    array_start = text.find("[", key_pos)
    if array_start < 0:
        return []

    objects: list[str] = []
    depth = 0
    start: int | None = None
    in_string = False
    escaped = False
    for index in range(array_start + 1, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth:
                depth -= 1
                if depth == 0 and start is not None:
                    objects.append(text[start : index + 1])
                    start = None
        elif char == "]" and depth == 0:
            break
    return objects


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = strip_markdown_fence(text)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as first_error:
        try:
            return json.loads(repair_json_text(stripped))
        except json.JSONDecodeError:
            repaired = escape_control_chars_in_strings(stripped)
            rows = []
            for key in ("augmented_examples", "examples"):
                for object_text in iter_object_strings_from_array(repaired, key):
                    try:
                        rows.append(json.loads(repair_json_text(object_text)))
                    except json.JSONDecodeError:
                        continue
                if rows:
                    return {"augmented_examples": rows}
            raise first_error


def call_openai_compatible(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    timeout: int,
    temperature: float,
    max_tokens: int,
    top_p: float,
    json_mode: bool,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You generate creative but faithful retrieval-training query augmentations. "
                    "Return strict JSON only. No markdown. No commentary."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"]
    return extract_json_object(content)


def build_prompt(
    batch: list[BaseTask],
    memories_by_id: dict[str, DiaryMemory],
    *,
    augmentations_per_example: int,
    styles: list[str],
    max_memory_chars: int,
) -> str:
    targets: list[dict[str, Any]] = []
    for task in batch:
        positives = [memories_by_id[item] for item in task.positive_ids if item in memories_by_id]
        targets.append(
            {
                "base_index": task.base_index,
                "base_key": task.base_key,
                "base_query": task.query,
                "base_intent": task.intent,
                "base_style": task.style,
                "source_mix": task.source_mix,
                "target_month": task.target_month,
                "target_date": task.target_date,
                "note": task.note,
                "positive_ids": list(task.positive_ids),
                "positive_memory_previews": [memory_preview(memory, max_memory_chars) for memory in positives[:8]],
            }
        )

    payload = {
        "targets": targets,
        "augmentations_per_target": augmentations_per_example,
        "desired_style_families": styles,
    }

    return f"""
You are creating STYLE AUGMENTATIONS for a diary-and-WhatsApp personal memory retriever named MIRA.

Goal:
Generate exactly {augmentations_per_example} new user queries for EACH target. The new queries must retrieve the SAME positive_ids as the base target, but must use a broader range of natural styles than the deterministic benchmark.

Why this matters:
The current model is weaker on query-style holdout. We need creative paraphrases: vague, conversational, multilingual, typo-heavy, emotional, indirect, voice-assistant-like, and cross-source questions. These are training queries, not answers.

Strict output JSON shape:
{{
  "augmented_examples": [
    {{
      "base_index": 0,
      "query": "natural user query",
      "augmentation_style": "one style family from the requested list",
      "intent": "short intent label",
      "style": "short_style_label",
      "positive_ids": ["exact ids copied from the target"],
      "target_month": "YYYY-MM",
      "target_date": "YYYY-MM-DD or null",
      "source_mix": "diary | whatsapp | mixed",
      "hard_negative_hint": "what nearby or similar memories could confuse the retriever",
      "why": "why this query helps style robustness"
    }}
  ]
}}

Rules:
- Return strict JSON only.
- Use each target's exact positive_ids. Do not create new ids.
- Do not leak IDs, labels, or the phrase "positive_ids" inside query text.
- Do not invent facts not supported by the previews.
- Do not use literal newline characters inside any JSON string value.
- Avoid raw double quote characters inside string values; use apostrophes instead.
- Put a comma between every object in the augmented_examples array.
- Do not make every query look like a benchmark label. Make them sound like the owner casually asking Jarvis.
- Include Turkish, English, and Turkish-English mixed phrasings when natural.
- Include some messy human queries: incomplete memory fragments, typos, no punctuation, "what was that day..." style.
- For WhatsApp targets, include questions about who said what, relationship context, message tone, and conversation windows.
- For diary targets, include questions about mood, events, people, time, contradictions, and life patterns.
- For mixed targets, ask for the life context that connects diary and conversations.
- Prefer retrieval-useful specificity, but vary where the specificity appears: date, emotion, person, event, source, or remembered phrase.
- Avoid generic queries such as "what happened that day" unless they include at least one useful clue.

Payload:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()


def normalize_augmented_example(
    raw: dict[str, Any],
    tasks_by_index: dict[int, BaseTask],
    model: str,
    batch_id: str,
) -> dict[str, Any] | None:
    try:
        base_index = int(raw.get("base_index"))
    except (TypeError, ValueError):
        return None
    task = tasks_by_index.get(base_index)
    if not task:
        return None

    query = str(raw.get("query") or "").strip()
    if len(query) < 4:
        return None

    requested_ids = tuple(sorted(str(item) for item in raw.get("positive_ids") or []))
    positives = requested_ids if set(requested_ids) == set(task.positive_ids) else task.positive_ids
    style = str(raw.get("style") or raw.get("augmentation_style") or "llm_style").strip()
    augmentation_style = str(raw.get("augmentation_style") or style).strip()
    if not style.startswith("aug_"):
        style = f"aug_{style}"

    return {
        "query": query,
        "intent": str(raw.get("intent") or task.intent).strip() or task.intent,
        "style": style[:80],
        "augmentation_style": augmentation_style[:80],
        "positive_ids": list(positives),
        "target_month": str(raw.get("target_month") or task.target_month),
        "target_date": raw.get("target_date") if raw.get("target_date") else task.target_date,
        "source_mix": str(raw.get("source_mix") or task.source_mix),
        "base_query": task.query,
        "base_intent": task.intent,
        "base_style": task.style,
        "base_key": task.base_key,
        "base_index": task.base_index,
        "hard_negative_hint": str(raw.get("hard_negative_hint") or "").strip(),
        "why": str(raw.get("why") or "").strip(),
        "generator": "deepseek_style_augmentation",
        "model": model,
        "batch_id": batch_id,
        "created_at": int(time.time()),
    }


def generate_batch(
    batch: list[BaseTask],
    memories_by_id: dict[str, DiaryMemory],
    *,
    base_url: str,
    api_key: str,
    model: str,
    augmentations_per_example: int,
    styles: list[str],
    max_memory_chars: int,
    timeout: int,
    temperature: float,
    max_tokens: int,
    top_p: float,
    json_mode: bool,
    retries: int,
) -> tuple[str, list[dict[str, Any]], str | None]:
    batch_id = hashlib.sha1(
        json.dumps([task.base_key for task in batch], sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    prompt = build_prompt(
        batch,
        memories_by_id,
        augmentations_per_example=augmentations_per_example,
        styles=styles,
        max_memory_chars=max_memory_chars,
    )
    if not api_key:
        return batch_id, [], "missing API key"

    last_error: str | None = None
    use_json_mode = json_mode
    for attempt in range(1, retries + 2):
        try:
            result = call_openai_compatible(
                base_url=base_url,
                api_key=api_key,
                model=model,
                prompt=prompt,
                timeout=timeout,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                json_mode=use_json_mode,
            )
            tasks_by_index = {task.base_index: task for task in batch}
            rows = []
            for item in result.get("augmented_examples") or result.get("examples") or []:
                normalized = normalize_augmented_example(item, tasks_by_index, model, batch_id)
                if normalized:
                    rows.append(normalized)
            return batch_id, rows, None
        except urllib.error.HTTPError as error:
            last_error = f"HTTPError: {error}"
            if use_json_mode and error.code in {400, 404, 422}:
                use_json_mode = False
                continue
            if attempt <= retries:
                time.sleep(min(30.0, 2.0 * attempt))
        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            KeyError,
            OSError,
        ) as error:
            last_error = f"{type(error).__name__}: {error}"
            if attempt <= retries:
                time.sleep(min(30.0, 2.0 * attempt))
    return batch_id, [], last_error


def parse_styles(value: str) -> list[str]:
    if not value.strip():
        return DEFAULT_STYLES
    return [item.strip() for item in value.split(",") if item.strip()]


def select_tasks(
    examples: list[EvalExample],
    *,
    limit: int,
    source_filter: str,
    style_filter: set[str] | None,
    seed: int,
) -> list[BaseTask]:
    filtered = []
    for index, example in enumerate(examples):
        source = example_source(example)
        if source_filter != "all" and source != source_filter:
            continue
        if style_filter and example.style not in style_filter:
            continue
        filtered.append(task_from_example(index, example))

    rng = random.Random(seed)
    rng.shuffle(filtered)
    if limit > 0:
        filtered = filtered[:limit]
    return filtered


def write_dry_run(path: Path, batches: list[list[BaseTask]], memories_by_id: dict[str, DiaryMemory], args: argparse.Namespace) -> None:
    styles = parse_styles(args.styles)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for batch in batches[:3]:
            prompt = build_prompt(
                batch,
                memories_by_id,
                augmentations_per_example=args.augmentations_per_example,
                styles=styles,
                max_memory_chars=args.max_memory_chars,
            )
            handle.write(
                json.dumps(
                    {
                        "base_indices": [task.base_index for task in batch],
                        "base_keys": [task.base_key for task in batch],
                        "prompt_preview": prompt[:4000],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def chunked(items: list[BaseTask], size: int) -> list[list[BaseTask]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate creative DeepSeek style augmentations for MIRA.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSONL output path.")
    parser.add_argument("--limit", type=int, default=500, help="Base deterministic examples to augment. Use 0 for all.")
    parser.add_argument("--augmentations-per-example", type=int, default=6, help="New queries per selected base example.")
    parser.add_argument("--batch-size", type=int, default=1, help="Base examples per API request.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel API requests.")
    parser.add_argument("--seed", type=int, default=13, help="Deterministic sampling seed.")
    parser.add_argument("--source-filter", choices=["all", "diary", "whatsapp", "mixed"], default="all")
    parser.add_argument("--style-filter", default="", help="Comma-separated deterministic base styles to augment.")
    parser.add_argument("--styles", default=",".join(DEFAULT_STYLES), help="Comma-separated augmentation style families.")
    parser.add_argument("--max-memory-chars", type=int, default=900, help="Max preview chars per positive memory.")
    parser.add_argument("--timeout", type=int, default=180, help="API timeout per request in seconds.")
    parser.add_argument("--heartbeat-seconds", type=int, default=30, help="Print progress while waiting for API batches.")
    parser.add_argument("--temperature", type=float, default=1.15, help="Generation temperature.")
    parser.add_argument("--top-p", type=float, default=0.92, help="Nucleus sampling top_p.")
    parser.add_argument("--max-tokens", type=int, default=6000, help="Max completion tokens per request.")
    parser.add_argument("--retries", type=int, default=2, help="Retries per failed API request.")
    parser.add_argument("--no-json-mode", action="store_true", help="Do not request OpenAI-compatible JSON mode.")
    parser.add_argument("--env", type=Path, default=JARVIS_ENV, help="Path to Jarvis .env.")
    parser.add_argument("--base-url", default=None, help="Override OpenAI-compatible base URL.")
    parser.add_argument("--model", default=None, help="Override model, e.g. deepseek-v4-flash.")
    parser.add_argument("--api-key-env", default="", help="Optional env var name containing the API key.")
    parser.add_argument("--dry-run", action="store_true", help="Write prompt previews instead of calling the API.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing output and overwrite from scratch.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env = {**load_env_file(args.env), **os.environ}
    explicit_key = env.get(args.api_key_env) if args.api_key_env else None
    model = args.model or env.get("DEEPSEEK_MODEL") or env.get("MIRA_AUG_MODEL") or "deepseek-v4-flash"
    wants_deepseek = "deepseek" in model.lower()
    base_url = (
        args.base_url
        or env.get("DEEPSEEK_BASE_URL")
        or ("https://api.deepseek.com" if wants_deepseek else env.get("LLM_BASE_URL"))
        or "https://api.deepseek.com"
    )
    api_key = (
        explicit_key
        or (env.get("DEEPSEEK_API_KEY") if wants_deepseek else None)
        or env.get("LLM_API_KEY")
        or env.get("DEEPSEEK_API_KEY")
        or env.get("NVIDIA_API_KEY")
    )

    memories = parse_all_memories()
    if not memories:
        print("No diary/WhatsApp memories found.", file=sys.stderr)
        return 1
    memories_by_id = {memory.entry_id: memory for memory in memories}
    all_examples = generate_benchmark(memories, args.seed)
    style_filter = {item.strip() for item in args.style_filter.split(",") if item.strip()} or None
    if args.limit > 0:
        # Keep the same source/style balance as honest_mira before the LLM expansion fans out.
        pool = stratified_limit_examples(all_examples, max(args.limit * 4, args.limit), args.seed)
    else:
        pool = all_examples
    tasks = select_tasks(
        pool,
        limit=args.limit,
        source_filter=args.source_filter,
        style_filter=style_filter,
        seed=args.seed,
    )

    if not tasks:
        print("No base examples matched the requested filters.", file=sys.stderr)
        return 1

    existing_counts: dict[str, int] = {}
    seen_rows: set[tuple[str, tuple[str, ...]]] = set()
    if args.output.exists() and not args.no_resume and not args.dry_run:
        existing_counts, seen_rows = load_existing_counts(args.output)
        tasks = [task for task in tasks if existing_counts.get(task.base_key, 0) < args.augmentations_per_example]

    batches = chunked(tasks, max(1, args.batch_size))
    styles = parse_styles(args.styles)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    diary_count = sum(1 for memory in memories if getattr(memory, "source_type", "diary") == "diary")
    whatsapp_count = len(memories) - diary_count
    print(f"Loaded {len(memories)} memories ({diary_count} diary, {whatsapp_count} whatsapp)")
    print(f"Selected {len(tasks)} base examples; {len(batches)} API batches")
    print(f"Model: {model}; Base URL: {base_url}")
    if not tasks:
        print("Nothing to do: existing output already has enough augmentations for the selected base examples.")
        return 0

    if args.dry_run:
        write_dry_run(args.output, batches, memories_by_id, args)
        print(f"Wrote dry-run prompt previews to {args.output}")
        return 0

    if not api_key:
        print("No API key found. Set DEEPSEEK_API_KEY, LLM_API_KEY, NVIDIA_API_KEY, or --api-key-env.", file=sys.stderr)
        return 1

    mode = "w" if args.no_resume else "a"
    total_written = 0
    total_failed = 0
    started = time.time()

    with args.output.open(mode, encoding="utf-8") as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            future_to_batch: dict[concurrent.futures.Future[tuple[str, list[dict[str, Any]], str | None]], str] = {}
            for batch in batches:
                expected_batch_id = hashlib.sha1(
                    json.dumps([task.base_key for task in batch], sort_keys=True).encode("utf-8")
                ).hexdigest()[:16]
                print(
                    f"submitted {expected_batch_id}: base examples {batch[0].base_index}..{batch[-1].base_index}",
                    flush=True,
                )
                future = executor.submit(
                    generate_batch,
                    batch,
                    memories_by_id,
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    augmentations_per_example=args.augmentations_per_example,
                    styles=styles,
                    max_memory_chars=args.max_memory_chars,
                    timeout=args.timeout,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    top_p=args.top_p,
                    json_mode=not args.no_json_mode,
                    retries=args.retries,
                )
                future_to_batch[future] = expected_batch_id

            pending = set(future_to_batch)
            done_count = 0
            while pending:
                done, pending = concurrent.futures.wait(
                    pending,
                    timeout=max(1, args.heartbeat_seconds),
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                if not done:
                    elapsed = max(time.time() - started, 1.0)
                    print(
                        f"waiting... completed {done_count}/{len(future_to_batch)}, "
                        f"pending {len(pending)}, wrote {total_written}, elapsed {elapsed / 60:.1f}m",
                        flush=True,
                    )
                    continue

                for future in done:
                    done_count += 1
                    expected_batch_id = future_to_batch[future]
                    try:
                        batch_id, rows, error = future.result()
                    except Exception as error:
                        total_failed += 1
                        print(
                            f"[{done_count}/{len(future_to_batch)}] {expected_batch_id} crashed: "
                            f"{type(error).__name__}: {error}",
                            file=sys.stderr,
                            flush=True,
                        )
                        continue
                    if error:
                        total_failed += 1
                        print(
                            f"[{done_count}/{len(future_to_batch)}] {batch_id} failed: {error}",
                            file=sys.stderr,
                            flush=True,
                        )
                        continue

                    written_now = 0
                    for row in rows:
                        key = (str(row["query"]).lower(), tuple(sorted(row["positive_ids"])))
                        if key in seen_rows:
                            continue
                        seen_rows.add(key)
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        written_now += 1
                    handle.flush()
                    total_written += written_now
                    elapsed = max(time.time() - started, 1.0)
                    rate = total_written / elapsed
                    print(
                        f"[{done_count}/{len(future_to_batch)}] {batch_id}: wrote {written_now} "
                        f"(total {total_written}, {rate:.2f}/s)",
                        flush=True,
                    )

    print(f"Wrote {total_written} new augmentations to {args.output}", flush=True)
    if total_failed:
        print(f"Failed batches: {total_failed}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
