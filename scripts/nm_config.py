"""Single source of truth for paths + LLM credentials.

Reads `.env` from the repo root, falls back to sensible defaults, and exposes
the resolved values as module-level constants. Importing this module is the
only thing scripts need to do to know where the diary lives, where feedback
comes from, and which LLM to call.

No third-party dependencies — there's a tiny in-house `.env` parser so the
lab can be cloned and used without an extra `pip install`.

Process environment wins over the `.env` file, so CLI overrides like
``DIARY_ROOT=/tmp/test python scripts/htema_core.py ...`` work without
editing the file.
"""

from __future__ import annotations

import os
from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parents[1]


def _load_env_file(path: Path) -> dict[str, str]:
    """Tiny dotenv parser. Ignores comments, trims whitespace, strips one
    layer of surrounding quotes. Returns {} when the file doesn't exist."""
    if not path.exists():
        return {}
    env: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        env[key.strip()] = value
    return env


_FILE_ENV = _load_env_file(LAB_ROOT / ".env")


def _get(key: str, default: str = "") -> str:
    """Process env wins (CLI overrides); .env file is the secondary source."""
    return os.environ.get(key) or _FILE_ENV.get(key, default)


def _resolve_path(value: str, default_relative: str) -> Path:
    """Resolve a path config value. Relative paths are anchored to LAB_ROOT
    so the repo is self-contained when configs use defaults."""
    raw = value or default_relative
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (LAB_ROOT / p)


# --- Paths ----------------------------------------------------------------
DIARY_ROOT = _resolve_path(_get("DIARY_ROOT"), "data/diaries")
FEEDBACK_PATH = _resolve_path(_get("FEEDBACK_PATH"), "data/feedback.jsonl")

_WA_RAW = _get("WHATSAPP_ROOT")
WHATSAPP_ROOT: Path | None = _resolve_path(_WA_RAW, "data/whatsapp") if _WA_RAW else None


def diary_files() -> list[Path]:
    """Resolve the list of diary files to use. Order of precedence:

    1. ``DIARY_FILES`` env var (comma-separated).
    2. Every ``*.md`` directly inside DIARY_ROOT (non-recursive, sorted).
    3. Empty list if DIARY_ROOT doesn't exist yet.

    Returned each call (not cached) so tests/CLIs can mutate env between runs.
    """
    explicit = _get("DIARY_FILES")
    if explicit:
        out: list[Path] = []
        for fragment in explicit.split(","):
            fragment = fragment.strip()
            if not fragment:
                continue
            p = Path(fragment).expanduser()
            out.append(p if p.is_absolute() else (DIARY_ROOT / p))
        return out
    if not DIARY_ROOT.exists():
        return []
    return sorted(DIARY_ROOT.glob("*.md"))


# --- LLM ------------------------------------------------------------------
# Falls through aliases so people coming from Jarvis (LLM_API_KEY), NVIDIA
# NIM (NVIDIA_API_KEY), or vanilla OpenAI (OPENAI_API_KEY) all work without
# renaming their existing env var.
LLM_BASE_URL = _get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = _get("LLM_MODEL", "gpt-4o-mini")
LLM_API_KEY = (
    _get("LLM_API_KEY")
    or _get("NVIDIA_API_KEY")
    or _get("OPENAI_API_KEY")
)
LLM_TIMEOUT_MS = int(_get("LLM_TIMEOUT_MS", "60000") or "60000")


def require_llm_api_key() -> str:
    """Friendly fast-fail for scripts that need the LLM. Raises with a
    pointer to the fix rather than letting an HTTP 401 explode later."""
    if not LLM_API_KEY:
        raise RuntimeError(
            "LLM_API_KEY is not set. Add it to neural-mira/.env:\n"
            "    LLM_API_KEY=sk-...\n"
            "(Or set LLM_BASE_URL to an OpenAI-compatible endpoint of your choice.)"
        )
    return LLM_API_KEY


def llm_env_dict() -> dict[str, str]:
    """Compat shim for code that used to expect the jarvis .env dict shape.
    Lets `load_jarvis_env()`-style callers swap without restructuring."""
    return {
        "LLM_BASE_URL": LLM_BASE_URL,
        "LLM_MODEL": LLM_MODEL,
        "LLM_API_KEY": LLM_API_KEY or "",
        "LLM_TIMEOUT_MS": str(LLM_TIMEOUT_MS),
    }
