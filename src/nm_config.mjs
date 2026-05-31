// Single source of truth for paths on the Node side. Mirrors scripts/nm_config.py.
// Reads `.env` from the repo root, falls back to sensible defaults, and exposes
// resolved paths as named exports.
//
// No dependency on `dotenv` — a few lines of inline parsing keeps the package
// import-free for callers who just need diary_attention.mjs to find their data.

import { readFileSync, existsSync, readdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const LAB_ROOT = path.resolve(__dirname, '..');

function loadEnvFile(filePath) {
  if (!existsSync(filePath)) return {};
  const env = {};
  for (let line of readFileSync(filePath, 'utf-8').split(/\r?\n/)) {
    line = line.trim();
    if (!line || line.startsWith('#') || !line.includes('=')) continue;
    const idx = line.indexOf('=');
    const key = line.slice(0, idx).trim();
    let value = line.slice(idx + 1).trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    env[key] = value;
  }
  return env;
}

const fileEnv = loadEnvFile(path.join(LAB_ROOT, '.env'));
const getEnv = (key, def = '') => process.env[key] ?? fileEnv[key] ?? def;

function resolvePath(value, defaultRelative) {
  let raw = value || defaultRelative;
  if (raw.startsWith('~')) {
    raw = path.join(process.env.HOME || process.env.USERPROFILE || '', raw.slice(1));
  }
  return path.isAbsolute(raw) ? raw : path.join(LAB_ROOT, raw);
}

export const LAB_ROOT_PATH = LAB_ROOT;
export const DIARY_ROOT = resolvePath(getEnv('DIARY_ROOT'), 'data/diaries');
export const FEEDBACK_PATH = resolvePath(getEnv('FEEDBACK_PATH'), 'data/feedback.jsonl');

/**
 * Resolve the list of diary files to read. Order of precedence:
 *   1. DIARY_FILES env var (comma-separated; entries can be absolute or
 *      relative-to-DIARY_ROOT).
 *   2. Every `*.md` directly in DIARY_ROOT (non-recursive, sorted).
 *   3. Empty array if DIARY_ROOT doesn't exist yet.
 *
 * Re-evaluated each call so tests / CLIs can mutate process.env between runs.
 */
export function diaryFiles() {
  const explicit = getEnv('DIARY_FILES');
  if (explicit) {
    return explicit
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean)
      .map((f) => (path.isAbsolute(f) ? f : path.join(DIARY_ROOT, f)));
  }
  if (!existsSync(DIARY_ROOT)) return [];
  return readdirSync(DIARY_ROOT)
    .filter((f) => f.endsWith('.md'))
    .sort()
    .map((f) => path.join(DIARY_ROOT, f));
}
