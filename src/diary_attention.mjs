import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';

import { LAB_ROOT_PATH, DIARY_ROOT, diaryFiles } from './nm_config.mjs';

// `repoRoot` is kept as the base for relative-path reporting in sourcePath
// fields. In the standalone repo it points at the lab root; downstream
// consumers don't care about the absolute prefix, only the relative shape.
const repoRoot = LAB_ROOT_PATH;
const diaryRoot = DIARY_ROOT;

// File list is discovered at import time from DIARY_ROOT (or DIARY_FILES env
// override). See src/nm_config.mjs and .env.example for details.
const DEFAULT_FILES = diaryFiles();

const MONTHS = {
  january: 1,
  jan: 1,
  february: 2,
  feb: 2,
  march: 3,
  mar: 3,
  april: 4,
  apr: 4,
  may: 5,
  june: 6,
  jun: 6,
  july: 7,
  jul: 7,
  august: 8,
  aug: 8,
  september: 9,
  sep: 9,
  october: 10,
  oct: 10,
  november: 11,
  nov: 11,
  december: 12,
  dec: 12
};

const STOPWORDS = new Set([
  'a',
  'about',
  'and',
  'are',
  'at',
  'be',
  'by',
  'day',
  'days',
  'did',
  'for',
  'from',
  'how',
  'i',
  'in',
  'into',
  'is',
  'it',
  'me',
  'my',
  'of',
  'on',
  'or',
  'the',
  'to',
  'was',
  'were',
  'what',
  'when',
  'with'
]);

const EMOTION_TERMS = {
  happy: { valence: 1, arousal: 0.45 },
  happiest: { valence: 1, arousal: 0.5, mood: 5 },
  excited: { valence: 0.8, arousal: 0.9 },
  enthusiastic: { valence: 0.8, arousal: 0.8 },
  hopeful: { valence: 0.75, arousal: 0.35 },
  proud: { valence: 0.75, arousal: 0.45 },
  relaxed: { valence: 0.65, arousal: -0.35 },
  calm: { valence: 0.55, arousal: -0.55 },
  refreshed: { valence: 0.65, arousal: -0.1 },
  sad: { valence: -0.8, arousal: -0.1, mood: 1 },
  saddest: { valence: -1, arousal: 0, mood: 1 },
  anxious: { valence: -0.75, arousal: 0.85 },
  angry: { valence: -0.8, arousal: 0.85 },
  tired: { valence: -0.45, arousal: -0.75 },
  stressed: { valence: -0.7, arousal: 0.75 },
  peaceful: { valence: 0.7, arousal: -0.65 },
  romantic: { valence: 0.85, arousal: 0.45 }
};

const DIARY_FEATURES = {
  travel: ['travel', 'trip', 'hotel', 'airport', 'flight', 'cappadocia', 'kapadokya', 'nevsehir', 'bali', 'vietnam', 'thailand'],
  relationship: ['sena', 'partner', 'romantic', 'relationship', 'date', 'ring'],
  music: ['music', 'vocal', 'song', 'guitar', 'concert', 'stage', 'recording'],
  work: ['work', 'office', 'company', 'project', 'startup', 'zephio', 'client'],
  health: ['health', 'doctor', 'pain', 'sleep', 'tired', 'sick'],
  social: ['friend', 'family', 'ugur', 'tansu', 'busra', 'conversation'],
  productivity: ['productive', 'study', 'learn', 'focus', 'task', 'plan']
};

const TOKEN_ALIASES = {
  cappadocia: ['kapadokya', 'nevşehir', 'nevsehir', 'peri', 'baca'],
  kapadokya: ['cappadocia', 'nevşehir', 'nevsehir'],
  nevsehir: ['nevşehir', 'kapadokya', 'cappadocia'],
  nevşehir: ['nevsehir', 'kapadokya', 'cappadocia'],
  newyear: ['yilbasi', 'yılbaşı', '00.00'],
  'new-year': ['yilbasi', 'yılbaşı', '00.00'],
  yilbasi: ['yılbaşı', 'newyear'],
  yılbaşı: ['yilbasi', 'newyear']
};

function normalizeDate(value) {
  const match = String(value || '').match(/(\d{4})[./-](\d{1,2})[./-](\d{1,2})/);
  if (!match) return null;
  return `${match[1]}-${match[2].padStart(2, '0')}-${match[3].padStart(2, '0')}`;
}

function ordinal(dateString) {
  return Math.floor(Date.parse(`${dateString}T00:00:00Z`) / 86_400_000);
}

function daysInMonth(year, month) {
  return new Date(Date.UTC(year, month, 0)).getUTCDate();
}

function tokenize(value) {
  const tokens = String(value || '')
    .toLocaleLowerCase('en-US')
    .split(/[^a-z0-9ğüşöçıİĞÜŞÖÇ]+/i)
    .filter((token) => token.length > 1 && !STOPWORDS.has(token));
  return tokens.flatMap((token) => [token, ...(TOKEN_ALIASES[token] || [])]);
}

function vectorize(tokens) {
  const vector = new Map();
  for (const token of tokens) {
    vector.set(token, (vector.get(token) || 0) + 1);
  }
  return vector;
}

function cosine(a, b) {
  let dot = 0;
  let aNorm = 0;
  let bNorm = 0;

  for (const value of a.values()) aNorm += value * value;
  for (const value of b.values()) bNorm += value * value;
  for (const [key, value] of a.entries()) dot += value * (b.get(key) || 0);

  if (!aNorm || !bNorm) return 0;
  return dot / Math.sqrt(aNorm * bNorm);
}

function clamp(value, min = 0, max = 1) {
  return Math.max(min, Math.min(max, value));
}

function parseMood(text) {
  const match = String(text || '').match(/\*\*Mood\*\*:\s*([1-5])|mood[^0-9]{0,20}([1-5])|(?:^|\s)([1-5])\/5(?:\s|$)/i);
  if (!match) return null;
  return Number(match[1] || match[2] || match[3]);
}

function parseIcons(text) {
  const match = String(text || '').match(/\*\*Icons\*\*:\s*([^\n]+)/i);
  if (!match) return [];
  return match[1]
    .split(',')
    .map((icon) => icon.trim().toLocaleLowerCase('en-US'))
    .filter(Boolean);
}

function emotionVector({ mood, icons, text }) {
  const tokens = new Set([...tokenize(text), ...icons.flatMap((icon) => tokenize(icon))]);
  let valence = mood ? (mood - 3) / 2 : 0;
  let arousal = 0;
  let hits = mood ? 1 : 0;

  for (const token of tokens) {
    const emotion = EMOTION_TERMS[token];
    if (!emotion) continue;
    valence += emotion.valence || 0;
    arousal += emotion.arousal || 0;
    hits += 1;
  }

  return {
    valence: clamp(valence / Math.max(hits, 1), -1, 1),
    arousal: clamp(arousal / Math.max(hits, 1), -1, 1),
    mood: mood || null
  };
}

function diaryFeatureVector(text, icons = []) {
  const haystack = `${text} ${icons.join(' ')}`.toLocaleLowerCase('en-US');
  const features = {};
  for (const [name, terms] of Object.entries(DIARY_FEATURES)) {
    const hits = terms.filter((term) => haystack.includes(term)).length;
    features[name] = clamp(hits / Math.max(terms.length / 2, 1));
  }
  return features;
}

function parseDiaryFiles(files = DEFAULT_FILES) {
  const memories = [];

  for (const file of files) {
    if (!existsSync(file)) continue;
    const raw = readFileSync(file, 'utf8');
    const parts = raw.split(/(?=^###\s+\d{4}[./-]\d{1,2}[./-]\d{1,2})/gm);

    for (const part of parts) {
      const date = normalizeDate(part.slice(0, 80));
      if (!date) continue;
      const mood = parseMood(part);
      const icons = parseIcons(part);
      const tokens = tokenize(part);

      memories.push({
        id: `${path.basename(file)}:${date}`,
        date,
        ordinal: ordinal(date),
        year: Number(date.slice(0, 4)),
        sourcePath: path.relative(repoRoot, file),
        mood,
        icons,
        text: part.trim(),
        key: {
          semantic: vectorize(tokens),
          emotion: emotionVector({ mood, icons, text: part }),
          diary: diaryFeatureVector(part, icons)
        },
        value: {
          sourcePath: path.relative(repoRoot, file),
          locator: date,
          summaryText: part.split('\n').slice(0, 12).join('\n'),
          fullText: part.trim()
        }
      });
    }
  }

  return memories.sort((a, b) => a.ordinal - b.ordinal);
}

function parseTimeWindow(query) {
  const lowered = query.toLocaleLowerCase('en-US');
  const year = Number(lowered.match(/\b(20\d{2})\b/)?.[1]);
  const monthName = Object.keys(MONTHS).find((name) => lowered.includes(name));

  if (year && monthName) {
    const month = MONTHS[monthName];
    const start = `${year}-${String(month).padStart(2, '0')}-01`;
    const end = `${year}-${String(month).padStart(2, '0')}-${String(daysInMonth(year, month)).padStart(2, '0')}`;
    return { start, end, center: ordinal(start) + Math.round((ordinal(end) - ordinal(start)) / 2), granularity: 'month' };
  }

  const explicitDates = [...query.matchAll(/(20\d{2})[./-](\d{1,2})[./-](\d{1,2})/g)].map((match) =>
    normalizeDate(match[0])
  );
  if (explicitDates.length) {
    const sorted = explicitDates.sort();
    const start = sorted[0];
    const end = sorted.at(-1);
    return { start, end, center: ordinal(start) + Math.round((ordinal(end) - ordinal(start)) / 2), granularity: 'day' };
  }

  if (year && /\b(end|last)\b/.test(lowered)) {
    const start = `${year}-12-01`;
    const end = `${year}-12-31`;
    return { start, end, center: ordinal('' + year + '-12-16'), granularity: 'range' };
  }

  if (year && /\blate\b/.test(lowered)) {
    const start = `${year}-10-01`;
    const end = `${year}-12-31`;
    return { start, end, center: ordinal('' + year + '-11-15'), granularity: 'range' };
  }

  if (year && /\b(start|early|beginning)\b/.test(lowered)) {
    const start = `${year}-01-01`;
    const end = `${year}-03-31`;
    return { start, end, center: ordinal('' + year + '-02-14'), granularity: 'range' };
  }

  if (year) {
    const start = `${year}-01-01`;
    const end = `${year}-12-31`;
    return { start, end, center: ordinal('' + year + '-07-01'), granularity: 'year' };
  }

  return null;
}

function parseEmotionTarget(query) {
  const lowered = query.toLocaleLowerCase('en-US');
  const tokens = tokenize(lowered);
  const directMood = lowered.match(/\bmood\s*([1-5])\b|([1-5])\/5/);
  let mood = directMood ? Number(directMood[1] || directMood[2]) : null;
  let valence = 0;
  let arousal = 0;
  let hits = 0;
  let mode = 'mixed';

  for (const token of tokens) {
    const emotion = EMOTION_TERMS[token];
    if (!emotion) continue;
    if (emotion.mood && !mood) mood = emotion.mood;
    valence += emotion.valence || 0;
    arousal += emotion.arousal || 0;
    hits += 1;
  }

  if (/happiest|happy|best|great|good/.test(lowered)) {
    mood = mood || 5;
    valence += 1;
    hits += 1;
    mode = 'happy';
  } else if (/saddest|sad|worst|bad/.test(lowered)) {
    mood = mood || 1;
    valence -= 1;
    hits += 1;
    mode = 'sad';
  } else if (/anxious|stress|stressed|nervous/.test(lowered)) {
    valence -= 0.7;
    arousal += 0.9;
    hits += 1;
    mode = 'anxious';
  } else if (/calm|peaceful|relaxed/.test(lowered)) {
    valence += 0.6;
    arousal -= 0.7;
    hits += 1;
    mode = 'calm';
  }

  if (!mood && !hits) return null;

  return {
    mood,
    valence: clamp(valence / Math.max(hits, 1), -1, 1),
    arousal: clamp(arousal / Math.max(hits, 1), -1, 1),
    mode
  };
}

function parseDiaryHints(query) {
  const lowered = query.toLocaleLowerCase('en-US');
  const features = {};
  for (const [name, terms] of Object.entries(DIARY_FEATURES)) {
    features[name] = terms.some((term) => lowered.includes(term)) ? 1 : 0;
  }
  return {
    features,
    exactTerms: tokenize(query)
  };
}

function buildQuery(raw) {
  const text = String(raw || '').trim();
  const semanticTerms = tokenize(text);
  return {
    raw: text,
    semanticTerms,
    semantic: vectorize(semanticTerms),
    timeWindow: parseTimeWindow(text),
    emotionTarget: parseEmotionTarget(text),
    diaryHints: parseDiaryHints(text)
  };
}

function temporalScore(memory, timeWindow) {
  if (!timeWindow) return 0.5;
  const start = ordinal(timeWindow.start);
  const end = ordinal(timeWindow.end);
  if (memory.ordinal >= start && memory.ordinal <= end) return 1;

  const distance = memory.ordinal < start ? start - memory.ordinal : memory.ordinal - end;
  const decay = timeWindow.granularity === 'day' ? 10 : timeWindow.granularity === 'month' ? 12 : 45;
  return Math.exp(-distance / decay);
}

function emotionScore(memory, target) {
  if (!target) return 0.5;
  const emotion = memory.key.emotion;
  const moodScore = target.mood && memory.mood ? 1 - Math.abs(memory.mood - target.mood) / 4 : 0.5;
  const valenceScore = 1 - Math.abs((emotion.valence || 0) - (target.valence || 0)) / 2;
  const arousalScore = target.mode === 'mixed' ? 0.5 : 1 - Math.abs((emotion.arousal || 0) - (target.arousal || 0)) / 2;
  return clamp(moodScore * 0.55 + valenceScore * 0.3 + arousalScore * 0.15);
}

function diaryFeatureScore(memory, hints) {
  const desired = Object.entries(hints.features).filter(([, value]) => value > 0);
  const termHitScore = hints.exactTerms.length
    ? hints.exactTerms.filter((term) => memory.text.toLocaleLowerCase('en-US').includes(term)).length / hints.exactTerms.length
    : 0;

  if (!desired.length) return clamp(termHitScore);
  const featureScore =
    desired.reduce((sum, [name]) => sum + (memory.key.diary[name] || 0), 0) / Math.max(desired.length, 1);
  return clamp(featureScore * 0.7 + termHitScore * 0.3);
}

function continuityScore(memory, memories, query) {
  const target = query.emotionTarget;
  const window = memories.filter((item) => Math.abs(item.ordinal - memory.ordinal) <= 2 && item.id !== memory.id);
  if (!window.length) return 0;

  const sameMoodArc = target?.mood
    ? window.filter((item) => item.mood && Math.abs(item.mood - target.mood) <= 1).length / window.length
    : window.filter((item) => item.mood && memory.mood && Math.abs(item.mood - memory.mood) <= 1).length / window.length;

  const sameFeatureArc = window.filter((item) => cosine(memory.key.semantic, item.key.semantic) > 0.12).length / window.length;
  return clamp(sameMoodArc * 0.65 + sameFeatureArc * 0.35);
}

function chooseWeights(query) {
  const weights = {
    semantic: 0.32,
    temporal: query.timeWindow ? 0.3 : 0.1,
    emotion: query.emotionTarget ? 0.28 : 0.12,
    diary: 0.18,
    continuity: /around|after|before|changed|timeline|led|during|streak/.test(query.raw.toLocaleLowerCase('en-US')) ? 0.22 : 0.1
  };

  const total = Object.values(weights).reduce((sum, value) => sum + value, 0);
  return Object.fromEntries(Object.entries(weights).map(([key, value]) => [key, value / total]));
}

function softmax(scores, temperature = 0.08) {
  const max = Math.max(...scores);
  const exp = scores.map((score) => Math.exp((score - max) / temperature));
  const total = exp.reduce((sum, value) => sum + value, 0);
  return exp.map((value) => value / total);
}

function scoreMemories(query, memories) {
  const weights = chooseWeights(query);
  const aroundQuery = /\b(around|after|before|changed|timeline|led|during|streak|arc)\b/.test(
    query.raw.toLocaleLowerCase('en-US')
  );
  const baseScored = memories.map((memory) => {
    const heads = {
      semantic: cosine(query.semantic, memory.key.semantic),
      temporal: temporalScore(memory, query.timeWindow),
      emotion: emotionScore(memory, query.emotionTarget),
      diary: diaryFeatureScore(memory, query.diaryHints),
      continuity: 0
    };
    return { memory, heads };
  });

  const anchorOrdinals = aroundQuery
    ? baseScored
        .filter((item) => item.heads.semantic >= 0.1)
        .sort((a, b) => b.heads.semantic + b.heads.diary - (a.heads.semantic + a.heads.diary))
        .slice(0, 5)
        .map((item) => item.memory.ordinal)
    : [];

  const scored = baseScored.map((item) => {
    const anchorContinuity = anchorOrdinals.length
      ? Math.max(...anchorOrdinals.map((anchor) => Math.exp(-Math.abs(item.memory.ordinal - anchor) / 1.8)))
      : 0;
    item.heads.continuity = aroundQuery
      ? anchorContinuity
      : continuityScore(item.memory, memories, query);
    item.score = Object.entries(item.heads).reduce((sum, [name, value]) => sum + value * weights[name], 0);
    return item;
  });

  const attention = softmax(scored.map((item) => item.score));
  return scored
    .map((item, index) => ({ ...item, attention: attention[index] }))
    .sort((a, b) => b.score - a.score || b.memory.ordinal - a.memory.ordinal);
}

function formatResult(item, index) {
  const { memory, score, attention, heads } = item;
  const headSummary = Object.entries(heads)
    .map(([name, value]) => `${name}:${value.toFixed(2)}`)
    .join(' ');
  const title = `${index + 1}. ${memory.date} mood=${memory.mood ?? 'n/a'} score=${score.toFixed(3)} attention=${attention.toFixed(3)}`;
  const icons = memory.icons.length ? `icons=${memory.icons.slice(0, 10).join(', ')}` : 'icons=n/a';
  const excerpt = memory.value.summaryText
    .replace(/^###\s+/gm, '')
    .split('\n')
    .filter((line) => line.trim())
    .slice(0, 7)
    .join('\n');

  return [title, `${memory.sourcePath} ${icons}`, headSummary, excerpt].join('\n');
}

function printUsage() {
  console.log('Usage: npm run query -- "happiest days at the end of 2024" [-- --limit 8] [-- --json]');
}

function parseArgs(argv) {
  const args = argv.slice(2);
  const flags = new Set(args.filter((arg) => arg.startsWith('--')));
  const limitIndex = args.indexOf('--limit');
  const limit = limitIndex >= 0 ? Number(args[limitIndex + 1]) : 6;
  const query = args.filter((arg, index) => !arg.startsWith('--') && args[index - 1] !== '--limit').join(' ').trim();
  return {
    query: query || 'happiest days at the end of 2024',
    limit: Number.isFinite(limit) ? Math.max(1, Math.min(20, limit)) : 6,
    json: flags.has('--json'),
    help: flags.has('--help') || flags.has('-h')
  };
}

function main() {
  const args = parseArgs(process.argv);
  if (args.help) {
    printUsage();
    return;
  }

  const memories = parseDiaryFiles();
  const query = buildQuery(args.query);
  const ranked = scoreMemories(query, memories).slice(0, args.limit);

  if (args.json) {
    console.log(
      JSON.stringify(
        {
          query: {
            raw: query.raw,
            semanticTerms: query.semanticTerms,
            timeWindow: query.timeWindow,
            emotionTarget: query.emotionTarget,
            diaryHints: query.diaryHints
          },
          count: memories.length,
          results: ranked.map((item) => ({
            date: item.memory.date,
            mood: item.memory.mood,
            score: item.score,
            attention: item.attention,
            heads: item.heads,
            sourcePath: item.memory.sourcePath,
            icons: item.memory.icons,
            text: item.memory.value.summaryText
          }))
        },
        null,
        2
      )
    );
    return;
  }

  console.log(`Query: ${query.raw}`);
  console.log(`Loaded ${memories.length} diary day tokens`);
  if (query.timeWindow) {
    console.log(`Time window: ${query.timeWindow.start} to ${query.timeWindow.end} (${query.timeWindow.granularity})`);
  }
  if (query.emotionTarget) {
    console.log(
      `Emotion target: mode=${query.emotionTarget.mode} mood=${query.emotionTarget.mood ?? 'n/a'} valence=${query.emotionTarget.valence.toFixed(2)} arousal=${query.emotionTarget.arousal.toFixed(2)}`
    );
  }
  console.log('');
  console.log(ranked.map(formatResult).join('\n\n---\n\n'));
}

main();
