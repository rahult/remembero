/**
 * Question paraphrasing for training examples. A strong model rewrites each
 * templated question; a paraphrase survives only if every entity and value
 * constant from the original question is still present verbatim, so the gold
 * program still fits. Results are cached on disk so reruns make no calls, and
 * any failure degrades to the templated question rather than dropping data.
 */
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import type { LlmClient } from '../llm/client.js';
import type { Example } from './verify.js';

export const PARAPHRASE_MODEL = 'openai/gpt-5.6-luna';

export interface Paraphraser {
  paraphrase(example: Example, count: number): Promise<string[]>;
}

/** Lowercase constants of the program that also appear in the question — these must survive. */
function anchoredConstants(example: Example): string[] {
  const constants = example.program.match(/\b[a-z][a-z0-9_]*\b/g) ?? [];
  const question = example.question.toLowerCase();
  return [...new Set(constants)].filter(
    (c) => !c.endsWith('_plus') && question.includes(c),
  );
}

export function preservesConstants(
  example: Example,
  paraphrase: string,
): boolean {
  const text = paraphrase.toLowerCase();
  // whole-word match: "search" must not be satisfied by "searched"; an
  // underscore may be rendered as a space or kept.
  return anchoredConstants(example).every((c) => {
    const escaped = c
      .replaceAll(/[.*+?^${}()|[\]\\]/g, '\\$&')
      .replaceAll('_', '[_ ]');
    return new RegExp(`(^|[^a-z0-9])${escaped}($|[^a-z0-9])`).test(text);
  });
}

const SYSTEM = `You rewrite questions about a small knowledge base. Return ONLY a JSON array of strings.
Rules: keep every proper name and value EXACTLY as written (same spelling, lowercase); keep the meaning
and the direction (above/below, depends on/depended on by) identical; vary sentence structure and
vocabulary; no numbering, no prose outside the JSON array.`;

export function createLlmParaphraser(
  client: LlmClient,
  cacheDir: string,
): Paraphraser {
  mkdirSync(cacheDir, { recursive: true });
  return {
    async paraphrase(example, count) {
      const key = createHash('sha256')
        .update(
          `${PARAPHRASE_MODEL}\n${count}\n${example.question}\n${example.program}`,
        )
        .digest('hex');
      const path = join(cacheDir, `paraphrase-${key}.json`);
      if (existsSync(path)) {
        return JSON.parse(readFileSync(path, 'utf8')) as string[];
      }
      const raw = await client.complete([
        { role: 'system', content: SYSTEM },
        {
          role: 'user',
          content: `Write ${count} paraphrases of: ${example.question}`,
        },
      ]);
      const start = raw.indexOf('[');
      const end = raw.lastIndexOf(']');
      let parsed: unknown = [];
      try {
        parsed = JSON.parse(raw.slice(start, end + 1));
      } catch {
        parsed = [];
      }
      const kept = (Array.isArray(parsed) ? parsed : [])
        .filter(
          (p): p is string => typeof p === 'string' && p.trim().length > 0,
        )
        .map((p) => p.trim())
        .filter((p) => p.toLowerCase() !== example.question.toLowerCase())
        .filter((p) => preservesConstants(example, p))
        .slice(0, count);
      writeFileSync(path, JSON.stringify(kept));
      return kept;
    },
  };
}

/** Returns every original example followed by its paraphrased copies. */
export async function paraphraseExamples(
  examples: Example[],
  paraphraser: Paraphraser,
  count: number,
  concurrency = 8,
): Promise<Example[]> {
  const out: Example[] = [];
  let index = 0;
  const worker = async () => {
    while (index < examples.length) {
      const example = examples[index++];
      out.push(example);
      if (count <= 0) continue;
      try {
        for (const question of await paraphraser.paraphrase(example, count)) {
          out.push({
            ...example,
            question,
            template: `${example.template}#paraphrase`,
          });
        }
      } catch {
        // degrade to the templated question only
      }
    }
  };
  await Promise.all(
    Array.from(
      { length: Math.max(1, Math.min(concurrency, examples.length)) },
      worker,
    ),
  );
  return out;
}
