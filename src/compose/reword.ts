/**
 * Reword a world's prose lines with a model, keeping a rewrite only if every fact the line states
 * is still stated on that one line (the same check extraction faces). Shared by the benchmark
 * build (v2+) and the training-set builder, so both see the same kind of reworded document.
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import type { LongMemEvalCompletionClient } from '../evals/longmemeval-answer.js';
import { mapConcurrent } from '../evals/map-concurrent.js';
import { unsupportedReason } from './extract.js';
import { LINE_PARAPHRASE_PROMPT } from './paraphrase.js';
import type { Section } from './render.js';

export const PROSE_DOCUMENTS = new Set(['Board minutes', 'Contract amendments', 'Supplier sites']);
export const LINE_CACHE_PATH = '.cache/compose/line-paraphrase.deepseek.json';

export interface RewordStats {
  kept: number;
  rejected: number;
}

export class LineRewriter {
  private readonly cache: Record<string, string>;
  readonly stats: RewordStats = { kept: 0, rejected: 0 };

  constructor(private readonly client: LongMemEvalCompletionClient, private readonly cachePath = LINE_CACHE_PATH) {
    this.cache = existsSync(cachePath) ? JSON.parse(readFileSync(cachePath, 'utf8')) : {};
  }

  /** Fetch rewrites for every prose line not yet cached. */
  async prepare(sectionsList: ReadonlyArray<readonly Section[]>, concurrency = 16): Promise<void> {
    const lines = new Set<string>();
    for (const sections of sectionsList) {
      for (const s of sections) if (PROSE_DOCUMENTS.has(s.document)) for (const l of s.lines) if (this.cache[l.text] === undefined) lines.add(l.text);
    }
    await mapConcurrent([...lines], concurrency, async (text) => {
      try {
        const reply = await this.client.completeWithUsage([
          { role: 'system', content: LINE_PARAPHRASE_PROMPT },
          { role: 'user', content: text },
        ]);
        this.cache[text] = reply.content.trim().split('\n')[0]!.trim();
      } catch {
        // an unreachable rewrite leaves the original line
      }
    });
    mkdirSync('.cache/compose', { recursive: true });
    writeFileSync(this.cachePath, JSON.stringify(this.cache, null, 1));
  }

  /** The sections with each faithful rewrite applied; unfaithful or missing rewrites keep the original. */
  apply(sections: readonly Section[]): Section[] {
    return sections.map((section) => {
      if (!PROSE_DOCUMENTS.has(section.document)) return section;
      return {
        ...section,
        lines: section.lines.map((line) => {
          const candidate = this.cache[line.text];
          if (candidate === undefined || candidate === '') return line;
          const faithful = (line.facts ?? []).every((fact) => unsupportedReason(fact, `${section.heading}\n${candidate}`) === undefined);
          if (!faithful) {
            this.stats.rejected += 1;
            return line;
          }
          this.stats.kept += 1;
          return { ...line, text: candidate };
        }),
      };
    });
  }
}
