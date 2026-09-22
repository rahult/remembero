/**
 * `npm run eval:doc-facts` — extract facts from every page of the corpus, keeping each fact's
 * page, so a fact search can point back at the pages that hold the evidence.
 *
 * This is the product's extraction step (its `extractionSystemPrompt`, its output normalizer, its
 * Datalog parser), called one page at a time against an empty schema. It is not `rememberText`:
 * that path takes the store's mutation lock per write and grows the prompt with the store's schema,
 * which over 7,500 pages is hours of serialized calls. The facts that come out are the same kind;
 * what is lost is predicate reuse across pages, which a page-pointer index does not need.
 *
 * Output: .cache/document-recall/<doc>.facts.<model>.json = { page: [fact, ...] }, written as it
 * goes, so an interrupted run resumes at the first page without an entry.
 */

import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { parseProgram } from '../engine/index.js';
import { OpenRouterClient } from '../llm/client.js';
import { normalizeExtractionOutput } from '../llm/extraction-guard.js';
import { extractionSystemPrompt, NOTHING_SENTINEL } from '../llm/prompts.js';
import { pagesFromTextFile } from './document-corpus.js';
import { mapConcurrent } from './map-concurrent.js';

export const FACT_PAGE_CHARACTERS = 12_000;

/**
 * The product's extraction prompt is written for personal memory — "the speaker", "durable facts",
 * skip anything transient — and on document prose it answers `% nothing` for 78% of pages (all
 * 100 pages of an FDA regulation). This prompt keeps the same Datalog contract, so its facts go in
 * the same store and parse with the same parser, but tells the writer what a document page holds.
 */
export const DOCUMENT_EXTRACTION_PROMPT = `You index one page of a long document into Datalog facts for a search system.

Output one ground fact per line and nothing else: no prose, no code fences, no rules, no variables.
- Predicates are lowercase snake_case. Constants are lowercase snake_case atoms or single-quoted strings; keep the document's own words for names, titles and section headings ('National Fire Protection Association', 'Section 4255.1').
- Numbers are bare with the unit in the predicate: budget_million_eur(programme, 4.2).
- Capture what the page states: named entities and what is said about them, numbers with what they measure, dates, definitions, requirements, and every item of a list or row of a table as its own fact.
- Record where things sit: in_section(item, 'Section heading') for items under a heading on this page, and page_heading('Heading') for each heading.
- Record references to other standards, laws, documents and bodies: references(source, 'Target').
- Never infer anything the page does not state. Skip page numbers, running headers and footers.
- At most 40 facts. If the page is blank, a cover, or only a table of contents, output exactly: % nothing`;

/** The ground facts in one extraction reply; rules, constraints and unparsable lines are dropped. */
export function groundFactsFrom(reply: string): string[] {
  if (reply.trim() === NOTHING_SENTINEL) return [];
  const facts: string[] = [];
  for (const line of normalizeExtractionOutput(reply)) {
    try {
      const clauses = parseProgram(line);
      if (clauses.length === 1 && clauses[0]!.body.length === 0 && clauses[0]!.integrity !== true) {
        facts.push(line.trim());
      }
    } catch {
      // an unparsable line is what the product's retry loop would reject; skip it here
    }
  }
  return facts;
}

export function factsCachePath(documentId: string, model: string, prompt: 'product' | 'document' = 'product'): string {
  const variant = prompt === 'product' ? '' : `.${prompt}`;
  return `.cache/document-recall/${documentId}.facts.${model.replace(/[^a-z0-9.-]+/gi, '_')}${variant}.json`;
}

/** Page number → facts, from the cache run-document-facts.ts wrote (default: deepseek, document prompt). */
export function loadFactsByPage(documentId: string, model = 'deepseek-chat', prompt: 'product' | 'document' = 'document'): Map<number, string[]> {
  const path = factsCachePath(documentId, model, prompt);
  if (!existsSync(path)) throw new Error(`no extracted facts for ${documentId} at ${path}; run npm run eval:doc-facts first`);
  const raw = JSON.parse(readFileSync(path, 'utf8')) as Record<string, string[]>;
  return new Map(Object.entries(raw).map(([page, facts]) => [Number(page), facts]));
}

async function main() {
  const argv = process.argv.slice(2);
  const flag = (name: string, fallback: string) => {
    const at = argv.indexOf(name);
    return at >= 0 && argv[at + 1] !== undefined ? argv[at + 1]! : fallback;
  };
  const spec = JSON.parse(readFileSync(flag('--spec', 'benchmarks/document-recall/spec.json'), 'utf8')) as {
    sources: Array<{ id: string }>;
  };
  const model = flag('--model', 'deepseek-chat');
  const concurrency = Number(flag('--concurrency', '16'));
  const price = flag('--price', '0.15,0.6').split(',').map(Number);
  const client = new OpenRouterClient({
    model,
    apiKey: flag('--api-key', process.env.DEEPSEEK_API_KEY ?? ''),
    baseUrl: flag('--base-url', 'https://api.deepseek.com/v1'),
    timeoutMs: 120_000,
  });
  const promptVariant = flag('--prompt', 'document');
  if (promptVariant !== 'product' && promptVariant !== 'document') throw new Error('--prompt is product or document');
  const system =
    promptVariant === 'document' ? DOCUMENT_EXTRACTION_PROMPT : extractionSystemPrompt('(the store is empty)');

  let promptTokens = 0;
  let completionTokens = 0;
  let calls = 0;
  let failures = 0;
  const started = Date.now();
  for (const source of spec.sources) {
    const path = factsCachePath(source.id, model, promptVariant);
    const facts: Record<string, string[]> = existsSync(path) ? JSON.parse(readFileSync(path, 'utf8')) : {};
    const pages = pagesFromTextFile(`.cache/document-recall/${source.id}.pages.txt`);
    const todo = pages.filter((page) => facts[String(page.page)] === undefined && page.text.trim().length > 0);
    let sinceSave = 0;
    await mapConcurrent(todo, concurrency, async (page) => {
      try {
        const reply = await client.completeWithUsage([
          { role: 'system', content: system },
          { role: 'user', content: page.text.slice(0, FACT_PAGE_CHARACTERS) },
        ], { maxTokens: 2_048 });
        promptTokens += reply.usage?.promptTokens ?? 0;
        completionTokens += reply.usage?.completionTokens ?? 0;
        calls += 1;
        facts[String(page.page)] = groundFactsFrom(reply.content);
      } catch {
        failures += 1;
      }
      sinceSave += 1;
      if (sinceSave >= 50) {
        sinceSave = 0;
        writeFileSync(path, JSON.stringify(facts));
      }
    });
    writeFileSync(path, JSON.stringify(facts));
    const count = Object.values(facts).reduce((sum, list) => sum + list.length, 0);
    const empty = Object.values(facts).filter((list) => list.length === 0).length;
    const cost = (promptTokens * price[0]! + completionTokens * price[1]!) / 1e6;
    console.log(
      `${source.id}: ${pages.length} pages, ${count} facts (${(count / pages.length).toFixed(1)}/page, ` +
        `${empty} pages with none) · running cost $${cost.toFixed(2)} · ${failures} failures · ` +
        `${Math.round((Date.now() - started) / 1000)}s`,
    );
  }
  console.log(`\n${calls} calls, ${promptTokens} prompt + ${completionTokens} completion tokens, ${failures} failures`);
}

if (process.argv[1]?.endsWith('run-document-facts.js')) {
  main().catch((error: unknown) => {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
  });
}
