import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { windowPages, windowsAsSessions } from '../src/evals/document-corpus.js';
import { buildDocumentRanker, interleaveFirst, llmDecomposer } from '../src/evals/document-rankers.js';
import type { Embed } from '../src/evals/document-index.js';
import type { LongMemEvalCompletionClient } from '../src/evals/longmemeval-answer.js';

const texts = [
  'Brady bonds restructured emerging market debt in 1989.',
  'Japanese government bonds are issued by the Ministry of Finance.',
  'Glossary and index of terms.',
  'Treasury bills mature within one year.',
];
const sessions = windowsAsSessions(
  windowPages(texts.map((text, index) => ({ page: index + 1, text })), { pagesPerWindow: 1 }),
);

const toyEmbed: Embed = async (inputs) =>
  inputs.map((text) => ['brady', 'japanese', 'glossary', 'treasury'].map((w) => (text.toLowerCase().includes(w) ? 1 : 0.01)));

describe('buildDocumentRanker', () => {
  it('ranks with bm25 without any model', async () => {
    const ranker = await buildDocumentRanker('bm25', sessions);
    expect((await ranker.rank('Brady bonds', 1))[0]).toBe('pages-0001-0001');
  });

  it('ranks with the product baseline', async () => {
    const ranker = await buildDocumentRanker('product', sessions);
    expect(await ranker.rank('Japanese government bonds', 2)).toContain('pages-0002-0002');
  });

  it('fuses bm25 and dense in hybrid', async () => {
    const ranker = await buildDocumentRanker('hybrid', sessions, { embed: toyEmbed });
    expect((await ranker.rank('treasury bills', 1))[0]).toBe('pages-0004-0004');
  });

  it('puts the top page of every sub-query into the result', async () => {
    const ranker = await buildDocumentRanker('decomposed', sessions, {
      embed: toyEmbed,
      decomposer: async () => ['Brady bonds', 'Japanese government bonds'],
    });
    const top = await ranker.rank('Which bond type is in the Brady section but not the JGB section?', 2);
    expect(top.sort()).toEqual(['pages-0001-0001', 'pages-0002-0002']);
  });

  it('refuses a dense ranker without an embedding function', async () => {
    await expect(buildDocumentRanker('dense', sessions)).rejects.toThrow(/embedding/);
  });
});

describe('interleaveFirst', () => {
  it('gives each list its best item before filling from the fused order', () => {
    expect(interleaveFirst([['a', 'b'], ['c', 'a']], ['a', 'b', 'c'], 3)).toEqual(['a', 'c', 'b']);
  });
});

describe('llmDecomposer', () => {
  it('parses one query per line, strips list markers and caches the answer', async () => {
    let calls = 0;
    const client: LongMemEvalCompletionClient = {
      model: 'fake',
      async completeWithUsage() {
        calls += 1;
        return {
          content: '1. Brady bonds section\n- JGB guarantee section\n\n',
          usage: { promptTokens: 1, completionTokens: 1, totalTokens: 2, cachedPromptTokens: 0, reasoningTokens: 0, costUsd: 0 },
        };
      },
    };
    const decompose = llmDecomposer(client, join(mkdtempSync(join(tmpdir(), 'dec-')), 'cache.json'));
    expect(await decompose('q?')).toEqual(['Brady bonds section', 'JGB guarantee section']);
    await decompose('q?');
    expect(calls).toBe(1);
  });
});

describe('fact-pointer rankers', () => {
  const factsByPage = new Map<number, string[]>([
    [1, ['issued_to_restructure(brady_bonds, emerging_market_debt).', 'plan_year(brady_plan, 1989).']],
    [2, ['issuer(japanese_government_bonds, ministry_of_finance).']],
    [4, ['maturity_within_years(treasury_bills, 1).']],
  ]);

  it('returns the page a matching fact came from', async () => {
    const ranker = await buildDocumentRanker('facts', sessions, { factsByPage });
    expect((await ranker.rank('Who issues Japanese government bonds?', 1))[0]).toBe('pages-0002-0002');
  });

  it('lets a page with several matching facts outrank a page with one', async () => {
    const ranker = await buildDocumentRanker('facts', sessions, { factsByPage });
    expect((await ranker.rank('Brady plan bonds', 2))[0]).toBe('pages-0001-0001');
  });

  it('never returns a page no fact points at', async () => {
    const ranker = await buildDocumentRanker('facts', sessions, { factsByPage });
    expect(await ranker.rank('glossary index', 3)).not.toContain('pages-0003-0003');
  });

  it('fuses fact pointers with the hybrid ranking', async () => {
    const ranker = await buildDocumentRanker('hybrid+facts', sessions, { embed: toyEmbed, factsByPage });
    expect(await ranker.rank('treasury bills maturity', 2)).toContain('pages-0004-0004');
  });

  it('refuses a fact ranker without facts', async () => {
    await expect(buildDocumentRanker('facts', sessions)).rejects.toThrow(/facts/);
  });
});
