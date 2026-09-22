import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  Bm25Index,
  DenseIndex,
  reciprocalRankFusion,
  tokenize,
  type Embed,
} from '../src/evals/document-index.js';

const windows = [
  { id: 'p1', text: 'Section 4255 Brady bonds were issued to restructure emerging market debt.' },
  { id: 'p2', text: 'Section 4235 Japanese government bonds are issued by the Ministry of Finance.' },
  { id: 'p3', text: 'Section 3000 investment securities section section section section.' },
  { id: 'p4', text: 'Glossary of terms used in this section.' },
];

describe('tokenize', () => {
  it('lower-cases, keeps digits and drops stopwords', () => {
    expect(tokenize('The Brady plan of 1989')).toEqual(['brady', 'plan', '1989']);
  });

  it('keeps accented words whole', () => {
    expect(tokenize('prejudiciële procedures')).toEqual(['prejudiciële', 'procedures']);
  });
});

describe('Bm25Index', () => {
  const index = new Bm25Index(windows);

  it('lets a rare term outrank a common one repeated many times', () => {
    expect(index.rank('Brady section', 2)[0]).toBe('p1');
  });

  it('weights a rare term above a common one', () => {
    expect(index.idf('brady')).toBeGreaterThan(index.idf('section'));
  });

  it('returns nothing for a query with no indexed terms', () => {
    expect(index.rank('zeppelin', 3)).toEqual([]);
  });

  it('cuts to the limit', () => {
    expect(index.rank('bonds issued', 1)).toHaveLength(1);
  });
});

/** An embedding that is a bag of three hand-picked words, so cosine is predictable. */
const toyEmbed: Embed = async (texts) =>
  texts.map((text) => {
    const lower = text.toLowerCase();
    return ['brady', 'japanese', 'glossary'].map((word) => (lower.includes(word) ? 1 : 0.01));
  });

describe('DenseIndex', () => {
  it('ranks by cosine similarity', async () => {
    const index = await DenseIndex.build(windows, toyEmbed);
    expect((await index.rank('japanese debt', 1))[0]).toBe('p2');
  });

  it('reuses cached vectors for identical windows and rebuilds when the text changes', async () => {
    const cachePath = join(mkdtempSync(join(tmpdir(), 'dense-')), 'index.json');
    let calls = 0;
    const counting: Embed = async (texts, kind) => {
      if (kind === 'document') calls += 1;
      return toyEmbed(texts, kind);
    };
    await DenseIndex.build(windows, counting, { cachePath });
    await DenseIndex.build(windows, counting, { cachePath });
    expect(calls).toBe(1);
    await DenseIndex.build([...windows.slice(0, 3), { id: 'p4', text: 'changed' }], counting, { cachePath });
    expect(calls).toBe(2);
  });
});

describe('reciprocalRankFusion', () => {
  it('promotes what both lists agree on', () => {
    expect(reciprocalRankFusion([['a', 'b'], ['c', 'b']], 1)).toEqual(['b']);
  });

  it('keeps an item only one list found', () => {
    expect(reciprocalRankFusion([['a'], ['z']], 5).sort()).toEqual(['a', 'z']);
  });
});
