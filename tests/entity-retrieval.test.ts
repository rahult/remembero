import { describe, expect, it } from 'vitest';
import { parseProgram, canonicalKey } from '../src/engine/index.js';
import {
  expandByEntities,
  interleaveSessions,
} from '../src/knowledge/entity-retrieval.js';
import type { MemorySource } from '../src/store/store.js';

function indexed(program: string, sessionOf: Record<string, string>) {
  const clauses = parseProgram(program);
  const sources = new Map<string, MemorySource[]>();
  for (const clause of clauses) {
    const text =
      clause.head.predicate +
      '(' +
      clause.head.args.map((a) => (a as { value: string }).value).join(',') +
      ')';
    const opId = sessionOf[text] ?? 'unknown';
    sources.set(canonicalKey(clause), [
      { namespace: 'x', opId, ts: `2024-01-0${opId.slice(-1)}T00:00:00.000Z` },
    ]);
  }
  return { clauses, sources };
}

describe('expandByEntities: one hop over the fact store from the question', () => {
  it('finds the sessions of every fact about the same relation and subject, even when their text shares no word with the question', () => {
    const { clauses, sources } = indexed(
      'model_kit(user, revell_f15). model_kit(user, tamiya_spitfire). bought(user, kayak). lives_in(user, osaka).',
      {
        'model_kit(user,revell_f15)': 's1',
        'model_kit(user,tamiya_spitfire)': 's2',
        'bought(user,kayak)': 's3',
        'lives_in(user,osaka)': 's4',
      },
    );
    const hits = expandByEntities(
      clauses,
      'How many model kits have I bought?',
      sources,
      { selfAtom: 'user' },
    );
    const ids = hits.map((h) => h.opId);
    // s1 and s2 through the model_kit relation (kits -> kit), s3 through bought; s4 has no link
    expect(ids.slice(0, 3).sort()).toEqual(['s1', 's2', 's3']);
    expect(ids).not.toContain('s4');
    expect(hits.find((h) => h.opId === 's2')?.facts).toEqual([
      'model_kit(user, tamiya_spitfire).',
    ]);
  });

  it('follows a named entity across sessions and ranks the session with the most related facts first', () => {
    const { clauses, sources } = indexed(
      "moved_to(rachel, 'the suburbs'). friend(user, rachel). moved_to(rachel, city). works_at(tom, acme).",
      {
        'moved_to(rachel,the suburbs)': 's1',
        'friend(user,rachel)': 's2',
        'moved_to(rachel,city)': 's3',
        'works_at(tom,acme)': 's4',
      },
    );
    const hits = expandByEntities(
      clauses,
      'Where did Rachel move to?',
      sources,
      { selfAtom: 'user' },
    );
    expect(hits.map((h) => h.opId)).toEqual(
      expect.arrayContaining(['s1', 's3', 's2']),
    );
    expect(hits.map((h) => h.opId)).not.toContain('s4');
    // a session whose fact matched the question directly outranks one reached only by expansion
    expect(['s1', 's3']).toContain(hits[0]?.opId);
  });

  it('returns nothing when no fact touches the question', () => {
    const { clauses, sources } = indexed('works_at(tom, acme).', {
      'works_at(tom,acme)': 's1',
    });
    expect(
      expandByEntities(clauses, 'What is my dentist called?', sources, {
        selfAtom: 'user',
      }),
    ).toEqual([]);
  });
});

describe('interleaveSessions: entity hits take alternate slots without wiping the lexical order', () => {
  it('alternates and skips duplicates', () => {
    expect(
      interleaveSessions(['a', 'b', 'c', 'd'], ['x', 'b', 'y'], 5),
    ).toEqual(['a', 'x', 'b', 'y', 'c']);
  });
  it('falls back to the lexical list when there are no entity hits', () => {
    expect(interleaveSessions(['a', 'b'], [], 5)).toEqual(['a', 'b']);
  });
});
