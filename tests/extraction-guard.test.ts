import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { parseProgram, serializeClause } from '../src/engine/index.js';
import {
  applyPredicateAliases,
  assertGroundedConstants,
  assertKnownVocabulary,
  normalizeExtractionOutput,
  predicateAliasesFrom,
  rewriteSelfAtoms,
  SELF_ATOM_SYNONYMS,
} from '../src/llm/extraction-guard.js';
import type { ChatMessage, LlmClient } from '../src/llm/client.js';
import { rememberText } from '../src/llm/pipeline.js';
import { MemoryStore } from '../src/store/store.js';

class ScriptedLlm implements LlmClient {
  readonly prompts: ChatMessage[][] = [];
  constructor(private readonly replies: string[]) {}
  async complete(messages: ChatMessage[]): Promise<string> {
    this.prompts.push(messages);
    const next = this.replies.shift();
    if (next === undefined) throw new Error('scripted LLM exhausted');
    return next;
  }
}

describe('normalizeExtractionOutput: lenient before the parser', () => {
  it('strips list markers, adds missing periods, drops blank lines, keeps retract lines', () => {
    const raw = [
      '- works_at(mira, acme)',
      '* lives_in(mira, sydney).',
      '1. prefers(mira, python)',
      '',
      'retract works_at(mira, _)',
      '```',
    ].join('\n');
    expect(normalizeExtractionOutput(raw)).toEqual([
      'works_at(mira, acme).',
      'lives_in(mira, sydney).',
      'prefers(mira, python).',
      'retract works_at(mira, _).',
    ]);
  });

  it('drops prose label lines that contain no clause, keeps prose that looks like a clause attempt', () => {
    expect(
      normalizeExtractionOutput('Here are the facts:\nworks_at(a, b)\nOutput:'),
    ).toEqual(['works_at(a, b).']);
    expect(normalizeExtractionOutput('works_at(a b')).toEqual(['works_at(a b']);
  });

  it('drops an invented assert keyword (small models mirror retract)', () => {
    expect(
      normalizeExtractionOutput(
        'assert works_at(mira, acme).\nretract works_at(mira, _).',
      ),
    ).toEqual(['works_at(mira, acme).', 'retract works_at(mira, _).']);
  });

  it('splits several facts written on one line without periods', () => {
    expect(
      normalizeExtractionOutput(
        'works_at(mira, acme) lives_in(mira, sydney) prefers(mira, python).',
      ),
    ).toEqual([
      'works_at(mira, acme).',
      'lives_in(mira, sydney).',
      'prefers(mira, python).',
    ]);
    // a rule stays one line
    expect(
      normalizeExtractionOutput(
        'eligible(X) :- employee(X), \\+ suspended(X).',
      ),
    ).toEqual(['eligible(X) :- employee(X), \\+ suspended(X).']);
  });
});

describe('rewriteSelfAtoms', () => {
  it('maps first-person synonyms to the configured self atom in facts and patterns', () => {
    const clauses = parseProgram(
      'lives_in(the_user, melbourne). dentist(me, dr_chen). works_at(you, acme). friend(user, zoe).',
    );
    const out = rewriteSelfAtoms(clauses, 'rahul').map(serializeClause);
    expect(out).toEqual([
      'lives_in(rahul, melbourne).',
      'dentist(rahul, dr_chen).',
      'works_at(rahul, acme).',
      'friend(rahul, zoe).',
    ]);
    expect(SELF_ATOM_SYNONYMS).toContain('i');
  });

  it('does not rewrite the synonyms when they are the configured atom already or appear quoted as names', () => {
    const clauses = parseProgram("nickname(zoe, 'Me'). lives_in(user, osaka).");
    expect(rewriteSelfAtoms(clauses, 'user').map(serializeClause)).toEqual([
      "nickname(zoe, 'Me').",
      'lives_in(user, osaka).',
    ]);
  });
});

describe('assertGroundedConstants: every new constant must appear in the input', () => {
  it('accepts constants present in the input, in the schema, or the self atom', () => {
    const input = 'Tom turns 40 in 2027 and works at Blue Harbour Analytics.';
    const clauses = parseProgram(
      "turns_age_in(tom, 40, 2027). works_at(tom, 'Blue Harbour Analytics'). lives_in(user, osaka).",
    );
    expect(() =>
      assertGroundedConstants(clauses, input, {
        known: new Set(['osaka']),
        selfAtom: 'user',
      }),
    ).not.toThrow();
  });

  it('rejects an inferred or fabricated constant with an actionable message', () => {
    const input = 'Tom turns 40 in 2027.';
    const clauses = parseProgram('birth_year(tom, 1987).');
    expect(() =>
      assertGroundedConstants(clauses, input, {
        known: new Set(),
        selfAtom: 'user',
      }),
    ).toThrow(/1987.*not in the input/s);
  });

  it('matches multi-word and hyphenated constants loosely', () => {
    const input =
      'Dana works on the check-out service and speaks Mandarin Chinese.';
    const clauses = parseProgram(
      "works_on(dana, checkout_service). speaks(dana, 'Mandarin Chinese').",
    );
    expect(() =>
      assertGroundedConstants(clauses, input, {
        known: new Set(),
        selfAtom: 'user',
      }),
    ).not.toThrow();
  });
});

describe('predicate aliases and closed vocabulary', () => {
  it('reads rembero_predicate_alias declarations and rewrites aliased predicates', () => {
    const store = parseProgram(
      'rembero_predicate_alias(employed_by, works_at). works_at(zoe, acme).',
    );
    const aliases = predicateAliasesFrom(store);
    expect(aliases.get('employed_by')).toBe('works_at');
    const out = applyPredicateAliases(
      parseProgram('employed_by(mira, initech).'),
      aliases,
    );
    expect(out.map(serializeClause)).toEqual(['works_at(mira, initech).']);
  });

  it('closed mode rejects a predicate outside the schema, naming the known ones', () => {
    const known = new Set(['works_at/2', 'lives_in/2']);
    expect(() =>
      assertKnownVocabulary(parseProgram('employer(mira, initech).'), known),
    ).toThrow(/employer\/2.*works_at\/2/s);
    expect(() =>
      assertKnownVocabulary(parseProgram('works_at(mira, initech).'), known),
    ).not.toThrow();
  });
});

describe('rememberText integration', () => {
  it('normalizes bullets, rewrites the self atom, and grounds constants against the input', async () => {
    const store = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-guard-')),
    );
    const llm = new ScriptedLlm([
      '- lives_in(me, melbourne)\n- works_at(me, acme)',
    ]);
    const result = await rememberText(
      { store, llm, selfAtom: 'rahul' },
      'I live in Melbourne and work at Acme.',
    );
    expect(result.added.sort()).toEqual([
      'lives_in(rahul, melbourne).',
      'works_at(rahul, acme).',
    ]);
  });

  it('retries with a grounding error when the model invents a constant', async () => {
    const store = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-guard-')),
    );
    const llm = new ScriptedLlm([
      'birth_year(tom, 1987).',
      'turns_age_in(tom, 40, 2027).',
    ]);
    const result = await rememberText({ store, llm }, 'Tom turns 40 in 2027.');
    expect(result.added).toEqual(['turns_age_in(tom, 40, 2027).']);
    expect(llm.prompts[1].at(-1)?.content).toMatch(/1987/);
  });

  it('closed vocabulary rewrites aliases and rejects unknown predicates', async () => {
    const store = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-guard-')),
    );
    store.assert(
      'default',
      'rembero_predicate_alias(employed_by, works_at). works_at(zoe, acme).',
      { opId: 'seed' },
    );
    const llm = new ScriptedLlm(['employed_by(mira, initech).']);
    const result = await rememberText(
      { store, llm, extractionVocabulary: 'closed' },
      'Mira is employed by Initech.',
    );
    expect(result.added).toEqual(['works_at(mira, initech).']);
    const rejecting = new ScriptedLlm([
      'job(tom, globex).',
      'works_at(tom, globex).',
    ]);
    const second = await rememberText(
      { store, llm: rejecting, extractionVocabulary: 'closed' },
      'Tom works at Globex.',
    );
    expect(second.added).toEqual(['works_at(tom, globex).']);
    expect(rejecting.prompts[1].at(-1)?.content).toMatch(
      /job\/2.*works_at\/2/s,
    );
  });
});
