import { describe, expect, it } from 'vitest';
import { factsCachePath, groundFactsFrom } from '../src/evals/run-document-facts.js';

describe('groundFactsFrom', () => {
  it('keeps ground facts and drops rules and junk', () => {
    const reply = [
      "budget_dollars(kestrel_programme, 4200000).",
      'eligible(X) :- member(X).',
      'this is prose, not datalog',
      "appointed_sro(northgate_transition, 'Dr Amara Osei').",
    ].join('\n');
    expect(groundFactsFrom(reply)).toEqual([
      'budget_dollars(kestrel_programme, 4200000).',
      "appointed_sro(northgate_transition, 'Dr Amara Osei').",
    ]);
  });

  it('returns nothing for the nothing sentinel', () => {
    expect(groundFactsFrom('% nothing')).toEqual([]);
  });
});

describe('factsCachePath', () => {
  it('keeps the two prompts in separate caches', () => {
    expect(factsCachePath('doc_000001', 'deepseek-chat', 'document')).toBe(
      '.cache/document-recall/doc_000001.facts.deepseek-chat.document.json',
    );
  });

  it('makes a model id safe for a file name', () => {
    expect(factsCachePath('doc_000001', 'openai/gpt-5.6-sol')).toBe(
      '.cache/document-recall/doc_000001.facts.openai_gpt-5.6-sol.json',
    );
  });
});
