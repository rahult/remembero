import { describe, it, expect, beforeEach, vi } from 'vitest';
import { existsSync, mkdtempSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { MemoryStore } from '../src/store/store.js';
import { parseProgram, serializeClause } from '../src/engine/index.js';
import { buildSchemaSummary } from '../src/llm/prompts.js';
import {
  deterministicRecallAnswer,
  rememberText,
  recallQuestion,
  retrieveQuestion,
} from '../src/llm/pipeline.js';
import {
  OpenRouterClient,
  addLlmUsage,
  emptyLlmUsageTotals,
  type ChatMessage,
  type LlmClient,
} from '../src/llm/client.js';
import { wrapTentativeFacts } from '../src/knowledge/trust.js';
import { assertTentativeFacts } from '../src/knowledge/trust-store.js';
import { searchKnowledge } from '../src/knowledge/search.js';

/** LlmClient returning scripted responses, recording every request. */
class ScriptedLlm implements LlmClient {
  calls: ChatMessage[][] = [];
  constructor(private responses: string[]) {}
  async complete(messages: ChatMessage[]): Promise<string> {
    this.calls.push(messages);
    const next = this.responses.shift();
    if (next === undefined) throw new Error('ScriptedLlm ran out of responses');
    return next;
  }
}

let store: MemoryStore;

beforeEach(() => {
  store = new MemoryStore(mkdtempSync(join(tmpdir(), 'rembero-llm-')));
});

describe('buildSchemaSummary', () => {
  it('lists predicates with arity, sample facts, and rules verbatim', () => {
    store.assert(
      'default',
      `works_at(rahul, acme). works_at(maya, acme). works_at(chen, initech). works_at(dee, initech).
       birth_year(rahul, 1985).
       colleague(X, Y) :- works_at(X, C), works_at(Y, C), X != Y.
       company_size(Company, Count) :- count(*) as Count where works_at(Person, Company).`,
    );
    const summary = buildSchemaSummary(store.clausesFor(['default']));
    expect(summary).toContain('works_at/2');
    expect(summary).toContain('works_at(rahul, acme).');
    expect(summary).toContain('birth_year/2');
    expect(summary).toContain(
      'colleague(X, Y) :- works_at(X, C), works_at(Y, C), X != Y.',
    );
    expect(summary).toContain(
      'company_size(Company, Count) :- count(*) as Count where works_at(Person, Company).',
    );
    // samples are capped at 3 per predicate
    expect(summary).not.toContain('works_at(dee, initech).');
  });

  it('says so when there are no memories yet', () => {
    expect(buildSchemaSummary([])).toContain('no memories yet');
  });

  it('keeps integrity policy out of the LLM-facing recall schema', () => {
    const summary = buildSchemaSummary(
      parseProgram('active(mira). :- active(X), suspended(X).'),
    );
    expect(summary).toContain('active/1');
    expect(summary).not.toContain('integrity');
    expect(summary).not.toContain('suspended');
    expect(summary).not.toContain(':-');
  });

  it('keeps entity identity metadata out of the LLM-facing schema', () => {
    const summary = buildSchemaSummary(
      parseProgram(
        "rembero_alias('Mira Patel', mira). rembero_entity_position(works_at, 2, 0). works_at(mira, acme).",
      ),
    );
    expect(summary).toContain('works_at/2');
    expect(summary).not.toContain('rembero_alias');
    expect(summary).not.toContain('rembero_entity_position');
  });

  it('keeps tentative metadata hidden unless recall explicitly includes its fact', () => {
    const summary = buildSchemaSummary(
      wrapTentativeFacts('status(mira, active).'),
    );
    expect(summary).toContain('no memories yet');
    expect(summary).not.toContain('rembero_tentative');
  });
});

describe('rememberText', () => {
  it('extracts, validates, and stores clauses', async () => {
    const llm = new ScriptedLlm([
      'works_at(rahul, acme).\nlives_in(rahul, sydney).',
    ]);
    const result = await rememberText(
      { store, llm },
      'Rahul works at Acme and lives in Sydney',
    );
    expect(result.added).toEqual([
      'works_at(rahul, acme).',
      'lives_in(rahul, sydney).',
    ]);
    expect(result.duplicates).toBe(0);
    expect(store.load('default')).toHaveLength(2);
    // the extraction prompt includes the schema and the raw text
    const [messages] = llm.calls;
    expect(messages[0].role).toBe('system');
    expect(messages[0].content).toContain('Datalog');
    expect(messages[1].content).toContain('Rahul works at Acme');
  });

  it('strips markdown code fences from the response', async () => {
    const llm = new ScriptedLlm(['```prolog\nworks_at(rahul, acme).\n```']);
    const result = await rememberText({ store, llm }, 'Rahul works at Acme');
    expect(result.added).toEqual(['works_at(rahul, acme).']);
  });

  it('retries once with the parse error, then succeeds', async () => {
    const llm = new ScriptedLlm([
      'works_at(X, acme).', // unsafe: non-ground fact
      'works_at(rahul, acme).',
    ]);
    const result = await rememberText({ store, llm }, 'Rahul works at Acme');
    expect(result.added).toEqual(['works_at(rahul, acme).']);
    expect(llm.calls).toHaveLength(2);
    const retryMessages = llm.calls[1];
    const lastUser = retryMessages[retryMessages.length - 1];
    expect(lastUser.content).toMatch(/ground|variable/i);
  });

  it('throws after a second failure, surfacing the error', async () => {
    const llm = new ScriptedLlm(['nonsense((', 'still nonsense((']);
    await expect(rememberText({ store, llm }, 'gibberish')).rejects.toThrow(
      /pars|expected/i,
    );
    expect(store.load('default')).toEqual([]);
  });

  it('never lets natural-language extraction create integrity policy', async () => {
    const llm = new ScriptedLlm([
      ':- active(X), suspended(X).',
      ':- active(X), terminated(X).',
    ]);
    await expect(
      rememberText({ store, llm }, 'Make sure active users are not suspended'),
    ).rejects.toThrow(/may not create integrity constraints/i);
    expect(store.load('default')).toEqual([]);
  });

  it('never lets natural-language extraction create entity identity metadata', async () => {
    const llm = new ScriptedLlm([
      "rembero_alias('Mira Patel', mira).",
      'rembero_entity_position(works_at, 2, 0).',
    ]);
    await expect(
      rememberText({ store, llm }, 'Mira Patel and Mira are the same person'),
    ).rejects.toThrow(/may not create entity identity metadata/i);
    expect(store.load('default')).toEqual([]);
  });

  it('instructs extraction not to replace identity authority with an inert ordinary fact', async () => {
    const llm = new ScriptedLlm(['% nothing']);
    await rememberText(
      { store, llm },
      'Mira Patel and Mira are the same person',
    );

    expect(llm.calls[0][0].content).toContain(
      'Do not replace it with same_person, alias, equivalent_to, or another ordinary fact.',
    );
    expect(store.load('default')).toEqual([]);
  });

  it('assigns tentative trust only from explicit caller authority', async () => {
    const tentativeLlm = new ScriptedLlm(['project(atlas).']);
    const result = await rememberText(
      { store, llm: tentativeLlm },
      'Atlas may be the active project',
      'default',
      { trust: 'tentative' },
    );
    expect(result).toMatchObject({
      added: ['project(atlas).'],
      duplicates: 0,
      retracted: 0,
      trust: 'tentative',
    });
    expect(store.load('default').map(serializeClause)).toEqual([
      "rembero_tentative('project(atlas).').",
    ]);
    expect(tentativeLlm.calls[0][0].content).toContain(
      'The caller explicitly authorized tentative storage.',
    );

    const defaultLlm = new ScriptedLlm([]);
    expect(
      await retrieveQuestion(
        { store, llm: defaultLlm },
        'Is Atlas the active project?',
      ),
    ).toMatchObject({ status: 'unanswerable', bindings: [] });
    expect(defaultLlm.calls).toHaveLength(0);

    const includedLlm = new ScriptedLlm([
      '?- project(atlas).',
      'Tentatively, Atlas is the active project.',
    ]);
    const included = await recallQuestion(
      { store, llm: includedLlm },
      'Is Atlas the active project?',
      ['default'],
      { trustMode: 'include_tentative', explain: true },
    );
    expect(included).toMatchObject({
      status: 'answered',
      trustMode: 'include_tentative',
      rowTrust: ['tentative'],
      bindings: [{}],
      answer: 'Tentatively, Atlas is the active project.',
      explanation: { rows: [{ proofs: [{ trust: 'tentative' }] }] },
    });
    expect(includedLlm.calls[1].at(-1)?.content).toContain(
      'Trust by result row: ["tentative"]',
    );
  });

  it('does not promote hedged claims when tentative authority is absent', async () => {
    const llm = new ScriptedLlm(['% nothing']);
    const result = await rememberText(
      { store, llm },
      'Atlas may be the active project',
    );

    expect(result).toEqual({ added: [], duplicates: 0, retracted: 0 });
    expect(llm.calls[0][0].content).toContain(
      'The caller did not authorize tentative storage.',
    );
    expect(store.load('default')).toEqual([]);
  });

  it('phrases an accepted duplicate as accepted in an opt-in trust view', async () => {
    store.assertTentative('default', 'status(mira, active).');
    store.assert('default', 'status(mira, active).');
    const llm = new ScriptedLlm([
      '?- status(mira, active).',
      'Mira is active.',
    ]);
    const result = await recallQuestion(
      { store, llm },
      'Is Mira active?',
      ['default'],
      { trustMode: 'include_tentative' },
    );
    expect(result).toMatchObject({
      status: 'answered',
      rowTrust: ['accepted'],
      answer: 'Mira is active.',
    });
    expect(llm.calls[1].at(-1)?.content).toContain(
      'Trust by result row: ["accepted"]',
    );
  });

  it('never lets model output assign its own trust metadata', async () => {
    const llm = new ScriptedLlm([
      "rembero_tentative('project(atlas).').",
      "rembero_tentative('project(beacon).').",
    ]);
    await expect(
      rememberText({ store, llm }, 'Maybe Atlas is active'),
    ).rejects.toThrow(/may not assign trust metadata/i);
    expect(store.load('default')).toEqual([]);
  });

  it('never lets natural-language extraction retract entity identity metadata', async () => {
    store.assert('default', "rembero_alias('Mira Patel', mira).");
    const llm = new ScriptedLlm([
      "retract rembero_alias('Mira Patel', _).",
      "retract rembero_alias('Mira Patel', mira).",
    ]);
    await expect(
      rememberText({ store, llm }, 'Forget that Mira Patel is Mira'),
    ).rejects.toThrow(/may not retract entity identity metadata/i);
    expect(store.load('default').map(serializeClause)).toEqual([
      "rembero_alias('Mira Patel', mira).",
    ]);
  });

  it('applies retract lines before asserting, superseding stale facts', async () => {
    store.assert('default', 'works_at(mira, acme).');
    const llm = new ScriptedLlm([
      'retract works_at(mira, _).\nworks_at(mira, initech).',
    ]);
    const result = await rememberText(
      { store, llm },
      'Mira now works at Initech',
    );
    expect(result.retracted).toBe(1);
    expect(result.added).toEqual(['works_at(mira, initech).']);
    expect(store.load('default').map(serializeClause)).toEqual([
      'works_at(mira, initech).',
    ]);
  });

  it('counts retractions that match nothing as zero without failing', async () => {
    const llm = new ScriptedLlm([
      'retract dentist(rahul, _).\ndentist(rahul, dr_chen).',
    ]);
    // the speaker is named by the self atom; "rahul" is not in the input text
    const result = await rememberText(
      { store, llm, selfAtom: 'rahul' },
      'My dentist is Dr Chen now',
    );
    expect(result.retracted).toBe(0);
    expect(result.added).toEqual(['dentist(rahul, dr_chen).']);
  });

  it('keeps deletion semantics by default when no valid-time mode is requested', async () => {
    store.assert('default', 'works_at(mira, acme).');
    const llm = new ScriptedLlm([
      'retract works_at(mira, _).\nworks_at(mira, initech).',
    ]);

    const result = await rememberText(
      { store, llm },
      'Mira now works at Initech',
    );

    expect(result.retracted).toBe(1);
    const clauses = store.load('default').map(serializeClause);
    expect(clauses).toEqual(['works_at(mira, initech).']);
    expect(clauses.some((clause) => clause.startsWith('works_at_until('))).toBe(
      false,
    );
  });

  it('uses archive_until supersession with a full ISO timestamp when valid-time mode is enabled', async () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-valid-time-'));
    const temporal = new MemoryStore(root);
    temporal.assert('default', 'works_at(mira, acme).', {
      opId: 'source-1',
      sourceText: 'Mira works at Acme.',
      at: new Date('2026-08-10T09:00:00.000Z'),
    });
    const llm = new ScriptedLlm([
      'retract works_at(mira, _).\nworks_at(mira, initech).',
    ]);

    const result = await (
      rememberText as unknown as (
        deps: { store: MemoryStore; llm: LlmClient },
        text: string,
        namespace?: string,
        options?: { validTimeMode?: 'delete' | 'archive_until'; at?: Date },
      ) => Promise<{
        added: string[];
        duplicates: number;
        retracted: number;
        archived: string[];
        opId: string;
      }>
    )({ store: temporal, llm }, 'Mira now works at Initech', 'default', {
      validTimeMode: 'archive_until',
      at: new Date('2026-08-16T16:59:00.000Z'),
    });

    expect(result).toMatchObject({
      retracted: 1,
      archived: ["works_at_until(mira, acme, '2026-08-16T16:59:00.000Z')."],
    });
    expect(temporal.load('default').map(serializeClause).sort()).toEqual(
      [
        'works_at(mira, initech).',
        "works_at_until(mira, acme, '2026-08-16T16:59:00.000Z').",
      ].sort(),
    );
    const journal = readFileSync(join(root, 'journal.log'), 'utf8');
    expect(journal).toContain('"op":"supersede"');
    expect(journal).toContain(
      "works_at_until(mira, acme, '2026-08-16T16:59:00.000Z').",
    );
  });

  it('honors the server-level valid-time mode when no per-call override is supplied', async () => {
    store.assert('default', 'works_at(mira, acme).');
    const llm = new ScriptedLlm([
      'retract works_at(mira, _).\nworks_at(mira, initech).',
    ]);

    const result = await rememberText(
      { store, llm, validTimeMode: 'archive_until' },
      'Mira now works at Initech',
    );

    expect(result.archived).toHaveLength(1);
    expect(store.load('default').map(serializeClause)).toEqual(
      expect.arrayContaining([
        'works_at(mira, initech).',
        expect.stringMatching(/^works_at_until\(mira, acme, '/),
      ]),
    );
  });

  it('journals the source text of what was remembered', async () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-journal-'));
    const s = new MemoryStore(root);
    const llm = new ScriptedLlm(['works_at(rahul, acme).']);
    await rememberText({ store: s, llm }, 'Rahul works at Acme');
    const journal = readFileSync(join(root, 'journal.log'), 'utf8');
    const entries = journal
      .trim()
      .split('\n')
      .map((line) => JSON.parse(line));
    const remember = entries.find((entry) => entry.op === 'remember');
    const assertion = entries.find((entry) => entry.op === 'assert');
    expect(remember.text).toBe('Rahul works at Acme');
    expect(assertion.sourceText).toBe('Rahul works at Acme');
    expect(assertion.opId).toBe(remember.opId);
  });

  it('refuses to send secrets to the external LLM or persist them', async () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-secret-journal-'));
    const s = new MemoryStore(root);
    const secret = 'ghp_supersecretvalue';
    const llm = new ScriptedLlm(['uses(rahul, github).']);

    await expect(
      rememberText({ store: s, llm }, `My GitHub token is ${secret}`),
    ).rejects.toThrow(/refusing to send sensitive memory text/i);
    expect(llm.calls).toHaveLength(0);
    expect(existsSync(join(root, 'journal.log'))).toBe(false);
  });

  it('keeps namespaces outside the explicit LLM allowlist local-only', async () => {
    const llm = new ScriptedLlm([]);
    const deps = { store, llm, llmAllowedNamespaces: new Set(['shared']) };

    await expect(
      rememberText(deps, 'Alice works at Acme', 'private'),
    ).rejects.toThrow(/namespace 'private' is local-only/i);
    expect(llm.calls).toHaveLength(0);
  });

  it('treats "% nothing" as a no-op', async () => {
    const llm = new ScriptedLlm(['% nothing']);
    const result = await rememberText({ store, llm }, 'hello there!');
    expect(result.added).toEqual([]);
    expect(store.load('default')).toEqual([]);
  });

  it('enforces a configured knowledge suite on direct natural-language memory', async () => {
    const llm = new ScriptedLlm(['forbidden(a).']);
    await expect(
      rememberText(
        {
          store,
          llm,
          knowledgeCheckEnforcement: {
            mode: 'strict',
            namespaces: ['default'],
            suite: {
              version: 1,
              checks: [
                {
                  name: 'forbidden stays absent',
                  query: 'forbidden(a)',
                  expect: { kind: 'empty' },
                },
              ],
            },
          },
        },
        'Remember forbidden A.',
      ),
    ).rejects.toMatchObject({ code: 'knowledge_check_enforcement' });
    expect(store.load('default')).toEqual([]);
  });
});

describe('recallQuestion', () => {
  beforeEach(() => {
    store.assert(
      'default',
      `works_at(rahul, acme). works_at(maya, acme).
       colleague(X, Y) :- works_at(X, C), works_at(Y, C), X != Y.`,
    );
  });

  it('answers immediately without any LLM call when memory is empty', async () => {
    const empty = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-empty-')),
    );
    const llm = new ScriptedLlm([]);
    const result = await recallQuestion(
      { store: empty, llm },
      'Where does Rahul work?',
    );
    expect(result.query).toBeNull();
    expect(result.answer).toMatch(/no (relevant )?memor/i);
    expect(llm.calls).toHaveLength(0);
  });

  it('refuses to expose sensitive stored facts to the external LLM', async () => {
    const sensitive = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-sensitive-')),
    );
    sensitive.assert('default', "password(rahul, 'do-not-send-this').");
    const llm = new ScriptedLlm(['?- password(rahul, Value).']);

    await expect(
      recallQuestion({ store: sensitive, llm }, 'What credential is stored?'),
    ).rejects.toThrow(/refusing to send sensitive memory schema/i);
    expect(llm.calls).toHaveLength(0);
  });

  it('rejects wildcard recall when it includes a local-only namespace', async () => {
    const scoped = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-scoped-')),
    );
    scoped.assert('shared', 'project(atlas).');
    scoped.assert('private', 'health_note(alice, stable).');
    const llm = new ScriptedLlm([]);

    await expect(
      retrieveQuestion(
        { store: scoped, llm, llmAllowedNamespaces: new Set(['shared']) },
        'What projects are stored?',
        '*',
      ),
    ).rejects.toThrow(/namespace 'private' is local-only/i);
    expect(llm.calls).toHaveLength(0);
  });

  it('generates a query, evaluates it, and phrases the answer', async () => {
    const llm = new ScriptedLlm([
      '?- colleague(rahul, Who).',
      'Maya is Rahul’s colleague.',
    ]);
    const result = await recallQuestion(
      { store, llm },
      'Who are Rahul’s colleagues?',
    );
    expect(result.query).toBe('colleague(rahul, Who)');
    expect(result.bindings).toEqual([{ Who: 'maya' }]);
    expect(result.answer).toBe('Maya is Rahul’s colleague.');
    // phrasing prompt received the bindings
    const phrasing = llm.calls[1];
    expect(phrasing[phrasing.length - 1].content).toContain('maya');
  });

  it('renders successful bindings locally in deterministic mode without phrasing', async () => {
    const llm = new ScriptedLlm(['?- works_at(Person, acme).']);
    const result = await recallQuestion(
      { store, llm },
      'Who works at Acme?',
      ['default'],
      { answerMode: 'deterministic' },
    );

    expect(result).toMatchObject({
      status: 'answered',
      answerMode: 'deterministic',
      query: 'works_at(Person, acme)',
      bindings: [{ Person: 'rahul' }, { Person: 'maya' }],
      answer:
        'Results for works_at(Person, acme):\n1. Person = rahul\n2. Person = maya',
    });
    expect(llm.calls).toHaveLength(1);
  });

  it('renders compact deterministic rule and source evidence without phrasing', async () => {
    const sourced = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-evidence-')),
    );
    sourced.assert(
      'default',
      `works_at(rahul, acme). works_at(maya, acme).
       colleague(X, Y) :- works_at(X, C), works_at(Y, C), X != Y.`,
      {
        opId: 'employment-source',
        sourceText: 'Rahul and Maya work at Acme.',
        at: new Date('2026-08-17T09:00:00.000Z'),
      },
    );
    const llm = new ScriptedLlm(['?- colleague(rahul, Who).']);
    const result = await recallQuestion(
      { store: sourced, llm },
      "Who are Rahul's colleagues?",
      ['default'],
      { answerMode: 'evidence' },
    );

    expect(result.answerMode).toBe('evidence');
    expect(result.answer).toBe(
      `Evidence for colleague(rahul, Who):
1. Who = maya
   Claims: colleague(rahul, maya); works_at(maya, acme); works_at(rahul, acme)
   Rules: #1 colleague(X, Y) :- works_at(X, C), works_at(Y, C), X != Y.
   Sources: default/employment-source@2026-08-17T09:00:00.000Z "Rahul and Maya work at Acme."`,
    );
    expect(result.explanation?.rows).toHaveLength(1);
    expect(llm.calls).toHaveLength(1);
  });

  it('renders absence, aggregate, and tentative evidence locally', async () => {
    const sourced = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-evidence-kinds-')),
    );
    sourced.assert(
      'default',
      `assigned(alice, admin).
       available(Person, Role) :- assigned(Person, Role), \\+ suspended(Person).
       member(red, alice). member(red, bob).
       team_size(Team, Count) :- count(*) as Count where member(Team, Person).`,
      { opId: 'evidence-source', at: new Date('2026-08-17T09:05:00.000Z') },
    );
    const absence = await recallQuestion(
      { store: sourced, llm: new ScriptedLlm(['?- available(alice, Role).']) },
      'What role is Alice available for?',
      ['default'],
      { answerMode: 'evidence' },
    );
    expect(absence.answer).toContain('Absent: suspended(alice)');

    const aggregate = await recallQuestion(
      { store: sourced, llm: new ScriptedLlm(['?- team_size(red, Count).']) },
      'How many members are on the red team?',
      ['default'],
      { answerMode: 'evidence' },
    );
    expect(aggregate.answer).toContain('Aggregates: count(*) = 2');

    const tentativeStore = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-evidence-tentative-')),
    );
    assertTentativeFacts(tentativeStore, 'default', 'status(mira, paused).', {
      opId: 'tentative-evidence',
      at: new Date('2026-08-17T09:10:00.000Z'),
    });
    const tentative = await recallQuestion(
      {
        store: tentativeStore,
        llm: new ScriptedLlm(['?- status(mira, State).']),
      },
      'What might Mira status be?',
      ['default'],
      { answerMode: 'evidence', trustMode: 'include_tentative' },
    );
    expect(tentative.answer).toContain('1. [tentative] State = paused');
    expect(tentative.answer).toContain('[tentative]');
  });

  it('renders boolean and tentative rows without losing trust labels', async () => {
    expect(deterministicRecallAnswer('project(atlas)', [{}])).toBe(
      'The query project(atlas) is supported.',
    );
    expect(
      deterministicRecallAnswer(
        'status(mira, State)',
        [{ State: 'paused' }],
        ['tentative'],
      ),
    ).toBe('Tentative result for status(mira, State): State = paused.');
    expect(
      deterministicRecallAnswer('count(*) as Count where employee(Person)', [
        { Count: '2' },
      ]),
    ).toBe('Result for count(*) as Count where employee(Person): Count = 2.');
    expect(
      deterministicRecallAnswer(
        'status(Person, State)',
        [
          { Person: 'mira', State: 'active' },
          { Person: 'zoe', State: 'paused' },
        ],
        ['accepted', 'tentative'],
      ),
    ).toBe(
      'Results for status(Person, State):\n1. Person = mira, State = active\n2. [tentative] Person = zoe, State = paused',
    );
    expect(() =>
      deterministicRecallAnswer(
        'status(Person, State)',
        [{ State: 'active' }],
        [],
      ),
    ).toThrow(/rowTrust must match binding row count/i);

    const tentativeStore = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-deterministic-trust-')),
    );
    tentativeStore.importClauses(
      'default',
      wrapTentativeFacts('status(mira, paused).'),
      { opId: 'tentative' },
    );
    const llm = new ScriptedLlm(['?- status(mira, State).']);
    const result = await recallQuestion(
      { store: tentativeStore, llm },
      'What may Mira status be?',
      ['default'],
      { trustMode: 'include_tentative', answerMode: 'deterministic' },
    );
    expect(result).toMatchObject({
      answerMode: 'deterministic',
      rowTrust: ['tentative'],
      answer: 'Tentative result for status(mira, State): State = paused.',
    });
    expect(llm.calls).toHaveLength(1);
  });

  it('fails closed on an unknown programmatic recall answer mode', async () => {
    const llm = new ScriptedLlm([]);
    await expect(
      recallQuestion(
        { store, llm, recallAnswerMode: 'creative' as never },
        'Who works at Acme?',
      ),
    ).rejects.toThrow(/natural.*deterministic.*evidence/i);
    expect(llm.calls).toHaveLength(0);
  });

  it('projects an alias only at declared positions during opt-in recall', async () => {
    const identityStore = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-recall-identity-')),
    );
    identityStore.assert(
      'default',
      `rembero_alias('Mira Patel', mira).
       rembero_entity_position(works_at, 2, 0).
       works_at('Mira Patel', acme).`,
    );
    const llm = new ScriptedLlm([
      "?- works_at('Mira Patel', Company).",
      'Mira Patel works at Acme.',
    ]);

    const result = await recallQuestion(
      { store: identityStore, llm },
      'Where does Mira Patel work?',
      ['default'],
      { entityIdentity: 'canonical' },
    );

    expect(result).toMatchObject({
      status: 'answered',
      query: 'works_at(mira, Company)',
      bindings: [{ Company: 'acme' }],
      answer: 'Mira Patel works at Acme.',
    });
    expect(llm.calls[0][0].content).toContain('works_at(mira, acme).');
    expect(llm.calls[0][0].content).not.toContain('rembero_alias');
  });

  it('can evaluate retrieval without a phrasing call and applies the grounded prompt', async () => {
    const llm = new ScriptedLlm(['?- works_at(rahul, Company).']);
    const result = await retrieveQuestion(
      { store, llm },
      'Where does Rahul work?',
      ['default'],
      { queryPromptVariant: 'grounded' },
    );
    expect(result).toEqual({
      status: 'answered',
      query: 'works_at(rahul, Company)',
      bindings: [{ Company: 'acme' }],
    });
    expect(llm.calls).toHaveLength(1);
    expect(llm.calls[0][0].content).toContain(
      'schema examples as syntax evidence only',
    );
  });

  it('corrects a semantically wrong non-empty predicate before accepting its rows', async () => {
    const confusable = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-recall-disambiguation-')),
    );
    confusable.assert(
      'default',
      'uses_language(atlas, rust). project_owner(atlas, rahul).',
    );
    const llm = new ScriptedLlm([
      '?- uses_language(atlas, Value).',
      '?- project_owner(atlas, Owner).',
    ]);

    const result = await retrieveQuestion(
      { store: confusable, llm },
      'Who owns Atlas?',
    );

    expect(result).toEqual({
      status: 'answered',
      query: 'project_owner(atlas, Owner)',
      bindings: [{ Owner: 'rahul' }],
      queryReviews: [
        {
          originalQuery: 'uses_language(atlas, Value)',
          reviewedQuery: 'project_owner(atlas, Owner)',
          reasons: ['competing_predicate'],
          competingPredicates: ['project_owner/2'],
          outcome: 'corrected',
        },
      ],
    });
    expect(llm.calls).toHaveLength(2);
    expect(llm.calls[1].at(-1)?.content).toContain('Returned-row sample');
    expect(llm.calls[1].at(-1)?.content).toContain('rust');
    expect(llm.calls[1].at(-1)?.content).toContain('project_owner/2');
  });

  it('corrects the inverse owner-versus-language non-empty confusion', async () => {
    const confusable = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-recall-language-disambiguation-')),
    );
    confusable.assert(
      'default',
      'uses_language(atlas, rust). project_owner(atlas, rahul).',
    );
    const llm = new ScriptedLlm([
      '?- project_owner(atlas, Value).',
      '?- uses_language(atlas, Language).',
    ]);

    const result = await retrieveQuestion(
      { store: confusable, llm },
      'What language does Atlas use?',
    );

    expect(result.query).toBe('uses_language(atlas, Language)');
    expect(result.bindings).toEqual([{ Language: 'rust' }]);
    expect(result.queryReviews?.[0]).toMatchObject({
      originalQuery: 'project_owner(atlas, Value)',
      reviewedQuery: 'uses_language(atlas, Language)',
      competingPredicates: ['uses_language/2'],
      outcome: 'corrected',
    });
  });

  it('keeps the one-call path for a semantically grounded non-empty query', async () => {
    const confusable = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-recall-safe-answer-')),
    );
    confusable.assert(
      'default',
      'uses_language(atlas, rust). project_owner(atlas, rahul).',
    );
    const llm = new ScriptedLlm(['?- project_owner(atlas, Owner).']);

    const result = await retrieveQuestion(
      { store: confusable, llm },
      'Who owns the Atlas project?',
    );

    expect(result).toEqual({
      status: 'answered',
      query: 'project_owner(atlas, Owner)',
      bindings: [{ Owner: 'rahul' }],
    });
    expect(llm.calls).toHaveLength(1);
  });

  it('records when the bounded review confirms an ambiguous query unchanged', async () => {
    const confusable = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-recall-review-repeat-')),
    );
    confusable.assert(
      'default',
      'lives_in(mira, sydney). works_at(mira, acme).',
    );
    const llm = new ScriptedLlm([
      '?- works_at(mira, Company).',
      '?- works_at(mira, Company).',
    ]);

    const result = await retrieveQuestion(
      { store: confusable, llm },
      'Who employs Mira?',
    );

    expect(result.bindings).toEqual([{ Company: 'acme' }]);
    expect(result.queryReviews).toEqual([
      {
        originalQuery: 'works_at(mira, Company)',
        reviewedQuery: 'works_at(mira, Company)',
        reasons: ['competing_predicate'],
        competingPredicates: ['lives_in/2'],
        outcome: 'repeated',
      },
    ]);
    expect(llm.calls).toHaveLength(2);
  });

  it('bounds non-empty review rows and competing predicate names', async () => {
    const bounded = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-recall-review-bounds-')),
    );
    bounded.assert(
      'default',
      `alpha_relation(atlas, one).
       beta_relation(atlas, two).
       delta_relation(atlas, three).
       echo_relation(atlas, four).
       foxtrot_relation(atlas, five).
       zzz_relation(atlas, first).
       zzz_relation(atlas, second).
       zzz_relation(atlas, third).
       zzz_relation(atlas, fourth).
       zzz_relation(atlas, fifth).`,
    );
    const llm = new ScriptedLlm([
      '?- zzz_relation(atlas, Value).',
      '?- zzz_relation(atlas, Value).',
    ]);

    const result = await retrieveQuestion(
      { store: bounded, llm },
      'Tell me about Atlas',
    );

    expect(result.queryReviews?.[0]?.competingPredicates).toHaveLength(4);
    const reviewPrompt = llm.calls[1].at(-1)?.content ?? '';
    expect(reviewPrompt).toContain('first');
    expect(reviewPrompt).toContain('second');
    expect(reviewPrompt).toContain('third');
    expect(reviewPrompt).not.toContain('fourth');
    expect(reviewPrompt).not.toContain('fifth');
  });

  it('uses the grounded query prompt by default', async () => {
    const llm = new ScriptedLlm(['?- works_at(rahul, Company).']);
    await retrieveQuestion({ store, llm }, 'Where does Rahul work?');
    expect(llm.calls[0][0].content).toContain(
      'Datalog variables represent requested unknown',
    );
    expect(llm.calls[0][0].content).toContain(
      'Never inline its body: helper variables used inside the rule are not requested answer columns.',
    );
    expect(llm.calls[0][0].content).toContain(
      'Every relational query containing variables MUST use "select Answer where goals"',
    );
  });

  it('keeps projected helper variables out of recall bindings and deterministic answers', async () => {
    store.assert(
      'default',
      'parent(alice, bob). parent(bob, carol). parent(carol, dan).',
    );
    const llm = new ScriptedLlm([
      '?- select Grandchild where parent(alice, Parent), parent(Parent, Grandchild).',
    ]);

    const result = await recallQuestion(
      { store, llm },
      "Who is Alice's grandchild?",
      ['default'],
      { answerMode: 'deterministic', explain: true },
    );

    expect(result).toMatchObject({
      status: 'answered',
      query:
        'select Grandchild where parent(alice, Parent), parent(Parent, Grandchild)',
      bindings: [{ Grandchild: 'carol' }],
      answer:
        'Result for select Grandchild where parent(alice, Parent), parent(Parent, Grandchild): Grandchild = carol.',
      explanation: { rows: [{ bindings: { Grandchild: 'carol' } }] },
    });
    expect(result.bindings[0]).not.toHaveProperty('Parent');
  });

  it('rejects unprojected multi-variable grounded output and retries once', async () => {
    store.assert('default', 'edge(a, b). edge(b, c).');
    const llm = new ScriptedLlm([
      '?- edge(a, Mid), edge(Mid, End).',
      '?- select End where edge(a, Mid), edge(Mid, End).',
    ]);

    const result = await retrieveQuestion(
      { store, llm },
      'What can A reach in two edge steps?',
    );

    expect(result).toMatchObject({
      query: 'select End where edge(a, Mid), edge(Mid, End)',
      bindings: [{ End: 'c' }],
    });
    expect(llm.calls).toHaveLength(2);
    expect(llm.calls[1].at(-1)?.content).toContain(
      'must use select to declare answer columns',
    );
  });

  it('recalls through a deterministic relevant schema slice with 100+ predicates', async () => {
    const scaled = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-recall-scale-')),
    );
    const noise = Array.from(
      { length: 120 },
      (_, index) =>
        `noise_${String(index).padStart(3, '0')}(subject_${index}, value_${index}).`,
    ).join('\n');
    scaled.assert(
      'default',
      `${noise}\nworks_at(alice, northwind).\nworks_at(mira, acme).`,
    );
    const llm = new ScriptedLlm(['?- works_at(mira, Company).']);

    const result = await retrieveQuestion(
      { store: scaled, llm, recallSchemaPredicateLimit: 8 },
      'Who is Mira employed by?',
    );

    expect(result).toMatchObject({
      status: 'answered',
      query: 'works_at(mira, Company)',
      bindings: [{ Company: 'acme' }],
      pruning: {
        totalPredicates: 121,
        selectedPredicates: expect.arrayContaining(['works_at/2']),
        catalogComplete: true,
      },
    });
    const prompt = llm.calls[0][0].content;
    expect(prompt).toContain('% selected predicates (8 of 121');
    expect(prompt).toContain('% e.g. works_at(mira, acme).');
    expect(Buffer.byteLength(prompt, 'utf8')).toBeLessThan(64 * 1024);
  });

  it('ranks recall schema by local source vocabulary without sending source text', async () => {
    const sourced = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-recall-source-rank-')),
    );
    sourced.assert('default', 'atlas_owner(atlas, rahul).', {
      opId: 'owner-source',
    });
    sourced.assert('default', 'fact_z(atlas, rust).', {
      opId: 'technology-source',
      sourceText: 'What technology stack does Atlas use?',
    });
    const llm = new ScriptedLlm([
      '?- fact_z(atlas, Technology).',
      '?- fact_z(atlas, Technology).',
    ]);

    const result = await retrieveQuestion(
      { store: sourced, llm, recallSchemaPredicateLimit: 1 },
      'What technology stack does Atlas use?',
    );

    expect(result).toMatchObject({
      status: 'answered',
      bindings: [{ Technology: 'rust' }],
      pruning: {
        selectedPredicates: ['fact_z/2'],
        sourceMatchedPredicates: ['fact_z/2'],
      },
    });
    const prompt = llm.calls[0][0].content;
    expect(prompt).toContain('% e.g. fact_z(atlas, rust).');
    expect(prompt).not.toContain('What technology stack does Atlas use?');
  });

  it('composes provenance ranking with canonical identity projection', async () => {
    const sourced = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-recall-source-identity-')),
    );
    sourced.assert('default', 'mira_owner(mira, rahul).', {
      opId: 'owner-source',
    });
    sourced.assert(
      'default',
      `rembero_alias('Mira Patel', mira).
       rembero_entity_position(fact_z, 2, 0).
       fact_z('Mira Patel', rust).`,
      {
        opId: 'technology-source',
        sourceText: 'What technology stack does Mira Patel use?',
      },
    );
    const llm = new ScriptedLlm([
      '?- fact_z(mira, Technology).',
      '?- fact_z(mira, Technology).',
    ]);

    const result = await retrieveQuestion(
      {
        store: sourced,
        llm,
        recallSchemaPredicateLimit: 1,
        entityIdentity: 'canonical',
      },
      'What technology stack does Mira Patel use?',
    );

    expect(result).toMatchObject({
      status: 'answered',
      bindings: [{ Technology: 'rust' }],
      pruning: {
        selectedPredicates: ['fact_z/2'],
        sourceMatchedPredicates: ['fact_z/2'],
      },
    });
    const prompt = llm.calls[0][0].content;
    expect(prompt).toContain('% e.g. fact_z(mira, rust).');
    expect(prompt).not.toContain('Mira Patel');
    expect(prompt).not.toContain('technology stack');
  });

  it('keeps derived-rule dependencies in a pruned recall-explain path', async () => {
    const scaled = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-recall-derived-')),
    );
    const noise = Array.from(
      { length: 100 },
      (_, index) =>
        `noise_${String(index).padStart(3, '0')}(subject_${index}).`,
    ).join('\n');
    scaled.assert(
      'default',
      `${noise}
       works_at(rahul, acme). works_at(mira, acme).
       colleague(X, Y) :- works_at(X, C), works_at(Y, C), X != Y.`,
    );
    const llm = new ScriptedLlm(['?- colleague(rahul, Who).']);

    const result = await retrieveQuestion(
      { store: scaled, llm, recallSchemaPredicateLimit: 6 },
      'Who are Rahul’s colleagues?',
      ['default'],
      { explain: true },
    );

    expect(result.status).toBe('answered');
    expect(result.bindings).toEqual([{ Who: 'mira' }]);
    expect(result.pruning?.selectedPredicates).toEqual(
      expect.arrayContaining(['colleague/2', 'works_at/2']),
    );
    expect(llm.calls[0][0].content).toContain(
      'colleague(X, Y) :- works_at(X, C), works_at(Y, C), X != Y.',
    );
    expect(result.explanation?.rows[0].proofs[0]).toMatchObject({
      predicate: 'colleague',
      because: [
        expect.objectContaining({ predicate: 'works_at' }),
        expect.objectContaining({ predicate: 'works_at' }),
      ],
    });
  });

  it('does not let schema ranking change the requested-namespace proof witness', async () => {
    const scaled = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-recall-witness-')),
    );
    scaled.assert('first', 'pet(rahul, luna).', {
      opId: 'first-source',
      sourceText: 'First namespace source.',
    });
    scaled.assert('second', 'pet(rahul, luna).', {
      opId: 'second-source',
      sourceText: 'Second namespace source.',
    });
    scaled.assert(
      'noise',
      Array.from(
        { length: 40 },
        (_, index) =>
          `noise_${String(index).padStart(3, '0')}(value_${index}).`,
      ).join('\n'),
    );
    const llm = new ScriptedLlm(['?- pet(rahul, Name).']);

    const result = await retrieveQuestion(
      { store: scaled, llm, recallSchemaPredicateLimit: 2 },
      'What is Rahul’s pet?',
      ['second', 'first', 'noise'],
      { explain: true },
    );

    expect(result.status).toBe('answered');
    expect(result.explanation?.rows[0].proofs[0]).toMatchObject({
      predicate: 'pet',
      sources: [
        expect.objectContaining({ namespace: 'second', opId: 'second-source' }),
      ],
    });
  });

  it('uses a complete name/arity catalog when the relevant predicate is outside details', async () => {
    const scaled = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-recall-catalog-')),
    );
    scaled.assert(
      'default',
      `${Array.from({ length: 40 }, (_, index) => `alpha_${index}(value_${index}).`).join('\n')}
       zeta_relation(target, answer).`,
    );
    const llm = new ScriptedLlm(['?- zeta_relation(target, Value).']);

    const result = await retrieveQuestion(
      { store: scaled, llm, recallSchemaPredicateLimit: 2 },
      'Find the requested information',
    );

    expect(result.status).toBe('answered');
    expect(result.bindings).toEqual([{ Value: 'answer' }]);
    expect(result.pruning).toMatchObject({
      selectedPredicates: expect.not.arrayContaining(['zeta_relation/2']),
      catalogComplete: true,
    });
    expect(llm.calls[0][0].content).toContain('zeta_relation/2');
  });

  it('widens deterministically before accepting unanswerable from a partial schema', async () => {
    const scaled = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-recall-widen-')),
    );
    scaled.assert(
      'default',
      `${Array.from({ length: 40 }, (_, index) => `alpha_${index}(value_${index}).`).join('\n')}
       zeta_relation(target, answer).`,
    );
    const llm = new ScriptedLlm([
      '?- unanswerable.',
      '?- zeta_relation(target, Value).',
    ]);

    const result = await retrieveQuestion(
      { store: scaled, llm, recallSchemaPredicateLimit: 2 },
      'Find the requested information',
    );

    expect(result).toMatchObject({
      status: 'answered',
      query: 'zeta_relation(target, Value)',
      bindings: [{ Value: 'answer' }],
      pruning: {
        schemaComplete: true,
        initialSelectedPredicates: ['alpha_0/1', 'alpha_1/1'],
        attempts: [
          expect.objectContaining({
            detailedPredicates: 2,
            outcome: 'unanswerable',
          }),
          expect.objectContaining({
            detailedPredicates: 41,
            outcome: 'answered',
          }),
        ],
      },
    });
    expect(llm.calls).toHaveLength(2);
  });

  it('fails closed instead of claiming unanswerable when the predicate catalog is bounded', async () => {
    const scaled = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-recall-budget-')),
    );
    scaled.assert(
      'default',
      Array.from(
        { length: 180 },
        (_, index) =>
          `very_long_predicate_name_${String(index).padStart(3, '0')}(entity_${index}, value_${index}).`,
      ).join('\n'),
    );
    const llm = new ScriptedLlm(['?- unanswerable.']);

    const result = await recallQuestion(
      {
        store: scaled,
        llm,
        recallSchemaPredicateLimit: 2,
        recallSchemaByteLimit: 512,
      },
      'What relationship is stored?',
      ['default'],
      { relatedKnowledge: { limit: 1 } },
    );

    expect(result).toMatchObject({
      status: 'schema_budget_exhausted',
      query: null,
      bindings: [],
      pruning: { catalogComplete: false },
      relatedKnowledge: { status: 'no_match', limit: 1 },
    });
    expect(result.answer).toMatch(/schema budget/i);
    expect(llm.calls).toHaveLength(1);
  });

  it('reports budget exhaustion when selected rule text cannot fit the byte cap', async () => {
    const scaled = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-recall-rule-budget-')),
    );
    const facts = Array.from(
      { length: 20 },
      (_, index) => `base_${String(index).padStart(3, '0')}(item).`,
    );
    const body = Array.from(
      { length: 20 },
      (_, index) => `base_${String(index).padStart(3, '0')}(X)`,
    ).join(', ');
    scaled.assert('default', `${facts.join('\n')}\nimportant(X) :- ${body}.`);
    const llm = new ScriptedLlm(['?- unanswerable.']);

    const result = await retrieveQuestion(
      { store: scaled, llm, recallSchemaByteLimit: 512 },
      'What is important?',
    );

    expect(result).toMatchObject({
      status: 'schema_budget_exhausted',
      query: null,
      pruning: {
        schemaComplete: false,
        omittedRules: 1,
        attempts: [expect.objectContaining({ outcome: 'unanswerable' })],
      },
    });
  });

  it('widens before recall when the relevant dependency closure exceeds the first-pass cap', async () => {
    const scaled = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-recall-closure-')),
    );
    scaled.assert(
      'default',
      `base_a(x). base_b(x). base_c(x).
       important(X) :- base_a(X), base_b(X), base_c(X).`,
    );
    const llm = new ScriptedLlm(['?- important(Value).']);

    const result = await retrieveQuestion(
      { store: scaled, llm, recallSchemaPredicateLimit: 2 },
      'What is important?',
    );

    expect(result).toMatchObject({
      status: 'answered',
      query: 'important(Value)',
      bindings: [{ Value: 'x' }],
    });
    expect(llm.calls).toHaveLength(1);
    expect(llm.calls[0][0].content).toContain(
      'important(X) :- base_a(X), base_b(X), base_c(X).',
    );
  });

  it('returns bounded exhaustion instead of throwing on oversized predicate names', async () => {
    const scaled = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-recall-name-budget-')),
    );
    const clauses = Array.from(
      { length: 4 },
      (_, index) => `${'p'.repeat(8_000)}_${index}(value).`,
    ).join('\n');
    scaled.assert('default', clauses);
    const llm = new ScriptedLlm([]);

    const result = await recallQuestion(
      { store: scaled, llm },
      'What is stored?',
    );

    expect(result).toEqual({
      status: 'schema_budget_exhausted',
      answer:
        'Recall reached its schema budget before it could rule out relevant memories.',
      query: null,
      bindings: [],
    });
    expect(llm.calls).toHaveLength(0);
  });

  it('generates exact count aggregation and treats zero as a real result', async () => {
    const llm = new ScriptedLlm([
      '?- count(*) as Count where works_at(Person, nowhere).',
    ]);
    const result = await retrieveQuestion(
      { store, llm },
      'How many people work at Nowhere?',
      ['default'],
      { explain: true },
    );

    expect(result).toMatchObject({
      query: 'count(*) as Count where works_at(Person, nowhere)',
      bindings: [{ Count: '0' }],
    });
    expect(result.explanation?.graph.nodes).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          kind: 'aggregate',
          op: 'count',
          value: 0,
          contributorCount: 0,
        }),
      ]),
    );
    expect(llm.calls).toHaveLength(1);
    expect(llm.calls[0][0].content).toContain('count(*) as Count where');
  });

  it('queries a reusable aggregate predicate without reducing it a second time', async () => {
    const aggregateStore = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-recall-aggregate-rule-')),
    );
    aggregateStore.assert(
      'default',
      `member(red, alice). member(red, bob).
       team_size(Team, Count) :- count(*) as Count where member(Team, Person).`,
    );
    const llm = new ScriptedLlm(['?- team_size(red, Count).']);

    const result = await retrieveQuestion(
      { store: aggregateStore, llm },
      'How many members are on the red team?',
      ['default'],
      { explain: true },
    );

    expect(result).toMatchObject({
      status: 'answered',
      query: 'team_size(red, Count)',
      bindings: [{ Count: '2' }],
      explanation: {
        rows: [
          {
            proofs: [
              {
                predicate: 'team_size',
                aggregate: { aggregated: true, op: 'count', value: 2 },
              },
            ],
          },
        ],
      },
    });
    expect(llm.calls).toHaveLength(1);
    expect(llm.calls[0][0].content).toContain(
      'query its head predicate directly',
    );
  });

  it('distinguishes a named or distributive aggregate value from counting its groups', async () => {
    const aggregateStore = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-recall-aggregate-groups-')),
    );
    aggregateStore.assert(
      'default',
      `member(red, alice). member(red, bob). member(blue, carol).
       team_size(Team, Count) :- count(*) as Count where member(Team, Person).`,
    );

    const groupCountLlm = new ScriptedLlm([
      '?- select Team, Size where team_size(Team, Size).',
      '?- count(*) as Count where team_size(Team, Size).',
    ]);
    const groupCount = await retrieveQuestion(
      { store: aggregateStore, llm: groupCountLlm },
      'How many teams are there?',
    );
    expect(groupCount).toMatchObject({
      query: 'count(*) as Count where team_size(Team, Size)',
      bindings: [{ Count: '2' }],
    });
    expect(groupCountLlm.calls).toHaveLength(2);

    const eachLlm = new ScriptedLlm([
      '?- select Team, Count where team_size(Team, Count).',
    ]);
    const each = await retrieveQuestion(
      { store: aggregateStore, llm: eachLlm },
      'How many members are on each team?',
    );
    expect(each.bindings).toEqual([
      { Team: 'red', Count: '2' },
      { Team: 'blue', Count: '1' },
    ]);
    expect(eachLlm.calls).toHaveLength(1);
  });

  it('allows an auxiliary goal to bind an aggregate rule group key', async () => {
    const aggregateStore = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-recall-aggregate-bound-group-')),
    );
    aggregateStore.assert(
      'default',
      `member(red, alice). member(red, bob). member(blue, carol).
       team_size(Team, Count) :- count(*) as Count where member(Team, Person).`,
    );
    const llm = new ScriptedLlm([
      '?- select Count where member(Team, alice), team_size(Team, Count).',
    ]);

    const result = await retrieveQuestion(
      { store: aggregateStore, llm },
      "How many members are on Alice's team?",
    );

    expect(result).toMatchObject({
      status: 'answered',
      query: 'select Count where member(Team, alice), team_size(Team, Count)',
      bindings: [{ Count: '2' }],
    });
    expect(llm.calls).toHaveLength(1);
  });

  it('generates and evaluates an explicit arithmetic threshold query', async () => {
    const numeric = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-arithmetic-')),
    );
    numeric.assert(
      'default',
      'age(alice, 30). age(bob, 20). age(carol, 38). age(dana, 27).',
    );
    const llm = new ScriptedLlm([
      '?- select Person where age(Person, Years), age(dana, DanaYears), Years > DanaYears + 5.',
    ]);

    const result = await retrieveQuestion(
      { store: numeric, llm },
      'Who is more than 5 years older than Dana?',
    );

    expect(result).toEqual({
      status: 'answered',
      query:
        'select Person where age(Person, Years), age(dana, DanaYears), Years > DanaYears + 5',
      bindings: [{ Person: 'carol' }],
    });
    expect(llm.calls[0][0].content).toContain('Years > DanaYears + 5');
  });

  it('recall-explain carries temporal source metadata for an archived fact', async () => {
    const temporal = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-recall-temporal-')),
    );
    temporal.assert('default', 'works_at(mira, acme).', {
      opId: 'source-1',
      sourceText: 'Mira works at Acme.',
      at: new Date('2026-08-10T09:00:00.000Z'),
    });
    await (
      rememberText as unknown as (
        deps: { store: MemoryStore; llm: LlmClient },
        text: string,
        namespace?: string,
        options?: { validTimeMode?: 'delete' | 'archive_until'; at?: Date },
      ) => Promise<unknown>
    )(
      {
        store: temporal,
        llm: new ScriptedLlm([
          'retract works_at(mira, _).\nworks_at(mira, initech).',
        ]),
      },
      'Mira now works at Initech',
      'default',
      {
        validTimeMode: 'archive_until',
        at: new Date('2026-08-16T16:59:00.000Z'),
      },
    );
    const llm = new ScriptedLlm([
      '?- select Company where works_at(mira, initech), works_at_until(mira, Company, Until).',
      'Mira worked at Acme until 16 August 2026.',
    ]);

    const result = await recallQuestion(
      { store: temporal, llm },
      'Where did Mira work before Initech?',
      ['default'],
      { explain: true },
    );

    expect(result.answer).toBe('Mira worked at Acme until 16 August 2026.');
    expect(result.explanation?.rows[0].proofs[1]).toMatchObject({
      predicate: 'works_at_until',
      sources: [
        expect.objectContaining({
          opId: expect.any(String),
          ts: expect.any(String),
          temporal: {
            kind: 'superseded',
            previousClause: 'works_at(mira, acme).',
            validUntil: '2026-08-16T16:59:00.000Z',
          },
        }),
      ],
    });
    expect(llm.calls[0][0].content).toContain(
      'works_at(mira, initech), works_at_until(mira, Company, Until)',
    );
  });

  it('reviews a non-empty historical query that omits the named later state', async () => {
    const temporal = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-recall-temporal-review-')),
    );
    temporal.assert(
      'default',
      "works_at(mira, initech). works_at_until(mira, acme, '2026-08-16T16:59:00.000Z').",
    );
    const llm = new ScriptedLlm([
      '?- select Company where works_at_until(mira, Company, Until).',
      '?- select Company where works_at(mira, initech), works_at_until(mira, Company, Until).',
    ]);

    const result = await retrieveQuestion(
      { store: temporal, llm },
      'Where did Mira work before Initech?',
    );

    expect(result.query).toBe(
      'select Company where works_at(mira, initech), works_at_until(mira, Company, Until)',
    );
    expect(result.bindings).toEqual([{ Company: 'acme' }]);
    expect(result.queryReviews).toEqual([
      {
        originalQuery:
          'select Company where works_at_until(mira, Company, Until)',
        reviewedQuery:
          'select Company where works_at(mira, initech), works_at_until(mira, Company, Until)',
        reasons: ['missing_temporal_context'],
        competingPredicates: ['works_at/2'],
        outcome: 'corrected',
      },
    ]);
  });

  it('does not let a review unanswerable decision bypass full-schema widening', async () => {
    const scaled = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-recall-review-widen-')),
    );
    const noise = Array.from(
      { length: 40 },
      (_, index) =>
        `noise_${String(index).padStart(3, '0')}(subject_${index}, value_${index}).`,
    ).join('\n');
    scaled.assert(
      'default',
      `${noise}\nuses_language(atlas, rust). project_owner(atlas, rahul).`,
    );
    const llm = new ScriptedLlm([
      '?- project_owner(atlas, Value).',
      '?- unanswerable.',
      '?- uses_language(atlas, Language).',
    ]);

    const result = await retrieveQuestion(
      { store: scaled, llm, recallSchemaPredicateLimit: 2 },
      'What language does Atlas use?',
    );

    expect(result).toMatchObject({
      status: 'answered',
      query: 'uses_language(atlas, Language)',
      bindings: [{ Language: 'rust' }],
      queryReviews: [
        {
          originalQuery: 'project_owner(atlas, Value)',
          reviewedQuery: null,
          outcome: 'unanswerable',
        },
      ],
      pruning: {
        schemaComplete: true,
        attempts: [
          expect.objectContaining({ outcome: 'unanswerable' }),
          expect.objectContaining({ outcome: 'answered' }),
        ],
      },
    });
    expect(llm.calls).toHaveLength(3);
  });

  it('fails closed before exporting a sensitive non-empty row sample for review', async () => {
    const sensitive = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-recall-review-sensitive-')),
    );
    sensitive.assert(
      'default',
      `uses_language(atlas, sk_secretvalue1).
       uses_language(atlas, alpha).
       uses_language(atlas, beta).
       uses_language(atlas, gamma).
       project_owner(atlas, rahul).`,
    );
    const llm = new ScriptedLlm(['?- uses_language(atlas, Value).']);

    await expect(
      retrieveQuestion({ store: sensitive, llm }, 'Who owns Atlas?'),
    ).rejects.toThrow(/sensitive query review evidence/i);
    expect(llm.calls).toHaveLength(1);
  });

  it('reviews the canonical executed query rather than the raw alias query', async () => {
    const identity = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-recall-review-identity-')),
    );
    identity.assert(
      'default',
      `rembero_alias('Atlas Project', atlas).
       rembero_entity_position(uses_language, 2, 0).
       rembero_entity_position(project_owner, 2, 0).
       uses_language('Atlas Project', rust).
       project_owner('Atlas Project', rahul).`,
    );
    const llm = new ScriptedLlm([
      "?- uses_language('Atlas Project', Value).",
      "?- project_owner('Atlas Project', Owner).",
    ]);

    const result = await retrieveQuestion(
      { store: identity, llm },
      'Who owns the Atlas Project?',
      ['default'],
      { entityIdentity: 'canonical' },
    );

    expect(result.query).toBe('project_owner(atlas, Owner)');
    expect(result.bindings).toEqual([{ Owner: 'rahul' }]);
    expect(result.queryReviews?.[0]).toMatchObject({
      originalQuery: 'uses_language(atlas, Value)',
      reviewedQuery: 'project_owner(atlas, Owner)',
      outcome: 'corrected',
    });
  });

  it('uses the selected recorded snapshot for disambiguation and evaluation', async () => {
    const recorded = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-recall-review-recorded-')),
    );
    recorded.assert(
      'default',
      'uses_language(atlas, rust). project_owner(atlas, rahul).',
      { opId: 'baseline' },
    );
    recorded.supersede(
      'default',
      ['project_owner(atlas, _)'],
      'project_owner(atlas, mira).',
      { opId: 'later' },
    );
    const llm = new ScriptedLlm([
      '?- uses_language(atlas, Value).',
      '?- project_owner(atlas, Owner).',
    ]);

    const result = await retrieveQuestion(
      { store: recorded, llm },
      'Who owns Atlas?',
      ['default'],
      { recordedSequence: 1 },
    );

    expect(result).toMatchObject({
      status: 'answered',
      query: 'project_owner(atlas, Owner)',
      bindings: [{ Owner: 'rahul' }],
      recordedSnapshot: { sequence: 1, journalEntries: 2 },
      queryReviews: [{ outcome: 'corrected' }],
    });
  });

  it('retries an aggregate the question did not explicitly request', async () => {
    const llm = new ScriptedLlm([
      '?- count(*) as Count where works_at(Person, acme).',
      '?- works_at(Person, acme).',
    ]);
    const result = await retrieveQuestion({ store, llm }, 'Who works at Acme?');

    expect(result.bindings).toEqual([{ Person: 'rahul' }, { Person: 'maya' }]);
    expect(llm.calls).toHaveLength(2);
    expect(llm.calls[1].at(-1)?.content).toContain(
      'requires the question to explicitly request',
    );
  });

  it('retries a relational query when the question explicitly requests a count', async () => {
    const llm = new ScriptedLlm([
      '?- works_at(Person, acme).',
      '?- count(*) as Count where works_at(Person, acme).',
    ]);
    const result = await retrieveQuestion(
      { store, llm },
      'How many people work at Acme?',
    );

    expect(result.bindings).toEqual([{ Count: '2' }]);
    expect(llm.calls).toHaveLength(2);
    expect(llm.calls[1].at(-1)?.content).toContain(
      'question explicitly requests count aggregation',
    );
  });

  it('can generate and explain a safe closed-world negation query', async () => {
    const employment = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-negation-')),
    );
    employment.assert(
      'default',
      'employee(alice). employee(bob). suspended(bob).',
      {
        opId: 'employment-source',
      },
    );
    const llm = new ScriptedLlm(['?- employee(X), \\+ suspended(X).']);

    const result = await retrieveQuestion(
      { store: employment, llm },
      'Which employees are not suspended?',
      ['default'],
      { explain: true },
    );

    expect(result.query).toBe('employee(X), \\+ suspended(X)');
    expect(result.bindings).toEqual([{ X: 'alice' }]);
    expect(result.explanation?.rows[0].proofs).toEqual([
      expect.objectContaining({ predicate: 'employee' }),
      {
        negated: true,
        predicate: 'suspended',
        pattern: ['alice'],
        stratum: 0,
      },
    ]);
    expect(llm.calls[0][0].content).toContain(
      'Closed-world negation is written \\+',
    );
  });

  it('allows a negated relation with no stored facts under closed-world recall', async () => {
    const employment = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-absent-relation-')),
    );
    employment.assert('default', 'employee(alice). employee(bob).');
    const llm = new ScriptedLlm(['?- employee(X), \\+ suspended(X).']);

    const result = await retrieveQuestion(
      { store: employment, llm },
      'Which employees are not suspended?',
    );

    expect(result.bindings).toEqual([{ X: 'alice' }, { X: 'bob' }]);
  });

  it('retries a misspelled negated predicate instead of proving the typo by absence', async () => {
    const employment = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-negation-typo-')),
    );
    employment.assert(
      'default',
      'employee(alice). employee(bob). suspended(bob).',
    );
    const llm = new ScriptedLlm([
      '?- employee(X), \\+ suspendd(X).',
      '?- employee(X), \\+ suspended(X).',
    ]);

    const result = await retrieveQuestion(
      { store: employment, llm },
      'Which employees are not suspended?',
    );

    expect(result.bindings).toEqual([{ X: 'alice' }]);
    expect(llm.calls).toHaveLength(2);
    expect(llm.calls[1].at(-1)?.content).toContain('resembles suspended/1');
  });

  it('rejects an absent negated relation that the question did not name', async () => {
    const employment = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-negation-ungrounded-')),
    );
    employment.assert('default', 'employee(alice).');
    const llm = new ScriptedLlm([
      '?- employee(X), \\+ blacklisted(X).',
      '?- employee(X), \\+ blacklisted(X).',
    ]);

    await expect(
      retrieveQuestion(
        { store: employment, llm },
        'Which employees are not suspended?',
      ),
    ).rejects.toThrow('must be explicitly named by the question');
  });

  it('short-circuits on unanswerable without calling the engine or phrasing', async () => {
    const llm = new ScriptedLlm(['?- unanswerable.']);
    const result = await recallQuestion(
      { store, llm },
      'What is the meaning of life?',
    );
    expect(result.query).toBeNull();
    expect(result.bindings).toEqual([]);
    expect(result.answer).toMatch(/no (relevant )?memor/i);
    expect(llm.calls).toHaveLength(1);
  });

  it('adds deterministic related knowledge to an unanswerable recall without another model call', async () => {
    store.assert('default', 'dentist(rahul, chen).', {
      opId: 'dentist-source',
      sourceText: 'Rahul dentist is Doctor Chen.',
    });
    const llm = new ScriptedLlm(['?- unanswerable.']);

    const result = await recallQuestion(
      { store, llm },
      'Who is Rahul dentist?',
      ['default'],
      { relatedKnowledge: { limit: 2, kinds: ['fact'] } },
    );

    expect(result).toMatchObject({
      status: 'unanswerable',
      query: null,
      bindings: [],
      relatedKnowledge: {
        status: 'matches',
        limit: 2,
      },
    });
    expect(result.relatedKnowledge?.results[0]).toMatchObject({
      rank: 1,
      clause: 'dentist(rahul, chen).',
      sources: [{ opId: 'dentist-source' }],
    });
    expect(result.answer).toMatch(/no relevant memories/i);
    expect(llm.calls).toHaveLength(1);
  });

  it('uses the exact recorded identity and trust view for related recall discovery', async () => {
    const temporal = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-related-view-')),
    );
    temporal.assert(
      'default',
      `rembero_alias('Mira Patel', mira).
       rembero_entity_position(works_at, 2, 0).
       works_at('Mira Patel', acme).`,
      { opId: 'identity-source' },
    );
    assertTentativeFacts(temporal, 'default', 'status(mira, paused).', {
      opId: 'tentative-source',
    });
    temporal.assert('default', 'status(mira, active).', {
      opId: 'later-source',
    });
    const snapshot = temporal.recordedSnapshot(['default'], 2);
    const expected = searchKnowledge(
      snapshot.clauses,
      'Mira paused work',
      snapshot.sources,
      {
        limit: 5,
        kinds: ['fact'],
        entityIdentity: 'canonical',
        trustMode: 'include_tentative',
      },
    );
    const llm = new ScriptedLlm(['?- unanswerable.']);

    const result = await recallQuestion(
      { store: temporal, llm },
      'Mira paused work',
      ['default'],
      {
        recordedSequence: 2,
        entityIdentity: 'canonical',
        trustMode: 'include_tentative',
        relatedKnowledge: { limit: 5, kinds: ['fact'] },
      },
    );

    expect(result.relatedKnowledge).toEqual(expected);
    expect(result.recordedSnapshot).toMatchObject({ sequence: 2 });
    expect(result.relatedKnowledge?.results).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          clause: 'works_at(mira, acme).',
          sources: [
            expect.objectContaining({
              projectedFrom: "works_at('Mira Patel', acme).",
            }),
          ],
        }),
        expect.objectContaining({
          clause: 'status(mira, paused).',
          trust: 'tentative',
        }),
      ]),
    );
    expect(
      result.relatedKnowledge?.results.some(
        ({ clause }) => clause === 'status(mira, active).',
      ),
    ).toBe(false);
    expect(llm.calls).toHaveLength(1);
  });

  it('does not attach related discovery to an answered recall', async () => {
    const llm = new ScriptedLlm([
      '?- works_at(rahul, Company).',
      'Rahul works at Acme.',
    ]);
    const result = await recallQuestion(
      { store, llm },
      'Where does Rahul work?',
      ['default'],
      { relatedKnowledge: true },
    );

    expect(result.status).toBe('answered');
    expect(result.relatedKnowledge).toBeUndefined();
    expect(result.answer).toBe('Rahul works at Acme.');
    expect(llm.calls).toHaveLength(2);
  });

  it('retries when the generated query uses an unknown predicate', async () => {
    const llm = new ScriptedLlm([
      '?- employed_by(rahul, X).',
      '?- works_at(rahul, X).',
      'Rahul works at Acme.',
    ]);
    const result = await recallQuestion(
      { store, llm },
      'Where does Rahul work?',
    );
    expect(result.query).toBe('works_at(rahul, X)');
    const retry = llm.calls[1];
    expect(retry[retry.length - 1].content).toContain('employed_by/2');
  });

  it('tries one alternative query when the first returns no rows', async () => {
    const llm = new ScriptedLlm([
      '?- works_at(maya, initech).', // plausible but wrong guess: no rows
      '?- works_at(maya, X).', // fallback attempt finds the answer
      'Maya works at Acme.',
    ]);
    const result = await recallQuestion(
      { store, llm },
      'Is Maya employed anywhere?',
    );
    expect(result.query).toBe('works_at(maya, X)');
    expect(result.bindings).toEqual([{ X: 'acme' }]);
    expect(result.answer).toBe('Maya works at Acme.');
    // the fallback prompt tells the model what came up empty
    const fallback = llm.calls[1];
    expect(fallback[fallback.length - 1].content).toContain(
      'works_at(maya, initech)',
    );
  });

  it('tells the model how each goal of an empty join fared on its own', async () => {
    const llm = new ScriptedLlm([
      '?- works_at(maya, X), works_at(X, acme).', // wrong join: X is a company, not a person
      '?- works_at(maya, X).',
      'Maya works at Acme.',
    ]);
    const result = await recallQuestion(
      { store, llm },
      'Where does Maya work?',
    );
    expect(result.bindings).toEqual([{ X: 'acme' }]);
    const fallback = llm.calls[1];
    const prompt = fallback[fallback.length - 1].content;
    expect(prompt).toContain('works_at(maya, X) alone matches 1 row');
    expect(prompt).toContain('works_at(X, acme) alone matches 2 rows');
    expect(prompt).toContain('together they match none');
  });

  it('phrases an honest answer when the fallback also returns no rows', async () => {
    const llm = new ScriptedLlm([
      '?- works_at(zoe, X).',
      '?- colleague(zoe, X).',
      "I don't have that in memory.",
    ]);
    const result = await recallQuestion({ store, llm }, 'Where does Zoe work?');
    expect(result.bindings).toEqual([]);
    expect(result.answer).toBe(
      'No stored result matches colleague(zoe, X). Required fact works_at(zoe, C) is missing.',
    );
    expect(result.whyNot).toMatchObject({
      status: 'blocked',
      summary: result.answer,
    });
    expect(llm.calls).toHaveLength(2);
  });

  it('preserves a valid empty query when the fallback repeats it unchanged', async () => {
    const llm = new ScriptedLlm([
      '?- works_at(zoe, X).',
      '?- works_at(zoe, X).',
      "I don't have that in memory.",
    ]);
    const result = await recallQuestion({ store, llm }, 'Where does Zoe work?');
    expect(result.bindings).toEqual([]);
    expect(result.query).toBe('works_at(zoe, X)');
    expect(result.answer).toBe(
      'No stored result matches works_at(zoe, X). Required fact works_at(zoe, X) is missing.',
    );
    expect(result.whyNot?.summary).toBe(result.answer);
    expect(llm.calls).toHaveLength(2);
    expect(llm.calls[1].at(-1)?.content).toContain(
      'empty result is valid evidence',
    );
  });

  it('keeps negative recall honest when complete why-not diagnostics exceed their bound', async () => {
    const crowded = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-negative-limit-')),
    );
    crowded.assert(
      'default',
      `${Array.from({ length: 40 }, (_, index) => `item(value_${index}).`).join('\n')}
       missing(X) :- absent(X).`,
    );
    const llm = new ScriptedLlm([
      '?- item(X), missing(X).',
      '?- item(X), missing(X).',
    ]);

    const result = await recallQuestion(
      { store: crowded, llm },
      'Which items are missing?',
    );

    expect(result).toMatchObject({
      status: 'no_match',
      answer: 'No stored result matches item(X), missing(X).',
      whyNotUnavailable: {
        reason: 'diagnostic_limit',
        message: expect.stringMatching(/frontier exceeded 32 bindings/i),
      },
    });
    expect(result.whyNot).toBeUndefined();
    expect(llm.calls).toHaveLength(2);
  });

  it('adds deterministic rule blockers to an empty recall-explain result', async () => {
    const llm = new ScriptedLlm([
      '?- colleague(rahul, zoe).',
      '?- colleague(rahul, zoe).',
    ]);
    const result = await retrieveQuestion(
      { store, llm },
      'Is Zoe a colleague of Rahul?',
      ['default'],
      { explain: true },
    );

    expect(result).toMatchObject({
      status: 'no_match',
      explanation: { rows: [] },
      whyNot: {
        status: 'blocked',
        failures: [
          {
            reason: 'rules_blocked',
            rules: [
              {
                failures: [
                  {
                    reason: 'missing_fact',
                    goal: 'works_at(zoe, acme)',
                  },
                ],
              },
            ],
          },
        ],
      },
    });
    expect(llm.calls).toHaveLength(2);
  });

  it('accepts structurally unanswerable as the fallback response and skips phrasing', async () => {
    const llm = new ScriptedLlm(['?- works_at(zoe, X).', '?- unanswerable.']);
    const result = await recallQuestion(
      { store, llm },
      'Why does Zoe work there?',
    );
    expect(result.bindings).toEqual([]);
    expect(result.query).toBeNull();
    expect(result.answer).toMatch(/no (relevant )?memor/i);
    expect(llm.calls).toHaveLength(2);
  });
});

describe('OpenRouterClient', () => {
  it('returns provider token, cache, reasoning, and charged-cost usage without a second request', async () => {
    const fakeFetch: typeof fetch = async () =>
      new Response(
        JSON.stringify({
          model: 'openai/gpt-5.6-luna',
          choices: [{ message: { content: 'tracked' } }],
          usage: {
            prompt_tokens: 120,
            completion_tokens: 8,
            total_tokens: 128,
            cost: 0.0000168,
            prompt_tokens_details: { cached_tokens: 20 },
            completion_tokens_details: { reasoning_tokens: 3 },
          },
        }),
        { status: 200, headers: { 'content-type': 'application/json' } },
      );
    const client = new OpenRouterClient(
      {
        apiKey: 'sk-test',
        baseUrl: 'https://openrouter.ai/api/v1',
        model: 'model',
      },
      fakeFetch,
    );
    const completion = await client.completeWithUsage([
      { role: 'user', content: 'hello' },
    ]);
    expect(completion).toEqual({
      content: 'tracked',
      model: 'openai/gpt-5.6-luna',
      usage: {
        promptTokens: 120,
        completionTokens: 8,
        totalTokens: 128,
        cachedPromptTokens: 20,
        reasoningTokens: 3,
        costUsd: 0.0000168,
      },
    });
    expect(addLlmUsage(emptyLlmUsageTotals(), completion.usage)).toMatchObject({
      calls: 1,
      usageResponses: 1,
      costResponses: 1,
      totalTokens: 128,
      costUsd: 0.0000168,
    });
  });

  it('sends an OpenAI-style chat request with auth header and retries on 5xx', async () => {
    const requests: { url: string; init: RequestInit }[] = [];
    let attempt = 0;
    const fakeFetch: typeof fetch = async (url, init) => {
      requests.push({ url: String(url), init: init! });
      attempt++;
      if (attempt === 1) return new Response('upstream error', { status: 502 });
      return new Response(
        JSON.stringify({ choices: [{ message: { content: 'hi there' } }] }),
        { status: 200, headers: { 'content-type': 'application/json' } },
      );
    };
    const client = new OpenRouterClient(
      {
        apiKey: 'sk-test',
        baseUrl: 'https://openrouter.ai/api/v1',
        model: 'openai/gpt-5.6-luna',
      },
      fakeFetch,
    );
    const completion = await client.completeWithUsage(
      [{ role: 'user', content: 'hello' }],
      { maxTokens: 42 },
    );
    expect(completion.content).toBe('hi there');
    expect(client.model).toBe('openai/gpt-5.6-luna');
    expect(requests).toHaveLength(2);
    expect(requests[0].url).toBe(
      'https://openrouter.ai/api/v1/chat/completions',
    );
    const headers = requests[0].init.headers as Record<string, string>;
    expect(headers.Authorization).toBe('Bearer sk-test');
    const body = JSON.parse(String(requests[0].init.body));
    expect(body.model).toBe('openai/gpt-5.6-luna');
    expect(body.messages).toEqual([{ role: 'user', content: 'hello' }]);
    expect(body.temperature).toBe(0);
    expect(body.max_tokens).toBe(42);
  });

  it('normalizes malformed Unicode before sending provider JSON', async () => {
    let requestBody = '';
    const client = new OpenRouterClient(
      { apiKey: 'sk-test', baseUrl: 'https://x.test/v1', model: 'm' },
      (async (_url, init) => {
        requestBody = String(init?.body);
        return new Response(
          JSON.stringify({
            choices: [{ message: { content: 'ok' } }],
          }),
          { status: 200 },
        );
      }) as typeof fetch,
    );
    await client.complete([{ role: 'user', content: `left\uD800right` }]);
    expect(JSON.parse(requestBody).messages[0].content).toBe('left�right');
  });

  it('rejects invalid completion limits before making a request', async () => {
    const fakeFetch = vi.fn() as unknown as typeof fetch;
    const client = new OpenRouterClient(
      { apiKey: 'sk-test', baseUrl: 'https://x.test/v1', model: 'm' },
      fakeFetch,
    );
    await expect(
      client.completeWithUsage([{ role: 'user', content: 'q' }], {
        maxTokens: 0,
      }),
    ).rejects.toThrow(/max tokens/i);
    expect(fakeFetch).not.toHaveBeenCalled();
  });

  it('throws a readable error on repeated failure, without leaking the key', async () => {
    const fakeFetch: typeof fetch = async () =>
      new Response('nope', { status: 500 });
    const client = new OpenRouterClient(
      { apiKey: 'sk-secret-value', baseUrl: 'https://x.test/v1', model: 'm' },
      fakeFetch,
    );
    await expect(
      client.complete([{ role: 'user', content: 'q' }]),
    ).rejects.toSatisfy(
      (e: Error) =>
        /500/.test(e.message) && !e.message.includes('sk-secret-value'),
    );
  });

  it('returns bounded provider diagnostics without echoing sensitive text', async () => {
    const actionable = new OpenRouterClient(
      { apiKey: 'sk-test', baseUrl: 'https://x.test/v1', model: 'm' },
      (async () =>
        new Response(
          JSON.stringify({
            error: { message: 'Prompt exceeds the configured context limit.' },
          }),
          { status: 400 },
        )) as typeof fetch,
    );
    await expect(
      actionable.complete([{ role: 'user', content: 'q' }]),
    ).rejects.toThrow(
      /status 400: Prompt exceeds the configured context limit/i,
    );

    const sensitive = new OpenRouterClient(
      { apiKey: 'sk-test', baseUrl: 'https://x.test/v1', model: 'm' },
      (async () =>
        new Response(
          JSON.stringify({
            error: { message: 'API key is sk-sensitive-value' },
          }),
          { status: 400 },
        )) as typeof fetch,
    );
    await expect(
      sensitive.complete([{ role: 'user', content: 'q' }]),
    ).rejects.toThrow(/^LLM request failed with status 400$/i);
  });
});
