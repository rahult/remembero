import { existsSync, mkdtempSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  evaluate,
  isComparison,
  isIntegrityConstraint,
  isNegation,
  parseProgram,
  parseQuery,
} from '../src/engine/index.js';
import { createRng } from '../src/training/rng.js';
import {
  chooseHeldoutWorlds,
  exportDataset,
  toConversation,
} from '../src/training/export.js';
import {
  createLlmParaphraser,
  paraphraseExamples,
  preservesConstants,
} from '../src/training/paraphrase.js';
import { generateTrainingData } from '../src/training/run.js';
import { generateCandidates } from '../src/training/templates.js';
import {
  assertRecursionFree,
  assertRejectionRates,
  executeProgram,
  isRejection,
  verifyAll,
  verifyCandidate,
} from '../src/training/verify.js';
import {
  BENCHMARK_PREDICATES,
  generateWorld,
  relationsOfKind,
  schemaListing,
  worldClauses,
} from '../src/training/worlds.js';

describe('training: rng', () => {
  it('is deterministic under a seed', () => {
    const a = createRng(7);
    const b = createRng(7);
    expect([a.int(100), a.int(100), a.int(100)]).toEqual([
      b.int(100),
      b.int(100),
      b.int(100),
    ]);
  });
});

describe('training: worlds', () => {
  it('produces byte-identical worlds for the same seed', () => {
    expect(JSON.stringify(generateWorld(3))).toBe(
      JSON.stringify(generateWorld(3)),
    );
    expect(JSON.stringify(generateWorld(3))).not.toBe(
      JSON.stringify(generateWorld(4)),
    );
  });

  it('satisfies the structural invariants on twenty seeds', () => {
    for (let seed = 1; seed <= 20; seed += 1) {
      const world = generateWorld(seed);
      const edges = [
        ...relationsOfKind(world, 'hierarchy'),
        ...relationsOfKind(world, 'dependency'),
      ];
      expect(edges.length).toBeGreaterThan(0);
      expect(relationsOfKind(world, 'attribute').length).toBeGreaterThan(0);
      expect(relationsOfKind(world, 'membership').length).toBeGreaterThan(0);
      expect(world.entities.length).toBeGreaterThanOrEqual(12);
      expect(world.entities.length).toBeLessThanOrEqual(30);
      for (const relation of world.relations) {
        expect(BENCHMARK_PREDICATES).not.toContain(relation.name);
      }
      const clauses = worldClauses(world);
      for (const edge of edges) {
        const three = evaluate(
          clauses,
          parseQuery(
            `${edge.name}(A, B), ${edge.name}(B, C), ${edge.name}(C, D).`,
          ),
        );
        expect(three.length, `${edge.name} in seed ${seed}`).toBeGreaterThan(0);
      }
      const gaps = relationsOfKind(world, 'attribute').some((attr) =>
        world.entities.some(
          (entity) =>
            evaluate(clauses, parseQuery(`${attr.name}(${entity}, V).`))
              .length === 0,
        ),
      );
      expect(gaps).toBe(true);
    }
  });

  it('lists the schema with argument names', () => {
    const world = generateWorld(1);
    const listing = schemaListing(world);
    for (const relation of world.relations) {
      expect(listing).toContain(
        `${relation.name}(${relation.args.join(', ')})`,
      );
    }
  });
});

describe('training: vocabulary direction', () => {
  // For each edge relation: the phrase fragment its UP phrase must contain,
  // where "up" = towards the parent/upstream end (the second argument of a
  // child-first edge, the first argument of a parent-first edge).
  const UP_MUST_CONTAIN: Record<string, string> = {
    manages: 'above',
    mentors: 'mentors',
    answers_to: 'above',
    reports_into: 'reports to',
    led_by: 'lead',
    hands_off_to: 'downstream',
    part_of: 'contains',
    owns: 'owner above',
    depends_on: 'depends on',
    feeds: 'upstream',
    calls: '{x} reaches',
    waits_for: '{x} ultimately waits for',
    blocks: 'blocks {x}',
    located_in: 'region containing',
    contains: 'region that contains',
    ships_to: 'goods from {x}',
    precedes: 'after {x}',
    supplied_by: 'goods from {x}',
    follows: 'after {x}',
  };

  it('every edge vocabulary has a checked up-phrase and its up literal returns the parent', () => {
    const seen = new Map<string, string>();
    for (let seed = 1; seed <= 90; seed += 1) {
      const world = generateWorld(seed);
      for (const edge of [
        ...relationsOfKind(world, 'hierarchy'),
        ...relationsOfKind(world, 'dependency'),
      ]) {
        if (seen.has(edge.name)) continue;
        seen.set(edge.name, edge.upPhrase ?? '');
        const chain =
          edge.orientation === 'child-first'
            ? [`${edge.name}(a, b).`, `${edge.name}(b, c).`]
            : [`${edge.name}(b, a).`, `${edge.name}(c, b).`];
        const clauses = parseProgram(chain.join('\n'));
        const upLiteral =
          edge.orientation === 'child-first'
            ? `${edge.name}_plus(b, Y).`
            : `${edge.name}_plus(Y, b).`;
        const up = evaluate(clauses, parseQuery(upLiteral)).map(
          (r) => (r.Y as { value: string }).value,
        );
        expect(up, `${edge.name} up literal`).toEqual(['c']);
      }
    }
    for (const [name, phrase] of seen) {
      const must = UP_MUST_CONTAIN[name];
      expect(
        must,
        `no direction expectation recorded for ${name}`,
      ).toBeDefined();
      expect(phrase, `${name} up-phrase "${phrase}"`).toContain(must);
    }
    expect(seen.size).toBe(Object.keys(UP_MUST_CONTAIN).length);
  });
});

describe('training: templates', () => {
  it('emits every category and balances chain directions', () => {
    const counts: Record<string, number> = {};
    const directions: Record<string, number> = {
      'anchor-first': 0,
      'anchor-second': 0,
    };
    for (let seed = 1; seed <= 20; seed += 1) {
      const world = generateWorld(seed);
      for (const candidate of generateCandidates(world, createRng(seed * 31))) {
        counts[candidate.category] = (counts[candidate.category] ?? 0) + 1;
        if (candidate.requiresClosure && candidate.direction !== 'none') {
          directions[candidate.direction] += 1;
        }
      }
    }
    for (const category of [
      'direct',
      'join',
      'multihop-up',
      'multihop-down',
      'root',
      'leaves',
      'chain-filter',
      'yes-no',
      'absence',
      'count',
    ]) {
      expect(counts[category], category).toBeGreaterThan(0);
    }
    const ratio = directions['anchor-first'] / directions['anchor-second'];
    expect(ratio).toBeGreaterThan(0.6);
    expect(ratio).toBeLessThan(1.7);
  });

  it('emits one-hop edge lookups in both directions without _plus', () => {
    let oneHop = 0;
    const directions = new Set<string>();
    for (let seed = 1; seed <= 20; seed += 1) {
      const world = generateWorld(seed);
      for (const candidate of generateCandidates(world, createRng(seed))) {
        if (candidate.template !== 'edge-one-hop') continue;
        oneHop += 1;
        directions.add(candidate.direction);
        expect(candidate.requiresClosure).toBe(false);
        expect(candidate.program).not.toContain('_plus');
      }
    }
    expect(oneHop).toBeGreaterThan(20);
    expect(directions).toEqual(new Set(['anchor-first', 'anchor-second']));
  });

  it('asks yes/no questions as ground goals that answer with a boolean row', () => {
    let seen = 0;
    for (let seed = 1; seed <= 20; seed += 1) {
      const world = generateWorld(seed);
      for (const candidate of generateCandidates(world, createRng(seed))) {
        if (candidate.category !== 'yes-no') continue;
        seen += 1;
        expect(candidate.program, candidate.question).toMatch(
          /^\?- [a-z_]+_plus\([a-z_0-9]+, [a-z_0-9]+\)\.$/,
        );
      }
    }
    expect(seen).toBeGreaterThan(10);
  });

  it('queries ternary schedule relations, including absence over them', () => {
    let lookups = 0;
    let absences = 0;
    for (let seed = 1; seed <= 40; seed += 1) {
      const world = generateWorld(seed);
      const schedule = relationsOfKind(world, 'schedule')[0];
      if (!schedule) continue;
      for (const candidate of generateCandidates(world, createRng(seed))) {
        if (candidate.template === 'schedule-lookup') {
          lookups += 1;
          expect(candidate.program).toMatch(
            new RegExp(
              `^q\\(D, W\\) :- ${schedule.name}\\([a-z_]+, D, W\\)\\.$`,
            ),
          );
        }
        if (candidate.template === 'schedule-absence') {
          absences += 1;
          expect(candidate.program).toContain(`\\+ ${schedule.name}(G, _, _)`);
        }
      }
    }
    expect(lookups).toBeGreaterThan(5);
    expect(absences).toBeGreaterThan(5);
  });

  it('never emits a recursive rule', () => {
    for (let seed = 1; seed <= 20; seed += 1) {
      const world = generateWorld(seed);
      for (const candidate of generateCandidates(world, createRng(seed))) {
        if (!candidate.program.includes(':-')) continue;
        for (const clause of parseProgram(candidate.program)) {
          if (isIntegrityConstraint(clause)) continue;
          for (const goal of clause.body) {
            if (isComparison(goal)) continue;
            const literal = isNegation(goal) ? goal.not : goal;
            expect(literal.predicate).not.toBe(clause.head.predicate);
          }
        }
      }
    }
  });

  it('mentions every entity constant of the program in the question', () => {
    const world = generateWorld(2);
    for (const candidate of generateCandidates(world, createRng(2))) {
      const constants = candidate.program.match(/\b[a-z][a-z0-9_]*\b/g) ?? [];
      for (const constant of constants.filter((c) =>
        world.entities.includes(c),
      )) {
        expect(candidate.question.toLowerCase(), candidate.program).toContain(
          constant,
        );
      }
    }
  });
});

describe('training: verify', () => {
  const world = generateWorld(5);

  it('executes rule programs and aggregate queries', () => {
    const clauses = worldClauses(world);
    const edge = [
      ...relationsOfKind(world, 'hierarchy'),
      ...relationsOfKind(world, 'dependency'),
    ][0];
    const rows = executeProgram(clauses, `q(A, B) :- ${edge.name}(A, B).`);
    expect(rows.length).toBeGreaterThan(0);
    const count = executeProgram(
      clauses,
      `count(*) as N where ${edge.name}(A, B)`,
    );
    expect(count).toHaveLength(1);
  });

  it('rejects recursive programs', () => {
    expect(() =>
      assertRecursionFree(
        'above(M) :- edge(x, M).\nabove(M) :- above(X), edge(X, M).',
      ),
    ).toThrow(/recurs/i);
    expect(() => assertRecursionFree('a(X) :- b(X).\nb(X) :- a(X).')).toThrow(
      /recurs/i,
    );
    expect(() => assertRecursionFree('q(Y) :- edge_plus(x, Y).')).not.toThrow();
  });

  it('rejects a closure example whose one-hop answer is identical', () => {
    const mini = {
      ...world,
      facts: ['edge(a, b).', 'edge(c, d).'],
      entities: ['a', 'b', 'c', 'd'],
      relations: [
        {
          name: 'edge',
          kind: 'hierarchy' as const,
          args: ['A', 'B'],
          orientation: 'child-first' as const,
        },
      ],
    };
    const result = verifyCandidate(mini, {
      world: mini.id,
      category: 'multihop-up',
      direction: 'anchor-first',
      template: 't',
      question: 'above a',
      program: 'q(Y) :- edge_plus(a, Y).',
      expectEmpty: false,
      requiresClosure: true,
    });
    expect(isRejection(result) && result.reason).toMatch(/closure/i);
  });

  it('keeps most generated candidates and records rejections by template', () => {
    const candidates = generateCandidates(world, createRng(5));
    const { examples, rejections } = verifyAll(world, candidates);
    expect(examples.length).toBeGreaterThan(candidates.length / 2);
    expect(() => assertRejectionRates(candidates, rejections)).not.toThrow();
    expect(() =>
      assertRejectionRates(
        candidates,
        candidates.map((c) => ({ candidate: c, reason: 'x' })),
      ),
    ).toThrow(/rejection rate/i);
    for (const example of examples) {
      if (example.expectEmpty) expect(example.answer).toEqual([]);
      else expect(example.answer.length).toBeGreaterThan(0);
    }
  });
});

describe('training: paraphrase', () => {
  const example = {
    world: 'w',
    category: 'multihop-up' as const,
    direction: 'anchor-first' as const,
    template: 't',
    question: 'List everyone above ana in the management chain.',
    program: 'q(Y) :- manages_plus(Y, ana).',
    expectEmpty: false,
    requiresClosure: true,
    answer: ['Y=bo'],
  };

  it('matches constants as whole words, not substrings', () => {
    const ex = {
      ...example,
      question: 'Everything under search via part_of.',
      program: 'q(L) :- part_of_plus(L, search).',
    };
    expect(preservesConstants(ex, 'What is under search in part_of?')).toBe(
      true,
    );
    expect(
      preservesConstants(ex, 'Which ones are found when searched via part_of?'),
    ).toBe(false);
    const multi = {
      ...example,
      question: 'Who is in db_primary?',
      program: 'q(X) :- in_team(X, db_primary).',
    };
    expect(preservesConstants(multi, 'List the members of db primary.')).toBe(
      true,
    );
  });

  it('keeps paraphrases that preserve every entity constant and drops those that do not', () => {
    expect(
      preservesConstants(example, 'Who is above Ana, all the way up?'),
    ).toBe(true);
    expect(
      preservesConstants(example, 'Who is above Bo, all the way up?'),
    ).toBe(false);
  });

  it('caches results so a second call makes no client request', async () => {
    let calls = 0;
    const client = {
      complete: async () => {
        calls += 1;
        return JSON.stringify([
          'Who sits above ana?',
          "Name ana's whole chain of command.",
          'Who is above bo?',
        ]);
      },
    };
    const dir = mkdtempSync(join(tmpdir(), 'para-'));
    const paraphraser = createLlmParaphraser(client, dir);
    const first = await paraphraser.paraphrase(example, 3);
    const second = await paraphraser.paraphrase(example, 3);
    expect(first).toEqual([
      'Who sits above ana?',
      "Name ana's whole chain of command.",
    ]);
    expect(second).toEqual(first);
    expect(calls).toBe(1);
  });

  it('falls back to the templated question when the client fails', async () => {
    const client = {
      complete: async () => {
        throw new Error('boom');
      },
    };
    const dir = mkdtempSync(join(tmpdir(), 'para-'));
    const out = await paraphraseExamples(
      [example],
      createLlmParaphraser(client, dir),
      2,
    );
    expect(out).toHaveLength(1);
    expect(out[0].question).toBe(example.question);
  });
});

describe('training: export', () => {
  it('emits Tinker conversation lines with system schema, user question, assistant program', () => {
    const world = generateWorld(9);
    const { examples, rejections } = verifyAll(
      world,
      generateCandidates(world, createRng(9)),
    );
    const worlds = [world, generateWorld(10)];
    const heldout = chooseHeldoutWorlds(worlds, 0.5);
    const {
      train,
      heldout: held,
      manifest,
    } = exportDataset({
      worlds,
      examples,
      heldoutWorldIds: heldout,
      rejections,
      seed: 9,
      paraphraseModel: null,
      paraphrasesPerExample: 0,
    });
    const lines = (heldout.has(world.id) ? held : train).trim().split('\n');
    expect(lines.length).toBe(examples.length);
    for (const line of lines) {
      const parsed = JSON.parse(line) as {
        messages: Array<{ role: string; content: string }>;
      };
      expect(parsed.messages.map((m) => m.role)).toEqual([
        'system',
        'user',
        'assistant',
      ]);
      expect(parsed.messages[0].content).toContain('p_plus');
      expect(parsed.messages[0].content).toContain(world.relations[0].name);
      expect(parsed.messages[2].content).toMatch(/^(q\(|count\(|\?- )/);
    }
    expect(manifest.train + manifest.heldout).toBe(examples.length);
    expect(Object.values(manifest.byCategory).reduce((a, b) => a + b, 0)).toBe(
      examples.length,
    );
  });

  it('adds repair-turn conversations for a share of the examples when asked', () => {
    const world = generateWorld(9);
    const { examples, rejections } = verifyAll(
      world,
      generateCandidates(world, createRng(9)),
    );
    const { train, manifest } = exportDataset({
      worlds: [world],
      examples,
      heldoutWorldIds: new Set(),
      rejections,
      seed: 9,
      paraphraseModel: null,
      paraphrasesPerExample: 0,
      repairShare: 1,
    });
    const lines = train
      .trim()
      .split('\n')
      .map(
        (l) =>
          JSON.parse(l) as {
            messages: Array<{ role: string; content: string }>;
          },
      );
    const repairs = lines.filter((l) => l.messages.length === 5);
    expect(repairs.length).toBeGreaterThan(0);
    expect(manifest.repairTurns).toBe(repairs.length);
    expect(lines.length).toBe(examples.length + repairs.length);
    for (const r of repairs) {
      expect(r.messages[3].content).toMatch(/returned no (rows|results)/);
    }
  });

  it('serializes the assistant program canonically', () => {
    const world = generateWorld(9);
    const example = {
      world: world.id,
      category: 'direct' as const,
      direction: 'none' as const,
      template: 't',
      question: 'x',
      program: 'q(V)   :-   tier( auth ,V ).',
      expectEmpty: false,
      requiresClosure: false,
      answer: [],
    };
    expect(toConversation(world, example).messages[2].content).toBe(
      'q(V) :- tier(auth, V).',
    );
  });
});

describe('training: run', () => {
  it('draws exactly the requested number of rounds when --rounds is given', async () => {
    const out = mkdtempSync(join(tmpdir(), 'train-rounds-'));
    const one = await generateTrainingData({
      examples: 0,
      rounds: 1,
      worlds: 4,
      paraphrases: 0,
      seed: 11,
      out,
      paraphrase: false,
    });
    const two = await generateTrainingData({
      examples: 0,
      rounds: 2,
      worlds: 4,
      paraphrases: 0,
      seed: 11,
      out,
      paraphrase: false,
    });
    expect(two.train + two.heldout).toBeGreaterThan(one.train + one.heldout);
    expect(two.rounds).toBe(2);
  });

  it('generates at least the requested number of verified examples without paraphrasing', async () => {
    const out = mkdtempSync(join(tmpdir(), 'train-'));
    const manifest = await generateTrainingData({
      examples: 200,
      worlds: 8,
      paraphrases: 0,
      seed: 11,
      out,
      paraphrase: false,
    });
    expect(manifest.train + manifest.heldout).toBeGreaterThanOrEqual(200);
    expect(existsSync(join(out, 'conversations.jsonl'))).toBe(true);
    expect(existsSync(join(out, 'heldout.jsonl'))).toBe(true);
    expect(
      JSON.parse(readFileSync(join(out, 'manifest.json'), 'utf8')).seed,
    ).toBe(11);
  });
});
