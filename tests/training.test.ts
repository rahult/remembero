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
import { generateCandidates } from '../src/training/templates.js';
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
