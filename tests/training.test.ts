import { describe, expect, it } from 'vitest';
import { evaluate, parseQuery } from '../src/engine/index.js';
import { createRng } from '../src/training/rng.js';
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
