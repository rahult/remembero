import { describe, expect, it } from 'vitest';
import { createRng } from '../src/training/rng.js';
import { generateCandidates } from '../src/training/templates.js';
import { verifyAll, executeProgram } from '../src/training/verify.js';
import { generateWorld, worldClauses } from '../src/training/worlds.js';
import { mutateToEmpty, toRepairConversation } from '../src/training/repair.js';

const world = generateWorld(1001);
const rng = () => createRng(5);
const examples = verifyAll(world, generateCandidates(world, rng())).examples;
const joins = examples.filter(
  (e) =>
    !e.expectEmpty &&
    e.program.includes(':-') &&
    e.program.split(',').length >= 2,
);

describe('repair-turn training data', () => {
  it('mutates a join program into one that runs but returns no rows, with engine feedback', () => {
    let found = 0;
    for (const example of joins) {
      const mutated = mutateToEmpty(world, example, rng());
      if (mutated === undefined) continue;
      found += 1;
      expect(mutated.program).not.toBe(example.program);
      expect(executeProgram(worldClauses(world), mutated.program)).toEqual([]);
      expect(mutated.feedback).toMatch(
        /alone matches|Did you mean|unknown predicate/,
      );
    }
    expect(found).toBeGreaterThan(0);
  });

  it('builds a five-message conversation ending in the verified program', () => {
    const example = joins.find(
      (e) => mutateToEmpty(world, e, rng()) !== undefined,
    )!;
    const conversation = toRepairConversation(world, example, rng())!;
    expect(conversation.messages.map((m) => m.role)).toEqual([
      'system',
      'user',
      'assistant',
      'user',
      'assistant',
    ]);
    expect(conversation.messages[1].content).toBe(example.question);
    expect(conversation.messages[3].content).toMatch(
      /returned no (rows|results)/,
    );
    expect(conversation.messages[4].content.replace(/\s+/g, ' ')).toContain(
      example.program.split('\n')[0].replace(/\s+/g, ' ').slice(0, 20),
    );
  });

  it('teaches repeating a correct program whose answer is legitimately empty', () => {
    const empty = examples.find((e) => e.expectEmpty)!;
    const conversation = toRepairConversation(world, empty, rng())!;
    expect(conversation.messages).toHaveLength(5);
    expect(conversation.messages[2].content).toBe(
      conversation.messages[4].content,
    );
  });
});
