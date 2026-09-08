/**
 * Export verified examples in the conversation JSONL shape that
 * tinker_cookbook.supervised.data.FromConversationFileBuilder reads:
 * one `{"messages":[{role, content}, ...]}` object per line, trained on the
 * assistant message. Whole worlds are held out so evaluation measures the
 * dialect rather than memorized predicate names.
 */
import { parseProgram, serializeClause } from '../engine/index.js';
import type { Example, Rejection } from './verify.js';
import { schemaListing, type World } from './worlds.js';

export const DIALECT_CARD = `Write ONE Datalog program that answers the question. Reply with ONLY the program.
- Rule shape: q(A, B) :- predicate(A), other(B, A).   The answer is the rule no other rule uses.
- Variables start uppercase (X, Person). Constants are lowercase exactly as listed.
- _ is a wildcard. \\+ predicate(X) means "no such fact". Comparisons: A != B, X = value.
- Any binary predicate p also answers p_plus(X, Y): Y is reachable from X in ONE OR MORE hops.
  Use p_plus for chains ("above", "below", "ultimately", "directly or transitively"). NEVER write recursive rules.
- Yes/no: ask the ground goal: ?- p_plus(x, target).   Answers yes = true or yes = false.
- Counting: count(*) as N where predicate(X, value)`;

export function systemPrompt(world: World): string {
  return `You query a knowledge base with these predicates:\n${schemaListing(world)}\n\n${DIALECT_CARD}`;
}

function canonicalProgram(program: string): string {
  if (!program.includes(':-')) return program.replace(/\s+/g, ' ').trim();
  return parseProgram(program).map(serializeClause).join('\n');
}

export interface Conversation {
  messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }>;
}

export function toConversation(world: World, example: Example): Conversation {
  return {
    messages: [
      { role: 'system', content: systemPrompt(world) },
      { role: 'user', content: example.question },
      { role: 'assistant', content: canonicalProgram(example.program) },
    ],
  };
}

export interface Manifest {
  seed: number;
  generatedAt: string;
  worlds: number;
  heldoutWorlds: string[];
  train: number;
  heldout: number;
  byCategory: Record<string, number>;
  byDirection: Record<string, number>;
  rejectionsByTemplate: Record<string, number>;
  paraphraseModel: string | null;
  paraphrasesPerExample: number;
  /** Candidate rounds drawn per world. */
  rounds: number;
}

/** Deterministic holdout: the highest-seeded worlds. */
export function chooseHeldoutWorlds(
  worlds: World[],
  fraction = 0.1,
): Set<string> {
  const count = Math.max(1, Math.round(worlds.length * fraction));
  return new Set(
    [...worlds]
      .sort((a, b) => b.seed - a.seed)
      .slice(0, count)
      .map((w) => w.id),
  );
}

function tally<T>(
  items: T[],
  key: (item: T) => string,
): Record<string, number> {
  const out: Record<string, number> = {};
  for (const item of items) out[key(item)] = (out[key(item)] ?? 0) + 1;
  return Object.fromEntries(
    Object.entries(out).sort(([a], [b]) => a.localeCompare(b)),
  );
}

export function exportDataset(input: {
  worlds: World[];
  examples: Example[];
  heldoutWorldIds: Set<string>;
  rejections: Rejection[];
  seed: number;
  paraphraseModel: string | null;
  paraphrasesPerExample: number;
  rounds?: number;
}): { train: string; heldout: string; manifest: Manifest } {
  const byId = new Map(input.worlds.map((w) => [w.id, w]));
  const train: string[] = [];
  const heldout: string[] = [];
  for (const example of input.examples) {
    const world = byId.get(example.world);
    if (!world) throw new Error(`unknown world ${example.world}`);
    const line = JSON.stringify(toConversation(world, example));
    (input.heldoutWorldIds.has(world.id) ? heldout : train).push(line);
  }
  return {
    train: `${train.join('\n')}\n`,
    heldout: `${heldout.join('\n')}\n`,
    manifest: {
      seed: input.seed,
      generatedAt: new Date().toISOString(),
      worlds: input.worlds.length,
      heldoutWorlds: [...input.heldoutWorldIds].sort(),
      train: train.length,
      heldout: heldout.length,
      byCategory: tally(input.examples, (e) => e.category),
      byDirection: tally(input.examples, (e) => e.direction),
      rejectionsByTemplate: tally(
        input.rejections,
        (r) => r.candidate.template,
      ),
      paraphraseModel: input.paraphraseModel,
      paraphrasesPerExample: input.paraphrasesPerExample,
      rounds: input.rounds ?? 0,
    },
  };
}
