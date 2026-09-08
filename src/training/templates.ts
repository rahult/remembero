/**
 * Question templates in the flat query dialect. Every template emits a natural
 * question plus the gold program that answers it on the given world. Chain
 * templates place the anchor entity in whichever argument position the edge's
 * orientation demands, so the data set teaches argument direction rather than
 * a fixed "anchor goes first" habit.
 */
import { type Clause, evaluate, parseQuery } from '../engine/index.js';
import type { Rng } from './rng.js';
import {
  relationsOfKind,
  worldClauses,
  type Relation,
  type World,
} from './worlds.js';

export type Category =
  | 'direct'
  | 'join'
  | 'multihop-up'
  | 'multihop-down'
  | 'root'
  | 'leaves'
  | 'chain-filter'
  | 'yes-no'
  | 'absence'
  | 'count';

export type Direction = 'anchor-first' | 'anchor-second' | 'none';

export interface Candidate {
  world: string;
  category: Category;
  direction: Direction;
  template: string;
  question: string;
  program: string;
  /** True when an empty answer is the correct answer (yes/no false). */
  expectEmpty: boolean;
  /** True when the program uses a _plus predicate. */
  requiresClosure: boolean;
}

/**
 * Closure literal from `anchor` towards `upward` (parents) or downward
 * (children), with the anchor placed by the edge's orientation.
 * child-first edge(child, parent): up = edge_plus(anchor, V); parent-first: up = edge_plus(V, anchor).
 */
function chainLiteral(
  edge: Relation,
  anchor: string,
  variable: string,
  upward: boolean,
): [string, Direction] {
  const anchorFirst = (edge.orientation === 'child-first') === upward;
  return anchorFirst
    ? [`${edge.name}_plus(${anchor}, ${variable})`, 'anchor-first']
    : [`${edge.name}_plus(${variable}, ${anchor})`, 'anchor-second'];
}

/** One-hop literal in the same orientation as chainLiteral, used for root/leaf negation. */
function oneHop(
  edge: Relation,
  anchor: string,
  variable: string,
  upward: boolean,
): string {
  const anchorFirst = (edge.orientation === 'child-first') === upward;
  return anchorFirst
    ? `${edge.name}(${anchor}, ${variable})`
    : `${edge.name}(${variable}, ${anchor})`;
}

function phrase(template: string | undefined, x: string): string {
  return (template ?? 'everything connected to {x}').replaceAll('{x}', x);
}

function atomValues(
  rows: Array<Record<string, { type: string; value?: unknown }>>,
  name: string,
): string[] {
  return [
    ...new Set(
      rows
        .map((row) => row[name])
        .filter((term) => term?.type === 'atom')
        .map((term) => String(term.value)),
    ),
  ].sort();
}

function nodesWithEdges(
  clauses: Clause[],
  world: World,
  edge: Relation,
): string[] {
  return world.entities.filter(
    (e) =>
      evaluate(clauses, parseQuery(`${edge.name}(${e}, _).`)).length > 0 ||
      evaluate(clauses, parseQuery(`${edge.name}(_, ${e}).`)).length > 0,
  );
}

function lower(argName: string): string {
  return argName.toLowerCase();
}

export function generateCandidates(world: World, rng: Rng): Candidate[] {
  const out: Candidate[] = [];
  const add = (c: Omit<Candidate, 'world'>) =>
    out.push({ world: world.id, ...c });
  const clauses = worldClauses(world);
  const edges = [
    ...relationsOfKind(world, 'hierarchy'),
    ...relationsOfKind(world, 'dependency'),
  ];
  const attributes = relationsOfKind(world, 'attribute');
  const groups = relationsOfKind(world, 'membership');
  const valuesOf = (attr: Relation) =>
    atomValues(evaluate(clauses, parseQuery(`${attr.name}(_, V).`)), 'V');
  const groupsOf = (group: Relation) =>
    atomValues(evaluate(clauses, parseQuery(`${group.name}(_, G).`)), 'G');

  // ---- direct -------------------------------------------------------------
  for (const attr of attributes) {
    const holders = atomValues(
      evaluate(clauses, parseQuery(`${attr.name}(X, _).`)),
      'X',
    );
    for (const x of rng.shuffle(holders).slice(0, 3)) {
      add({
        category: 'direct',
        direction: 'none',
        template: 'direct-value',
        expectEmpty: false,
        requiresClosure: false,
        question: rng.pick([
          `What is the ${lower(attr.args[1])} of ${x}?`,
          `Which ${lower(attr.args[1])} does ${x} have?`,
          `${x}: ${lower(attr.args[1])}?`,
        ]),
        program: `q(V) :- ${attr.name}(${x}, V).`,
      });
    }
    for (const v of rng.shuffle(valuesOf(attr)).slice(0, 2)) {
      add({
        category: 'direct',
        direction: 'none',
        template: 'direct-holders',
        expectEmpty: false,
        requiresClosure: false,
        question: rng.pick([
          `Who has ${lower(attr.args[1])} ${v}?`,
          `List everything whose ${lower(attr.args[1])} is ${v}.`,
        ]),
        program: `q(X) :- ${attr.name}(X, ${v}).`,
      });
    }
  }

  // ---- join ---------------------------------------------------------------
  for (const group of groups) {
    const groupNames = groupsOf(group);
    for (const attr of attributes.slice(0, 2)) {
      // pick a (group, value) pair that actually has a member with that value
      const pairs = groupNames.flatMap((g) =>
        valuesOf(attr)
          .filter(
            (v) =>
              evaluate(
                clauses,
                parseQuery(`${group.name}(X, ${g}), ${attr.name}(X, ${v}).`),
              ).length > 0,
          )
          .map((v) => [g, v] as const),
      );
      if (pairs.length === 0) continue;
      const [g, v] = rng.pick(pairs);
      add({
        category: 'join',
        direction: 'none',
        template: 'join-group-attr',
        expectEmpty: false,
        requiresClosure: false,
        question: rng.pick([
          `Which members of ${g} have ${lower(attr.args[1])} ${v}?`,
          `In ${g}, who has ${lower(attr.args[1])} ${v}?`,
        ]),
        program: `q(X) :- ${group.name}(X, ${g}), ${attr.name}(X, ${v}).`,
      });
    }
    // 2..6 members keeps the ordered-pair answer within the 30-row cap
    const populous = groupNames.filter((g) => {
      const size = evaluate(
        clauses,
        parseQuery(`${group.name}(X, ${g}).`),
      ).length;
      return size >= 2 && size <= 6;
    });
    if (populous.length > 0) {
      const g = rng.pick(populous);
      add({
        category: 'join',
        direction: 'none',
        template: 'join-pairs',
        expectEmpty: false,
        requiresClosure: false,
        question: `Which pairs of different members both belong to ${g}?`,
        program: `q(A, B) :- ${group.name}(A, ${g}), ${group.name}(B, ${g}), A != B.`,
      });
    }
  }

  // ---- chains -------------------------------------------------------------
  for (const edge of edges) {
    const nodes = rng.shuffle(nodesWithEdges(clauses, world, edge));
    if (nodes.length === 0) continue;
    const reachable = (x: string, upward: boolean) =>
      atomValues(
        evaluate(
          clauses,
          parseQuery(`${chainLiteral(edge, x, 'Y', upward)[0]}.`),
        ),
        'Y',
      );

    const oneHopOf = (x: string, upward: boolean) =>
      atomValues(
        evaluate(clauses, parseQuery(`${oneHop(edge, x, 'Y', upward)}.`)),
        'Y',
      );
    // anchors where the closure reaches strictly more than the direct edge,
    // so every chain example genuinely needs _plus
    const ups = nodes.filter(
      (x) => reachable(x, true).length > oneHopOf(x, true).length,
    );
    const downs = nodes.filter(
      (x) => reachable(x, false).length > oneHopOf(x, false).length,
    );

    // one-hop edge lookups: the direct neighbour only, no closure. Teaches the
    // model that "direct" / "immediately" / "one step" means the base predicate.
    const argUp = lower(edge.args[edge.orientation === 'child-first' ? 1 : 0]);
    const argDown = lower(
      edge.args[edge.orientation === 'child-first' ? 0 : 1],
    );
    for (const x of rng.shuffle(nodes).slice(0, 2)) {
      for (const upward of [true, false]) {
        if (
          evaluate(clauses, parseQuery(`${oneHop(edge, x, 'Y', upward)}.`))
            .length === 0
        )
          continue;
        const literal = oneHop(edge, x, 'Y', upward);
        const direction: Direction = literal.startsWith(`${edge.name}(${x},`)
          ? 'anchor-first'
          : 'anchor-second';
        const role = upward ? argUp : argDown;
        add({
          category: 'direct',
          direction,
          template: 'edge-one-hop',
          expectEmpty: false,
          requiresClosure: false,
          question: rng.pick([
            `What is the direct ${role} of ${x} via ${edge.name}? One step only.`,
            `Immediately via ${edge.name}, which ${role} does ${x} connect to? Not the whole chain.`,
            `Give ${x}'s direct ${role} under ${edge.name}, one hop.`,
          ]),
          program: `q(Y) :- ${literal}.`,
        });
      }
    }

    for (const x of ups.slice(0, 3)) {
      const [up, upDir] = chainLiteral(edge, x, 'Y', true);
      add({
        category: 'multihop-up',
        direction: upDir,
        template: 'chain-up',
        expectEmpty: false,
        requiresClosure: true,
        question: `List ${phrase(edge.upPhrase, x)}.`,
        program: `q(Y) :- ${up}.`,
      });
    }
    for (const x of downs.slice(0, 3)) {
      const [down, downDir] = chainLiteral(edge, x, 'Y', false);
      add({
        category: 'multihop-down',
        direction: downDir,
        template: 'chain-down',
        expectEmpty: false,
        requiresClosure: true,
        question: `List ${phrase(edge.downPhrase, x)}.`,
        program: `q(Y) :- ${down}.`,
      });
    }
    for (const x of ups.slice(3, 5)) {
      const [up, upDir] = chainLiteral(edge, x, 'R', true);
      add({
        category: 'root',
        direction: upDir,
        template: 'chain-root',
        expectEmpty: false,
        requiresClosure: true,
        question: `Following ${edge.name} from ${x} to the end: what is the final one that has nothing further above it?`,
        program: `q(R) :- ${up}, \\+ ${oneHop(edge, 'R', '_', true)}.`,
      });
    }
    for (const x of downs.slice(3, 5)) {
      const [down, downDir] = chainLiteral(edge, x, 'L', false);
      add({
        category: 'leaves',
        direction: downDir,
        template: 'chain-leaves',
        expectEmpty: false,
        requiresClosure: true,
        question: `Under ${x} via ${edge.name}, which ones have nothing further below them?`,
        program: `q(L) :- ${down}, \\+ ${oneHop(edge, 'L', '_', false)}.`,
      });
    }
    if (attributes.length > 0 && ups.length > 0) {
      const attr = rng.pick(attributes);
      const x = rng.pick(ups);
      const [up, upDir] = chainLiteral(edge, x, 'Y', true);
      // values where the filtered closure answer is larger than the filtered
      // one-hop answer, so the example still needs _plus
      const values = valuesOf(attr).filter(
        (v) =>
          evaluate(clauses, parseQuery(`${up}, ${attr.name}(Y, ${v}).`))
            .length >
          evaluate(
            clauses,
            parseQuery(`${oneHop(edge, x, 'Y', true)}, ${attr.name}(Y, ${v}).`),
          ).length,
      );
      if (values.length > 0) {
        const v = rng.pick(values);
        add({
          category: 'chain-filter',
          direction: upDir,
          template: 'chain-filter-up',
          expectEmpty: false,
          requiresClosure: true,
          question: `Among ${phrase(edge.upPhrase, x)}, which have ${lower(attr.args[1])} ${v}?`,
          program: `q(Y) :- ${up}, ${attr.name}(Y, ${v}).`,
        });
      }
    }
    // yes/no: one true pair and one false pair per edge
    const pairs = rng.shuffle(
      nodes.flatMap((a) =>
        nodes.filter((b) => b !== a).map((b) => [a, b] as const),
      ),
    );
    // true pairs must be at least two hops apart so the closure is required
    const truePair = pairs.find(
      ([a, b]) =>
        reachable(a, true).includes(b) && !oneHopOf(a, true).includes(b),
    );
    const falsePair = pairs.find(([a, b]) => !reachable(a, true).includes(b));
    for (const [pair, truth] of [
      [truePair, true],
      [falsePair, false],
    ] as const) {
      if (!pair) continue;
      const [a, b] = pair;
      const [up, upDir] = chainLiteral(edge, a, 'Y', true);
      add({
        category: 'yes-no',
        direction: upDir,
        template: `yes-no-${truth}`,
        expectEmpty: !truth,
        requiresClosure: true,
        question: rng.pick([
          `Is ${b} anywhere above ${a} via ${edge.name}? Yes or no.`,
          `Does ${a} ultimately reach ${b} through ${edge.name}?`,
        ]),
        program: `q(Y) :- ${up}, Y = ${b}.`,
      });
    }
    if (ups.length > 0) {
      const x = rng.pick(ups);
      const [up, upDir] = chainLiteral(edge, x, 'Y', true);
      add({
        category: 'count',
        direction: upDir,
        template: 'count-chain',
        expectEmpty: false,
        requiresClosure: true,
        question: `How many are there in total among ${phrase(edge.upPhrase, x)}?`,
        program: `count(*) as N where ${up}`,
      });
    }
  }

  // ---- absence ------------------------------------------------------------
  for (const group of groups) {
    for (const attr of attributes) {
      add({
        category: 'absence',
        direction: 'none',
        template: 'absence-attr',
        expectEmpty: false,
        requiresClosure: false,
        question: rng.pick([
          `Which members of any ${lower(group.args[1])} have no recorded ${lower(attr.args[1])}?`,
          `Who is in a ${lower(group.args[1])} but has no ${lower(attr.args[1])} on file?`,
        ]),
        program: `q(E) :- ${group.name}(E, _), \\+ ${attr.name}(E, _).`,
      });
    }
    const groupNames = groupsOf(group);
    if (groupNames.length > 0) {
      const g = rng.pick(groupNames);
      add({
        category: 'count',
        direction: 'none',
        template: 'count-group',
        expectEmpty: false,
        requiresClosure: false,
        question: `How many members does ${g} have?`,
        program: `count(*) as N where ${group.name}(X, ${g})`,
      });
    }
  }
  return out;
}
