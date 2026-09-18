/*
 * First-proof demo: the sixty-second Remembero loop, executed live.
 *
 * A visitor "says" two things, stores them as facts, compiles one rule,
 * and asks one question. Every step below runs Remembero's deterministic
 * TypeScript engine in the visitor's browser — the same program parser and
 * proof-producing evaluator the grounded-agent lab uses. No model is
 * involved, nothing is sent anywhere, and the result never varies.
 */

import {
  evaluateQuerySpecWithProof,
  parseProgram,
  parseQuerySpec,
  type Clause,
  type QueryProof,
} from "./engine";

const ENGINE_OPTIONS = {
  maxFacts: 100,
  maxIterations: 16,
  maxRows: 8,
  maxProofDepth: 16,
  maxProofNodes: 256,
  maxProofsPerRow: 1,
  maxAggregateRows: 128,
  maxAggregateProofRows: 32,
} as const;

export interface DemoUtterance {
  said: string;
  when: string;
  plain: string;
  fact: string;
}

export const DEMO_UTTERANCES: readonly DemoUtterance[] = [
  {
    said: "Dana is my manager.",
    when: "1:1 planning · 3 Jun",
    plain: "Dana is your manager",
    fact: "manager(dana, you)",
  },
  {
    said: "We meet on Tuesdays.",
    when: "1:1 planning · 3 Jun",
    plain: "Dana and you meet on Tuesdays",
    fact: "meeting(dana, tuesday)",
  },
];

export const DEMO_RULE = `one_on_one(Person, Day) :-
  manager(Person, you),
  meeting(Person, Day).`;

export const DEMO_RULE_PLAIN = "a meeting with your manager is a one-on-one";

export const DEMO_QUESTION = "Do I have a 1:1 this week?";
export const DEMO_QUERY = "one_on_one(Person, Day)";

export const DEMO_MAGIC =
  "The phrase “1:1” appears nowhere in what you said. The rule derived it — that is knowledge, not retrieval.";

export interface DemoResult {
  answer: string;
  query: string;
  proofChain: string[];
  durationMs: number;
}

function claimText(proof: QueryProof): string {
  if ("aggregated" in proof) {
    return `${proof.op}(${proof.input}) as ${proof.as} = ${proof.value}`;
  }
  if ("negated" in proof) {
    return `not ${proof.predicate}(${proof.pattern
      .map((value) => (value === null ? "_" : value))
      .join(", ")})`;
  }
  return `${proof.predicate}(${proof.values
    .map((value) =>
      typeof value === "number" || typeof value === "string" ? String(value) : "_",
    )
    .join(", ")})`;
}

function collectProofLines(proof: QueryProof, lines: string[]): void {
  if (!lines.includes(claimText(proof))) lines.push(claimText(proof));
  if ("aggregated" in proof) {
    for (const contributor of proof.contributors) {
      for (const child of contributor.proofs) collectProofLines(child, lines);
    }
    return;
  }
  if ("negated" in proof) return;
  for (const child of proof.because ?? []) collectProofLines(child, lines);
  for (const contributor of proof.aggregate?.contributors ?? []) {
    for (const child of contributor.proofs) collectProofLines(child, lines);
  }
}

function demoProgram(): Clause[] {
  const facts = DEMO_UTTERANCES.map((utterance) => `${utterance.fact}.`).join("\n");
  return parseProgram(`${facts}\n\n${DEMO_RULE}`);
}

export function runFirstProofDemo(): DemoResult {
  const started = performance.now();
  const rows = evaluateQuerySpecWithProof(
    demoProgram(),
    parseQuerySpec(DEMO_QUERY),
    ENGINE_OPTIONS,
  );
  const first = rows[0];
  if (first === undefined) {
    throw new Error("first-proof demo produced no rows");
  }
  const proofChain: string[] = [];
  const proof = first.proofs[0];
  if (proof !== undefined) collectProofLines(proof, proofChain);
  const person = first.bindings.Person;
  const day = first.bindings.Day;
  const personName = termToName(person);
  const dayName = termToName(day);
  return {
    answer:
      personName && dayName
        ? `Yes — ${dayName}, with ${personName}.`
        : "Yes — you have a one-on-one this week.",
    query: DEMO_QUERY,
    proofChain,
    durationMs: performance.now() - started,
  };
}

function termToName(term: unknown): string | undefined {
  if (
    term !== undefined &&
    term !== null &&
    typeof term === "object" &&
    "type" in term &&
    "value" in term &&
    typeof (term as { value: unknown }).value === "string"
  ) {
    const name = (term as { value: string }).value;
    return `${name[0].toUpperCase()}${name.slice(1)}`;
  }
  return undefined;
}
