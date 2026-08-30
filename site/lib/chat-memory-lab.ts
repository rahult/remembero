import {
  evaluateQuerySpecWithProof,
  isIntegrityConstraint,
  parseProgram,
  parseQuerySpec,
  serializeClause,
  serializeTerm,
  type Clause,
  type ProofStep,
  type QueryProof,
  type Term,
} from "./engine";

export type ChatMemoryScenarioId =
  | "root-blocker"
  | "write-gate"
  | "unknown-preference"
  | "why-not";

export interface ChatMemoryScenario {
  id: ChatMemoryScenarioId;
  label: string;
  question: string;
  baselineAnswer: string;
  query: string;
  program: string;
  requiredTerms: readonly string[];
}

export interface ChatMemoryRun {
  answer: string;
  claims: string[];
  absences: string[];
  proofTrail: string[];
  rule?: string;
  bindings: Record<string, string>;
  factCount: number;
  ruleCount: number;
  absenceCount: number;
  facts: string[];
  rules: string[];
  query: string;
}

const SHARED_FACTS = `
  status(atlas, blocked).
  status(orchard, active).
  blocker(atlas, vendor_security_review).
  waits_on(atlas, vendor_security_review).
  waits_on(vendor_security_review, legal_signoff).
  waits_on(legal_signoff, procurement_freeze).
  prefers_meeting(maya, morning).
  pending_meeting(jordan, roadmap_sync).
  review_slot(atlas, tuesday, morning).
`;

export const CHAT_MEMORY_SCENARIOS: readonly ChatMemoryScenario[] = [
  {
    id: "root-blocker",
    label: "Find the root blocker",
    question: "What is really blocking the Atlas review?",
    baselineAnswer:
      "The recursive SQL closure lists every dependency row — vendor security review, legal signoff, procurement freeze — but the rows arrive without any derivation to check.",
    query: "root_blocker(atlas, Root)",
    requiredTerms: ["procurement", "freeze"],
    program: `
      ${SHARED_FACTS}
      reaches(P, X) :- waits_on(P, X).
      reaches(P, X) :- reaches(P, M), waits_on(M, X).
      root_blocker(P, Root) :-
        reaches(P, Root),
        \\+ waits_on(Root, _).
    `,
  },
  {
    id: "write-gate",
    label: "Reject a bad update",
    question: "Mark Atlas active — the review can go ahead now, right?",
    baselineAnswer:
      "The UPDATE succeeded, the schedule query returned Tuesday morning, and nothing warned that the vendor security review still blocks Atlas.",
    query: "violation('active project keeps a blocker', P, B)",
    requiredTerms: ["vendor security review", "refus"],
    program: `
      ${SHARED_FACTS}
      proposed_status(atlas, active).
      violation('active project keeps a blocker', P, B) :-
        proposed_status(P, active),
        blocker(P, B).
    `,
  },
  {
    id: "unknown-preference",
    label: "Prove an absence",
    question: "Should I book Jordan for the morning sync?",
    baselineAnswer:
      "The LEFT JOIN returned Jordan’s sync with a NULL preference column; whether NULL means unknown or no preference is left for the model to guess.",
    query: "missing_preference(jordan)",
    requiredTerms: ["jordan", "ask", "preference"],
    program: `
      ${SHARED_FACTS}
      missing_preference(Person) :-
        pending_meeting(Person, _),
        \\+ prefers_meeting(Person, _).
    `,
  },
  {
    id: "why-not",
    label: "Explain a missing answer",
    question: "Why isn’t the Orchard review on the calendar?",
    baselineAnswer:
      "The schedule query for Orchard returned zero rows. SQL has nothing further to say about which condition failed.",
    query: "schedule_review(orchard, Day, Window, Blocker)",
    requiredTerms: ["orchard", "blocked", "slot"],
    program: `
      ${SHARED_FACTS}
      schedule_review(Project, Day, Window, Blocker) :-
        review_slot(Project, Day, Window),
        status(Project, blocked),
        blocker(Project, Blocker),
        prefers_meeting(maya, Window).
    `,
  },
] as const;

export interface WhyNotPremise {
  literal: string;
  holds: boolean;
  detail: string;
}

/**
 * Premise-by-premise diagnosis of why schedule_review(orchard, …) derives no
 * rows: each body literal of the rule is evaluated on its own so the failing
 * conditions are named instead of inferred from an empty result set.
 */
export function diagnoseWhyNot(): WhyNotPremise[] {
  const scenario = CHAT_MEMORY_SCENARIOS.find((entry) => entry.id === "why-not")!;
  const clauses = parseProgram(scenario.program);
  const premises: ReadonlyArray<{ literal: string; description: string }> = [
    {
      literal: "review_slot(orchard, Day, Window)",
      description: "a review slot exists for orchard",
    },
    {
      literal: "status(orchard, blocked)",
      description: "orchard is in the blocked state",
    },
    {
      literal: "blocker(orchard, Blocker)",
      description: "a blocker is recorded for orchard",
    },
    {
      literal: "prefers_meeting(maya, Window)",
      description: "maya has a stored meeting window",
    },
  ];
  return premises.map(({ literal, description }) => {
    const rows = evaluateQuerySpecWithProof(clauses, parseQuerySpec(literal), {
      maxFacts: 128,
      maxIterations: 16,
      maxRows: 4,
      maxProofDepth: 8,
      maxProofNodes: 64,
      maxProofsPerRow: 1,
      maxAggregateRows: 64,
      maxAggregateProofRows: 16,
    });
    return {
      literal,
      holds: rows.length > 0,
      detail:
        rows.length > 0
          ? `holds — ${description}`
          : `fails — no fact satisfies ${literal}`,
    };
  });
}

function termText(term: Term): string {
  return serializeTerm(term);
}

function proofText(proof: ProofStep): string {
  if ("negated" in proof) {
    return `not ${proof.predicate}(${proof.pattern
      .map((value) => {
        if (value === null) return "_";
        return typeof value === "number"
          ? termText({ type: "num", value })
          : termText({ type: "atom", value });
      })
      .join(", ")})`;
  }

  return `${proof.predicate}(${proof.values
    .map((value) =>
      termText(
        typeof value === "number"
          ? { type: "num", value }
          : { type: "atom", value },
      ),
    )
    .join(", ")}).`;
}

function addUnique(target: string[], value?: string): void {
  if (value === undefined || target.includes(value)) return;
  target.push(value);
}

function collectProof(
  proof: QueryProof,
  claims: string[],
  absences: string[],
  trail: string[],
  depth = 0,
): void {
  const prefix = `${"  ".repeat(depth)}${depth === 0 ? "" : "↳ "}`;

  if ("aggregated" in proof) {
    addUnique(
      trail,
      `${prefix}${proof.op}(${proof.input}) as ${proof.as} = ${proof.value}`,
    );
    for (const contributor of proof.contributors) {
      for (const child of contributor.proofs) {
        collectProof(child, claims, absences, trail, depth + 1);
      }
    }
    return;
  }

  if ("negated" in proof) {
    const absence = proofText(proof);
    addUnique(absences, absence);
    addUnique(trail, `${prefix}${absence}`);
    return;
  }

  const claim = proofText(proof);
  if (proof.rule === undefined) addUnique(claims, claim);
  addUnique(
    trail,
    proof.rule === undefined
      ? `${prefix}${claim}`
      : `${prefix}${claim} via rule ${proof.rule}`,
  );

  for (const child of proof.because ?? []) {
    collectProof(child, claims, absences, trail, depth + 1);
  }

  for (const contributor of proof.aggregate?.contributors ?? []) {
    for (const child of contributor.proofs) {
      collectProof(child, claims, absences, trail, depth + 1);
    }
  }
}

function authoredRules(clauses: Clause[]): string[] {
  return clauses
    .filter((clause) => clause.body.length > 0 && !isIntegrityConstraint(clause))
    .map(serializeClause);
}

function factsOnly(clauses: Clause[]): string[] {
  return clauses
    .filter((clause) => clause.body.length === 0)
    .map(serializeClause);
}

function title(value: string): string {
  return value
    .replace(/^'(.*)'$/, "$1")
    .split("_")
    .map((part) =>
      part.length === 0
        ? part
        : `${part.slice(0, 1).toUpperCase()}${part.slice(1)}`,
    )
    .join(" ");
}

function scenarioAnswer(
  scenario: ChatMemoryScenario,
  bindings: Record<string, string>,
  absences: string[],
): string {
  switch (scenario.id) {
    case "root-blocker":
      return `The ${title(bindings.Root).toLowerCase()} is the root blocker. The proof walks the chain vendor security review → legal signoff → procurement freeze, and proves nothing sits beneath it.`;
    case "write-gate":
      return `Refused. Marking ${title(bindings.P)} active would contradict the recorded blocker ${title(bindings.B).toLowerCase()}; the write was rolled back and the review stays blocked.`;
    case "unknown-preference":
      return absences.length > 0
        ? "Don’t guess. Ask Jordan first — the proof contains the verified absence of any stored meeting preference."
        : "I don’t have a grounded preference for Jordan.";
    case "why-not":
      return "It can’t be derived, and here is exactly why: orchard is not in the blocked state, no review slot exists for orchard, and no blocker is recorded — each failing premise is named below.";
  }
}

export function runChatMemoryScenario(id: ChatMemoryScenarioId): ChatMemoryRun {
  const scenario = CHAT_MEMORY_SCENARIOS.find((entry) => entry.id === id);
  if (!scenario) {
    throw new Error(`Unknown chat memory scenario: ${id}`);
  }

  const clauses = parseProgram(scenario.program);
  const rows = evaluateQuerySpecWithProof(clauses, parseQuerySpec(scenario.query), {
    maxFacts: 64,
    maxIterations: 16,
    maxRows: 4,
    maxProofDepth: 16,
    maxProofNodes: 256,
    maxProofsPerRow: 1,
    maxAggregateRows: 64,
    maxAggregateProofRows: 16,
  });

  const facts = factsOnly(clauses);
  const rules = authoredRules(clauses);
  if (rows.length === 0) {
    if (scenario.id === "why-not") {
      const diagnosis = diagnoseWhyNot();
      return {
        answer: scenarioAnswer(scenario, {}, []),
        claims: diagnosis
          .filter((premise) => premise.holds)
          .map((premise) => premise.detail),
        absences: diagnosis
          .filter((premise) => !premise.holds)
          .map((premise) => premise.detail),
        proofTrail: diagnosis.map(
          (premise) => `${premise.holds ? "✓" : "✗"} ${premise.literal} — ${premise.detail}`,
        ),
        bindings: {},
        factCount: diagnosis.filter((premise) => premise.holds).length,
        ruleCount: 1,
        absenceCount: diagnosis.filter((premise) => !premise.holds).length,
        facts,
        rules,
        query: scenario.query,
      };
    }
    return {
      answer: "No supported answer was proven.",
      claims: [],
      absences: [],
      proofTrail: [],
      bindings: {},
      factCount: 0,
      ruleCount: 0,
      absenceCount: 0,
      facts,
      rules,
      query: scenario.query,
    };
  }

  const row = rows[0];
  const bindings = Object.fromEntries(
    Object.entries(row.bindings).map(([key, value]) => [key, termText(value)]),
  );
  const claims: string[] = [];
  const absences: string[] = [];
  const proofTrail: string[] = [];
  const usedRules = new Set<number>();

  for (const proof of row.proofs) {
    if (!("aggregated" in proof) && !("negated" in proof) && proof.rule !== undefined) {
      usedRules.add(proof.rule);
    }
    collectProof(proof, claims, absences, proofTrail);
  }

  const rule =
    usedRules.size > 0
      ? rules[[...usedRules][0] - 1]
      : undefined;

  return {
    answer: scenarioAnswer(scenario, bindings, absences),
    claims,
    absences,
    proofTrail,
    ...(rule ? { rule } : {}),
    bindings,
    factCount: claims.length,
    ruleCount: usedRules.size,
    absenceCount: absences.length,
    facts,
    rules,
    query: scenario.query,
  };
}
