import { createHash } from 'node:crypto';
import {
  type AbsenceProof,
  type AggregateProof,
  type Bindings,
  type Clause,
  type DerivationProof,
  type EvaluateOptions,
  type ProofStep,
  type QueryProof,
  type Term,
  canonicalKey,
  evaluateQuerySpecWithProof,
  expandClosurePredicates,
  isIntegrityConstraint,
  parseQueryProgram,
  serializeClause,
  serializeTerm,
} from '../engine/index.js';
import type { MemorySource } from '../store/store.js';
import {
  canonicalizeKnowledge,
  literalKnowledge,
  type EntityAlias,
  type EntityIdentityMode,
  type EntityProjection,
  type EntityRewrite,
  type EntityResolver,
} from './identity.js';
import type { TrustViewMode } from './trust.js';
import {
  selectExplanationGraph,
  type ExplanationGraphSelection,
  type ExplanationGraphSelector,
} from './graph-navigation.js';

export interface SourcedDerivationProof extends Omit<
  DerivationProof,
  'because' | 'aggregate'
> {
  because?: SourcedProofStep[];
  aggregate?: SourcedAggregateProof;
  sources?: MemorySource[];
  /** Additional active namespace witnesses, emitted only during alternative-proof inspection. */
  sourceAlternatives?: MemorySource[];
  /** Present when this claim exists only through an opt-in identity or trust projection. */
  projectedFrom?: string;
  identityRewrites?: EntityRewrite[];
  trust?: 'tentative';
}

export type SourcedAbsenceProof = AbsenceProof;
export type SourcedProofStep = SourcedDerivationProof | SourcedAbsenceProof;

export interface SourcedAggregateProof extends Omit<
  AggregateProof,
  'contributors'
> {
  contributors: Array<{
    bindings: Record<string, string>;
    proofs: SourcedProofStep[];
  }>;
  trust?: 'tentative';
}

export type SourcedQueryProof = SourcedProofStep | SourcedAggregateProof;

export interface ExplainedKnowledgeRow {
  bindings: Record<string, string>;
  proofs: SourcedQueryProof[];
  /** Additional complete proof vectors, ordered after the unchanged first witness. */
  alternativeProofs?: SourcedQueryProof[][];
}

export interface ExplanationRule {
  number: number;
  clause: string;
  projectedFrom?: string;
  identityRewrites?: EntityRewrite[];
}

export interface ResultGraphNode {
  id: string;
  kind: 'result';
  bindings: Record<string, string>;
}

export interface ClaimGraphNode {
  id: string;
  kind: 'claim';
  predicate: string;
  values: (string | number)[];
  derived: boolean;
  rule?: number;
  sources?: MemorySource[];
  sourceAlternatives?: MemorySource[];
  projectedFrom?: string;
  identityRewrites?: EntityRewrite[];
  trust?: 'tentative';
}

export interface EntityGraphNode {
  id: string;
  kind: 'entity';
  value: string | number;
  valueType: 'atom' | 'number';
  /** Explicit aliases that resolve to this canonical atom. */
  aliases?: EntityAlias[];
}

export interface AbsenceGraphNode {
  id: string;
  kind: 'absence';
  predicate: string;
  pattern: (string | number | null)[];
  stratum: number;
}

export interface AggregateGraphNode {
  id: string;
  kind: 'aggregate';
  op: AggregateProof['op'];
  input: '*' | string;
  as: string;
  value: string | number;
  contributorCount: number;
  trust?: 'tentative';
}

export interface ProofGraphNode {
  id: string;
  kind: 'proof';
  predicate: string;
  values: (string | number)[];
  rule?: number;
  sources?: MemorySource[];
  sourceAlternatives?: MemorySource[];
  projectedFrom?: string;
  identityRewrites?: EntityRewrite[];
  trust?: 'tentative';
}

export interface ConflictGraphNode {
  id: string;
  kind: 'conflict';
  focus: string | null;
  violationCount: number;
  constraintIds: string[];
}

export type ExplanationGraphNode =
  | ResultGraphNode
  | ClaimGraphNode
  | EntityGraphNode
  | AbsenceGraphNode
  | AggregateGraphNode
  | ProofGraphNode
  | ConflictGraphNode;

export interface ExplanationGraphEdge {
  id: string;
  kind:
    'answers' | 'because' | 'arg' | 'input' | 'witness' | 'proves' | 'contains';
  from: string;
  to: string;
  position?: number;
  /** One-based index into a row's alternativeProofs; absent for the primary witness. */
  alternative?: number;
}

export interface ExplanationGraph {
  nodes: ExplanationGraphNode[];
  edges: ExplanationGraphEdge[];
}

export interface ExplainKnowledgeResult {
  rows: ExplainedKnowledgeRow[];
  rules: ExplanationRule[];
  graph: ExplanationGraph;
  graphSelection?: ExplanationGraphSelection;
  trustMode?: TrustViewMode;
}

export interface ExplainKnowledgeOptions extends EvaluateOptions {
  entityIdentity?: EntityIdentityMode;
  trustMode?: TrustViewMode;
  graphSelector?: ExplanationGraphSelector;
}

function proofClauseKey(proof: DerivationProof): string {
  const args: Term[] = proof.values.map((value) =>
    typeof value === 'number'
      ? { type: 'num', value }
      : { type: 'atom', value },
  );
  return canonicalKey({ head: { predicate: proof.predicate, args }, body: [] });
}

function isAbsenceProof(
  proof: ProofStep | SourcedProofStep,
): proof is AbsenceProof {
  return 'negated' in proof;
}

function isAggregateProof(
  proof: QueryProof | SourcedQueryProof,
): proof is AggregateProof | SourcedAggregateProof {
  return 'aggregated' in proof;
}

function addAggregateSources(
  proof: AggregateProof,
  sourceIndex: Map<string, MemorySource[]>,
  exactClaims: ReadonlySet<string>,
  projectionIndex: ReadonlyMap<string, EntityProjection[]>,
  includeOtherSources: boolean,
): SourcedAggregateProof {
  const contributors = proof.contributors.map((contributor) => ({
    bindings: bindingStrings(contributor.bindings),
    proofs: contributor.proofs.map((child) =>
      addSources(
        child,
        sourceIndex,
        exactClaims,
        projectionIndex,
        includeOtherSources,
      ),
    ),
  }));
  const tentative = contributors.some((contributor) =>
    contributor.proofs.some(sourcedProofUsesTentative),
  );
  return {
    aggregated: true,
    op: proof.op,
    input: proof.input,
    as: proof.as,
    value: proof.value,
    contributors,
    ...(proof.witnessPositions === undefined
      ? {}
      : { witnessPositions: [...proof.witnessPositions] }),
    ...(tentative ? { trust: 'tentative' as const } : {}),
  };
}

function sourcedProofUsesTentative(proof: SourcedProofStep): boolean {
  if (isAbsenceProof(proof)) return false;
  return (
    proof.trust === 'tentative' ||
    (proof.because ?? []).some(sourcedProofUsesTentative) ||
    proof.aggregate?.trust === 'tentative'
  );
}

function addSources(
  proof: ProofStep,
  sourceIndex: Map<string, MemorySource[]>,
  exactClaims: ReadonlySet<string>,
  projectionIndex: ReadonlyMap<string, EntityProjection[]>,
  includeOtherSources = false,
): SourcedProofStep {
  if (isAbsenceProof(proof)) return { ...proof, pattern: [...proof.pattern] };
  const key = proofClauseKey(proof);
  const sources = proof.rule === undefined ? sourceIndex.get(key) : undefined;
  const exactSources = sources?.filter(
    (source) => source.projectedFrom === undefined,
  );
  const witnessSources =
    exactSources !== undefined && exactSources.length > 0
      ? exactSources.slice(0, 1)
      : exactClaims.has(key)
        ? undefined
        : sources?.slice(0, 1);
  const sourceProjection =
    witnessSources?.[0]?.projectedFrom === undefined
      ? undefined
      : {
          projectedFrom: witnessSources[0].projectedFrom,
          identityRewrites: witnessSources[0].identityRewrites ?? [],
          ...(witnessSources[0].trust === undefined
            ? {}
            : { trust: witnessSources[0].trust }),
        };
  const projection =
    sourceProjection ??
    (!exactClaims.has(key) ? projectionIndex.get(key)?.[0] : undefined);
  const alternativeSources =
    witnessSources === undefined ? sources : sources?.slice(1);
  const because = proof.because?.map((child) =>
    addSources(
      child,
      sourceIndex,
      exactClaims,
      projectionIndex,
      includeOtherSources,
    ),
  );
  const aggregate =
    proof.aggregate === undefined
      ? undefined
      : addAggregateSources(
          proof.aggregate,
          sourceIndex,
          exactClaims,
          projectionIndex,
          includeOtherSources,
        );
  const trust =
    projection?.trust === 'tentative' ||
    (because ?? []).some(sourcedProofUsesTentative) ||
    aggregate?.trust === 'tentative'
      ? ('tentative' as const)
      : undefined;
  return {
    predicate: proof.predicate,
    values: proof.values,
    ...(proof.rule === undefined ? {} : { rule: proof.rule }),
    ...(because === undefined ? {} : { because }),
    ...(aggregate === undefined ? {} : { aggregate }),
    ...(witnessSources === undefined || witnessSources.length === 0
      ? {}
      : { sources: witnessSources }),
    ...(projection === undefined ? {} : projection),
    ...(trust === undefined ? {} : { trust }),
    ...(!includeOtherSources ||
    alternativeSources === undefined ||
    alternativeSources.length === 0
      ? {}
      : { sourceAlternatives: alternativeSources }),
  };
}

function addQuerySources(
  proof: QueryProof,
  sourceIndex: Map<string, MemorySource[]>,
  exactClaims: ReadonlySet<string>,
  projectionIndex: ReadonlyMap<string, EntityProjection[]>,
  includeOtherSources = false,
): SourcedQueryProof {
  if (!isAggregateProof(proof)) {
    return addSources(
      proof,
      sourceIndex,
      exactClaims,
      projectionIndex,
      includeOtherSources,
    );
  }
  return addAggregateSources(
    proof,
    sourceIndex,
    exactClaims,
    projectionIndex,
    includeOtherSources,
  );
}

function bindingStrings(bindings: Bindings): Record<string, string> {
  return Object.fromEntries(
    Object.entries(bindings)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([name, term]) => [name, serializeTerm(term)]),
  );
}

function typedValue(
  value: string | number,
): ['atom' | 'number', string | number] {
  return [typeof value === 'number' ? 'number' : 'atom', value];
}

function entityId(value: string | number): string {
  return `entity:${JSON.stringify(typedValue(value))}`;
}

function claimId(proof: SourcedDerivationProof): string {
  return `claim:${JSON.stringify([
    proof.predicate,
    proof.values.map((value) => typedValue(value)),
  ])}`;
}

function absenceId(proof: SourcedAbsenceProof): string {
  return `absence:${JSON.stringify([
    proof.predicate,
    proof.pattern.map((value) =>
      value === null ? ['wildcard'] : typedValue(value),
    ),
    proof.stratum,
  ])}`;
}

function aggregateId(proof: SourcedAggregateProof, scope?: string): string {
  const hash = createHash('sha256');
  hash.update(
    JSON.stringify([
      scope ?? null,
      proof.op,
      proof.input,
      proof.as,
      typedValue(proof.value),
    ]),
  );
  for (const contributor of proof.contributors) {
    hash.update(JSON.stringify(Object.entries(contributor.bindings)));
    hash.update(
      JSON.stringify(
        contributor.proofs.map((child) =>
          isAbsenceProof(child) ? absenceId(child) : claimId(child),
        ),
      ),
    );
  }
  return `aggregate:${proof.op}:${hash.digest('hex')}`;
}

function aggregateStructure(proof: SourcedAggregateProof): unknown {
  return [
    proof.op,
    proof.input,
    proof.as,
    typedValue(proof.value),
    proof.contributors.map((contributor) => [
      Object.entries(contributor.bindings),
      contributor.proofs.map(proofStructure),
    ]),
    proof.witnessPositions ?? null,
    proof.trust ?? null,
  ];
}

function proofStructure(proof: SourcedProofStep): unknown {
  if (isAbsenceProof(proof)) {
    return ['absence', proof.predicate, proof.pattern, proof.stratum];
  }
  return [
    'claim',
    proof.predicate,
    proof.values.map((value) => typedValue(value)),
    proof.rule ?? null,
    (proof.because ?? []).map(proofStructure),
    proof.aggregate === undefined ? null : aggregateStructure(proof.aggregate),
    proof.trust ?? null,
  ];
}

function proofId(proof: SourcedDerivationProof): string {
  return `proof:${createHash('sha256')
    .update(JSON.stringify(proofStructure(proof)))
    .digest('hex')}`;
}

function resultId(bindings: Record<string, string>): string {
  return `result:${JSON.stringify(Object.entries(bindings))}`;
}

function contributorResultId(
  aggregate: string,
  position: number,
  bindings: Record<string, string>,
): string {
  return `result:input:${JSON.stringify([aggregate, position, Object.entries(bindings)])}`;
}

function edge(
  kind: ExplanationGraphEdge['kind'],
  from: string,
  to: string,
  position?: number,
  alternative?: number,
): ExplanationGraphEdge {
  return {
    id: `edge:${JSON.stringify([kind, from, to, position ?? null, alternative ?? null])}`,
    kind,
    from,
    to,
    ...(position === undefined ? {} : { position }),
    ...(alternative === undefined ? {} : { alternative }),
  };
}

export function buildExplanationGraph(
  rows: ExplainedKnowledgeRow[],
  resolver?: EntityResolver,
): ExplanationGraph {
  const nodes = new Map<string, ExplanationGraphNode>();
  const edges = new Map<string, ExplanationGraphEdge>();
  const expandedProofs = rows.some(
    (row) => (row.alternativeProofs?.length ?? 0) > 0,
  );

  const addEdge = (value: ExplanationGraphEdge) => edges.set(value.id, value);
  const addEntity = (
    value: string | number,
    predicate: string,
    arity: number,
    position: number,
  ): string => {
    const id = entityId(value);
    const declaredAliases =
      typeof value === 'string' &&
      resolver?.isEntityPosition(predicate, arity, position)
        ? resolver.aliasesFor(value)
        : undefined;
    const existing = nodes.get(id);
    const existingAliases =
      existing?.kind === 'entity' ? existing.aliases : undefined;
    const aliases = [
      ...(existingAliases ?? []),
      ...(declaredAliases ?? []),
    ].filter(
      (alias, index, all) =>
        all.findIndex(
          (candidate) =>
            candidate.alias === alias.alias &&
            candidate.target === alias.target &&
            candidate.canonical === alias.canonical,
        ) === index,
    );
    nodes.set(id, {
      id,
      kind: 'entity',
      value,
      valueType: typeof value === 'number' ? 'number' : 'atom',
      ...(aliases.length === 0 ? {} : { aliases }),
    });
    return id;
  };
  const addProof = (proof: SourcedProofStep): string => {
    if (isAbsenceProof(proof)) {
      const id = absenceId(proof);
      nodes.set(id, {
        id,
        kind: 'absence',
        predicate: proof.predicate,
        pattern: proof.pattern,
        stratum: proof.stratum,
      });
      for (const [position, value] of proof.pattern.entries()) {
        if (value === null) continue;
        const target = addEntity(
          value,
          proof.predicate,
          proof.pattern.length,
          position,
        );
        addEdge(edge('arg', id, target, position));
      }
      return id;
    }
    const id = claimId(proof);
    nodes.set(id, {
      id,
      kind: 'claim',
      predicate: proof.predicate,
      values: proof.values,
      derived: proof.rule !== undefined,
      ...(proof.rule === undefined ? {} : { rule: proof.rule }),
      ...(proof.sources === undefined ? {} : { sources: proof.sources }),
      ...(proof.sourceAlternatives === undefined
        ? {}
        : { sourceAlternatives: proof.sourceAlternatives }),
      ...(proof.projectedFrom === undefined
        ? {}
        : { projectedFrom: proof.projectedFrom }),
      ...(proof.identityRewrites === undefined
        ? {}
        : { identityRewrites: proof.identityRewrites }),
      ...(proof.trust === undefined ? {} : { trust: proof.trust }),
    });
    for (const [position, value] of proof.values.entries()) {
      const target = addEntity(
        value,
        proof.predicate,
        proof.values.length,
        position,
      );
      addEdge(edge('arg', id, target, position));
    }
    for (const [position, child] of (proof.because ?? []).entries()) {
      const target = addProof(child);
      addEdge(edge('because', id, target, position));
    }
    if (proof.aggregate !== undefined) {
      addEdge(
        edge(
          'because',
          id,
          addAggregate(proof.aggregate, id),
          proof.because?.length ?? 0,
        ),
      );
    }
    return id;
  };

  const addProofInstance = (proof: SourcedProofStep): string => {
    if (isAbsenceProof(proof)) return addProof(proof);
    const claim = claimId(proof);
    if (!nodes.has(claim)) {
      nodes.set(claim, {
        id: claim,
        kind: 'claim',
        predicate: proof.predicate,
        values: proof.values,
        derived: proof.rule !== undefined,
        ...(proof.rule === undefined ? {} : { rule: proof.rule }),
        ...(proof.sources === undefined ? {} : { sources: proof.sources }),
        ...(proof.sourceAlternatives === undefined
          ? {}
          : { sourceAlternatives: proof.sourceAlternatives }),
        ...(proof.projectedFrom === undefined
          ? {}
          : { projectedFrom: proof.projectedFrom }),
        ...(proof.identityRewrites === undefined
          ? {}
          : { identityRewrites: proof.identityRewrites }),
        ...(proof.trust === undefined ? {} : { trust: proof.trust }),
      });
      for (const [position, value] of proof.values.entries()) {
        const target = addEntity(
          value,
          proof.predicate,
          proof.values.length,
          position,
        );
        addEdge(edge('arg', claim, target, position));
      }
    }

    const id = proofId(proof);
    nodes.set(id, {
      id,
      kind: 'proof',
      predicate: proof.predicate,
      values: proof.values,
      ...(proof.rule === undefined ? {} : { rule: proof.rule }),
      ...(proof.sources === undefined ? {} : { sources: proof.sources }),
      ...(proof.sourceAlternatives === undefined
        ? {}
        : { sourceAlternatives: proof.sourceAlternatives }),
      ...(proof.projectedFrom === undefined
        ? {}
        : { projectedFrom: proof.projectedFrom }),
      ...(proof.identityRewrites === undefined
        ? {}
        : { identityRewrites: proof.identityRewrites }),
      ...(proof.trust === undefined ? {} : { trust: proof.trust }),
    });
    addEdge(edge('proves', id, claim));
    for (const [position, child] of (proof.because ?? []).entries()) {
      addEdge(edge('because', id, addProofInstance(child), position));
    }
    if (proof.aggregate !== undefined) {
      addEdge(
        edge(
          'because',
          id,
          addAggregate(proof.aggregate, id),
          proof.because?.length ?? 0,
        ),
      );
    }
    return id;
  };

  const addAggregate = (
    proof: SourcedAggregateProof,
    scope?: string,
  ): string => {
    const id = aggregateId(proof, scope);
    nodes.set(id, {
      id,
      kind: 'aggregate',
      op: proof.op,
      input: proof.input,
      as: proof.as,
      value: proof.value,
      contributorCount: proof.contributors.length,
      ...(proof.trust === undefined ? {} : { trust: proof.trust }),
    });
    const witnesses = new Set(proof.witnessPositions ?? []);
    for (const [position, contributor] of proof.contributors.entries()) {
      const contributorId = contributorResultId(
        id,
        position,
        contributor.bindings,
      );
      nodes.set(contributorId, {
        id: contributorId,
        kind: 'result',
        bindings: contributor.bindings,
      });
      for (const [proofPosition, child] of contributor.proofs.entries()) {
        addEdge(edge('answers', contributorId, addProof(child), proofPosition));
      }
      addEdge(edge('input', id, contributorId, position));
      if (witnesses.has(position)) {
        addEdge(edge('witness', id, contributorId, position));
      }
    }
    return id;
  };

  for (const row of rows) {
    const id = resultId(row.bindings);
    nodes.set(id, { id, kind: 'result', bindings: row.bindings });
    for (const [position, proof] of row.proofs.entries()) {
      addEdge(
        edge(
          'answers',
          id,
          isAggregateProof(proof)
            ? addAggregate(proof)
            : expandedProofs
              ? addProofInstance(proof)
              : addProof(proof),
          position,
        ),
      );
    }
    for (const [alternativeIndex, alternative] of (
      row.alternativeProofs ?? []
    ).entries()) {
      for (const [position, proof] of alternative.entries()) {
        if (isAggregateProof(proof)) continue;
        addEdge(
          edge(
            'answers',
            id,
            addProofInstance(proof),
            position,
            alternativeIndex + 1,
          ),
        );
      }
    }
  }

  return {
    nodes: [...nodes.values()].sort((left, right) =>
      left.id.localeCompare(right.id),
    ),
    edges: [...edges.values()].sort((left, right) =>
      left.id.localeCompare(right.id),
    ),
  };
}

export function explainKnowledge(
  clauses: Clause[],
  query: string,
  sourceIndex: Map<string, MemorySource[]> = new Map(),
  options: ExplainKnowledgeOptions = {},
): ExplainKnowledgeResult {
  const { entityIdentity, trustMode, graphSelector, ...evaluateOptions } =
    options;
  const view =
    entityIdentity === 'canonical'
      ? canonicalizeKnowledge(clauses, sourceIndex, trustMode)
      : literalKnowledge(clauses, sourceIndex, trustMode);
  // Same normalizer as the query tool: goal list, `?-`, or a rule program whose
  // sink rule is the target. Authored rules join the view so proofs and the
  // rule catalog can reference them.
  const program = parseQueryProgram(query);
  const querySpec =
    entityIdentity === 'canonical'
      ? view.resolver.canonicalizeQuery(program.query).query
      : program.query;
  const viewClauses = [...view.clauses, ...program.clauses];
  const explained = evaluateQuerySpecWithProof(
    viewClauses,
    querySpec,
    evaluateOptions,
  );
  const includeAlternatives = (evaluateOptions.maxProofsPerRow ?? 1) > 1;
  const rows = explained.map(({ bindings, proofs, alternativeProofs }) => {
    const serializedBindings = bindingStrings(bindings);
    const sourcedProofs = proofs.map((proof) =>
      addQuerySources(
        proof,
        view.sources,
        view.exactClaims,
        view.projections,
        includeAlternatives,
      ),
    );
    const sourcedAlternatives = alternativeProofs?.map((alternative) =>
      alternative.map((proof) =>
        addQuerySources(
          proof,
          view.sources,
          view.exactClaims,
          view.projections,
          true,
        ),
      ),
    );
    return {
      bindings: serializedBindings,
      proofs: sourcedProofs,
      ...(sourcedAlternatives === undefined || sourcedAlternatives.length === 0
        ? {}
        : {
            alternativeProofs: sourcedAlternatives,
          }),
    };
  });
  const result: ExplainKnowledgeResult = {
    rows,
    // Synthesized closure rules (p_plus) are numbered after the authored rules
    // by the evaluator, so the catalog must include them for proofs to resolve.
    rules: expandClosurePredicates(viewClauses, querySpec.goals)
      .filter(
        (clause) => clause.body.length > 0 && !isIntegrityConstraint(clause),
      )
      .map((clause, index) => ({
        number: index + 1,
        clause: serializeClause(clause),
        ...(view.projections.get(canonicalKey(clause))?.[0] ?? {}),
      })),
    graph: buildExplanationGraph(
      rows,
      entityIdentity === 'canonical' ? view.resolver : undefined,
    ),
    ...(trustMode === undefined || trustMode === 'accepted'
      ? {}
      : { trustMode }),
  };
  return graphSelector === undefined
    ? result
    : selectExplanationGraph(result, graphSelector);
}
