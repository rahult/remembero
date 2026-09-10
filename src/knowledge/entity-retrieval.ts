/**
 * Entity-keyed retrieval over the fact store.
 *
 * Lexical search treats a fact as a short document and ranks it by word overlap with the
 * question. That misses the two things a fact store knows and a transcript does not: which
 * entity a fact is about, and which relation it belongs to. "How many model kits have I
 * bought" should reach every `model_kit(user, …)` fact whatever the session said, and "where
 * did Rachel move" should reach every fact that mentions rachel.
 *
 * This is one hop over the graph the facts already form. Seed facts are those whose predicate
 * or constant words overlap the question; each seed then pulls in the facts that share its
 * relation-and-subject or any of its non-self constants. Sessions are scored by how many
 * seed terms they matched directly plus how many related facts they contributed.
 */
import {
  canonicalKey,
  isIntegrityConstraint,
  type Clause,
} from '../engine/index.js';
import type { MemorySource } from '../store/store.js';

export interface EntityRetrievalOptions {
  selfAtom?: string;
  /** Sessions returned at most (default 20). */
  maxSessions?: number;
  /** Related facts pulled per seed fact at most (default 40). */
  maxRelatedPerSeed?: number;
}

export interface EntitySessionHit {
  opId: string;
  ts: string;
  /** Distinct question terms matched by this session's own facts. */
  seedMatches: number;
  /** Facts reached through a shared relation-and-subject or a shared entity. */
  relatedFacts: number;
  score: number;
  facts: string[];
}

const STOPWORDS = new Set(
  'the a an and or of to in on at for with from by as is are was were be been being have has had do does did what which who whom whose when where why how many much i me my mine you your we our us they them their it its this that these those there here about into over after before during since ago last next first time day days week weeks month months year years ever any some all both each per did does done get got give gave go went come came take took make made say said tell told ask asked use used want wanted like liked also just still yet again more most less least very really something anything nothing one two three four five'.split(
    ' ',
  ),
);

/** Lowercase words of three letters or more, with a crude plural fold, minus stopwords. */
export function questionTerms(question: string): Set<string> {
  const terms = new Set<string>();
  for (const raw of question.toLowerCase().split(/[^a-z0-9]+/)) {
    if (raw.length < 3 || STOPWORDS.has(raw)) continue;
    terms.add(raw);
    if (raw.endsWith('ies') && raw.length > 4)
      terms.add(`${raw.slice(0, -3)}y`);
    else if (raw.endsWith('es') && raw.length > 4) terms.add(raw.slice(0, -2));
    if (raw.endsWith('s') && raw.length > 3) terms.add(raw.slice(0, -1));
  }
  return terms;
}

function words(value: string): string[] {
  return value
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((w) => w.length >= 3);
}

interface Fact {
  clause: Clause;
  text: string;
  predicate: string;
  constants: string[];
  subject: string | undefined;
  sources: MemorySource[];
}

function groundFacts(
  clauses: readonly Clause[],
  sources: Map<string, MemorySource[]>,
): Fact[] {
  const out: Fact[] = [];
  for (const clause of clauses) {
    if (clause.body.length > 0 || isIntegrityConstraint(clause)) continue;
    const constants: string[] = [];
    for (const term of clause.head.args) {
      if (term.type === 'atom' || term.type === 'num')
        constants.push(String(term.value));
    }
    const text = `${clause.head.predicate}(${clause.head.args
      .map((t) => {
        if (t.type === 'atom')
          return /^[a-z][a-z0-9_]*$/.test(t.value) ? t.value : `'${t.value}'`;
        if (t.type === 'num') return String(t.value);
        return '_';
      })
      .join(', ')}).`;
    out.push({
      clause,
      text,
      predicate: clause.head.predicate,
      constants,
      subject: constants[0],
      sources: sources.get(canonicalKey(clause)) ?? [],
    });
  }
  return out;
}

export function expandByEntities(
  clauses: readonly Clause[],
  question: string,
  sources: Map<string, MemorySource[]>,
  options: EntityRetrievalOptions = {},
): EntitySessionHit[] {
  const selfAtom = (options.selfAtom ?? 'user').toLowerCase();
  const maxSessions = options.maxSessions ?? 20;
  const maxRelatedPerSeed = options.maxRelatedPerSeed ?? 40;
  const terms = questionTerms(question);
  if (terms.size === 0) return [];
  const facts = groundFacts(clauses, sources);

  // seeds: predicate words or constant words in the question
  const seedTerms = new Map<Fact, Set<string>>();
  for (const fact of facts) {
    const matched = new Set<string>();
    for (const w of words(fact.predicate)) if (terms.has(w)) matched.add(w);
    for (const c of fact.constants) {
      if (c.toLowerCase() === selfAtom) continue;
      for (const w of words(c)) if (terms.has(w)) matched.add(w);
    }
    if (matched.size > 0) seedTerms.set(fact, matched);
  }
  if (seedTerms.size === 0) return [];

  // one hop: same relation about the same subject, or a shared non-self entity
  const related = new Set<Fact>();
  const byRelationSubject = new Map<string, Fact[]>();
  const byEntity = new Map<string, Fact[]>();
  for (const fact of facts) {
    if (fact.subject !== undefined) {
      const key = `${fact.predicate}|${fact.subject.toLowerCase()}`;
      byRelationSubject.set(key, [...(byRelationSubject.get(key) ?? []), fact]);
    }
    for (const c of fact.constants) {
      const entity = c.toLowerCase();
      if (entity === selfAtom || /^\d+(\.\d+)?$/.test(entity)) continue;
      byEntity.set(entity, [...(byEntity.get(entity) ?? []), fact]);
    }
  }
  for (const seed of seedTerms.keys()) {
    let pulled = 0;
    const candidates = [
      ...(seed.subject === undefined
        ? []
        : (byRelationSubject.get(
            `${seed.predicate}|${seed.subject.toLowerCase()}`,
          ) ?? [])),
      ...seed.constants
        .filter((c) => c.toLowerCase() !== selfAtom)
        .flatMap((c) => byEntity.get(c.toLowerCase()) ?? []),
    ];
    for (const fact of candidates) {
      if (fact === seed || seedTerms.has(fact) || related.has(fact)) continue;
      related.add(fact);
      pulled += 1;
      if (pulled >= maxRelatedPerSeed) break;
    }
  }

  // sessions
  const hits = new Map<string, EntitySessionHit & { termSet: Set<string> }>();
  const touch = (fact: Fact, seed: Set<string> | undefined) => {
    const source = fact.sources[0];
    if (source === undefined) return;
    const hit = hits.get(source.opId) ?? {
      opId: source.opId,
      ts: source.ts,
      seedMatches: 0,
      relatedFacts: 0,
      score: 0,
      facts: [],
      termSet: new Set<string>(),
    };
    if (seed !== undefined) for (const t of seed) hit.termSet.add(t);
    else hit.relatedFacts += 1;
    if (hit.facts.length < 24) hit.facts.push(fact.text);
    hits.set(source.opId, hit);
  };
  for (const [fact, matched] of seedTerms) touch(fact, matched);
  for (const fact of related) touch(fact, undefined);

  return [...hits.values()]
    .map(({ termSet, ...hit }) => ({
      ...hit,
      seedMatches: termSet.size,
      score: termSet.size * 3 + hit.relatedFacts,
    }))
    .sort((a, b) => b.score - a.score || a.ts.localeCompare(b.ts))
    .slice(0, maxSessions);
}

/** Alternate lexical and entity-ranked sessions, skipping repeats, up to `limit`. */
export function interleaveSessions(
  lexical: readonly string[],
  entity: readonly string[],
  limit: number,
): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  let li = 0;
  let ei = 0;
  const next = (list: readonly string[], from: number): number => {
    let i = from;
    while (i < list.length && seen.has(list[i]!)) i += 1;
    return i;
  };
  while (out.length < limit) {
    li = next(lexical, li);
    if (li < lexical.length) {
      seen.add(lexical[li]!);
      out.push(lexical[li]!);
      li += 1;
    }
    if (out.length >= limit) break;
    ei = next(entity, ei);
    if (ei < entity.length) {
      seen.add(entity[ei]!);
      out.push(entity[ei]!);
      ei += 1;
    }
    if (li >= lexical.length && ei >= entity.length) break;
  }
  return out;
}
