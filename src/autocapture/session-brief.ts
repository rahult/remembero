import type { Clause } from '../engine/ast.js';
import { serializeClause } from '../engine/index.js';
import type { MemoryStore } from '../store/store.js';

export const DEFAULT_SESSION_BRIEF_BYTES = 4096;
const MIN_SESSION_BRIEF_BYTES = 512;
const MAX_SESSION_BRIEF_BYTES = 16 * 1024;
const SAMPLE_FACTS_PER_PREDICATE = 3;

export interface SessionBriefOptions {
  maxBytes?: number;
}

interface PredicateGroup {
  key: string;
  facts: Clause[];
}

function isFact(clause: Clause): boolean {
  return clause.integrity !== true && clause.aggregate === undefined && clause.body.length === 0;
}

function isRule(clause: Clause): boolean {
  return clause.integrity !== true && (clause.body.length > 0 || clause.aggregate !== undefined);
}

function plural(count: number, noun: string): string {
  return `${count} ${noun}${count === 1 ? '' : 's'}`;
}

/**
 * A deterministic, bounded summary of one namespace for SessionStart context
 * injection. Empty namespaces produce an empty brief so quiet stores add no
 * context; no LLM call is ever involved.
 */
export function buildSessionBrief(
  store: MemoryStore,
  namespace: string,
  options: SessionBriefOptions = {}
): string {
  const maxBytes = options.maxBytes ?? DEFAULT_SESSION_BRIEF_BYTES;
  if (
    !Number.isSafeInteger(maxBytes) ||
    maxBytes < MIN_SESSION_BRIEF_BYTES ||
    maxBytes > MAX_SESSION_BRIEF_BYTES
  ) {
    throw new Error(
      `session brief byte limit must be an integer between ${MIN_SESSION_BRIEF_BYTES} and ${MAX_SESSION_BRIEF_BYTES}`
    );
  }
  const clauses = store.load(namespace);
  if (clauses.length === 0) return '';

  const facts = clauses.filter(isFact);
  const rules = clauses.filter(isRule);
  const constraints = clauses.filter((clause) => clause.integrity === true);

  const groups: PredicateGroup[] = [];
  const byKey = new Map<string, PredicateGroup>();
  for (const fact of facts) {
    const key = `${fact.head.predicate}/${fact.head.args.length}`;
    let group = byKey.get(key);
    if (group === undefined) {
      group = { key, facts: [] };
      byKey.set(key, group);
      groups.push(group);
    }
    group.facts.push(fact);
  }

  const headerParts = [plural(facts.length, 'fact')];
  if (rules.length > 0) headerParts.push(plural(rules.length, 'rule'));
  if (constraints.length > 0) headerParts.push(plural(constraints.length, 'constraint'));
  const header = `Remembero memory — namespace '${namespace}': ${headerParts.join(', ')}.`;
  const footer =
    'Read this memory with the `recall` or `query` MCP tools; store new durable facts with `remember`.';

  const lines: string[] = [header];
  const elisionReserve = Buffer.byteLength(
    `- …and ${groups.length} more predicates — use list_memories for the full set.\n`,
    'utf8'
  );
  let used =
    Buffer.byteLength(header, 'utf8') + 1 + Buffer.byteLength(footer, 'utf8') + elisionReserve;
  let included = 0;
  for (const group of groups) {
    const samples = group.facts
      .slice(0, SAMPLE_FACTS_PER_PREDICATE)
      .map((fact) => serializeClause(fact))
      .join(' ');
    const suffix = group.facts.length > SAMPLE_FACTS_PER_PREDICATE ? ' …' : '';
    const line = `- ${group.key} (${group.facts.length}): ${samples}${suffix}`;
    const bytes = Buffer.byteLength(line, 'utf8') + 1;
    if (used + bytes > maxBytes) break;
    lines.push(line);
    used += bytes;
    included += 1;
  }
  if (included < groups.length) {
    lines.push(
      `- …and ${groups.length - included} more predicates — use list_memories for the full set.`
    );
  }
  lines.push(footer);
  return lines.join('\n');
}
