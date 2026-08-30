import { readFileSync, statSync, writeFileSync } from 'node:fs';
import { serializeClause } from '../engine/index.js';
import type { MemoryStore } from '../store/store.js';
import {
  createKnowledgeBundle,
  MAX_KNOWLEDGE_BUNDLE_BYTES,
  serializeKnowledgeBundle,
  verifyKnowledgeBundle,
  type KnowledgeBundle,
  type KnowledgeBundleVerification,
} from './bundle.js';

export interface RestoreResult {
  sha256: string;
  namespaces: string[];
  clausesAdded: number;
}

/**
 * Disaster-recovery export: one verified content-addressed bundle of every
 * namespace, written to a regular file. The bundle format already carries
 * clause text and durable sources; this wrapper names the workflow.
 */
export function backupKnowledge(
  store: MemoryStore,
  filePath: string
): KnowledgeBundleVerification {
  const bundle = createKnowledgeBundle(store, { namespaces: '*' });
  const text = serializeKnowledgeBundle(bundle);
  writeFileSync(filePath, `${text}\n`, { encoding: 'utf8', mode: 0o600 });
  return verifyKnowledgeBundle(readFileSync(filePath, 'utf8').trimEnd());
}

function readBundleFile(filePath: string): { bundle: KnowledgeBundle; sha256: string } {
  const size = statSync(filePath).size;
  if (size > MAX_KNOWLEDGE_BUNDLE_BYTES + 16) {
    throw new Error(`backup file exceeds ${MAX_KNOWLEDGE_BUNDLE_BYTES} bytes`);
  }
  const text = readFileSync(filePath, 'utf8').trimEnd();
  const verification = verifyKnowledgeBundle(text);
  return { bundle: JSON.parse(text) as KnowledgeBundle, sha256: verification.sha256 };
}

function canonicalClauses(store: MemoryStore, namespace: string): string[] {
  return store
    .load(namespace)
    .map((clause) => serializeClause(clause))
    .sort();
}

/**
 * Restore a verified backup. Each bundle namespace must be empty in the target
 * store, or already contain exactly the bundle's clauses (making retries
 * idempotent); anything else fails before any write. Restored clauses record a
 * restore source; original per-clause provenance stays readable in the backup
 * file itself.
 */
export function restoreKnowledge(store: MemoryStore, filePath: string): RestoreResult {
  const { bundle, sha256 } = readBundleFile(filePath);

  const plans: { namespace: string; clauses: string[] }[] = [];
  for (const entry of bundle.namespaces) {
    const bundleClauses = entry.clauses.map(({ clause }) => clause);
    const existing = canonicalClauses(store, entry.namespace);
    if (existing.length === 0) {
      plans.push({ namespace: entry.namespace, clauses: bundleClauses });
      continue;
    }
    const wanted = [...bundleClauses].sort();
    if (
      existing.length === wanted.length &&
      existing.every((clause, index) => clause === wanted[index])
    ) {
      continue; // already restored — idempotent
    }
    throw new Error(
      `refusing to restore into non-empty namespace '${entry.namespace}'; ` +
        'forget its clauses first or restore into a fresh REMBERO_HOME'
    );
  }

  let clausesAdded = 0;
  for (const plan of plans) {
    const { added } = store.assert(plan.namespace, plan.clauses.join('\n'), {
      sourceText: `Restored from backup ${sha256.slice(0, 12)}`,
    });
    clausesAdded += added.length;
  }
  return {
    sha256,
    namespaces: bundle.namespaces.map(({ namespace }) => namespace),
    clausesAdded,
  };
}
