/**
 * The append-only verdict store. Every decision persists its whole input basis — ticket
 * snapshot, pack snapshot, the content hash of every admitted claim, rule versions, verdict,
 * proof — so the same file is simultaneously the compliance audit log, the re-eval corpus
 * that makes engine-swap invariance testable, and the source for the PROV-O projection.
 * Nothing is ever updated or deleted; a re-decision appends.
 */

import { appendFileSync, existsSync, readFileSync } from 'node:fs';
import { hashJson } from './claims.js';
import type { Proof } from './engine.js';

export interface VerdictRecord {
  seq: number;
  at: string;
  /** Hash of the proof minus volatile fields — two runs on the same basis hash equal. */
  decisionHash: string;
  pack: { id: string; version: string; contentHash: string };
  claims: Array<{ id: string; hash: string; predicate: string }>;
  proof: Proof;
}

export class VerdictStore {
  private seq = 0;

  constructor(private readonly path: string) {
    if (existsSync(path)) {
      for (const line of readFileSync(path, 'utf8').split('\n')) {
        if (line.trim().length > 0) this.seq = Math.max(this.seq, (JSON.parse(line) as VerdictRecord).seq);
      }
    }
  }

  append(proof: Proof, pack: { id: string; version: string; contentHash: string }, at = new Date().toISOString()): VerdictRecord {
    // The hash is over the decision's CONTENT — verdict, summary, provenance, computation —
    // never volatile fields (decidedAt) or allocated ids, so two runs on the same basis hash equal.
    const decision = {
      rule: proof.rule,
      pack: proof.pack,
      shape: proof.shape,
      params: proof.params,
      verdict: proof.verdict,
      summary: proof.summary,
      facts: proof.facts.map((f) => [f.predicate, f.args, f.source, f.at ?? null]),
      computation: proof.computation ?? null,
      unknown: proof.unknown ?? null,
    };
    const record: VerdictRecord = {
      seq: this.seq + 1,
      at,
      decisionHash: hashJson(decision),
      pack,
      claims: proof.facts.map((f) => ({
        id: f.id,
        hash: hashJson([f.predicate, f.args, f.source]),
        predicate: f.predicate,
      })),
      proof,
    };
    this.seq += 1;
    appendFileSync(this.path, `${JSON.stringify(record)}\n`);
    return record;
  }

  readAll(): VerdictRecord[] {
    if (!existsSync(this.path)) return [];
    return readFileSync(this.path, 'utf8')
      .split('\n')
      .filter((l) => l.trim().length > 0)
      .map((l) => JSON.parse(l) as VerdictRecord);
  }
}
