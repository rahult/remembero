/**
 * Structured evidence: the memory's own facts about the question, made readable by code.
 *
 * The writer extracted the facts at write time (model output). Before a reader sees them,
 * deterministic rules date each fact (by a temporal expression inside it, resolved against the
 * session it came from, else by the session date), ground it against the session text it was
 * extracted from (a fact whose content words never appear there is dropped), deduplicate, and
 * mark the later of two conflicting values of the same claim as current. The result goes before
 * the chats, so the reader composes from dated claims and only then checks the wording.
 */

import { questionKeywords, resolveTemporalExpressions } from './computed-notes.js';

const STOP = new Set('the a an and or of to in on at for with by from as is are was were be been being i me my mine we our you your it its this that these those have has had do does did not no yes now then there their they them his her he she him about into over than also just very really some any all each every both more most other such only own same so too can will would should could'.split(' '));

export interface EvidenceSource {
  ts: string;
  text: string;
  /** Facts the writer extracted from this session that matched the question. */
  facts?: string[];
}

interface DatedFact {
  text: string;
  iso: string;
  how: 'stated' | 'session';
  sessionDay: string;
  key: string;
}

function dayOf(ts: string): string {
  const m = /(\d{4})[/-](\d{2})[/-](\d{2})/.exec(ts);
  return m ? `${m[1]}-${m[2]}-${m[3]}` : ts.slice(0, 10);
}

function contentWords(text: string): string[] {
  return [...new Set(text.toLowerCase().replace(/[^a-z0-9 ]/g, ' ').split(/\s+/).filter((w) => w.length >= 3 && !STOP.has(w) && !/^\d+$/.test(w)))];
}

/** A fact is grounded when its content words appear in the session it was extracted from. */
function grounded(fact: string, sessionText: string): boolean {
  const words = contentWords(fact);
  if (words.length === 0) return false;
  const low = sessionText.toLowerCase();
  const hits = words.filter((w) => low.includes(w.endsWith('s') ? w.slice(0, -1) : w)).length;
  return hits >= Math.min(2, words.length);
}

/** The claim a fact makes, minus its value: the words that are not numbers or capitalised names. */
function claimKey(fact: string): string {
  const words = fact.toLowerCase().replace(/[^a-z0-9 ]/g, ' ').split(/\s+/).filter((w) => w.length >= 3 && !STOP.has(w) && !/^\d+$/.test(w));
  const proper = new Set(fact.split(/\s+/).filter((w) => /^[A-Z]/.test(w)).map((w) => w.toLowerCase().replace(/[^a-z0-9]/g, '')));
  return words.filter((w) => !proper.has(w)).slice(0, 3).join(' ');
}

export function buildStructuredEvidence(question: string, questionDate: string, sources: EvidenceSource[], limits: { maxFacts?: number } = {}): string {
  const maxFacts = limits.maxFacts ?? 16;
  const keywords = questionKeywords(question);
  const facts: DatedFact[] = [];
  const seen = new Set<string>();
  for (const source of [...sources].sort((l, r) => l.ts.localeCompare(r.ts))) {
    for (const fact of source.facts ?? []) {
      const norm = fact.toLowerCase().replace(/\s+/g, ' ').trim();
      if (seen.has(norm)) continue;
      if (!grounded(fact, source.text)) continue;
      // relevance: the fact shares a content word with the question, or nothing in the question is specific enough to check
      if (keywords.length > 0 && !keywords.some((k) => norm.includes(k))) continue;
      seen.add(norm);
      const sessionDay = dayOf(source.ts);
      const stated = resolveTemporalExpressions(`USER: ${fact}`, source.ts).find((e) => e.kind !== 'absolute' || true);
      facts.push({ text: fact, iso: stated ? stated.iso : sessionDay, how: stated ? 'stated' : 'session', sessionDay, key: claimKey(fact) });
    }
  }
  if (facts.length === 0) return '';
  facts.sort((l, r) => l.iso.localeCompare(r.iso));
  // supersession: same claim key, different text → the latest is current, the rest superseded
  const byKey = new Map<string, DatedFact[]>();
  for (const f of facts) byKey.set(f.key, [...(byKey.get(f.key) ?? []), f]);
  const status = new Map<DatedFact, string>();
  for (const group of byKey.values()) {
    if (group.length < 2 || group[0]!.key === '') continue;
    const last = group[group.length - 1]!;
    for (const f of group) status.set(f, f === last ? 'current' : 'superseded');
  }
  const shown = facts.slice(0, maxFacts);
  const lines = ['### Dated facts (structured evidence: the memory\'s own facts about the question, each dated by the session it was said in or by a date inside it; where the same claim has several values the latest is marked current)'];
  for (const f of shown) {
    const when = f.how === 'stated' ? `${f.iso} (stated in the fact; said ${f.sessionDay})` : `${f.iso} (session date)`;
    const tag = status.get(f);
    lines.push(`- ${when}: ${f.text}${tag ? ` [${tag}]` : ''}`);
  }
  if (facts.length > shown.length) lines.push(`- ${facts.length - shown.length} more dated facts not shown`);
  return lines.join('\n') + '\n';
}
