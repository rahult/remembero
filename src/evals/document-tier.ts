/**
 * Assemble an exact-page-count tier out of labelled documents.
 *
 * The tiers are 100, 500 and 1000 pages because that is the sweep the platform is being judged
 * on, and no public labelled PDF is exactly those lengths. So a tier concatenates whole documents
 * in a fixed order and truncates the last one to land on the number exactly. Every question's
 * evidence page is shifted by the offset its document landed at; a question whose evidence falls
 * past the truncation is dropped rather than kept and scored against pages that are not there.
 *
 * Concatenating text, not PDFs, keeps this deterministic: page numbers are tier-level positions,
 * so a retrieval's window id and a label's page number mean the same thing.
 */

import type { DocumentPage } from './document-corpus.js';
import type { DocumentQuestion, DocumentTier, TierMember } from './document-recall.js';

/** A labelled document, already extracted to pages, with its own 1-based evidence pages. */
export interface TierSource {
  id: string;
  title: string;
  sourceUrl: string;
  sha256: string;
  pages: DocumentPage[];
  questions: Array<{
    id: string;
    question: string;
    answer: string | null;
    /** Pages within THIS document, 1-based. */
    evidencePages: number[];
    datasetKind?: string;
  }>;
}

export interface AssembledTier {
  tier: DocumentTier;
  /** The tier's pages, renumbered 1..N across the concatenation. */
  pages: DocumentPage[];
  /** Questions left behind because the truncation cut their evidence away. */
  droppedQuestions: Array<{ id: string; reason: string }>;
}

/**
 * Take sources in order until the tier has exactly `targetPages` pages. A source is truncated
 * only if it is the one that crosses the target; sources after the target are not used.
 */
export function assembleTier(
  tierName: string,
  targetPages: number,
  sources: TierSource[],
): AssembledTier {
  if (targetPages <= 0) throw new Error(`a tier needs a positive page target, got ${targetPages}`);

  const pages: DocumentPage[] = [];
  const members: TierMember[] = [];
  const questions: DocumentQuestion[] = [];
  const droppedQuestions: Array<{ id: string; reason: string }> = [];

  for (const source of sources) {
    if (pages.length >= targetPages) break;
    const offset = pages.length;
    const room = targetPages - offset;
    const used = source.pages.slice(0, room);
    for (const page of used) pages.push({ page: pages.length + 1, text: page.text });
    members.push({
      id: source.id,
      title: source.title,
      sourceUrl: source.sourceUrl,
      sha256: source.sha256,
      pages: used.length,
      pageOffset: offset,
    });

    for (const question of source.questions) {
      const beyond = question.evidencePages.filter((page) => page > used.length);
      if (beyond.length > 0) {
        droppedQuestions.push({
          id: question.id,
          reason: `evidence page ${beyond.join(', ')} of ${source.id} falls past the ${used.length}-page truncation`,
        });
        continue;
      }
      questions.push({
        id: question.id,
        question: question.question,
        answer: question.answer,
        evidencePages: question.evidencePages.map((page) => page + offset),
        sourceDocument: source.id,
        datasetKind: question.datasetKind,
      });
    }
  }

  if (pages.length !== targetPages) {
    throw new Error(
      `tier ${tierName} wanted ${targetPages} pages but the sources supplied ${pages.length}`,
    );
  }

  return {
    tier: { tier: tierName, pages: pages.length, members, questions },
    pages,
    droppedQuestions,
  };
}

/**
 * Pick the documents for a tier so the questions are spread through the pages rather than bunched
 * at the front: a tier whose labels all sit in its first 50 pages would flatter retrieval at
 * every size. Sources are ordered so labelled and unlabelled documents alternate.
 */
export function interleaveSources(sources: TierSource[]): TierSource[] {
  const labelled = sources.filter((source) => source.questions.length > 0);
  const filler = sources.filter((source) => source.questions.length === 0);
  const ordered: TierSource[] = [];
  const longest = Math.max(labelled.length, filler.length);
  for (let index = 0; index < longest; index += 1) {
    if (index < labelled.length) ordered.push(labelled[index]!);
    if (index < filler.length) ordered.push(filler[index]!);
  }
  return ordered;
}
