/**
 * Real transcripts as extraction training data, and a direct measure of extraction quality.
 *
 * The synthetic worlds render one or two sentences per turn; real sessions are paragraphs with
 * the memorable fact buried in an aside, and two fine-tunes that tie on the synthetic
 * benchmarks differed by eight LongMemEval answers. So: label real sessions with the frontier
 * model through the product's own extraction path (guards included), train on those, and
 * score the small model's extraction against the same labels.
 *
 * Sessions come from LongMemEval-S haystacks that appear in no development-split question,
 * are the evidence for no question, and repeat no development-split text, so nothing here can
 * leak into the development evaluation. (They are filler in test-split haystacks; a future
 * test-split claim must say so.)
 */
import {
  NOTHING_SENTINEL,
  transcriptExtractionSystemPrompt,
} from '../llm/prompts.js';
import type { LongMemEvalInstance } from '../evals/longmemeval.js';
import { longMemEvalTranscript } from '../evals/longmemeval-answer.js';

export type Session = LongMemEvalInstance['haystack_sessions'][number];

export interface SelectedSession {
  id: string;
  date: string;
  session: Session;
}

/** Cut each assistant turn to this many characters, as the LongMemEval evaluation does. */
export const REAL_ASSISTANT_CHARACTERS = 600;
export const REAL_TRANSCRIPT_CHARACTERS = 16_000;

function textSignature(session: Session): string {
  return session.map((turn) => turn.content.slice(0, 200)).join('|');
}

/**
 * Sessions safe to train on for a development-split evaluation: not in any development
 * haystack, evidence for no question, and not a textual duplicate of a development session.
 */
export function selectTrainingSessions(
  instances: readonly LongMemEvalInstance[],
  split: (questionId: string) => 'dev' | 'test',
): SelectedSession[] {
  const devIds = new Set<string>();
  const devSignatures = new Set<string>();
  const evidence = new Set<string>();
  for (const instance of instances) {
    for (const id of instance.answer_session_ids) evidence.add(id);
    if (split(instance.question_id) !== 'dev') continue;
    instance.haystack_session_ids.forEach((id, index) => {
      devIds.add(id);
      devSignatures.add(textSignature(instance.haystack_sessions[index]!));
    });
  }
  const chosen = new Map<string, SelectedSession>();
  for (const instance of instances) {
    if (split(instance.question_id) !== 'test') continue;
    instance.haystack_session_ids.forEach((id, index) => {
      if (devIds.has(id) || evidence.has(id) || chosen.has(id)) return;
      const session = instance.haystack_sessions[index]!;
      if (devSignatures.has(textSignature(session))) return;
      chosen.set(id, { id, date: instance.haystack_dates[index]!, session });
    });
  }
  return [...chosen.values()];
}

/** The transcript as the extractor sees it in the evaluation. */
export function realTranscript(session: Session): string {
  return longMemEvalTranscript(session, {
    assistantCharacters: REAL_ASSISTANT_CHARACTERS,
  }).slice(0, REAL_TRANSCRIPT_CHARACTERS);
}

function constantsOf(fact: string, selfAtom: string): Set<string> {
  const inside = fact.slice(fact.indexOf('(') + 1, fact.lastIndexOf(')'));
  const out = new Set<string>();
  for (const raw of inside.split(',')) {
    const value = raw
      .trim()
      .replace(/^'|'$/g, '')
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '_')
      .replace(/^_|_$/g, '');
    if (value && value !== selfAtom.toLowerCase()) out.add(value);
  }
  return out;
}

export interface FactSetComparison {
  referenceFacts: number;
  modelFacts: number;
  matchedReference: number;
  matchedModel: number;
  recall: number;
  precision: number;
}

/**
 * Loose agreement between two extractions of one session: a fact matches when it shares at
 * least one non-self constant with a fact on the other side. Predicate names are ignored,
 * because two extractors legitimately name the same relation differently.
 */
export function compareFactSets(
  reference: readonly string[],
  model: readonly string[],
  selfAtom: string,
): FactSetComparison {
  const ref = reference.map((f) => constantsOf(f, selfAtom));
  const mod = model.map((f) => constantsOf(f, selfAtom));
  const shares = (a: Set<string>, b: Set<string>) =>
    [...a].some((c) => b.has(c));
  const matchedReference = ref.filter((r) =>
    mod.some((m) => shares(r, m)),
  ).length;
  const matchedModel = mod.filter((m) => ref.some((r) => shares(r, m))).length;
  return {
    referenceFacts: reference.length,
    modelFacts: model.length,
    matchedReference,
    matchedModel,
    recall:
      reference.length === 0
        ? model.length === 0
          ? 1
          : 0
        : matchedReference / reference.length,
    precision:
      model.length === 0
        ? reference.length === 0
          ? 1
          : 0
        : matchedModel / model.length,
  };
}

/** One training conversation: the product transcript prompt over an empty store, the compacted transcript, the facts. */
export function toRealConversation(
  session: Session,
  facts: readonly string[],
  selfAtom: string,
): {
  messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }>;
} {
  return {
    messages: [
      {
        role: 'system',
        content: transcriptExtractionSystemPrompt(
          '% (no memories yet)',
          selfAtom,
        ),
      },
      { role: 'user', content: realTranscript(session) },
      {
        role: 'assistant',
        content: facts.length === 0 ? NOTHING_SENTINEL : facts.join('\n'),
      },
    ],
  };
}
