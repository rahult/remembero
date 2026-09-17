/**
 * The five answering flags of `knowledge/question-kind.ts` read by a model instead of by
 * rules: ONE TypeSafe ("System One") request per question carrying five nouls, one per flag,
 * evaluated independently over the same state (https://docs.typesafe.ai). Jev is literal, so
 * each instruction states the exact condition and nothing else. The client, its disk cache,
 * its limiter and its retries are typesafe-rerank.ts's.
 */
import {
  DEFAULT_QUESTION_KIND_THRESHOLD,
  QUESTION_KIND_FLAGS,
  type QuestionKind,
} from '../knowledge/question-kind.js';
import { assertSafeForExternalLlm } from '../safety.js';
import type { TypesafeNouls, TypesafeQuestion } from './typesafe-rerank.js';

export { DEFAULT_QUESTION_KIND_THRESHOLD };

/** One noul per flag; the wording is the condition, because Jev answers it literally. */
export const QUESTION_KIND_QUESTIONS = {
  aggregation: {
    type: 'noul',
    instructions:
      'The question asks for a count, a sum, a total, a maximum or a comparison that has to be gathered from several separate things the user said at different times, not read off one single statement.',
  },
  temporal: {
    type: 'noul',
    instructions:
      'The question asks about time: a date, how long something took, how much time passed between two events, the order in which events happened, or it anchors what it asks about to a named past day, month or period.',
  },
  update: {
    type: 'noul',
    instructions:
      'The question asks for the value that holds now of something about the user that has changed over time, so an earlier statement about it is out of date.',
  },
  preference: {
    type: 'noul',
    instructions:
      'The question asks the assistant for a recommendation, a suggestion, advice or an opinion that should be personalised to the user, rather than asking for a fact from the past.',
  },
  assistantRecall: {
    type: 'noul',
    instructions:
      'The question asks what the assistant said, recommended or explained in the past, not what the user said.',
  },
} as const satisfies Record<keyof QuestionKind, TypesafeQuestion>;

export interface TypesafeQuestionKindResult {
  kind: QuestionKind;
  /** The probability each noul came back with, before the threshold. */
  nouls: Record<string, number>;
  /** The provider's input_tokens (reported for a cached answer too, which costs nothing). */
  inputTokens: number;
  cached: boolean;
}

/**
 * One request, five nouls, one threshold. Throws when the question cannot be sent (it fails
 * the external-LLM safety check) or when the request fails after its retries; the caller
 * falls back to the text rules and records that it did.
 */
export async function typesafeQuestionKind(
  question: string,
  nouls: TypesafeNouls,
  options: { threshold?: number } = {},
): Promise<TypesafeQuestionKindResult> {
  const threshold = options.threshold ?? DEFAULT_QUESTION_KIND_THRESHOLD;
  if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) {
    throw new Error('question-kind threshold must be from 0 to 1');
  }
  const state = { question };
  assertSafeForExternalLlm(JSON.stringify(state), 'TypeSafe question kind');
  const answer = await nouls(state, QUESTION_KIND_QUESTIONS);
  const kind = Object.fromEntries(
    QUESTION_KIND_FLAGS.map((flag) => [flag, answer.nouls[flag]! >= threshold]),
  ) as unknown as QuestionKind;
  return {
    kind,
    nouls: answer.nouls,
    inputTokens: answer.inputTokens,
    cached: answer.cached,
  };
}
