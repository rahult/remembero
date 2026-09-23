/**
 * Why did each wrong answer go wrong? A cause for every miss, so the next piece of work is chosen
 * by where the misses are rather than by guess.
 *
 * Three layers, cheapest and most certain first:
 *
 *   1. deterministic, from the page labels — no gold page reached the reader (`not_found`), or
 *      some did and some did not (`partly_found`);
 *   2. paired against the same reader's oracle run — if it answers correctly when handed exactly
 *      the gold pages, the miss belongs to retrieval even when a gold page was shown;
 *   3. an LLM label for what is left, given the benchmark's own verbatim evidence quotes: the
 *      reader had the evidence and still missed, and the label says how.
 *
 * Every question also gets a reasoning shape (lookup, comparison, count, arithmetic, date,
 * identity, set difference, multi-hop), which says which misses a Datalog engine could own.
 */

export const READING_CAUSES = [
  'misread_value',
  'wrong_passage',
  'incomplete_multi_part',
  'computation_error',
  'declined_despite_evidence',
  'answered_unanswerable',
  'correct_but_phrased_differently',
  'gold_questionable',
] as const;
export type ReadingCause = (typeof READING_CAUSES)[number];

export const SHAPES = [
  'lookup',
  'comparison',
  'count',
  'arithmetic',
  'date_or_duration',
  'identity',
  'set_difference',
  'multi_hop',
] as const;
export type Shape = (typeof SHAPES)[number];

export type MissCause = 'not_found' | 'partly_found' | 'retrieval_order' | ReadingCause;

export interface MissInput {
  id: string;
  question: string;
  gold: string | null;
  answer: string;
  evidencePages: number[];
  pageRecall: number;
  evidenceHit: boolean;
  /** Did the same reader answer correctly when given only the gold pages? */
  oracleCorrect?: boolean;
}

/** Layers 1 and 2: the causes that need no model. Undefined means "ask the labeller". */
export function deterministicCause(miss: MissInput): MissCause | undefined {
  if (miss.gold === null) return 'answered_unanswerable';
  if (!miss.evidenceHit) return 'not_found';
  if (miss.pageRecall < 1) return 'partly_found';
  // every gold page was in the context, yet the oracle run (only those pages) got it right:
  // the distractor pages around them are what cost the answer
  if (miss.oracleCorrect === true) return 'retrieval_order';
  return undefined;
}

export function buildTaxonomyPrompt(miss: MissInput, quotes: string[]): string {
  return [
    'A reader answered a question about a long document and was judged wrong. Every evidence page was in front of it.',
    'Classify why, and classify the question.',
    '',
    `Question: ${miss.question}`,
    `Gold answer: ${miss.gold}`,
    `Reader's answer: ${miss.answer}`,
    'Gold evidence quotes from the document:',
    ...quotes.map((quote) => `- ${quote.slice(0, 600)}`),
    '',
    'cause, exactly one of:',
    '  misread_value — used the right passage but took the wrong number, name or detail from it',
    '  wrong_passage — answered from a different part of the document than the evidence',
    '  incomplete_multi_part — the question needs several pieces and the answer used only some',
    '  computation_error — had the right pieces but counted, compared, subtracted or ordered them wrongly',
    '  declined_despite_evidence — said it did not know although the evidence answers it',
    '  correct_but_phrased_differently — the answer is actually right; only the wording differs',
    '  gold_questionable — the gold answer looks wrong or the question is ambiguous',
    'shape, exactly one of: lookup, comparison, count, arithmetic, date_or_duration, identity, set_difference, multi_hop',
    '',
    'Reply on two lines exactly:',
    'cause: <cause>',
    'shape: <shape>',
  ].join('\n');
}

export function parseTaxonomyReply(reply: string): { cause: ReadingCause; shape: Shape } {
  const cause = /cause:\s*([a-z_]+)/i.exec(reply)?.[1]?.toLowerCase();
  const shape = /shape:\s*([a-z_]+)/i.exec(reply)?.[1]?.toLowerCase();
  if (!(READING_CAUSES as readonly string[]).includes(cause ?? '')) {
    throw new Error(`taxonomy reply has no valid cause: ${reply.slice(0, 120)}`);
  }
  if (!(SHAPES as readonly string[]).includes(shape ?? '')) {
    throw new Error(`taxonomy reply has no valid shape: ${reply.slice(0, 120)}`);
  }
  return { cause: cause as ReadingCause, shape: shape as Shape };
}

export function buildShapePrompt(question: string): string {
  return [
    'Classify the reasoning this question about a long document needs. Reply with exactly one word from:',
    'lookup, comparison, count, arithmetic, date_or_duration, identity, set_difference, multi_hop',
    '',
    `Question: ${question}`,
  ].join('\n');
}

export function parseShape(reply: string): Shape {
  const word = reply.trim().toLowerCase().match(/[a-z_]+/)?.[0];
  if (!(SHAPES as readonly string[]).includes(word ?? '')) throw new Error(`not a shape: ${reply.slice(0, 80)}`);
  return word as Shape;
}
