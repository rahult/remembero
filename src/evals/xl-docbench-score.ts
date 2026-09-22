/**
 * XL-DocBench's own scorer, ported.
 *
 * The benchmark ships `evaluate.py` (490 lines of stdlib, no model calls) and its published
 * baselines — SimpleDoc+GPT-5.4 44.0%, Claude Opus 4.6 40.5% — are computed with it. A judge
 * model of our own would produce numbers that cannot be compared to those, so the primary metric
 * here is their rule, ported line for line: relaxed accuracy by answer type, token F1 and ANLS.
 *
 * The port is checked against the original rather than trusted: the runner writes a predictions
 * file in the shape `evaluate.py` reads, so the same run can be scored by both and the two
 * accuracies compared (see `benchmarks/document-recall/README.md`).
 */

/** Their ANSWER_FORMAT_MAP, plus the verification-rule fallback in `answer_format`. */
export type XlAnswerType = 'entity' | 'numeric' | 'percentage' | 'boolean' | 'single_choice' | 'unanswerable';

const FORMAT_MAP: Record<string, XlAnswerType> = {
  Str: 'entity',
  Int: 'numeric',
  Float: 'numeric',
  None: 'unanswerable',
  Bool: 'boolean',
  Boolean: 'boolean',
  Percentage: 'percentage',
};

/** The answer type this question is scored as, from its format and verification rule. */
export function xlAnswerType(format: string | undefined, verificationRule?: string): XlAnswerType {
  const raw = format === undefined || format === '' ? 'Str' : format;
  const mapped = FORMAT_MAP[raw];
  if (mapped !== undefined) return mapped;
  const rule = verificationRule ?? '';
  if (rule.includes('numeric') || rule.includes('tolerance')) return 'numeric';
  if (rule.includes('choice')) return 'single_choice';
  return raw.toLowerCase() as XlAnswerType;
}

/** `normalize_answer`: strip an answer preamble, punctuation and articles, fold case. */
export function normalizeAnswer(text: string): string {
  let value = text.trim().toLowerCase();
  for (const prefix of ['the answer is', 'answer:', 'answer is']) {
    if (value.startsWith(prefix)) {
      value = value.slice(prefix.length).trim();
    }
  }
  value = value.replace(/[^\w\s.\-%]/gu, '');
  value = value.replace(/\b(a|an|the)\b/g, ' ');
  return value.replace(/\s+/g, ' ').trim();
}

/** `extract_number`: the first number in the text, commas and spaces removed first. */
export function extractNumber(text: string): number | undefined {
  const stripped = text.replace(/,/g, '').replace(/ /g, '');
  const match = /[-+]?\d*\.?\d+/.exec(stripped);
  if (match === null) return undefined;
  const value = Number(match[0]);
  return Number.isFinite(value) ? value : undefined;
}

/** Plain Levenshtein distance, as their implementation computes it. */
export function levenshteinDistance(left: string, right: string): number {
  if (left.length < right.length) return levenshteinDistance(right, left);
  if (right.length === 0) return left.length;
  let previous = Array.from({ length: right.length + 1 }, (_unused, index) => index);
  for (let leftIndex = 0; leftIndex < left.length; leftIndex += 1) {
    const current = [leftIndex + 1];
    for (let rightIndex = 0; rightIndex < right.length; rightIndex += 1) {
      const substitution = left[leftIndex] === right[rightIndex] ? 0 : 1;
      current.push(
        Math.min(
          current[rightIndex]! + 1,
          previous[rightIndex + 1]! + 1,
          previous[rightIndex]! + substitution,
        ),
      );
    }
    previous = current;
  }
  return previous[previous.length - 1]!;
}

/** `normalized_levenshtein_similarity`, on the trimmed lower-cased strings. */
export function normalizedLevenshteinSimilarity(prediction: string, gold: string): number {
  const left = prediction.trim().toLowerCase();
  const right = gold.trim().toLowerCase();
  if (left === '' && right === '') return 1;
  if (left === '' || right === '') return 0;
  return 1 - levenshteinDistance(left, right) / Math.max(left.length, right.length);
}

/** ANLS at their 0.5 threshold. */
export function anlsScore(prediction: string, gold: string, threshold = 0.5): number {
  const similarity = normalizedLevenshteinSimilarity(prediction, gold);
  return similarity >= threshold ? similarity : 0;
}

const UNANSWERABLE_PHRASES = [
  'not answerable',
  'unanswerable',
  'cannot be determined',
  'cannot be answered',
  'not enough information',
  'context_overflow',
];

/**
 * `accuracy_score`: 1 or 0 per question. Relaxed by type — a numeric answer within 5 per cent,
 * a gold string contained in the prediction, or an 0.8 normalised edit similarity all count.
 */
export function xlAccuracyScore(prediction: string, gold: string, answerType: XlAnswerType): number {
  const predictionNorm = normalizeAnswer(prediction);
  const goldNorm = normalizeAnswer(gold);

  if (answerType === 'unanswerable') {
    return UNANSWERABLE_PHRASES.some((phrase) => predictionNorm.includes(phrase)) ? 1 : 0;
  }

  if (answerType === 'boolean') {
    const truthy = ['yes', 'true', 'correct'].some((word) => predictionNorm.includes(word));
    const falsy = ['no', 'false', 'incorrect'].some((word) => predictionNorm.includes(word));
    const goldBool = ['yes', 'true', 'correct'].some((word) => goldNorm.includes(word));
    if (truthy) return goldBool ? 1 : 0;
    if (falsy) return goldBool ? 0 : 1;
    return 0;
  }

  if (answerType === 'numeric' || answerType === 'percentage') {
    const predictionNumber = extractNumber(predictionNorm);
    const goldNumber = extractNumber(goldNorm);
    if (predictionNumber !== undefined && goldNumber !== undefined) {
      if (goldNumber === 0) return Math.abs(predictionNumber) < 1e-6 ? 1 : 0;
      return Math.abs(predictionNumber - goldNumber) / Math.abs(goldNumber) <= 0.05 ? 1 : 0;
    }
  }

  if (answerType === 'single_choice') {
    const predictionOption = /\b([A-D])\b/.exec(prediction.trim().toUpperCase());
    const goldOption = /\b([A-D])\b/.exec(gold.trim().toUpperCase());
    if (predictionOption !== null && goldOption !== null) {
      return predictionOption[1] === goldOption[1] ? 1 : 0;
    }
  }

  if (goldNorm !== '' && predictionNorm.includes(goldNorm)) return 1;
  return normalizedLevenshteinSimilarity(predictionNorm, goldNorm) >= 0.8 ? 1 : 0;
}

/** `token_f1_score` over normalised token sets. */
export function xlTokenF1(prediction: string, gold: string): number {
  const predictionTokens = new Set(normalizeAnswer(prediction).split(' ').filter((token) => token !== ''));
  const goldTokens = new Set(normalizeAnswer(gold).split(' ').filter((token) => token !== ''));
  if (goldTokens.size === 0) return predictionTokens.size === 0 ? 1 : 0;
  if (predictionTokens.size === 0) return 0;
  const overlap = [...predictionTokens].filter((token) => goldTokens.has(token));
  if (overlap.length === 0) return 0;
  const precision = overlap.length / predictionTokens.size;
  const recall = overlap.length / goldTokens.size;
  return (2 * precision * recall) / (precision + recall);
}

/** The three numbers their evaluator reports for one question. */
export function scoreAgainstXlDocBench(
  prediction: string,
  gold: string,
  answerType: XlAnswerType,
): { accuracy: number; tokenF1: number; anls: number } {
  return {
    accuracy: xlAccuracyScore(prediction, gold, answerType),
    tokenF1: xlTokenF1(prediction, gold),
    anls: anlsScore(prediction, gold),
  };
}
