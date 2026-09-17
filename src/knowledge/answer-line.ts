/**
 * The reader's final answer, taken out of a reply that shows its working.
 *
 * The Notes-then-Answer reading (session-retrieval.ts) asks the reader for its notes first
 * and the answer on a last line, so the product has to read that last line and show only
 * it: a plain `recall` never shows the working. The benchmark harness has needed the same
 * function since reader v5 and holds a copy in `src/evals/longmemeval-answer.ts`; this is
 * the product's side of the same contract, in `knowledge/` because `src/llm` may not depend
 * on `src/evals` (the harness imports `llm/pipeline.ts`, so the edge would be a cycle).
 * The eval copy should re-export this one; it is under review in another branch today.
 */
export function finalAnswerLine(reply: string): string {
  const index = reply.lastIndexOf('Answer:');
  if (index < 0) return reply.trim();
  const answer = reply.slice(index + 'Answer:'.length).trim();
  return answer === '' ? reply.trim() : answer;
}
