import { describe, expect, it } from 'vitest';
import {
  parseTemporalRange,
  temporalRangePrompt,
} from '../src/evals/longmemeval-answer.js';
import { createRng } from '../src/training/rng.js';
import {
  generateTimeRangeExamples,
  toTimeRangeConversation,
} from '../src/training/timerange-data.js';

describe('time-range training data', () => {
  const examples = generateTimeRangeExamples(createRng(3), 400);

  it('mixes ranged questions with refusals, refusals in the majority', () => {
    const refusals = examples.filter((e) => e.range === null).length;
    expect(refusals / examples.length).toBeGreaterThan(0.5);
    expect(refusals / examples.length).toBeLessThan(0.85);
  });

  it('computes ranges that contain the cue and widen at the edges', () => {
    const lastMonth = examples.find((e) => e.cue === 'last month');
    expect(lastMonth).toBeDefined();
    const [year, month] = lastMonth!.questionDate.split('-').map(Number);
    const previous = new Date(Date.UTC(year, month - 2, 1));
    const firstOfPrevious = previous.toISOString().slice(0, 10);
    const endOfPrevious = new Date(Date.UTC(year, month - 1, 0))
      .toISOString()
      .slice(0, 10);
    expect(lastMonth!.range!.start <= firstOfPrevious).toBe(true);
    expect(lastMonth!.range!.end >= endOfPrevious).toBe(true);
    // widened by days, not weeks
    expect(
      lastMonth!.range!.start >=
        new Date(previous.getTime() - 7 * 86_400_000)
          .toISOString()
          .slice(0, 10),
    ).toBe(true);
  });

  it('renders the exact prompt the evaluation uses and a reply its parser accepts', () => {
    for (const example of examples.slice(0, 50)) {
      const conversation = toTimeRangeConversation(example);
      expect(conversation.messages[0].content).toBe(
        temporalRangePrompt(example.question, example.questionDate),
      );
      const parsed = parseTemporalRange(conversation.messages[1].content);
      if (example.range === null) expect(parsed).toBeNull();
      else expect(parsed).toEqual(example.range);
    }
  });

  it('gives "how long ago" and "when did" questions no range', () => {
    const agoQuestions = examples.filter((e) =>
      /how (many weeks|many months|long has it been) .*(ago|since)/i.test(
        e.question,
      ),
    );
    expect(agoQuestions.length).toBeGreaterThan(0);
    for (const e of agoQuestions) expect(e.range).toBeNull();
  });
});
