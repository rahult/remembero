import { describe, expect, it } from 'vitest';
import type { ChatMessage, LlmCompletion } from '../src/llm/client.js';
import {
  evaluateLongMemEvalAnswerInstance,
  summarizeLongMemEvalAnswers,
  type LongMemEvalCompletionClient,
} from '../src/evals/longmemeval-answer.js';
import type { LongMemEvalInstance } from '../src/evals/longmemeval.js';
import {
  COUNT_QUESTIONS,
  countHistoryInstances,
  countedNoteLine,
  selectCountCandidates,
} from '../src/evals/typesafe-count.js';
import type { TypesafeNouls } from '../src/evals/typesafe-rerank.js';
import { countedThing, isCountQuestion } from '../src/knowledge/count-question.js';

describe('isCountQuestion', () => {
  const positives = [
    'How many books have I read this year?',
    'How many different types of tea have I tried?',
    'How many times have I been to the dentist?',
    'How many plants did I buy?',
    'How many charity events did I participate in?',
    'What is the total number of marathons I have run?',
    'What is the count of gyms I have joined?',
    'How many of my friends have I introduced to climbing?',
    'How many concerts did I attend since moving to Boston?',
  ];
  const negatives = [
    // a span of time, not a tally
    'How long have I been collecting stamps?',
    'How many days ago did I attend the service?',
    'How many weeks passed between the two trips?',
    'How many months before my anniversary did Rachel get engaged?',
    // an amount of one thing
    'How much did I spend on the new bike?',
    'How much money did I raise for the shelter?',
    'How old was I when I moved here?',
    // a rate the user simply stated
    'How many times a week do I go to the gym?',
    'How many cups of coffee do I drink per day?',
    // one thing's stated property
    'How many bedrooms does my new apartment have?',
    'How many people were at the wedding I attended?',
    'How many pages is the novel I am reading?',
    // the assistant's own earlier answer
    'How many alternatives did you suggest for the trip?',
    // not about the user at all
    'How many countries are in the European Union?',
  ];
  it('calls a tally of the user\'s own instances a count question', () => {
    for (const question of positives) {
      expect(isCountQuestion(question), question).toBe(true);
    }
  });
  it('rejects durations, amounts, rates, single details and assistant recall', () => {
    for (const question of negatives) {
      expect(isCountQuestion(question), question).toBe(false);
    }
  });
});

describe('countedThing', () => {
  it('takes the noun phrase verbatim from the question', () => {
    expect(countedThing('How many books have I read this year?')).toBe('books');
    expect(countedThing('How many different types of tea have I tried?')).toBe(
      'different types of tea',
    );
    expect(countedThing('How many times have I been to the dentist?')).toBe('times');
    expect(countedThing('What is the total number of marathons I have run?')).toBe(
      'marathons',
    );
    expect(countedThing('How many of my friends have I told?')).toBe('of my friends');
    expect(countedThing('What did I do last week?')).toBe('items');
  });
});

const sources = [
  {
    ts: '2024/03/02 (Sat) 10:00',
    text: 'USER: I finished the Berlin marathon today. It rained.\nASSISTANT: Nice, a marathon in the rain.\nUSER: Unrelated: my printer is broken.',
  },
  {
    ts: '2024/01/05 (Fri) 09:00',
    text: 'USER: I ran my first marathon in Boston.',
  },
];

describe('selectCountCandidates', () => {
  it('keeps the user sentences sharing a content word, in session-date order', () => {
    const candidates = selectCountCandidates(
      'How many marathons have I run?',
      sources,
    );
    expect(candidates.map((c) => [c.sessionDate, c.sentence])).toEqual([
      ['2024-01-05', 'I ran my first marathon in Boston.'],
      ['2024-03-02', 'I finished the Berlin marathon today.'],
    ]);
  });

  it('keeps the user turns only, whichever case the role markers are in', () => {
    const lower = [
      {
        ts: '2024/01/05 (Fri) 09:00',
        text: 'user: I ran my first marathon in Boston.\nassistant: Congratulations on the marathon!',
      },
    ];
    expect(
      selectCountCandidates('How many marathons have I run?', lower).map(
        (c) => c.sentence,
      ),
    ).toEqual(['I ran my first marathon in Boston.']);
  });

  it('caps at max by descending overlap and returns reading order', () => {
    const many = [
      {
        ts: '2024/01/01 (Mon) 09:00',
        text: 'USER: I ran a marathon in Boston. I like marathon running a lot and ran one more marathon. I bought a marathon shirt.',
      },
    ];
    const capped = selectCountCandidates('How many marathons have I run?', many, 2);
    expect(capped).toHaveLength(2);
    // the two sentences with the most overlap ("marathon" plus "run"), in reading order
    expect(capped.map((c) => c.sentence)).toEqual([
      'I ran a marathon in Boston.',
      'I like marathon running a lot and ran one more marathon.',
    ]);
  });
});

describe('countedNoteLine', () => {
  it('states the count, then every instance with its session date', () => {
    const counted = selectCountCandidates('How many marathons have I run?', sources);
    expect(countedNoteLine('How many marathons have I run?', counted)).toBe(
      'Counted 2 marathons the history states: 2024-01-05 — I ran my first marathon in Boston.; 2024-03-02 — I finished the Berlin marathon today.',
    );
  });

  it('always states the count, cutting the list to the budget', () => {
    const counted = selectCountCandidates('How many marathons have I run?', sources);
    const line = countedNoteLine('How many marathons have I run?', counted, 40);
    expect(line).toBe(
      'Counted 2 marathons the history states: 2024-01-05 — I ran my first marathon in Boston.; … and 1 more not shown',
    );
  });

  it('has no line without a counted instance', () => {
    expect(countedNoteLine('How many marathons have I run?', [])).toBe('');
  });
});

describe('countHistoryInstances', () => {
  const stub = (
    scores: Record<string, number>,
  ): { nouls: TypesafeNouls; states: Array<Record<string, unknown>> } => {
    const states: Array<Record<string, unknown>> = [];
    const nouls: TypesafeNouls = async (state, questions) => {
      expect(questions).toBe(COUNT_QUESTIONS);
      const { sentence } = state as { sentence: string };
      states.push(state as Record<string, unknown>);
      return {
        nouls: { instance: scores[sentence] ?? 0 },
        inputTokens: 40,
        cached: sentence.includes('Boston'),
      };
    };
    return { nouls, states };
  };

  it('sends one request per candidate and counts the ones above the threshold', async () => {
    const { nouls, states } = stub({
      'I ran my first marathon in Boston.': 0.9,
      'I finished the Berlin marathon today.': 0.8,
      'Unrelated: my printer is broken.': 0.01,
    });
    const result = await countHistoryInstances({
      question: 'How many marathons have I run?',
      sources,
      nouls,
    });
    expect(states).toEqual([
      {
        question: 'How many marathons have I run?',
        sentence: 'I ran my first marathon in Boston.',
        session_date: '2024-01-05',
      },
      {
        question: 'How many marathons have I run?',
        sentence: 'I finished the Berlin marathon today.',
        session_date: '2024-03-02',
      },
    ]);
    expect(result.stats).toEqual({
      candidates: 2,
      counted: 2,
      calls: 1,
      cached: 1,
      inputTokens: 40,
    });
    expect(result.line).toContain('Counted 2 marathons the history states: ');
  });

  it('leaves out a candidate below the threshold', async () => {
    const { nouls } = stub({
      'I ran my first marathon in Boston.': 0.9,
      'I finished the Berlin marathon today.': 0.2,
    });
    const result = await countHistoryInstances({
      question: 'How many marathons have I run?',
      sources,
      nouls,
      threshold: 0.5,
    });
    expect(result.stats.counted).toBe(1);
    expect(result.line).toBe(
      'Counted 1 marathons the history states: 2024-01-05 — I ran my first marathon in Boston.',
    );
  });

  it('never sends a sensitive candidate and does not count it', async () => {
    const { nouls, states } = stub({});
    const result = await countHistoryInstances({
      question: 'How many marathons have I run?',
      sources: [
        {
          ts: '2024/01/05 (Fri) 09:00',
          text: 'USER: I ran a marathon and my api key is sk-abcdefghijklmnopqrstuvwxyz0123456789.',
        },
      ],
      nouls,
    });
    expect(states).toEqual([]);
    expect(result.stats).toEqual({
      candidates: 1,
      counted: 0,
      calls: 0,
      cached: 0,
      inputTokens: 0,
    });
    expect(result.line).toBe('');
  });
});

class ScriptedCompletionClient implements LongMemEvalCompletionClient {
  readonly prompts: string[] = [];
  constructor(
    readonly model: string,
    private readonly outputs: string[],
  ) {}
  async completeWithUsage(messages: ChatMessage[]): Promise<LlmCompletion> {
    this.prompts.push(messages.map((message) => message.content).join('\n'));
    const content = this.outputs.shift();
    if (content === undefined) throw new Error('scripted completion exhausted');
    return {
      content,
      model: this.model,
      usage: {
        promptTokens: 1,
        completionTokens: 1,
        totalTokens: 2,
        cachedPromptTokens: 0,
        reasoningTokens: 0,
        costUsd: 0,
      },
    };
  }
}

function fixture(): LongMemEvalInstance {
  return {
    question_id: 'count-1',
    question_type: 'multi-session',
    question: 'How many marathons have I run?',
    answer: '2',
    question_date: '2024/03/10 (Sun) 09:00',
    haystack_session_ids: ['boston', 'berlin', 'noise'],
    haystack_dates: [
      '2024/01/05 (Fri) 09:00',
      '2024/03/02 (Sat) 10:00',
      '2024/02/01 (Thu) 09:00',
    ],
    haystack_sessions: [
      [{ role: 'user', content: 'I ran my first marathon in Boston.', has_answer: true }],
      [{ role: 'user', content: 'I finished the Berlin marathon today.', has_answer: true }],
      [{ role: 'user', content: 'Compare credit card rewards for travel.' }],
    ],
    answer_session_ids: ['boston', 'berlin'],
  };
}

describe('counted items inside the answer harness', () => {
  const nouls: TypesafeNouls = async (state) => {
    const { sentence } = state as { sentence: string };
    return {
      nouls: { instance: sentence.toLowerCase().includes('marathon') ? 0.9 : 0.05 },
      inputTokens: 30,
      cached: false,
    };
  };

  it('puts the counted line first in the computed-notes block and accounts the tokens', async () => {
    const reader = new ScriptedCompletionClient('reader', ['2 marathons']);
    const judge = new ScriptedCompletionClient('judge', ['yes']);
    const observation = await evaluateLongMemEvalAnswerInstance(
      fixture(),
      reader,
      judge,
      {
        topK: 3,
        contextBytes: 8_192,
        computedNotes: true,
        typesafeCount: { nouls },
      },
    );
    expect(observation.countedItems).toEqual({
      candidates: 2,
      counted: 2,
      calls: 2,
      cached: 0,
      inputTokens: 60,
    });
    const prompt = reader.prompts[0]!;
    const block = prompt.slice(prompt.indexOf('### Computed from the history'));
    expect(block.split('\n')[1]).toBe(
      'Counted 2 marathons the history states: 2024-01-05 — I ran my first marathon in Boston.; 2024-03-02 — I finished the Berlin marathon today.',
    );
    const summary = summarizeLongMemEvalAnswers([observation]);
    expect(summary.countUsage).toEqual({
      candidates: 2,
      counted: 2,
      calls: 2,
      cached: 0,
      inputTokens: 60,
      costUsd: (60 * 0.042) / 1_000_000,
    });
  });

  it('leaves the block alone when the flag is off', async () => {
    const reader = new ScriptedCompletionClient('reader', ['2 marathons']);
    const judge = new ScriptedCompletionClient('judge', ['yes']);
    const observation = await evaluateLongMemEvalAnswerInstance(
      fixture(),
      reader,
      judge,
      { topK: 3, contextBytes: 8_192, computedNotes: true },
    );
    expect(observation.countedItems).toBeUndefined();
    expect(reader.prompts[0]!).not.toContain('Counted ');
    expect(summarizeLongMemEvalAnswers([observation]).countUsage).toBeUndefined();
  });

  it('does not count a question that is not a count question', async () => {
    const reader = new ScriptedCompletionClient('reader', ['Boston']);
    const judge = new ScriptedCompletionClient('judge', ['yes']);
    const observation = await evaluateLongMemEvalAnswerInstance(
      { ...fixture(), question: 'Which marathon did I run first?' },
      reader,
      judge,
      {
        topK: 3,
        contextBytes: 8_192,
        computedNotes: true,
        typesafeCount: { nouls },
      },
    );
    expect(observation.countedItems).toBeUndefined();
    expect(reader.prompts[0]!).not.toContain('Counted ');
  });
});

describe('count flags', () => {
  it('are off by default and recorded when on', async () => {
    const { parseArgs } = await import('../src/evals/run-longmemeval-answer.js');
    const off = parseArgs([]);
    expect(off.typesafeCount).toBe(false);
    const on = parseArgs([
      '--typesafe-count',
      '--typesafe-count-max',
      '20',
      '--typesafe-count-threshold',
      '0.7',
    ]);
    expect(on.typesafeCount).toBe(true);
    expect(on.typesafeCountMax).toBe(20);
    expect(on.typesafeCountThreshold).toBe(0.7);
    expect(() => parseArgs(['--typesafe-count-threshold', '1.5'])).toThrow(
      /--typesafe-count-threshold/,
    );
  });
});
