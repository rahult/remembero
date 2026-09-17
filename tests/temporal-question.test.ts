import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { temporalQuestionAgreement } from '../src/evals/temporal-question-agreement.js';
import { isTemporalQuestion } from '../src/knowledge/temporal-question.js';

describe('isTemporalQuestion', () => {
  const positives = [
    // distances and durations between events
    'How many days ago did I attend the Maundy Thursday service?',
    'How many days passed between my visit to MoMA and the Met exhibit?',
    'How many weeks had passed since I recovered from the flu when I went jogging?',
    'How many months before my anniversary did Rachel get engaged?',
    'How many days did it take me to finish the novel?',
    'How long had I been bird watching when I attended the workshop?',
    'How long have I been working before I started my current job at NovaTech?',
    // ordering
    'Which event happened first, my cousin\'s wedding or the engagement party?',
    'Who graduated first, second and third among Emma, Rachel and Alex?',
    'Which seeds were started first, the tomatoes or the marigolds?',
    'What is the order of the six museums I visited from earliest to latest?',
    'Which mode of transport did I use most recently, a bus or a train?',
    'Which streaming service did I start using most recently?',
    'What was the first issue I had with my new car after its first service?',
    'How many charity events did I participate in before the Run for the Cure event?',
    // dates
    'When did I book the Airbnb in Sacramento?',
    'What was the date on which I attended the first BBQ event in June?',
    'Which month did I start the course?',
    // relative periods
    'Which book did I finish a week ago?',
    'I received a piece of jewelry last Saturday from whom?',
    'Which bike did I fix the past weekend?',
    'What was the airline that I flew with on Valentine\'s day?',
    'Which airline did I fly with the most in March and April?',
    'Which pair of shoes did I clean last month?',
    'What did I do with Rachel on the Wednesday two months ago?',
    'What did I cook the weekend before the party?',
  ];
  const negatives = [
    // counting, not time
    'How many plants have I bought?',
    'How many Korean restaurants have I tried in my city?',
    'How many plants did I acquire in the last month?',
    'How much total money did I spend on workshops in the last four months?',
    'How many short stories have I written since I started writing regularly?',
    // a remembered duration or a schedule, not a distance between events
    'How long is my daily commute to work?',
    'How long have I been collecting vintage cameras?',
    'How many days a week do I attend fitness classes?',
    'How many years older is my grandma than me?',
    'What time do I usually go to the gym?',
    'What day of the week do I take a cocktail-making class?',
    // "first" and "before" as plain words
    'What was the discount I got on my first purchase from the new brand?',
    'What was my last name before I changed it?',
    'Where did I go on my most recent family trip?',
    // plans and preferences
    'Can you recommend some cultural events happening around me this weekend?',
    'I\'m planning my meal prep next week, any suggestions?',
    'I\'ve got some free time tonight, any documentary recommendations?',
    'Can you remind me what color the Plesiosaur was in our previous chat?',
    'I remember you told me about the refineries earlier. Which process is used at Lake Charles?',
    'What is my current highest score in Ticket to Ride?',
  ];

  it.each(positives)('temporal: %s', (question) => {
    expect(isTemporalQuestion(question)).toBe(true);
  });

  it.each(negatives)('not temporal: %s', (question) => {
    expect(isTemporalQuestion(question)).toBe(false);
  });

  it('separates counting from durations', () => {
    expect(isTemporalQuestion('How many shirts have I bought?')).toBe(false);
    expect(isTemporalQuestion('How many days ago did I buy a smoker?')).toBe(true);
  });
});

describe('temporalQuestionAgreement', () => {
  it('counts precision, recall and the confusion by type', () => {
    const report = temporalQuestionAgreement([
      { question_type: 'temporal-reasoning', question: 'How many days ago did I go?' },
      { question_type: 'temporal-reasoning', question: 'What did I buy?' },
      { question_type: 'multi-session', question: 'When did I submit the paper?' },
      { question_type: 'multi-session', question: 'How many shirts have I bought?' },
    ]);
    expect(report.truePositives).toBe(1);
    expect(report.falsePositives).toBe(1);
    expect(report.falseNegatives).toBe(1);
    expect(report.precision).toBe(0.5);
    expect(report.recall).toBe(0.5);
    expect(report.byType).toEqual({
      'multi-session': { total: 2, temporal: 1 },
      'temporal-reasoning': { total: 2, temporal: 1 },
    });
    expect(report.misses).toEqual(['What did I buy?']);
    expect(report.falseAlarms).toEqual([
      { questionType: 'multi-session', question: 'When did I submit the paper?' },
    ]);
  });

  const dataset = '.cache/longmemeval/longmemeval_s_cleaned.json';
  it.skipIf(!existsSync(dataset))(
    'holds its agreement on the LongMemEval 500',
    () => {
      const instances = JSON.parse(readFileSync(dataset, 'utf8')) as Array<{
        question: string;
        question_type: string;
      }>;
      const report = temporalQuestionAgreement(instances);
      expect(report.recall).toBeGreaterThanOrEqual(0.96);
      expect(report.precision).toBeGreaterThanOrEqual(0.88);
    },
  );
});
