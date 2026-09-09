import { describe, expect, it } from 'vitest';
import { parseProgram, serializeClause } from '../src/engine/index.js';
import {
  generateExtractionExamples,
  renderRequestText,
  toExtractionConversation,
  verifyRendering,
  type Renderer,
} from '../src/training/extraction-data.js';
import { createRng } from '../src/training/rng.js';
import { generateWorld } from '../src/training/worlds.js';

/** Fake renderer: states each fact as "<subj> <pred words> <obj>." and honours first person. */
const fakeRenderer: Renderer = async (request) => {
  const display = (value: string) => request.display?.[value] ?? value;
  const sentences = request.facts.map((fact, i) => {
    const [subject, ...rest] = fact.args;
    const words = fact.predicate.replaceAll('_', ' ');
    let subj =
      request.firstPerson && subject === request.selfAtom
        ? 'I'
        : request.pluralAtom === subject
          ? 'We'
          : display(subject);
    if (request.pronoun && i > 0) subj = 'She';
    return `${subj} ${words} ${rest.map(display).join(' ')}.`;
  });
  if (request.negated)
    return sentences
      .map((s) => s.replace(/^(\S+) /, '$1 no longer '))
      .join(' ');
  if (request.hedged)
    return sentences.map((s) => s.replace(/^(\S+) /, '$1 might ')).join(' ');
  return `${request.distractor ?? ''}${sentences.join(' ')}`.trim();
};

describe('extraction training data', () => {
  it('verifies a rendering mentions every constant and no other world entity', () => {
    const world = generateWorld(2);
    const [a, b] = world.entities;
    const facts = parseProgram(`${world.relations[0].name}(${a}, ${b}).`);
    expect(
      verifyRendering(
        `${a} ${world.relations[0].name.replaceAll('_', ' ')} ${b}.`,
        facts,
        world,
        'user',
      ),
    ).toBe(true);
    expect(verifyRendering(`${a} does something.`, facts, world, 'user')).toBe(
      false,
    );
    const c = world.entities[2];
    expect(verifyRendering(`${a} and ${c}: ${b}.`, facts, world, 'user')).toBe(
      false,
    );
  });

  it('generates every example kind with exact gold and verified text', async () => {
    const world = generateWorld(3);
    const examples = await generateExtractionExamples(
      world,
      createRng(3),
      fakeRenderer,
      { selfAtom: 'rahul' },
    );
    const kinds = new Set(examples.map((e) => e.kind));
    for (const kind of [
      'state',
      'first_person',
      'supersession',
      'negation',
      'hedge',
      'distractor',
      'coreference',
      'normalization',
      'date_number',
      'quoted_name',
      'implicit_subject',
      'transcript',
    ]) {
      expect(kinds.has(kind as never), kind).toBe(true);
    }
    for (const example of examples) {
      expect(example.input.length).toBeGreaterThan(0);
      expect(() => parseProgram(example.initialProgram)).not.toThrow();
      expect(() =>
        parseProgram(example.expectedAdded.join('\n')),
      ).not.toThrow();
      if (example.kind === 'negation' || example.kind === 'hedge') {
        expect(example.expectedAdded).toEqual([]);
      }
      if (example.kind === 'supersession') {
        expect(example.expectedRetract.length).toBe(1);
        expect(example.initialProgram).not.toBe('');
      }
      if (example.kind === 'coreference') {
        expect(example.expectedAdded.length).toBe(2);
        expect(example.input).toMatch(/\b(she|he|they)\b/i);
      }
      if (example.kind === 'normalization') {
        expect(example.input).toMatch(/[A-Z]/);
        expect(example.expectedAdded.join(' ')).toMatch(
          /^[a-z_]+\([a-z0-9_, ]+\)\.$/,
        );
      }
      if (example.kind === 'quoted_name') {
        // a multi-word name not in the schema is stored quoted, capitals kept
        expect(example.expectedAdded.join(' ')).toMatch(
          /'[A-Z][A-Za-z]+( [A-Z][A-Za-z]+)+'/,
        );
        expect(example.input).toMatch(/[A-Z][a-z]+ [A-Z][a-z]+/);
      }
      if (example.kind === 'date_number') {
        expect(example.expectedAdded.join(' ')).toMatch(/\d/);
        if (example.initialProgram !== '')
          expect(example.initialProgram).toMatch(
            /headcount|started_on|deadline/,
          );
        if (/started_on|deadline/.test(example.expectedAdded.join(' '))) {
          expect(example.expectedAdded.join(' ')).toMatch(
            /'\d{4}-\d{2}-\d{2}'/,
          );
        }
      }
      if (example.kind === 'first_person') {
        expect(example.expectedAdded.join(' ')).toContain('rahul');
        expect(example.input).toMatch(/\b(I|my|me)\b/i);
      }
      if (example.kind === 'implicit_subject') {
        // the subject is a generic noun (team, project, service); it is either in the
        // text or spoken as "we"/"our"; no world entity is involved
        expect(example.expectedAdded.length).toBe(1);
        const subject = example.expectedAdded[0].match(/\((\w+),/)![1];
        expect(['team', 'project', 'service']).toContain(subject);
        expect(
          new RegExp(`\\b(${subject}|we|our)\\b`, 'i').test(example.input),
        ).toBe(true);
        for (const entity of world.entities)
          expect(example.expectedAdded[0]).not.toContain(`(${entity},`);
      }
      if (example.kind === 'transcript') {
        expect(example.mode).toBe('transcript');
        expect(example.input).toMatch(/^USER: |^ASSISTANT: /m);
        expect(example.input).toMatch(/\nASSISTANT: |^ASSISTANT: /);
        expect(example.expectedRetract).toEqual([]);
      } else if (example.kind === 'event') {
        expect(['text', 'transcript']).toContain(example.mode);
      } else {
        expect(example.mode).toBe('text');
      }
    }
  });

  it('puts three-place schedule facts into state examples so argument order is learned', async () => {
    // world 1 has a schedule relation; with enough draws a state example uses it
    const world = generateWorld(1);
    expect(world.relations.some((r) => r.kind === 'schedule')).toBe(true);
    const examples = await generateExtractionExamples(
      world,
      createRng(1),
      fakeRenderer,
      { selfAtom: 'rahul', perKind: 48 },
    );
    expect(
      examples.some(
        (e) =>
          e.kind === 'state' &&
          e.expectedAdded.some((f) => f.split(',').length === 3),
      ),
    ).toBe(true);
  });

  it('writes some first-person examples over relations ("my manager is ..."), not only attributes', async () => {
    let relational = false;
    for (let seed = 1; seed <= 3 && !relational; seed += 1) {
      const world = generateWorld(seed);
      const attributes = new Set(
        world.relations
          .filter((r) => r.kind === 'attribute')
          .map((r) => r.name),
      );
      const examples = await generateExtractionExamples(
        world,
        createRng(seed),
        fakeRenderer,
        { selfAtom: 'rahul', perKind: 24 },
      );
      relational = examples.some(
        (e) =>
          e.kind === 'first_person' &&
          !attributes.has(
            e.expectedAdded[0].slice(0, e.expectedAdded[0].indexOf('(')),
          ),
      );
    }
    expect(relational).toBe(true);
  });

  it('transcript examples never store what only the assistant said', async () => {
    const world = generateWorld(4);
    const examples = await generateExtractionExamples(
      world,
      createRng(4),
      fakeRenderer,
      { selfAtom: 'rahul', perKind: 3 },
    );
    const transcripts = examples.filter((e) => e.kind === 'transcript');
    expect(transcripts.length).toBeGreaterThan(0);
    // at least one transcript has an assistant turn that states a fact (a guess or a
    // summary) and that fact is absent from the gold
    const withGuess = transcripts.filter((e) =>
      /ASSISTANT: (It looks like|I assume|Summary|Just to confirm)/.test(
        e.input,
      ),
    );
    expect(withGuess.length).toBeGreaterThan(0);
    for (const e of transcripts) {
      const userText = e.input
        .split(/\n\n/)
        .filter((turn) => turn.startsWith('USER: '))
        .join(' ')
        .toLowerCase();
      for (const fact of e.expectedAdded) {
        // every gold constant other than the self atom is in a USER turn (or confirmed there)
        const constants = fact
          .slice(fact.indexOf('(') + 1, fact.lastIndexOf(')'))
          .split(',')
          .map((c) => c.trim().replace(/'/g, ''))
          .filter((c) => c !== 'rahul');
        const confirmed = /USER: Yes/.test(e.input);
        if (!confirmed)
          for (const c of constants)
            expect(userText.replace(/[^a-z0-9]+/g, ''), e.input).toContain(
              c.toLowerCase().replace(/[^a-z0-9]+/g, ''),
            );
      }
    }
  });

  it('embeds facts inside long chatty user turns with long assistant replies, like real transcripts', async () => {
    let embedded = 0;
    let longAssistant = 0;
    for (let seed = 1; seed <= 3; seed += 1) {
      const world = generateWorld(seed);
      const examples = await generateExtractionExamples(
        world,
        createRng(seed),
        fakeRenderer,
        { selfAtom: 'rahul', perKind: 12 },
      );
      for (const e of examples.filter((x) => x.kind === 'transcript')) {
        const userTurns = e.input
          .split(/\n\n(?=USER: |ASSISTANT: )/)
          .filter((t) => t.startsWith('USER: '));
        const assistantTurns = e.input
          .split(/\n\n(?=USER: |ASSISTANT: )/)
          .filter((t) => t.startsWith('ASSISTANT: '));
        if (userTurns.some((t) => t.length > 220) && e.expectedAdded.length > 0)
          embedded += 1;
        if (assistantTurns.some((t) => t.length > 250)) longAssistant += 1;
      }
    }
    expect(embedded).toBeGreaterThan(3);
    expect(longAssistant).toBeGreaterThan(3);
  });

  it('leaves the schema empty in a share of examples so predicate naming is learned from the text', async () => {
    let empty = 0;
    let total = 0;
    for (let seed = 1; seed <= 3; seed += 1) {
      const world = generateWorld(seed);
      const examples = await generateExtractionExamples(
        world,
        createRng(seed),
        fakeRenderer,
        { selfAtom: 'rahul', perKind: 3 },
      );
      for (const e of examples) {
        total += 1;
        if (e.initialProgram === '') {
          empty += 1;
          // an empty store cannot hold anything to retract or supersede
          expect(e.expectedRetract).toEqual([]);
          expect(['supersession', 'negation']).not.toContain(e.kind);
        }
      }
    }
    expect(empty / total).toBeGreaterThan(0.12);
    expect(empty / total).toBeLessThan(0.4);
  });

  it('teaches episodic asides: "by the way, I just ..." events with the self atom as subject', async () => {
    let asides = 0;
    for (let seed = 1; seed <= 3; seed += 1) {
      const world = generateWorld(seed);
      const examples = await generateExtractionExamples(
        world,
        createRng(seed),
        fakeRenderer,
        { selfAtom: 'rahul', perKind: 12 },
      );
      for (const e of examples.filter((x) => x.kind === 'event')) {
        asides += 1;
        expect(e.input).toMatch(
          /\b(By the way|Also|Oh, and|Incidentally|Speaking of which)\b/,
        );
        expect(e.expectedAdded.length).toBeGreaterThan(0);
        // every event fact is about the speaker, and its constants are in the user's words
        for (const fact of e.expectedAdded) {
          expect(fact).toMatch(/^[a-z_]+\(rahul, /);
          const constants = fact
            .slice(fact.indexOf('(') + 1, fact.lastIndexOf(')'))
            .split(', ')
            .slice(1)
            .map((c) =>
              c
                .replace(/'/g, '')
                .toLowerCase()
                .replace(/[^a-z0-9]+/g, ''),
            );
          const userText = (
            e.mode === 'transcript'
              ? e.input
                  .split(/\n\n(?=USER: |ASSISTANT: )/)
                  .filter((t) => t.startsWith('USER: '))
                  .join(' ')
              : e.input
          )
            .toLowerCase()
            .replace(/[^a-z0-9]+/g, '');
          for (const c of constants) expect(userText, e.input).toContain(c);
        }
      }
    }
    expect(asides).toBeGreaterThan(10);
  });

  it('exports transcript examples with the transcript prompt and additive facts only', () => {
    const world = generateWorld(3);
    const conversation = toExtractionConversation(
      world,
      {
        world: world.id,
        kind: 'transcript',
        input: 'USER: I work at Acme.\n\nASSISTANT: Noted.',
        initialProgram: 'works_at(zoe, globex).',
        expectedAdded: ['works_at(rahul, acme).'],
        expectedRetract: [],
        mode: 'transcript',
      },
      'rahul',
    );
    expect(conversation.messages[0].content).toContain('transcript');
    expect(conversation.messages[0].content).not.toContain('retract works_at');
    expect(conversation.messages[2].content).toBe('works_at(rahul, acme).');
  });

  it('verifyRendering exempts the plural atom spoken as "we"', () => {
    const world = generateWorld(2);
    const facts = parseProgram('deploy_day(team, friday).');
    expect(verifyRendering('We deploy on Friday.', facts, world, 'user')).toBe(
      false,
    );
    expect(
      verifyRendering('We deploy on Friday.', facts, world, 'user', 'team'),
    ).toBe(true);
  });

  it('capitalizes single-word attribute values too, since companies and cities are proper nouns in English', async () => {
    let valueCapitalized = false;
    for (let seed = 1; seed <= 8 && !valueCapitalized; seed += 1) {
      const world = generateWorld(seed);
      const examples = await generateExtractionExamples(
        world,
        createRng(seed),
        fakeRenderer,
        { selfAtom: 'rahul' },
      );
      // some attribute example shows a capitalized value that is not a world entity
      valueCapitalized = examples
        .filter((e) => e.kind === 'state' || e.kind === 'supersession')
        .some((e) => {
          const caps = e.input.match(/\b[A-Z][a-z]+\b/g) ?? [];
          return caps.some(
            (w) =>
              !world.entities.includes(w.toLowerCase()) &&
              !/^(She|He|They|I)$/.test(w),
          );
        });
    }
    expect(valueCapitalized).toBe(true);
  });

  it('shows single-word entity names capitalized in most ordinary examples while the fact keeps the lowercase atom', async () => {
    const world = generateWorld(3);
    const examples = await generateExtractionExamples(
      world,
      createRng(3),
      fakeRenderer,
      { selfAtom: 'rahul' },
    );
    const states = examples.filter(
      (e) =>
        e.kind === 'state' ||
        e.kind === 'distractor' ||
        e.kind === 'supersession',
    );
    const capitalized = states.filter((e) => /\b[A-Z][a-z]+\b/.test(e.input));
    expect(capitalized.length).toBeGreaterThan(states.length / 2);
    for (const e of states) {
      for (const fact of e.expectedAdded)
        expect(fact).toMatch(
          /^[a-z_]+\((?:'[^']+'|[a-z0-9_.-]+)(?:, (?:'[^']+'|[a-z0-9_.-]+))*\)\.$/,
        );
    }
  });

  it('prepends distractor prose itself, so the renderer cannot drop it', async () => {
    const world = generateWorld(3);
    // a renderer that ignores request.distractor, like the real one
    const bare: Renderer = async (request) =>
      request.facts
        .map(
          (f) =>
            `${f.args[0]} ${f.predicate.replaceAll('_', ' ')} ${f.args.slice(1).join(' ')}.`,
        )
        .join(' ');
    const examples = await generateExtractionExamples(
      world,
      createRng(3),
      bare,
      { selfAtom: 'rahul' },
    );
    const distractors = examples.filter((e) => e.kind === 'distractor');
    expect(distractors.length).toBeGreaterThan(0);
    for (const example of distractors) {
      expect(example.input).toMatch(
        /meetings|Quick note|vendor call|office is closed/,
      );
      expect(example.expectedAdded.length).toBe(1);
    }
  });

  it('exports a conversation with the product extraction prompt and clause lines as the answer', () => {
    const world = generateWorld(3);
    const conversation = toExtractionConversation(
      world,
      {
        world: world.id,
        kind: 'supersession',
        input: 'Mira now works at Initech.',
        initialProgram: 'works_at(mira, acme).',
        expectedAdded: ['works_at(mira, initech).'],
        expectedRetract: ['works_at(mira, _)'],
        mode: 'text',
      },
      'rahul',
    );
    expect(conversation.messages.map((m) => m.role)).toEqual([
      'system',
      'user',
      'assistant',
    ]);
    expect(conversation.messages[0].content).toContain('Datalog clauses');
    expect(conversation.messages[0].content).toContain('rahul');
    expect(conversation.messages[2].content).toBe(
      'retract works_at(mira, _).\nworks_at(mira, initech).',
    );
    const nothing = toExtractionConversation(
      world,
      {
        world: world.id,
        kind: 'hedge',
        input: 'Mira might move.',
        initialProgram: '',
        expectedAdded: [],
        expectedRetract: [],
        mode: 'text',
      },
      'rahul',
    );
    expect(nothing.messages[2].content).toBe('% nothing');
  });

  it('renders a request into an unambiguous instruction for the rendering model', () => {
    const world = generateWorld(3);
    const text = renderRequestText({
      facts: parseProgram(
        `${world.relations[0].name}(${world.entities[0]}, ${world.entities[1]}).`,
      ).map((c) => ({
        predicate: c.head.predicate,
        args: c.head.args.map((t) => String((t as { value: string }).value)),
      })),
      argNames: world.relations[0].args,
      firstPerson: false,
      selfAtom: 'user',
      negated: false,
      hedged: false,
    });
    expect(text).toContain(world.entities[0]);
    expect(text).toContain('exactly');
    void serializeClause;
  });
});

describe('training run with both tasks', () => {
  it('writes extraction conversations alongside query ones and records them in the manifest', async () => {
    const { mkdtempSync, readFileSync } = await import('node:fs');
    const { tmpdir } = await import('node:os');
    const { join } = await import('node:path');
    const { generateTrainingData } = await import('../src/training/run.js');
    const out = mkdtempSync(join(tmpdir(), 'train-both-'));
    const manifest = await generateTrainingData(
      {
        examples: 0,
        rounds: 1,
        worlds: 4,
        paraphrases: 0,
        seed: 21,
        out,
        paraphrase: false,
        selfAtom: 'rahul',
      },
      undefined,
      fakeRenderer,
    );
    expect(manifest.tasks).toEqual(['query', 'extraction']);
    expect(manifest.extraction?.count).toBeGreaterThan(0);
    const lines = readFileSync(join(out, 'conversations.jsonl'), 'utf8')
      .trim()
      .split('\n')
      .map((l) => JSON.parse(l));
    const extraction = lines.filter((l) =>
      l.messages[0].content.includes('Datalog clauses'),
    );
    const query = lines.filter((l) => l.messages[0].content.includes('p_plus'));
    expect(extraction.length).toBeGreaterThan(0);
    expect(query.length).toBeGreaterThan(0);
    expect(manifest.train + manifest.heldout).toBe(
      lines.length +
        readFileSync(join(out, 'heldout.jsonl'), 'utf8')
          .trim()
          .split('\n')
          .filter(Boolean).length,
    );
  });
});
