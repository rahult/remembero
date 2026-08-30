import {
  closeSync,
  linkSync,
  mkdirSync,
  mkdtempSync,
  openSync,
  readFileSync,
  symlinkSync,
  unlinkSync,
  utimesSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { beforeEach, describe, expect, it } from 'vitest';
import {
  autoCaptureClaudeStop,
  type AutoCaptureOptions,
} from '../src/autocapture/capture.js';
import {
  MANAGED_HOOK_MARKER,
  installClaudeHook,
  removeClaudeHook,
} from '../src/autocapture/hooks.js';
import {
  parseClaudeStopHookInput,
  readClaudeTranscriptTail,
} from '../src/autocapture/transcript.js';
import type { ChatMessage, LlmClient } from '../src/llm/client.js';
import { MemoryStore } from '../src/store/store.js';

class ScriptedLlm implements LlmClient {
  calls: ChatMessage[][] = [];

  constructor(private readonly responses: string[]) {}

  async complete(messages: ChatMessage[]): Promise<string> {
    this.calls.push(messages);
    const response = this.responses.shift();
    if (response === undefined) throw new Error('ScriptedLlm ran out of responses');
    return response;
  }
}

let root: string;
let claudeConfigDir: string;
let transcriptPath: string;
let store: MemoryStore;

function transcriptLine(role: 'user' | 'assistant', content: unknown): string {
  return JSON.stringify({ type: role, message: { role, content } });
}

function writeTranscript(lines: string[]): void {
  writeFileSync(transcriptPath, `${lines.join('\n')}\n`, 'utf8');
}

function stopInput(overrides: Record<string, unknown> = {}): string {
  return JSON.stringify({
    session_id: 'session-1',
    transcript_path: transcriptPath,
    cwd: root,
    permission_mode: 'default',
    hook_event_name: 'Stop',
    stop_hook_active: false,
    last_assistant_message: 'Understood. I will remember that preference.',
    ...overrides,
  });
}

function captureOptions(overrides: Partial<AutoCaptureOptions> = {}): AutoCaptureOptions {
  return {
    namespace: 'default',
    dailyCap: 5,
    tailBytes: 16 * 1024,
    claudeConfigDir,
    now: new Date('2026-08-17T02:00:00.000Z'),
    ...overrides,
  };
}

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), 'rembero-autocapture-'));
  claudeConfigDir = join(root, '.claude');
  const projectDir = join(claudeConfigDir, 'projects', 'project');
  transcriptPath = join(projectDir, 'session-1.jsonl');
  mkdirSync(projectDir, { recursive: true });
  store = new MemoryStore(join(root, 'memory'));
});

describe('Claude Stop transcript ingress', () => {
  it('validates hook input and extracts only bounded user/assistant text', () => {
    writeTranscript([
      JSON.stringify({ type: 'queue-operation', operation: 'enqueue' }),
      transcriptLine('user', [
        { type: 'text', text: 'Please remember that I prefer dark mode.' },
        { type: 'tool_result', content: 'secret tool output' },
      ]),
      transcriptLine('assistant', [
        { type: 'thinking', thinking: 'private reasoning' },
        { type: 'text', text: '```ts\nconst noisy = true;\n```\nNoted.' },
      ]),
    ]);

    const input = parseClaudeStopHookInput(stopInput());
    const tail = readClaudeTranscriptTail(input, {
      claudeConfigDir,
      tailBytes: 16 * 1024,
    });

    expect(tail.text).toContain('USER: Please remember that I prefer dark mode.');
    expect(tail.text).toContain('ASSISTANT: Noted.');
    expect(tail.text).toContain('ASSISTANT: Understood. I will remember that preference.');
    expect(tail.text).not.toContain('secret tool output');
    expect(tail.text).not.toContain('private reasoning');
    expect(tail.text).not.toContain('const noisy');
    expect(tail.bytes).toBeLessThanOrEqual(16 * 1024);
  });

  it('keeps the latest user statement even when later assistant output is large', () => {
    writeTranscript([
      transcriptLine('user', 'My durable preference is tea.'),
      ...Array.from({ length: 8 }, (_, index) =>
        transcriptLine('assistant', `status ${index} ${'x'.repeat(2000)}`)
      ),
    ]);
    const input = parseClaudeStopHookInput(
      stopInput({ last_assistant_message: `final ${'y'.repeat(2000)}` })
    );
    const tail = readClaudeTranscriptTail(input, { claudeConfigDir, tailBytes: 1024 });

    expect(tail.text).toContain('USER: My durable preference is tea.');
    expect(tail.userMessageCount).toBe(1);
    expect(tail.bytes).toBeLessThanOrEqual(1024);
  });

  it('finds the latest user message when tool output pushes it past the base read window', () => {
    // Agentic sessions bury the user's turn under hundreds of KB of tool_result
    // lines; the reader must widen its backward scan until user text is found.
    const noise = Array.from({ length: 60 }, () =>
      transcriptLine('user', [{ type: 'tool_result', content: 'x'.repeat(2000) }])
    );
    writeTranscript([
      transcriptLine('user', 'My timezone is Australia/Sydney.'),
      ...noise,
    ]);
    const input = parseClaudeStopHookInput(stopInput());
    const tail = readClaudeTranscriptTail(input, { claudeConfigDir, tailBytes: 1024 });

    expect(tail.text).toContain('USER: My timezone is Australia/Sydney.');
    expect(tail.userMessageCount).toBe(1);
  });

  it('rejects non-Stop payloads and transcripts outside the Claude config root', () => {
    expect(() =>
      parseClaudeStopHookInput(stopInput({ hook_event_name: 'PreToolUse' }))
    ).toThrow(/Stop hook/i);

    const outside = join(root, 'outside.jsonl');
    writeFileSync(outside, transcriptLine('user', 'outside'), 'utf8');
    const input = parseClaudeStopHookInput(stopInput({ transcript_path: outside }));
    expect(() =>
      readClaudeTranscriptTail(input, { claudeConfigDir, tailBytes: 4096 })
    ).toThrow(/trusted Claude transcript root/i);
  });

  it('rejects a symlinked transcript instead of following it', () => {
    const target = join(root, 'target.jsonl');
    writeFileSync(target, transcriptLine('user', 'outside'), 'utf8');
    const link = join(claudeConfigDir, 'projects', 'project', 'linked.jsonl');
    symlinkSync(target, link);
    const input = parseClaudeStopHookInput(stopInput({ transcript_path: link }));
    expect(() =>
      readClaudeTranscriptTail(input, { claudeConfigDir, tailBytes: 4096 })
    ).toThrow(/symbolic link/i);
  });

  it('rejects a hard-linked file from outside the trusted transcript root', () => {
    const target = join(root, 'outside-hard-link.jsonl');
    writeFileSync(target, transcriptLine('user', 'outside'), 'utf8');
    const link = join(claudeConfigDir, 'projects', 'project', 'hard-linked.jsonl');
    linkSync(target, link);
    const input = parseClaudeStopHookInput(stopInput({ transcript_path: link }));

    expect(() =>
      readClaudeTranscriptTail(input, { claudeConfigDir, tailBytes: 4096 })
    ).toThrow(/hard-linked/i);
  });
});

describe('auto-capture pipeline', () => {
  it('stores additive ground facts with capture provenance but never the raw transcript', async () => {
    writeTranscript([
      transcriptLine('user', 'I prefer dark mode and live in Melbourne.'),
      transcriptLine('assistant', 'I will remember those preferences.'),
    ]);
    const llm = new ScriptedLlm([
      'prefers_theme(user, dark).\nlives_in(user, melbourne).',
    ]);

    const result = await autoCaptureClaudeStop(
      { store, llm },
      stopInput(),
      captureOptions()
    );

    expect(result.status).toBe('captured');
    expect(result.added).toEqual([
      'prefers_theme(user, dark).',
      'lives_in(user, melbourne).',
    ]);
    expect(llm.calls[0][0].content).toContain('transcript');
    expect(llm.calls[0][0].content).toContain('additive ground facts');

    const journal = readFileSync(join(root, 'memory', 'journal.log'), 'utf8');
    expect(journal).toContain('"op":"auto_capture"');
    expect(journal).toContain('"status":"started"');
    expect(journal).toContain('"status":"captured"');
    expect(journal).toContain('"origin":"claude-stop"');
    expect(journal).not.toContain('I prefer dark mode and live in Melbourne');
  });

  it('enforces configured knowledge checks on ambient capture writes', async () => {
    writeTranscript([transcriptLine('user', 'Remember forbidden A.')]);
    const llm = new ScriptedLlm(['forbidden(a).']);

    await expect(
      autoCaptureClaudeStop(
        {
          store,
          llm,
          knowledgeCheckEnforcement: {
            mode: 'strict',
            namespaces: ['default'],
            suite: {
              version: 1,
              checks: [
                {
                  name: 'forbidden stays absent',
                  query: 'forbidden(a)',
                  expect: { kind: 'empty' },
                },
              ],
            },
          },
        },
        stopInput(),
        captureOptions()
      )
    ).rejects.toMatchObject({ code: 'knowledge_check_enforcement' });
    expect(store.load('default')).toEqual([]);
  });

  it('rejects rules and retractions from auto-capture even after retry', async () => {
    writeTranscript([transcriptLine('user', 'Remember my current preference.')]);
    const llm = new ScriptedLlm([
      'retract prefers_theme(user, _).',
      'preference(X) :- requested(X).',
    ]);

    await expect(
      autoCaptureClaudeStop({ store, llm }, stopInput(), captureOptions())
    ).rejects.toThrow(/additive ground facts/i);
    expect(store.load('default')).toEqual([]);
    const journal = readFileSync(join(root, 'memory', 'journal.log'), 'utf8');
    expect(journal).toContain('"status":"failed"');
  });

  it('sends only user-authored text to the extraction model', async () => {
    // Assistant prose is not memory authority; keeping it out of the prompt
    // makes hallucinated "facts" from the assistant's own words impossible.
    writeTranscript([
      transcriptLine('user', 'I prefer dark mode.'),
      transcriptLine('assistant', 'Noted. Also, the folio plan review feature is disabled.'),
    ]);
    const llm = new ScriptedLlm(['prefers_theme(user, dark).']);
    const result = await autoCaptureClaudeStop(
      { store, llm },
      stopInput({ last_assistant_message: 'The folio feature stays disabled.' }),
      captureOptions()
    );

    expect(result.status).toBe('captured');
    const prompts = llm.calls.flat().map((message) => message.content).join('\n');
    expect(prompts).toContain('I prefer dark mode.');
    expect(prompts).not.toContain('folio');
  });

  it('deduplicates the same transcript before spending another LLM call', async () => {
    writeTranscript([transcriptLine('user', 'I prefer dark mode.')]);
    const llm = new ScriptedLlm(['prefers_theme(user, dark).']);

    const first = await autoCaptureClaudeStop(
      { store, llm },
      stopInput(),
      captureOptions()
    );
    const duplicate = await autoCaptureClaudeStop(
      { store, llm },
      stopInput(),
      captureOptions()
    );

    expect(first.status).toBe('captured');
    expect(duplicate).toMatchObject({ status: 'skipped', reason: 'duplicate' });
    expect(llm.calls).toHaveLength(1);
  });

  it('applies the same atomic integrity guard and journals a safe failure reason', async () => {
    store.assert(
      'default',
      'active(mira). :- active(Person), suspended(Person).'
    );
    writeTranscript([transcriptLine('user', 'Mira is suspended.')]);
    const llm = new ScriptedLlm(['suspended(mira).']);

    await expect(
      autoCaptureClaudeStop(
        { store, llm, integrityEnforcement: { mode: 'strict' } },
        stopInput(),
        captureOptions()
      )
    ).rejects.toMatchObject({ code: 'integrity_violation' });
    expect(store.load('default').map((clause) => clause.head.predicate)).not.toContain(
      'suspended'
    );
    expect(
      store.reviewAutoCaptures({
        days: 7,
        now: new Date('2026-08-18T02:00:00.000Z'),
      }).captures
    ).toEqual([
      expect.objectContaining({ status: 'failed', reason: 'integrity_violation' }),
    ]);
  });

  it('enforces the per-namespace UTC-day cap before the LLM call', async () => {
    writeTranscript([transcriptLine('user', 'I prefer dark mode.')]);
    const llm = new ScriptedLlm(['prefers_theme(user, dark).']);
    await autoCaptureClaudeStop(
      { store, llm },
      stopInput(),
      captureOptions({ dailyCap: 1 })
    );

    writeTranscript([transcriptLine('user', 'I prefer compact layouts.')]);
    const capped = await autoCaptureClaudeStop(
      { store, llm },
      stopInput({ last_assistant_message: 'Noted compact layouts.' }),
      captureOptions({ dailyCap: 1 })
    );

    expect(capped).toMatchObject({ status: 'skipped', reason: 'daily_cap' });
    expect(llm.calls).toHaveLength(1);
  });

  it('keeps a live old lock exclusive and records journal contention in the fallback log', async () => {
    writeTranscript([transcriptLine('user', 'I prefer dark mode.')]);
    const memoryRoot = join(root, 'memory');
    mkdirSync(memoryRoot, { recursive: true });
    const lockPath = join(memoryRoot, '.journal.lock');
    const descriptor = openSync(lockPath, 'wx', 0o600);
    writeFileSync(
      descriptor,
      `${JSON.stringify({ pid: process.pid, createdAt: '2026-08-17T00:00:00.000Z' })}\n`,
      'utf8'
    );
    const old = new Date(Date.now() - 60_000);
    utimesSync(lockPath, old, old);
    const llm = new ScriptedLlm([]);

    try {
      await expect(
        autoCaptureClaudeStop({ store, llm }, stopInput(), captureOptions())
      ).rejects.toThrow(/timed out waiting for memory lock 'journal'/i);
    } finally {
      closeSync(descriptor);
      unlinkSync(lockPath);
    }

    expect(llm.calls).toHaveLength(0);
    expect(readFileSync(join(memoryRoot, 'capture-errors.log'), 'utf8')).toContain(
      '"reason":"journal_unavailable"'
    );
    expect(
      store.reviewAutoCaptures({
        days: 7,
        now: new Date('2026-08-18T02:00:00.000Z'),
      }).captures
    ).toEqual([
      expect.objectContaining({ status: 'failed', reason: 'journal_unavailable' }),
    ]);
  });

  it('fails closed on sensitive transcript text and journals a safe reason', async () => {
    writeTranscript([transcriptLine('user', 'My API key is sk-supersecretvalue.')]);
    const llm = new ScriptedLlm([]);

    await expect(
      autoCaptureClaudeStop({ store, llm }, stopInput(), captureOptions())
    ).rejects.toThrow(/sensitive transcript/i);
    expect(llm.calls).toHaveLength(0);
    const journal = readFileSync(join(root, 'memory', 'journal.log'), 'utf8');
    expect(journal).toContain('"reason":"sensitive_text"');
    expect(journal).not.toContain('sk-supersecretvalue');
  });

  it('reviews every capture and prunes selected current facts explicitly', async () => {
    writeTranscript([transcriptLine('user', 'I prefer dark mode and tea.')]);
    const llm = new ScriptedLlm([
      'prefers_theme(user, dark).\nprefers_drink(user, tea).',
    ]);
    await autoCaptureClaudeStop(
      { store, llm },
      stopInput(),
      captureOptions()
    );

    const review = store.reviewAutoCaptures({
      days: 7,
      now: new Date('2026-08-18T02:00:00.000Z'),
    });
    expect(review.captures).toHaveLength(1);
    expect(review.captures[0]).toMatchObject({ status: 'captured', namespace: 'default' });
    expect(review.facts.map((fact) => fact.clause)).toEqual([
      'prefers_drink(user, tea).',
      'prefers_theme(user, dark).',
    ]);
    expect(review.facts.every((fact) => fact.current)).toBe(true);

    const pruned = store.pruneAutoCaptureFacts([review.facts[0]], {
      now: new Date('2026-08-18T02:01:00.000Z'),
    });
    expect(pruned.removed).toBe(1);
    expect(
      store.reviewAutoCaptures({
        days: 7,
        now: new Date('2026-08-18T02:02:00.000Z'),
      }).facts
    ).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ clause: 'prefers_drink(user, tea).', current: false }),
        expect.objectContaining({ clause: 'prefers_theme(user, dark).', current: true }),
      ])
    );
  });

  it('preflights integrity-enforced pruning before removing any reviewed fact', async () => {
    writeTranscript([transcriptLine('user', 'Alice is managed by Mira.')]);
    await autoCaptureClaudeStop(
      { store, llm: new ScriptedLlm(['manager(alice, mira).']) },
      stopInput(),
      captureOptions()
    );
    store.assert(
      'default',
      'employee(alice). :- employee(Person), \\+ manager(Person, _).'
    );
    const review = store.reviewAutoCaptures({
      days: 7,
      now: new Date('2026-08-18T02:00:00.000Z'),
    });

    expect(() =>
      store.pruneAutoCaptureFacts(review.facts, {
        integrity: { mode: 'strict' },
      })
    ).toThrowError(expect.objectContaining({ code: 'integrity_violation' }));
    expect(store.load('default').map((clause) => clause.head.predicate)).toContain(
      'manager'
    );
  });

  it('does not let a caller retarget a valid review identity at another fact', async () => {
    writeTranscript([transcriptLine('user', 'I prefer dark mode.')]);
    const llm = new ScriptedLlm(['prefers_theme(user, dark).']);
    await autoCaptureClaudeStop(
      { store, llm },
      stopInput(),
      captureOptions()
    );
    store.assert('default', 'manual_fact(keep_me).');
    const [reviewed] = store.reviewAutoCaptures({
      days: 7,
      now: new Date('2026-08-18T02:00:00.000Z'),
    }).facts;

    expect(() =>
      store.pruneAutoCaptureFacts([{ ...reviewed, clause: 'manual_fact(keep_me).' }])
    ).toThrow(/not present in the journal/i);
    expect(store.load('default').map((clause) => clause.head.predicate)).toContain(
      'manual_fact'
    );
  });

  it('does not prune a fact that was later removed and explicitly re-added manually', async () => {
    writeTranscript([transcriptLine('user', 'I prefer dark mode.')]);
    const llm = new ScriptedLlm(['prefers_theme(user, dark).']);
    await autoCaptureClaudeStop(
      { store, llm },
      stopInput(),
      captureOptions()
    );
    store.retract('default', 'prefers_theme(user, dark)', {
      opId: 'manual-remove',
      at: new Date('2026-08-17T02:10:00.000Z'),
    });
    store.assert('default', 'prefers_theme(user, dark).', {
      opId: 'manual-add',
      sourceText: 'The user explicitly re-added dark mode',
      at: new Date('2026-08-17T02:20:00.000Z'),
    });
    const review = store.reviewAutoCaptures({
      days: 7,
      now: new Date('2026-08-18T02:00:00.000Z'),
    });

    expect(review.facts[0]).toMatchObject({
      clause: 'prefers_theme(user, dark).',
      current: false,
    });
    expect(store.pruneAutoCaptureFacts([review.facts[0]]).removed).toBe(0);
    expect(store.load('default')).toHaveLength(1);
  });
});

describe('Claude hook installer', () => {
  it('merges one safe async exec-form hook idempotently and preserves unrelated settings', () => {
    const settingsPath = join(claudeConfigDir, 'settings.json');
    writeFileSync(
      settingsPath,
      JSON.stringify({
        permissions: { deny: ['Read(.env)'] },
        hooks: {
          Stop: [
            {
              matcher: '',
              hooks: [{ type: 'command', command: 'notify-existing' }],
            },
          ],
        },
      }),
      'utf8'
    );

    const options = {
      settingsPath,
      nodePath: '/usr/local/bin/node',
      cliPath: '/opt/rembero/dist/cli.js',
      namespace: 'personal',
      dailyCap: 7,
      tailBytes: 12_000,
    };
    expect(installClaudeHook(options).changed).toBe(true);
    expect(installClaudeHook(options).changed).toBe(false);

    const settings = JSON.parse(readFileSync(settingsPath, 'utf8'));
    expect(settings.permissions).toEqual({ deny: ['Read(.env)'] });
    const handlers = settings.hooks.Stop.flatMap(
      (group: { hooks: Record<string, unknown>[] }) => group.hooks
    );
    expect(handlers).toContainEqual({ type: 'command', command: 'notify-existing' });
    expect(handlers).toContainEqual({
      type: 'command',
      command: '/usr/local/bin/node',
      args: [
        '/opt/rembero/dist/cli.js',
        'remember',
        '--batch',
        '--managed-by',
        MANAGED_HOOK_MARKER,
        '--namespace',
        'personal',
        '--daily-cap',
        '7',
        '--tail-bytes',
        '12000',
      ],
      async: true,
      timeout: 120,
    });
  });

  it('installs a managed SessionStart brief hook alongside Stop and removes both', () => {
    const settingsPath = join(claudeConfigDir, 'settings.json');
    installClaudeHook({
      settingsPath,
      nodePath: '/usr/local/bin/node',
      cliPath: '/opt/rembero/dist/cli.js',
      namespace: 'personal',
      dailyCap: 5,
      tailBytes: 8192,
    });
    const settings = JSON.parse(readFileSync(settingsPath, 'utf8'));
    const startHandlers = settings.hooks.SessionStart.flatMap(
      (group: { hooks: Record<string, unknown>[] }) => group.hooks
    );
    expect(startHandlers).toContainEqual({
      type: 'command',
      command: '/usr/local/bin/node',
      args: [
        '/opt/rembero/dist/cli.js',
        'session-brief',
        '--managed-by',
        MANAGED_HOOK_MARKER,
        '--namespace',
        'personal',
      ],
      timeout: 15,
    });

    expect(removeClaudeHook({ settingsPath }).changed).toBe(true);
    const after = JSON.parse(readFileSync(settingsPath, 'utf8'));
    expect(after.hooks?.SessionStart).toBeUndefined();
    const stopHandlers = (after.hooks?.Stop ?? []).flatMap(
      (group: { hooks: Record<string, unknown>[] }) => group.hooks
    );
    expect(stopHandlers).toEqual([]);
  });

  it('removes only the managed hook and leaves unrelated hooks intact', () => {
    const settingsPath = join(claudeConfigDir, 'settings.json');
    installClaudeHook({
      settingsPath,
      nodePath: '/usr/local/bin/node',
      cliPath: '/opt/rembero/dist/cli.js',
      namespace: 'default',
      dailyCap: 5,
      tailBytes: 8192,
    });
    const settings = JSON.parse(readFileSync(settingsPath, 'utf8'));
    settings.hooks.Stop.unshift({
      matcher: '',
      hooks: [{ type: 'command', command: 'notify-existing' }],
    });
    writeFileSync(settingsPath, JSON.stringify(settings), 'utf8');

    expect(removeClaudeHook({ settingsPath }).changed).toBe(true);
    expect(removeClaudeHook({ settingsPath }).changed).toBe(false);
    const after = JSON.parse(readFileSync(settingsPath, 'utf8'));
    expect(after.hooks.Stop).toEqual([
      {
        matcher: '',
        hooks: [{ type: 'command', command: 'notify-existing' }],
      },
    ]);
  });
});
