import { spawn } from 'node:child_process';
import {
  evaluateQuerySpecWithProof,
  isAggregateRule,
  isIntegrityConstraint,
  parseProgram,
  parseQuerySpec,
  type Clause,
  type ProofStep,
  type Term,
} from '../engine/index.js';
import {
  MEMORY_STACK_PROTOCOL_VERSION,
  publicCase,
  type MemoryCell,
  type MemoryRow,
  type MemoryStackAdapter,
  type MemoryStackAdapterDescriptor,
  type MemoryStackCase,
  type MemoryStackCaseObservation,
  type MemoryStackEvent,
  type MemoryStackQuestion,
  type RankedMemory,
} from './memory-stack-contract.js';

const DEFAULT_LIMITS = {
  maxFacts: 25_000,
  maxIterations: 10_000,
  maxRows: 2_000,
  maxProofDepth: 64,
  maxProofNodes: 10_000,
  maxProofsPerRow: 1,
} as const;

function termCell(term: Term): MemoryCell {
  if (term.type === 'atom') return { type: 'atom', value: term.value };
  if (term.type === 'num') return { type: 'number', value: term.value };
  throw new Error(`answer column resolved to non-ground ${term.type}`);
}

function answerRows(
  explained: ReturnType<typeof evaluateQuerySpecWithProof>,
  question: MemoryStackQuestion
): MemoryRow[] {
  return explained.map(({ bindings }) =>
    question.answerColumns.map((column) => {
      const value = bindings[column];
      if (value === undefined) throw new Error(`query did not bind answer column ${column}`);
      return termCell(value);
    })
  );
}

function factKey(predicate: string, values: readonly (string | number)[]): string {
  return JSON.stringify([predicate, ...values]);
}

interface ProgramSources {
  factSources: Map<string, string>;
  ruleSources: Map<number, string>;
}

function addProgram(
  events: readonly MemoryStackEvent[],
  includeTentative: boolean,
  directFactsOnly: boolean
): { clauses: Clause[]; sources: ProgramSources } {
  const clauses: Clause[] = [];
  const factSources = new Map<string, string>();
  const ruleSources = new Map<number, string>();
  let ruleIndex = 0;
  for (const event of events) {
    if (event.trust === 'tentative' && !includeTentative) continue;
    for (const clause of parseProgram(event.clauses)) {
      if (isIntegrityConstraint(clause)) continue;
      const hasBody = isAggregateRule(clause) || clause.body.length > 0;
      if (hasBody) {
        ruleIndex++;
        if (!directFactsOnly) {
          clauses.push(clause);
          ruleSources.set(ruleIndex, event.id);
        }
        continue;
      }
      clauses.push(clause);
      factSources.set(
        factKey(
          clause.head.predicate,
          clause.head.args.map((term) => {
            if (term.type === 'atom' || term.type === 'num') return term.value;
            throw new Error(`fixture fact ${event.id} is not ground`);
          })
        ),
        event.id
      );
    }
  }
  return { clauses, sources: { factSources, ruleSources } };
}

function collectStepSources(
  proof: ProofStep,
  sources: ProgramSources,
  found: Set<string>
): void {
  if ('negated' in proof) return;
  const factSource = sources.factSources.get(factKey(proof.predicate, proof.values));
  if (factSource !== undefined) found.add(factSource);
  if (proof.rule !== undefined) {
    const ruleSource = sources.ruleSources.get(proof.rule);
    if (ruleSource !== undefined) found.add(ruleSource);
  }
  for (const child of proof.because ?? []) collectStepSources(child, sources, found);
  for (const contributor of proof.aggregate?.contributors ?? []) {
    for (const child of contributor.proofs) collectStepSources(child, sources, found);
  }
}

function proofSources(
  explained: ReturnType<typeof evaluateQuerySpecWithProof>,
  sources: ProgramSources
): string[] {
  const found = new Set<string>();
  for (const row of explained) {
    for (const proof of row.proofs) {
      if ('aggregated' in proof) {
        for (const contributor of proof.contributors) {
          for (const child of contributor.proofs) collectStepSources(child, sources, found);
        }
      } else {
        collectStepSources(proof, sources, found);
      }
    }
  }
  return [...found];
}

function engineAdapter(
  descriptor: MemoryStackAdapterDescriptor,
  directFactsOnly: boolean
): MemoryStackAdapter {
  return {
    describe: () => descriptor,
    async runCase(testCase) {
      const questions = testCase.questions.map((question) => {
        const started = performance.now();
        try {
          const { clauses, sources } = addProgram(
            testCase.events,
            question.includeTentative === true,
            directFactsOnly
          );
          const explained = evaluateQuerySpecWithProof(
            clauses,
            parseQuerySpec(question.query),
            DEFAULT_LIMITS
          );
          const citations = proofSources(explained, sources);
          return {
            questionId: question.id,
            status: explained.length === 0 ? ('no_match' as const) : ('answered' as const),
            answerRows: answerRows(explained, question),
            retrieved: citations.map((eventId, index) => ({ eventId, rank: index + 1 })),
            citations,
            wallMs: performance.now() - started,
          };
        } catch (error) {
          return {
            questionId: question.id,
            status: 'error' as const,
            answerRows: [],
            retrieved: [],
            citations: [],
            wallMs: performance.now() - started,
            error: error instanceof Error ? error.message : String(error),
          };
        }
      });
      return { caseId: testCase.id, questions };
    },
  };
}

export function createRemberoMemoryAdapter(): MemoryStackAdapter {
  return engineAdapter(
    {
      id: 'rembero-engine',
      version: '1',
      capabilities: {
        answerRows: true,
        rankedRetrieval: true,
        citations: true,
        rules: true,
        temporalUpdates: true,
        trustViews: true,
      },
    },
    false
  );
}

export function createDirectFactAdapter(): MemoryStackAdapter {
  return engineAdapter(
    {
      id: 'direct-fact-scan',
      version: '1',
      capabilities: {
        answerRows: true,
        rankedRetrieval: true,
        citations: true,
        rules: false,
        temporalUpdates: false,
        trustViews: true,
      },
    },
    true
  );
}

function tokens(text: string): Set<string> {
  return new Set(
    text
      .toLocaleLowerCase('en-US')
      .normalize('NFKC')
      .match(/[\p{L}\p{N}_]+/gu) ?? []
  );
}

function rankedObservation(
  ranker: (
    testCase: MemoryStackCase,
    question: MemoryStackQuestion
  ) => RankedMemory[],
  adapterId: string
): MemoryStackAdapter {
  return {
    describe: () => ({
      id: adapterId,
      version: '1',
      capabilities: {
        answerRows: false,
        rankedRetrieval: true,
        citations: false,
        rules: false,
        temporalUpdates: adapterId === 'recency-top-k',
        trustViews: false,
      },
    }),
    async runCase(testCase) {
      return {
        caseId: testCase.id,
        questions: testCase.questions.map((question) => {
          const started = performance.now();
          return {
            questionId: question.id,
            status: 'unsupported',
            answerRows: [],
            retrieved: ranker(testCase, question),
            citations: [],
            wallMs: performance.now() - started,
          };
        }),
      };
    },
  };
}

export function createLexicalAdapter(): MemoryStackAdapter {
  return rankedObservation(
    (testCase, question) => {
      const queryTokens = tokens(question.text);
      const limit = question.topK ?? 5;
      return testCase.events
        .map((event) => {
          const eventTokens = tokens(event.text);
          const overlap = [...queryTokens].filter((token) => eventTokens.has(token)).length;
          return { event, overlap };
        })
        .filter(({ overlap }) => overlap > 0)
        .sort(
          (left, right) =>
            right.overlap - left.overlap || left.event.id.localeCompare(right.event.id)
        )
        .slice(0, limit)
        .map(({ event }, index) => ({ eventId: event.id, rank: index + 1 }));
    },
    'lexical-overlap-top-k'
  );
}

export function createRecencyAdapter(): MemoryStackAdapter {
  return rankedObservation(
    (testCase, question) =>
      [...testCase.events]
        .sort(
          (left, right) =>
            right.at.localeCompare(left.at) || left.id.localeCompare(right.id)
        )
        .slice(0, question.topK ?? 5)
        .map((event, index) => ({ eventId: event.id, rank: index + 1 })),
    'recency-top-k'
  );
}

export interface ExternalCommandAdapterOptions {
  executable: string;
  args?: string[];
  workingDirectory?: string;
  timeoutMs?: number;
  maxOutputBytes?: number;
  env?: Record<string, string>;
}

function validateExternalObservation(
  value: unknown,
  expectedCaseId: string
): MemoryStackCaseObservation {
  if (value === null || typeof value !== 'object') {
    throw new Error('external adapter returned a non-object');
  }
  const observation = value as Partial<MemoryStackCaseObservation>;
  if (observation.caseId !== expectedCaseId || !Array.isArray(observation.questions)) {
    throw new Error('external adapter returned an invalid case observation');
  }
  return observation as MemoryStackCaseObservation;
}

export function createExternalCommandAdapter(
  descriptor: MemoryStackAdapterDescriptor,
  options: ExternalCommandAdapterOptions
): MemoryStackAdapter {
  const timeoutMs = options.timeoutMs ?? 30_000;
  const maxOutputBytes = options.maxOutputBytes ?? 1_000_000;
  return {
    describe: () => descriptor,
    async runCase(testCase) {
      return await new Promise<MemoryStackCaseObservation>((resolve, reject) => {
        const child = spawn(options.executable, options.args ?? [], {
          shell: false,
          stdio: ['pipe', 'pipe', 'pipe'],
          cwd: options.workingDirectory,
          env: {
            PATH: process.env.PATH ?? '/usr/bin:/bin',
            LANG: process.env.LANG ?? 'C.UTF-8',
            ...options.env,
          },
        });
        const stdout: Buffer[] = [];
        let outputBytes = 0;
        let diagnosticBytes = 0;
        let settled = false;
        const finish = (error?: Error, value?: MemoryStackCaseObservation) => {
          if (settled) return;
          settled = true;
          clearTimeout(timer);
          if (error !== undefined) reject(error);
          else resolve(value!);
        };
        const timer = setTimeout(() => {
          child.kill('SIGKILL');
          finish(new Error(`external adapter timed out after ${timeoutMs}ms`));
        }, timeoutMs);
        child.stdout.on('data', (chunk: Buffer) => {
          outputBytes += chunk.length;
          if (outputBytes > maxOutputBytes) {
            child.kill('SIGKILL');
            finish(new Error(`external adapter stdout exceeded ${maxOutputBytes} bytes`));
            return;
          }
          stdout.push(chunk);
        });
        child.stderr.on('data', (chunk: Buffer) => {
          diagnosticBytes += chunk.length;
        });
        child.on('error', (error) => finish(error));
        // A child may exit before reading its request (crash, refusal); the
        // close handler reports that outcome, so a broken stdin pipe is not
        // itself an error worth crashing the process for.
        child.stdin.on('error', (error: NodeJS.ErrnoException) => {
          if (error.code !== 'EPIPE' && error.code !== 'ERR_STREAM_DESTROYED') {
            finish(error);
          }
        });
        child.on('close', (code, signal) => {
          if (settled) return;
          if (code !== 0) {
            finish(
              new Error(
                `external adapter exited with ${signal ?? code}; stderr suppressed (${diagnosticBytes} bytes)`
              )
            );
            return;
          }
          try {
            const parsed = JSON.parse(Buffer.concat(stdout).toString('utf8')) as unknown;
            finish(undefined, validateExternalObservation(parsed, testCase.id));
          } catch (error) {
            finish(error instanceof Error ? error : new Error(String(error)));
          }
        });
        child.stdin.end(
          JSON.stringify({
            protocolVersion: MEMORY_STACK_PROTOCOL_VERSION,
            case: publicCase(testCase),
          })
        );
      });
    },
  };
}
