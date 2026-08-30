import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import {
  StdioClientTransport,
  getDefaultEnvironment,
} from '@modelcontextprotocol/sdk/client/stdio.js';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { resolve } from 'node:path';
import { benchmarkDigest } from './memory-stack-contract.js';
import { createRemberoMemoryAdapter } from './memory-stack-adapters.js';
import {
  MEMORY_STACK_CASES,
  MEMORY_STACK_LABELS,
  MEMORY_STACK_SUITE,
} from './memory-stack-fixtures.js';
import { runMemoryStackBenchmark } from './memory-stack-score.js';
import {
  runAgentDbScaleSweep,
  type AgentDbScaleSweep,
} from './agent-db-scale.js';

export const AGENT_DB_SCORECARD_VERSION = 'remembero.agent-db-scorecard.v1' as const;

export interface AgentDbScorecard {
  schemaVersion: typeof AGENT_DB_SCORECARD_VERSION;
  generatedAt: string;
  evidenceDigest: string;
  accuracy: {
    questions: number;
    answerAccuracy: number;
    answerabilityAccuracy: number;
    retrievalPrecisionAtK: number;
    citationPrecision: number;
    citationRecall: number;
    staleLeakageRate: number;
    operationalErrors: number;
  };
  speed: {
    repetitions: number;
    warmupRepetitions: number;
    observations: number;
    engineMedianMs: number;
    engineP95Ms: number;
    mcpStartupMs: number;
    mcpToolDiscoveryMs: number;
    mcpExplainRoundTripMs: number;
    scale: AgentDbScaleSweep;
  };
  cost: {
    structuredQueryModelCalls: 0;
    structuredQueryEmbeddingCalls: 0;
    structuredQueryRemoteNetworkCalls: 0;
    structuredQueryRequiredApiKeys: 0;
    structuredQueryMarginalProviderCostUsd: 0;
    boundary: string;
  };
  ease: {
    installCommand: string;
    serverCommand: string;
    setupCommandCount: 2;
    readTool: 'explain_query';
    naturalLanguageReadTool: 'recall_explain';
    discoveredTools: number;
    proofRoundTripPassed: boolean;
    outputBytes: number;
  };
  gates: {
    passed: boolean;
    failures: string[];
    thresholds: {
      engineP95Ms: number;
      mcpExplainRoundTripMs: number;
      scaleParseMs: number;
      scaleQueryP95Ms: number;
      scaleProofP95Ms: number;
    };
  };
}

interface McpProbe {
  startupMs: number;
  toolDiscoveryMs: number;
  explainRoundTripMs: number;
  discoveredTools: number;
  proofRoundTripPassed: boolean;
  outputBytes: number;
}

function percentile(values: readonly number[], quantile: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.ceil((sorted.length - 1) * quantile)];
}

function textContent(result: unknown): string {
  if (result === null || typeof result !== 'object') {
    throw new Error('MCP tool returned a malformed payload');
  }
  const payload = result as { isError?: boolean; content?: unknown };
  if (!Array.isArray(payload.content)) {
    throw new Error('MCP tool returned no content array');
  }
  const text = payload.content.find(
    (item): item is { type: 'text'; text: string } =>
      item !== null &&
      typeof item === 'object' &&
      (item as { type?: unknown }).type === 'text' &&
      typeof (item as { text?: unknown }).text === 'string'
  );
  if (payload.isError === true || text === undefined) {
    throw new Error('MCP tool returned no successful text payload');
  }
  return text.text;
}

async function probeRealMcpProcess(): Promise<McpProbe> {
  const root = await mkdtemp(resolve(tmpdir(), 'remembero-agent-db-scorecard-'));
  const transport = new StdioClientTransport({
    command: process.execPath,
    args: [resolve(process.cwd(), 'dist', 'cli.js'), 'serve'],
    env: {
      ...getDefaultEnvironment(),
      REMBERO_HOME: root,
    },
    stderr: 'pipe',
  });
  transport.stderr?.on('data', () => undefined);
  const client = new Client({ name: 'remembero-agent-db-scorecard', version: '1.0.0' });
  const startupStarted = performance.now();
  try {
    await client.connect(transport);
    const startupMs = performance.now() - startupStarted;
    const discoveryStarted = performance.now();
    const tools = await client.listTools();
    const toolDiscoveryMs = performance.now() - discoveryStarted;
    const names = new Set(tools.tools.map(({ name }) => name));
    if (!names.has('explain_query') || !names.has('recall_explain')) {
      throw new Error('MCP server did not expose required agent read tools');
    }

    await client.callTool({
      name: 'assert_facts',
      arguments: {
        namespace: 'agent',
        opId: 'agent-db-scorecard-seed',
        clauses: `
          project_owner(atlas, rahul).
          project_contributor(atlas, maya).
          collaborator(Person, Project) :-
            project_owner(Project, Owner),
            project_contributor(Project, Person),
            Owner != Person.
        `,
      },
    });

    const explainStarted = performance.now();
    const explained = await client.callTool({
      name: 'explain_query',
      arguments: {
        query: 'collaborator(Person, atlas)',
        namespaces: ['agent'],
        proofLimit: 4,
      },
    });
    const explainRoundTripMs = performance.now() - explainStarted;
    const payloadText = textContent(explained);
    const payload = JSON.parse(payloadText) as unknown;
    const serialized = JSON.stringify(payload);
    const proofRoundTripPassed =
      serialized.includes('maya') &&
      serialized.includes('atlas') &&
      serialized.includes('project_owner') &&
      serialized.includes('project_contributor');
    if (!proofRoundTripPassed) {
      throw new Error('MCP proof round trip did not contain the expected answer and support');
    }
    return {
      startupMs,
      toolDiscoveryMs,
      explainRoundTripMs,
      discoveredTools: names.size,
      proofRoundTripPassed,
      outputBytes: Buffer.byteLength(payloadText, 'utf8'),
    };
  } finally {
    await client.close().catch(() => undefined);
    await rm(root, { recursive: true, force: true });
  }
}

function requiredMetric(value: number | null, label: string): number {
  if (value === null) throw new Error(`${label} was not measured`);
  return value;
}

export async function buildAgentDbScorecard(options: {
  repetitions?: number;
  warmupRepetitions?: number;
  generatedAt?: string;
  engineP95ThresholdMs?: number;
  mcpRoundTripThresholdMs?: number;
  scaleFactCounts?: readonly number[];
  scaleRepetitions?: number;
  scaleParseThresholdMs?: number;
  scaleQueryP95ThresholdMs?: number;
  scaleProofP95ThresholdMs?: number;
} = {}): Promise<AgentDbScorecard> {
  const repetitions = options.repetitions ?? 10;
  if (!Number.isInteger(repetitions) || repetitions < 1 || repetitions > 100) {
    throw new Error('repetitions must be an integer between 1 and 100');
  }
  const warmupRepetitions = options.warmupRepetitions ?? 1;
  if (!Number.isInteger(warmupRepetitions) || warmupRepetitions < 0 || warmupRepetitions > 10) {
    throw new Error('warmupRepetitions must be an integer between 0 and 10');
  }
  // The engine gate measures steady-state evaluation latency, not first-call JIT
  // compilation; untimed warmup passes keep process warmup out of the percentile.
  for (let index = 0; index < warmupRepetitions; index++) {
    await runMemoryStackBenchmark({
      suite: MEMORY_STACK_SUITE,
      cases: MEMORY_STACK_CASES,
      labels: MEMORY_STACK_LABELS,
      adapter: createRemberoMemoryAdapter(),
      generatedAt: options.generatedAt,
    });
  }
  const wallTimes: number[] = [];
  let reference: Awaited<ReturnType<typeof runMemoryStackBenchmark>> | null = null;
  for (let index = 0; index < repetitions; index++) {
    const run = await runMemoryStackBenchmark({
      suite: MEMORY_STACK_SUITE,
      cases: MEMORY_STACK_CASES,
      labels: MEMORY_STACK_LABELS,
      adapter: createRemberoMemoryAdapter(),
      generatedAt: options.generatedAt,
    });
    reference ??= run;
    for (const testCase of run.cases) {
      for (const question of testCase.observation.questions) wallTimes.push(question.wallMs);
    }
  }
  if (reference === null) throw new Error('scorecard produced no benchmark run');
  const scale = runAgentDbScaleSweep({
    ...(options.scaleFactCounts === undefined
      ? {}
      : { factCounts: options.scaleFactCounts }),
    repetitions: options.scaleRepetitions ?? 3,
  });
  const mcp = await probeRealMcpProcess();
  const engineP95ThresholdMs = options.engineP95ThresholdMs ?? 25;
  const mcpRoundTripThresholdMs = options.mcpRoundTripThresholdMs ?? 500;
  const scaleParseThresholdMs = options.scaleParseThresholdMs ?? 2_000;
  const scaleQueryP95ThresholdMs = options.scaleQueryP95ThresholdMs ?? 250;
  const scaleProofP95ThresholdMs = options.scaleProofP95ThresholdMs ?? 500;
  const answerAccuracy = requiredMetric(reference.summary.answerAccuracy, 'answer accuracy');
  const answerabilityAccuracy = requiredMetric(
    reference.summary.answerabilityAccuracy,
    'answerability accuracy'
  );
  const retrievalPrecisionAtK = requiredMetric(
    reference.summary.retrievalPrecisionAtK,
    'retrieval precision at k'
  );
  const citationPrecision = requiredMetric(
    reference.summary.citationPrecision,
    'citation precision'
  );
  const citationRecall = requiredMetric(reference.summary.citationRecall, 'citation recall');
  const failures: string[] = [];
  if (answerAccuracy !== 1) failures.push('answer accuracy must be 100%');
  if (answerabilityAccuracy !== 1) failures.push('answerability accuracy must be 100%');
  if (retrievalPrecisionAtK !== 1) failures.push('retrieval precision at k must be 100%');
  if (citationPrecision !== 1 || citationRecall !== 1) {
    failures.push('citation precision and recall must be 100%');
  }
  if (reference.summary.staleLeakageRate !== 0) failures.push('stale leakage must be zero');
  if (reference.summary.operationalErrors !== 0) failures.push('operational errors must be zero');
  const engineP95Ms = percentile(wallTimes, 0.95);
  if (engineP95Ms > engineP95ThresholdMs) {
    failures.push(`engine p95 ${engineP95Ms.toFixed(2)}ms exceeds ${engineP95ThresholdMs}ms`);
  }
  if (mcp.explainRoundTripMs > mcpRoundTripThresholdMs) {
    failures.push(
      `MCP explain round trip ${mcp.explainRoundTripMs.toFixed(2)}ms exceeds ${mcpRoundTripThresholdMs}ms`
    );
  }
  if (!mcp.proofRoundTripPassed) failures.push('real MCP proof round trip failed');
  for (const result of scale.cases) {
    if (!result.rowsAndProofsCorrect) failures.push(`${result.facts}-fact row/proof mismatch`);
    if (result.indexedRelationLookups < 1) {
      failures.push(`${result.facts}-fact query did not use relation indexes`);
    }
    if (result.candidateFactsVisited > 10) {
      failures.push(
        `${result.facts}-fact query visited ${result.candidateFactsVisited} candidates`
      );
    }
  }
  if (scale.maxima.parseMs > scaleParseThresholdMs) {
    failures.push(
      `scale parse ${scale.maxima.parseMs.toFixed(2)}ms exceeds ${scaleParseThresholdMs}ms`
    );
  }
  if (scale.maxima.queryP95Ms > scaleQueryP95ThresholdMs) {
    failures.push(
      `scale query p95 ${scale.maxima.queryP95Ms.toFixed(2)}ms exceeds ${scaleQueryP95ThresholdMs}ms`
    );
  }
  if (scale.maxima.proofP95Ms > scaleProofP95ThresholdMs) {
    failures.push(
      `scale proof p95 ${scale.maxima.proofP95Ms.toFixed(2)}ms exceeds ${scaleProofP95ThresholdMs}ms`
    );
  }

  const semanticEvidence = {
    suiteDigest: reference.suite.digest,
    semanticDigest: reference.semanticDigest,
    answerAccuracy,
    answerabilityAccuracy,
    retrievalPrecisionAtK,
    citationPrecision,
    citationRecall,
    staleLeakageRate: reference.summary.staleLeakageRate,
    mcpProofRoundTripPassed: mcp.proofRoundTripPassed,
    modelCalls: 0,
    embeddingCalls: 0,
    remoteNetworkCalls: 0,
    requiredApiKeys: 0,
    scaleFacts: scale.maxima.facts,
    scaleRowsAndProofsCorrect: scale.cases.every(({ rowsAndProofsCorrect }) =>
      Boolean(rowsAndProofsCorrect)
    ),
  };

  return {
    schemaVersion: AGENT_DB_SCORECARD_VERSION,
    generatedAt: options.generatedAt ?? new Date().toISOString(),
    evidenceDigest: benchmarkDigest(semanticEvidence),
    accuracy: {
      questions: reference.summary.questions,
      answerAccuracy,
      answerabilityAccuracy,
      retrievalPrecisionAtK,
      citationPrecision,
      citationRecall,
      staleLeakageRate: reference.summary.staleLeakageRate,
      operationalErrors: reference.summary.operationalErrors,
    },
    speed: {
      repetitions,
      warmupRepetitions,
      observations: wallTimes.length,
      engineMedianMs: percentile(wallTimes, 0.5),
      engineP95Ms,
      mcpStartupMs: mcp.startupMs,
      mcpToolDiscoveryMs: mcp.toolDiscoveryMs,
      mcpExplainRoundTripMs: mcp.explainRoundTripMs,
      scale,
    },
    cost: {
      structuredQueryModelCalls: 0,
      structuredQueryEmbeddingCalls: 0,
      structuredQueryRemoteNetworkCalls: 0,
      structuredQueryRequiredApiKeys: 0,
      structuredQueryMarginalProviderCostUsd: 0,
      boundary:
        'Deterministic structured query and explain path only; natural-language translation and answer phrasing may use a configured model.',
    },
    ease: {
      installCommand: 'npm install -g remembero',
      serverCommand: 'remembero serve',
      setupCommandCount: 2,
      readTool: 'explain_query',
      naturalLanguageReadTool: 'recall_explain',
      discoveredTools: mcp.discoveredTools,
      proofRoundTripPassed: mcp.proofRoundTripPassed,
      outputBytes: mcp.outputBytes,
    },
    gates: {
      passed: failures.length === 0,
      failures,
      thresholds: {
        engineP95Ms: engineP95ThresholdMs,
        mcpExplainRoundTripMs: mcpRoundTripThresholdMs,
        scaleParseMs: scaleParseThresholdMs,
        scaleQueryP95Ms: scaleQueryP95ThresholdMs,
        scaleProofP95Ms: scaleProofP95ThresholdMs,
      },
    },
  };
}
