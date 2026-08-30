import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';
import {
  entityIdentityFromEnv,
  integrityEnforcementFromEnv,
  knowledgeCheckEnforcementFromEnv,
  recallAnswerModeFromEnv,
  recallSchemaPredicateLimitFromEnv,
  validTimeModeFromEnv,
} from '../env.js';
import type {
  PipelineDeps,
  RecallRelatedKnowledgeOptions,
} from '../llm/pipeline.js';
import { MAX_RECALL_SCHEMA_PREDICATES } from '../llm/schema.js';
import { MAX_PROOFS_PER_ROW } from '../engine/index.js';
import {
  MAX_INPUT_BYTES,
  MAX_NAMESPACE_COUNT,
  assertBoundedOutput,
  stringifyBoundedResult,
} from '../safety.js';
import {
  checkIntegrityTool,
  conflictViewsTool,
  assertFactsTool,
  assertTentativeTool,
  checkpointJournalTool,
  explainQueryTool,
  forgetTool,
  historyTool,
  listMemoriesTool,
  listCheckpointsTool,
  queryTool,
  recallExplainTool,
  recallTool,
  rememberTool,
  proposeMemoryTool,
  applyMemoryProposalTool,
  knowledgeHealthTool,
  resolveTentativeTool,
  reviewTentativeTool,
  supersedeFactsTool,
  whatIfTool,
  applyRuleChangeProposalTool,
  whyNotTool,
  topologyTool,
  recordedDiffTool,
  repairPlanTool,
  auditRulesTool,
  searchKnowledgeTool,
  semanticSearchKnowledgeTool,
  prepareSemanticKnowledgeTool,
  browseKnowledgeGraphTool,
  connectKnowledgeGraphTool,
  exportKnowledgeBundleTool,
  verifyKnowledgeBundleTool,
  runKnowledgeChecksTool,
  profileKnowledgeTool,
} from './tools.js';
import { lazyEmbeddingClientFromEnv } from '../llm/embeddings.js';
import {
  FileEmbeddingCache,
  LayeredEmbeddingCache,
  MemoryEmbeddingCache,
} from '../knowledge/semantic-search.js';
import {
  IncompleteHistoryError,
  MAX_HISTORY_EVENTS,
  MAX_SUPERSEDE_PATTERNS,
} from '../store/store.js';
import { MAX_INTEGRITY_VIOLATIONS } from '../knowledge/integrity.js';
import { MAX_CONFLICT_FOCUS_BYTES } from '../knowledge/conflicts.js';
import {
  MAX_COUNTERFACTUAL_ASSUMPTIONS,
  MAX_COUNTERFACTUAL_RETRACTIONS,
  MAX_COUNTERFACTUAL_RULE_ADDITIONS,
  MAX_COUNTERFACTUAL_RULE_REMOVALS,
} from '../knowledge/counterfactual.js';
import {
  MAX_WHY_NOT_CANDIDATES,
  MAX_WHY_NOT_DEPTH,
  MAX_WHY_NOT_EVIDENCE,
  MAX_WHY_NOT_FAILURES,
} from '../knowledge/why-not.js';
import { MAX_TOPOLOGY_FOCUS_BYTES } from '../knowledge/topology.js';
import {
  MAX_RULE_CHANGE_PROPOSAL_BYTES,
  RuleChangeCheckError,
} from '../knowledge/rule-change.js';
import { KnowledgeCheckEnforcementError } from '../knowledge/check-enforcement.js';
import {
  MAX_MEMORY_PROPOSAL_BYTES,
  MemoryChangeCheckError,
} from '../knowledge/memory-application.js';
import {
  MAX_REPAIR_PLANS,
  MAX_REPAIR_SEARCH_STATES,
  MAX_REPAIR_STEPS,
} from '../knowledge/repair.js';
import {
  MAX_KNOWLEDGE_SEARCH_LIMIT,
  type KnowledgeSearchClauseKind,
} from '../knowledge/search.js';
import {
  MAX_BROWSE_ENTITY_FOCUS_BYTES,
  MAX_BROWSE_GRAPH_CLAIMS,
  MAX_BROWSE_GRAPH_DEPTH,
  MAX_BROWSE_PREDICATE_FOCUS_BYTES,
} from '../knowledge/browse.js';
import {
  MAX_KNOWLEDGE_PATH_DEPTH,
  MAX_KNOWLEDGE_PATHS,
} from '../knowledge/paths.js';
import {
  MAX_KNOWLEDGE_BUNDLE_BYTES,
  serializeKnowledgeBundle,
} from '../knowledge/bundle.js';
import { MAX_KNOWLEDGE_CHECK_SUITE_BYTES } from '../knowledge/checks.js';
import { captureRememberoVersion } from '../ledger/remembero-version.js';
import {
  promoteRememberoReview,
  reviewRememberoCandidate,
} from '../ledger/remembero-review.js';
import { TrustMetadataError } from '../knowledge/trust.js';
import {
  IntegrityViolationError,
  type IntegrityEnforcementMode,
  type IntegrityEnforcementOptions,
} from '../knowledge/enforcement.js';
import type { EntityIdentityMode } from '../knowledge/identity.js';
import {
  MAX_OPERATION_ID_BYTES,
  OperationConflictError,
  MemoryChangeStaleError,
  RuleChangeStaleError,
} from '../store/store.js';
import {
  MAX_GRAPH_NEIGHBOR_DEPTH,
  MAX_GRAPH_NODE_ID_BYTES,
  MAX_GRAPH_RESULT_ROW,
  type ExplanationGraphSelector,
} from '../knowledge/graph-navigation.js';

const namespaceField = z
  .string()
  .optional()
  .describe('Memory namespace (default: "default")');
const namespacesField = z
  .union([z.array(z.string()).max(MAX_NAMESPACE_COUNT), z.literal('*')])
  .optional()
  .describe('Namespaces to search: a list, or "*" for all (default: ["default"])');
const schemaPredicateLimitField = z
  .number()
  .int()
  .min(1)
  .max(MAX_RECALL_SCHEMA_PREDICATES)
  .optional()
  .describe('Maximum predicates receiving detailed recall context');
const proofLimitField = z
  .number()
  .int()
  .min(1)
  .max(MAX_PROOFS_PER_ROW)
  .optional()
  .describe('Total deterministic proof witnesses per result, including the primary witness');
const maxViolationsField = z
  .number()
  .int()
  .min(1)
  .max(MAX_INTEGRITY_VIOLATIONS)
  .optional()
  .describe('Maximum complete integrity-violation rows returned across all constraints');
const whyNotFailureLimitField = z
  .number()
  .int()
  .min(1)
  .max(MAX_WHY_NOT_FAILURES)
  .optional()
  .describe('Maximum complete blocker nodes (default: 32)');
const whyNotDepthField = z
  .number()
  .int()
  .min(1)
  .max(MAX_WHY_NOT_DEPTH)
  .optional()
  .describe('Maximum nested rule-diagnostic depth (default: 8)');
const whyNotCandidateLimitField = z
  .number()
  .int()
  .min(1)
  .max(MAX_WHY_NOT_CANDIDATES)
  .optional()
  .describe('Maximum nearby sourced facts per blocker (default: 4)');
const whyNotEvidenceLimitField = z
  .number()
  .int()
  .min(1)
  .max(MAX_WHY_NOT_EVIDENCE)
  .optional()
  .describe('Maximum distinct nearby facts carrying proof evidence (default: 16)');
const topologyFocusField = z
  .string()
  .min(1)
  .max(MAX_TOPOLOGY_FOCUS_BYTES)
  .optional()
  .describe("Optional 'predicate' or 'predicate/arity' focus");
const topologyDirectionField = z
  .enum(['upstream', 'downstream', 'both'])
  .optional()
  .describe('Focused dependency direction (default: both; requires focus)');
const repairPlanLimitField = z
  .number()
  .int()
  .min(1)
  .max(MAX_REPAIR_PLANS)
  .optional()
  .describe('Maximum complete minimal repair plans (default: 8)');
const repairStepLimitField = z
  .number()
  .int()
  .min(1)
  .max(MAX_REPAIR_STEPS)
  .optional()
  .describe('Maximum iterative blocker-repair depth (default: 4)');
const repairSearchStateLimitField = z
  .number()
  .int()
  .min(1)
  .max(MAX_REPAIR_SEARCH_STATES)
  .optional()
  .describe('Maximum candidate edit states inspected (default: 128)');
const knowledgeSearchLimitField = z
  .number()
  .int()
  .min(1)
  .max(MAX_KNOWLEDGE_SEARCH_LIMIT)
  .optional()
  .describe('Maximum ranked local knowledge matches (default: 20)');
const knowledgeSearchKindsField = z
  .array(z.enum(['fact', 'rule', 'constraint']))
  .min(1)
  .max(3)
  .optional()
  .describe('Optional fact, rule, or constraint filters');
const semanticPrepareCursorField = z
  .string()
  .min(1)
  .max(MAX_INPUT_BYTES)
  .optional()
  .describe('Opaque nextCursor from the previous preparation batch');
const relatedKnowledgeField = z
  .boolean()
  .optional()
  .describe('Include local discovery evidence when recall cannot answer');
const relatedKnowledgeLimitField = z
  .number()
  .int()
  .min(1)
  .max(MAX_KNOWLEDGE_SEARCH_LIMIT)
  .optional()
  .describe('Maximum related local knowledge matches (default: 20)');
const relatedKnowledgeKindsField = z
  .array(z.enum(['fact', 'rule', 'constraint']))
  .min(1)
  .max(3)
  .optional()
  .describe('Optional related fact, rule, or constraint filters');
const browseEntityFocusField = z
  .union([
    z.string().max(MAX_BROWSE_ENTITY_FOCUS_BYTES),
    z.number().finite(),
  ])
  .optional()
  .describe('Optional exact atom string or numeric entity seed');
const browsePredicateField = z
  .string()
  .min(1)
  .max(MAX_BROWSE_PREDICATE_FOCUS_BYTES)
  .optional()
  .describe("Optional seed predicate as 'name' or 'name/arity'");
const browseDepthField = z
  .number()
  .int()
  .min(1)
  .max(MAX_BROWSE_GRAPH_DEPTH)
  .optional()
  .describe('Explicit fact-neighborhood depth (default: 1)');
const browseClaimLimitField = z
  .number()
  .int()
  .min(1)
  .max(MAX_BROWSE_GRAPH_CLAIMS)
  .optional()
  .describe('Maximum complete explicit claims (default: 100)');
const pathEndpointField = z
  .union([
    z.string().max(MAX_BROWSE_ENTITY_FOCUS_BYTES),
    z.number().finite(),
  ])
  .describe('Exact atom string or numeric path endpoint');
const pathDepthField = z
  .number()
  .int()
  .min(1)
  .max(MAX_KNOWLEDGE_PATH_DEPTH)
  .optional()
  .describe('Maximum explicit-claim hops (default: 4)');
const pathLimitField = z
  .number()
  .int()
  .min(1)
  .max(MAX_KNOWLEDGE_PATHS)
  .optional()
  .describe('Maximum complete shortest paths (default: 3)');
const conflictFocusField = z
  .string()
  .min(1)
  .max(MAX_CONFLICT_FOCUS_BYTES)
  .optional()
  .describe('Optional ground Datalog atom or number selecting one conflict focus');
const integrityModeField = z
  .enum(['strict', 'no_new_violations'])
  .optional()
  .describe('Atomically reject writes that violate policy; cannot weaken a server default');
const integrityNamespacesField = z
  .union([z.array(z.string()).max(MAX_NAMESPACE_COUNT), z.literal('*')])
  .optional()
  .describe('Knowledge view governed by write enforcement; must include the target namespace');
const entityIdentityField = z
  .literal('canonical')
  .optional()
  .describe('Project aliases only at explicitly declared predicate positions');
const knowledgeTrustField = z
  .enum(['accepted', 'tentative'])
  .optional()
  .describe('Store extracted facts as accepted (default) or explicitly tentative');
const trustViewField = z
  .enum(['accepted', 'include_tentative'])
  .optional()
  .describe('Accepted knowledge only (default), or opt in to tentative claims');
const recallAnswerModeField = z
  .enum(['natural', 'deterministic', 'evidence'])
  .optional()
  .describe('LLM phrasing, exact local bindings, or compact local proof/source evidence');
const graphSelectorField = z
  .discriminatedUnion('kind', [
    z.object({
      kind: z.literal('result'),
      row: z.number().int().min(1).max(MAX_GRAPH_RESULT_ROW),
    }),
    z.object({
      kind: z.literal('support'),
      nodeId: z.string().min(1).max(MAX_GRAPH_NODE_ID_BYTES),
    }),
    z.object({
      kind: z.literal('neighbors'),
      nodeId: z.string().min(1).max(MAX_GRAPH_NODE_ID_BYTES),
      depth: z.number().int().min(1).max(MAX_GRAPH_NEIGHBOR_DEPTH).optional(),
    }),
  ])
  .optional()
  .describe('Select one complete result support chain, node support closure, or bounded neighborhood');
const boundedText = (description?: string) => {
  const field = z.string().max(MAX_INPUT_BYTES);
  return description ? field.describe(description) : field;
};
const operationIdField = z
  .string()
  .min(1)
  .max(MAX_OPERATION_ID_BYTES)
  .optional()
  .describe('Caller-stable idempotency key for safe retries');
const recordedSequenceField = z
  .number()
  .int()
  .min(0)
  .optional()
  .describe('Read the deterministic knowledge snapshot after global journal entry n; 0 is empty');
const recordedDiffSequenceField = z
  .number()
  .int()
  .min(0)
  .describe('Exact global journal sequence (0 is the empty initial state)');
const validTimeInstantField = z
  .string()
  .max(64)
  .optional()
  .describe('Canonical UTC valid-until instant, e.g. 2026-08-16T16:59:00.000Z');

function asContent(result: unknown) {
  return { content: [{ type: 'text' as const, text: stringifyBoundedResult(result, 'MCP result') }] };
}

function asRawContent(text: string, label: string) {
  assertBoundedOutput(text, label);
  return { content: [{ type: 'text' as const, text }] };
}

function asError(e: unknown) {
  let text: string;
  if (
    e instanceof OperationConflictError ||
    e instanceof IncompleteHistoryError ||
    e instanceof TrustMetadataError ||
    e instanceof MemoryChangeStaleError ||
    e instanceof MemoryChangeCheckError ||
    e instanceof KnowledgeCheckEnforcementError ||
    e instanceof RuleChangeStaleError ||
    e instanceof RuleChangeCheckError
  ) {
    text = stringifyBoundedResult(e.toJSON(), 'MCP structured error');
  } else if (e instanceof IntegrityViolationError) {
    try {
      text = stringifyBoundedResult(e.toJSON(), 'MCP integrity rejection');
    } catch {
      text = JSON.stringify({
        error: 'integrity_rejection_output_exceeded',
        message: 'write was rejected, but complete evidence exceeds the MCP output bound',
        mode: e.mode,
        baselineViolationCount: e.baselineViolationCount,
        blockingViolationCount: e.blockingViolations.length,
        introducedViolationCount: e.introducedViolations.length,
      });
    }
  } else {
    text = e instanceof Error ? e.message : String(e);
  }
  return { content: [{ type: 'text' as const, text }], isError: true };
}

function requestedIntegrity(
  fallback: IntegrityEnforcementOptions | false | undefined,
  mode: IntegrityEnforcementMode | undefined,
  namespaces: string[] | '*' | undefined,
  proofLimit: number | undefined,
  maxViolations: number | undefined,
  entityIdentity: EntityIdentityMode | undefined,
  graphSelector: ExplanationGraphSelector | undefined
): IntegrityEnforcementOptions | undefined {
  const activeFallback = fallback === false ? undefined : fallback;
  if (mode === undefined) {
    if (
      activeFallback === undefined &&
      (
        namespaces !== undefined ||
        proofLimit !== undefined ||
        maxViolations !== undefined ||
        graphSelector !== undefined
      )
    ) {
      throw new Error('integrity write options require integrityMode or a server default');
    }
    return activeFallback === undefined
      ? undefined
      : {
          ...activeFallback,
          ...(namespaces === undefined ? {} : { namespaces }),
          ...(proofLimit === undefined ? {} : { maxProofsPerRow: proofLimit }),
          ...(maxViolations === undefined ? {} : { maxViolations }),
          ...(entityIdentity === undefined ? {} : { entityIdentity }),
          ...(graphSelector === undefined ? {} : { graphSelector }),
        };
  }
  if (activeFallback?.mode === 'strict' && mode !== 'strict') {
    throw new Error('tool call cannot weaken strict server integrity enforcement');
  }
  return {
    ...(activeFallback ?? {}),
    mode,
    ...(namespaces === undefined ? {} : { namespaces }),
    ...(proofLimit === undefined ? {} : { maxProofsPerRow: proofLimit }),
    ...(maxViolations === undefined ? {} : { maxViolations }),
    ...(entityIdentity === undefined ? {} : { entityIdentity }),
    ...(graphSelector === undefined ? {} : { graphSelector }),
  };
}

function requestedRelatedKnowledge(
  enabled: boolean | undefined,
  limit: number | undefined,
  kinds: KnowledgeSearchClauseKind[] | undefined
): boolean | RecallRelatedKnowledgeOptions | undefined {
  if (enabled === false) {
    if (limit !== undefined || kinds !== undefined) {
      throw new Error(
        'related knowledge limits or kinds cannot be used when relatedKnowledge is false'
      );
    }
    return false;
  }
  if (enabled !== true && limit === undefined && kinds === undefined) return undefined;
  if (limit === undefined && kinds === undefined) return true;
  return {
    ...(limit === undefined ? {} : { limit }),
    ...(kinds === undefined ? {} : { kinds }),
  };
}

export function createServer(deps: PipelineDeps): McpServer {
  const entityIdentity = deps.entityIdentity ?? entityIdentityFromEnv();
  const configuredIntegrity = deps.integrityEnforcement ?? integrityEnforcementFromEnv();
  const configuredChecks =
    deps.knowledgeCheckEnforcement ?? knowledgeCheckEnforcementFromEnv();
  const resolvedDeps: PipelineDeps = {
    ...deps,
    validTimeMode: deps.validTimeMode ?? validTimeModeFromEnv(),
    recallSchemaPredicateLimit:
      deps.recallSchemaPredicateLimit ?? recallSchemaPredicateLimitFromEnv(),
    recallAnswerMode: deps.recallAnswerMode ?? recallAnswerModeFromEnv(),
    integrityEnforcement:
      configuredIntegrity === undefined || configuredIntegrity === false
        ? configuredIntegrity
        : {
            ...configuredIntegrity,
            ...(configuredIntegrity.entityIdentity !== undefined || entityIdentity !== 'canonical'
              ? {}
              : { entityIdentity }),
          },
    knowledgeCheckEnforcement: configuredChecks,
    entityIdentity,
  };
  const embeddings = deps.embeddings ?? lazyEmbeddingClientFromEnv();
  const semanticCache = deps.semanticCache ?? new LayeredEmbeddingCache(
    new MemoryEmbeddingCache(),
    new FileEmbeddingCache(resolvedDeps.store.semanticEmbeddingCacheRoot())
  );
  const server = new McpServer({ name: 'rembero', version: '0.54.0' });

  const semanticLedger = () => {
    if (resolvedDeps.semanticLedger === undefined) {
      throw new Error('semantic version authority is not configured for this MCP server');
    }
    return resolvedDeps.semanticLedger;
  };

  server.registerTool(
    'capture_semantic_version',
    {
      title: 'Capture semantic version',
      description:
        'Capture the exact Remembero knowledge head plus document, rule, integrity-policy, model, runtime, and evaluation-suite objects into the SQLite semantic ledger. This does not mutate memory. If the ref does not exist, it initializes that ref.',
      inputSchema: {
        label: z.string().max(256).optional(),
        ref: z.string().max(256).optional(),
      },
    },
    async ({ label, ref }) => {
      try {
        const ledger = semanticLedger();
        const targetRef = ref ?? 'main';
        const parent = ledger.getRef(targetRef)?.versionDigest;
        const capture = captureRememberoVersion({
          ledger,
          store: resolvedDeps.store,
          ...(parent === undefined ? {} : { parents: [parent] }),
          label: label ?? `remembero@mcp-${Date.now()}`,
          metadata: { source: 'mcp' },
        });
        if (parent === undefined) {
          ledger.setRef({
            name: targetRef,
            versionDigest: capture.version.digest,
            operationId: `remembero-mcp-version-initialize-${targetRef}`,
            reason: 'Initialize semantic version ref',
          });
        }
        return asContent({
          version: capture.version,
          baselineVersionDigest: parent,
          recordedSnapshot: {
            sequence: capture.recordedSnapshot.sequence,
            journalEntries: capture.recordedSnapshot.journalEntries,
            namespaces: capture.recordedSnapshot.namespaces,
          },
        });
      } catch (e) {
        return asError(e);
      }
    }
  );

  server.registerTool(
    'list_semantic_versions',
    {
      title: 'List semantic versions',
      description: 'List current semantic refs and recent immutable Remembero versions.',
      inputSchema: {},
    },
    async () => {
      try {
        const ledger = semanticLedger();
        return asContent({ refs: ledger.listRefs(), versions: ledger.listVersions() });
      } catch (e) {
        return asError(e);
      }
    }
  );

  server.registerTool(
    'inspect_semantic_version',
    {
      title: 'Inspect semantic version',
      description: 'Resolve an exact digest, immutable label, or mutable ref to its full semantic version.',
      inputSchema: { reference: z.string().min(1).max(256) },
    },
    async ({ reference }) => {
      try {
        return asContent(semanticLedger().resolveVersion(reference));
      } catch (e) {
        return asError(e);
      }
    }
  );

  server.registerTool(
    'diff_semantic_versions',
    {
      title: 'Diff semantic versions',
      description: 'Compare semantic members, typed edges, contracts, evidence metrics, and compatibility for two exact versions.',
      inputSchema: {
        from: z.string().min(1).max(256),
        to: z.string().min(1).max(256),
      },
    },
    async ({ from, to }) => {
      try {
        const ledger = semanticLedger();
        const left = ledger.resolveVersion(from);
        const right = ledger.resolveVersion(to);
        return asContent(ledger.diffVersions(left.digest, right.digest));
      } catch (e) {
        return asError(e);
      }
    }
  );

  server.registerTool(
    'review_semantic_version',
    {
      title: 'Review semantic version',
      description:
        'Run deterministic document evidence and record a compatibility vector for a candidate version. This is non-mutating and does not move refs.',
      inputSchema: {
        candidate: z.string().min(1).max(256),
        includeDocumentEvaluation: z.boolean().optional(),
      },
    },
    async ({ candidate, includeDocumentEvaluation }) => {
      try {
        const ledger = semanticLedger();
        const version = ledger.resolveVersion(candidate);
        return asContent(
          reviewRememberoCandidate({
            ledger,
            store: resolvedDeps.store,
            candidateVersionDigest: version.digest,
            baselineVersionDigest: version.parents[0],
            includeDocumentEvaluation,
          })
        );
      } catch (e) {
        return asError(e);
      }
    }
  );

  server.registerTool(
    'promote_semantic_version',
    {
      title: 'Promote reviewed semantic version',
      description:
        'Move a ref only after an exact compatibility assessment. Failed and blocked dimensions always reject; review dimensions require explicit acceptance. This is mutation authority and should be called only after human review.',
      inputSchema: {
        ref: z.string().min(1).max(256),
        candidate: z.string().min(1).max(256),
        assessment: z.string().regex(/^[a-f0-9]{64}$/),
        operationId: z.string().min(1).max(MAX_OPERATION_ID_BYTES),
        acceptedReviewDimensions: z.array(z.string().min(1).max(128)).max(256).optional(),
        reason: z.string().max(4096).optional(),
      },
    },
    async ({ ref, candidate, assessment, operationId, acceptedReviewDimensions, reason }) => {
      try {
        const ledger = semanticLedger();
        const version = ledger.resolveVersion(candidate);
        const current = ledger.getRef(ref);
        return asContent(
          promoteRememberoReview({
            ledger,
            ref,
            candidateVersionDigest: version.digest,
            assessmentDigest: assessment,
            operationId,
            expectedCurrentVersionDigest: current?.versionDigest,
            acceptedReviewDimensions,
            reason,
          })
        );
      } catch (e) {
        return asError(e);
      }
    }
  );

  server.registerTool(
    'semantic_ref_history',
    {
      title: 'Semantic ref history',
      description: 'List immutable movements of one semantic ref, including promotion decisions.',
      inputSchema: { ref: z.string().min(1).max(256) },
    },
    async ({ ref }) => {
      try {
        return asContent({ ref, history: semanticLedger().refHistory(ref) });
      } catch (e) {
        return asError(e);
      }
    }
  );

  server.registerTool(
    'remember',
    {
      title: 'Remember',
      description:
        "Store a natural-language statement in long-term memory as logical facts/rules. Use proactively when the user states something durable: preferences, relationships, decisions, project facts, biography ('my dentist is Dr Chen', 'we picked Postgres', 'Mira now works at Initech' — updates supersede old facts). Do NOT store secrets (passwords, keys) or transient context (today's error message).",
      inputSchema: {
        text: boundedText('What to remember, in plain language'),
        namespace: namespaceField,
        integrityMode: integrityModeField,
        integrityNamespaces: integrityNamespacesField,
        proofLimit: proofLimitField,
        maxViolations: maxViolationsField,
        entityIdentity: entityIdentityField,
        trust: knowledgeTrustField,
        graphSelector: graphSelectorField,
      },
    },
    async ({
      text,
      namespace,
      integrityMode,
      integrityNamespaces,
      proofLimit,
      maxViolations,
      entityIdentity,
      trust,
      graphSelector,
    }) => {
      try {
        return asContent(
          await rememberTool(resolvedDeps, {
            text,
            namespace,
            integrityEnforcement: requestedIntegrity(
              resolvedDeps.integrityEnforcement,
              integrityMode,
              integrityNamespaces,
              proofLimit,
              maxViolations,
              entityIdentity,
              graphSelector
            ),
            entityIdentity,
            trust,
          })
        );
      } catch (e) {
        return asError(e);
      }
    }
  );

  server.registerTool(
    'propose_memory',
    {
      title: 'Propose accepted personal memory changes for review',
      description:
        'Use the identical natural-language extraction and validation path as remember, but never mutate. Returns exact additions/removals, candidate integrity and rule-audit impact, and a content-addressed accepted-memory proposal bound to the current baseline. Secrets fail before the LLM. Use this when human review should precede accepted memory authority.',
      inputSchema: {
        text: boundedText('What may be remembered, in plain language'),
        namespace: namespaceField,
        namespaces: namespacesField,
        validTimeMode: z.enum(['delete', 'archive_until']).optional(),
        at: validTimeInstantField,
        checkSuite: z
          .string()
          .max(MAX_KNOWLEDGE_CHECK_SUITE_BYTES)
          .optional()
          .describe('Optional JSON v1 checks and semantic coverage required by the proposal'),
        integrityMode: integrityModeField,
        integrityNamespaces: integrityNamespacesField,
        proofLimit: proofLimitField,
        maxViolations: maxViolationsField,
        entityIdentity: entityIdentityField,
        graphSelector: graphSelectorField,
      },
    },
    async ({
      text,
      namespace,
      namespaces,
      validTimeMode,
      at,
      checkSuite,
      integrityMode,
      integrityNamespaces,
      proofLimit,
      maxViolations,
      entityIdentity,
      graphSelector,
    }) => {
      try {
        return asContent(
          await proposeMemoryTool(resolvedDeps, {
            text,
            namespace,
            namespaces,
            validTimeMode,
            at,
            checkSuite,
            integrityEnforcement: requestedIntegrity(
              resolvedDeps.integrityEnforcement,
              integrityMode,
              integrityNamespaces,
              proofLimit,
              maxViolations,
              entityIdentity,
              graphSelector
            ),
            entityIdentity,
          })
        );
      } catch (e) {
        return asError(e);
      }
    }
  );

  server.registerTool(
    'apply_memory_proposal',
    {
      title: 'Apply a reviewed accepted-memory proposal',
      description:
        'Apply only an explicitly reviewed v1 proposal emitted by propose_memory. Revalidates proposal digest, exact current governed baseline, candidate rule audit, and no-new-integrity-violations policy under one mutation lock before a crash-safe idempotent memory_change journal commit. This is accepted-memory mutation authority: call only after human review.',
      inputSchema: {
        proposal: z
          .string()
          .max(MAX_MEMORY_PROPOSAL_BYTES)
          .describe('Standalone proposal JSON or complete propose_memory JSON containing one'),
        opId: z
          .string()
          .min(1)
          .max(MAX_OPERATION_ID_BYTES)
          .describe('Caller-stable idempotency key for this reviewed application'),
        maxViolations: maxViolationsField,
      },
    },
    async ({ proposal, opId, maxViolations }) => {
      try {
        return asContent(
          applyMemoryProposalTool(resolvedDeps, {
            proposal,
            opId,
            maxViolations,
          })
        );
      } catch (e) {
        return asError(e);
      }
    }
  );

  server.registerTool(
    'knowledge_health',
    {
      title: 'Inspect deterministic personal knowledge health',
      description:
        'Build one immutable current or recorded health snapshot combining integrity, rule audit/topology, pending tentative claims, identity metadata, provenance completeness, and an optional knowledge/coverage suite. Returns a content digest and stable findings without an LLM or mutation.',
      inputSchema: {
        namespaces: namespacesField,
        recordedSequence: recordedSequenceField,
        entityIdentity: entityIdentityField,
        trustMode: trustViewField,
        checkSuite: z
          .string()
          .max(MAX_KNOWLEDGE_CHECK_SUITE_BYTES)
          .optional()
          .describe('Optional serialized JSON v1 knowledge check and coverage suite'),
        proofLimit: proofLimitField,
        maxViolations: maxViolationsField,
      },
    },
    async ({
      namespaces,
      recordedSequence,
      entityIdentity,
      trustMode,
      checkSuite,
      proofLimit,
      maxViolations,
    }) => {
      try {
        return asContent(
          knowledgeHealthTool(resolvedDeps, {
            namespaces,
            recordedSequence,
            entityIdentity,
            trustMode,
            checkSuite,
            proofLimit,
            maxViolations,
          })
        );
      } catch (e) {
        return asError(e);
      }
    }
  );

  server.registerTool(
    'recall',
    {
      title: 'Recall',
      description:
        "Answer a question from current or explicitly selected recorded long-term memory using logical inference over stored facts and rules. Use when the user asks about anything previously discussed or personal ('who is my dentist?', 'what did we decide about the database?'), and at the start of tasks where remembered context would help. Returns an explicit recall status plus the query, bindings, and bounded schema diagnostics when pruning activates. Optional related knowledge is same-snapshot local discovery evidence only; it never changes the answer status or adds a model call.",
      inputSchema: {
        question: boundedText(),
        namespaces: namespacesField,
        schemaPredicateLimit: schemaPredicateLimitField,
        entityIdentity: entityIdentityField,
        trustMode: trustViewField,
        answerMode: recallAnswerModeField,
        relatedKnowledge: relatedKnowledgeField,
        relatedLimit: relatedKnowledgeLimitField,
        relatedKinds: relatedKnowledgeKindsField,
        recordedSequence: recordedSequenceField,
      },
    },
    async ({
      question,
      namespaces,
      schemaPredicateLimit,
      entityIdentity,
      trustMode,
      answerMode,
      relatedKnowledge,
      relatedLimit,
      relatedKinds,
      recordedSequence,
    }) => {
      try {
        const related = requestedRelatedKnowledge(
          relatedKnowledge,
          relatedLimit,
          relatedKinds
        );
        return asContent(
          await recallTool(resolvedDeps, {
            question,
            namespaces,
            schemaPredicateLimit,
            entityIdentity,
            trustMode,
            answerMode,
            ...(related === undefined ? {} : { relatedKnowledge: related }),
            recordedSequence,
          })
        );
      } catch (e) {
        return asError(e);
      }
    }
  );

  server.registerTool(
    'recall_explain',
    {
      title: 'Recall with explanation',
      description:
        'Answer from long-term memory and return the recall status, generated query, bindings, deterministic derivation proofs, durable source statements, query-scoped knowledge graph, and bounded schema diagnostics when pruning activates. Optional related knowledge is same-snapshot local discovery evidence only; it never changes the answer status or adds a model call.',
      inputSchema: {
        question: boundedText(),
        namespaces: namespacesField,
        schemaPredicateLimit: schemaPredicateLimitField,
        proofLimit: proofLimitField,
        entityIdentity: entityIdentityField,
        trustMode: trustViewField,
        answerMode: recallAnswerModeField,
        relatedKnowledge: relatedKnowledgeField,
        relatedLimit: relatedKnowledgeLimitField,
        relatedKinds: relatedKnowledgeKindsField,
        graphSelector: graphSelectorField,
        recordedSequence: recordedSequenceField,
      },
    },
    async ({
      question,
      namespaces,
      schemaPredicateLimit,
      proofLimit,
      entityIdentity,
      trustMode,
      answerMode,
      relatedKnowledge,
      relatedLimit,
      relatedKinds,
      graphSelector,
      recordedSequence,
    }) => {
      try {
        const related = requestedRelatedKnowledge(
          relatedKnowledge,
          relatedLimit,
          relatedKinds
        );
        return asContent(
          await recallExplainTool(resolvedDeps, {
            question,
            namespaces,
            schemaPredicateLimit,
            proofLimit,
            entityIdentity,
            trustMode,
            answerMode,
            ...(related === undefined ? {} : { relatedKnowledge: related }),
            graphSelector,
            recordedSequence,
          })
        );
      } catch (e) {
        return asError(e);
      }
    }
  );

  server.registerTool(
    'assert_facts',
    {
      title: 'Assert facts',
      description:
        "Store raw Datalog clauses directly, no LLM translation. Accepts facts like 'works_at(rahul, acme).', rules like 'senior(X) :- years(X, Y), Y >= 10 + 5.', and explicit integrity constraints like ':- active(X), suspended(X).'. Arithmetic is allowed only in comparison filters.",
      inputSchema: {
        clauses: boundedText(),
        namespace: namespaceField,
        opId: operationIdField,
        integrityMode: integrityModeField,
        integrityNamespaces: integrityNamespacesField,
        proofLimit: proofLimitField,
        maxViolations: maxViolationsField,
        entityIdentity: entityIdentityField,
        graphSelector: graphSelectorField,
      },
    },
    async ({
      clauses,
      namespace,
      opId,
      integrityMode,
      integrityNamespaces,
      proofLimit,
      maxViolations,
      entityIdentity,
      graphSelector,
    }) => {
      try {
        return asContent(
          assertFactsTool({
            store: resolvedDeps.store,
            knowledgeCheckEnforcement: resolvedDeps.knowledgeCheckEnforcement,
          }, {
            clauses,
            namespace,
            opId,
            integrityEnforcement: requestedIntegrity(
              resolvedDeps.integrityEnforcement,
              integrityMode,
              integrityNamespaces,
              proofLimit,
              maxViolations,
              entityIdentity,
              graphSelector
            ),
          })
        );
      } catch (e) {
        return asError(e);
      }
    }
  );

  server.registerTool(
    'assert_tentative',
    {
      title: 'Assert tentative facts',
      description:
        'Store explicit ground Datalog facts as tentative claims. They are journaled but excluded from ordinary query, recall, integrity, and conflict views until accepted or explicitly included.',
      inputSchema: {
        clauses: boundedText('One or more ordinary ground facts'),
        namespace: namespaceField,
        opId: operationIdField,
        integrityMode: integrityModeField,
        integrityNamespaces: integrityNamespacesField,
        proofLimit: proofLimitField,
        maxViolations: maxViolationsField,
        entityIdentity: entityIdentityField,
        graphSelector: graphSelectorField,
      },
    },
    async ({
      clauses,
      namespace,
      opId,
      integrityMode,
      integrityNamespaces,
      proofLimit,
      maxViolations,
      entityIdentity,
      graphSelector,
    }) => {
      try {
        return asContent(
          assertTentativeTool(
            {
              store: resolvedDeps.store,
              knowledgeCheckEnforcement: resolvedDeps.knowledgeCheckEnforcement,
            },
            {
              clauses,
              namespace,
              opId,
              integrityEnforcement: requestedIntegrity(
                resolvedDeps.integrityEnforcement,
                integrityMode,
                integrityNamespaces,
                proofLimit,
                maxViolations,
                entityIdentity,
                graphSelector
              ),
            }
          )
        );
      } catch (e) {
        return asError(e);
      }
    }
  );

  server.registerTool(
    'review_tentative',
    {
      title: 'Review tentative claims',
      description:
        'List bounded tentative facts awaiting explicit acceptance or rejection, with stable IDs and durable sources.',
      inputSchema: { namespaces: namespacesField },
    },
    async ({ namespaces }) => {
      try {
        return asContent(
          reviewTentativeTool(
            {
              store: resolvedDeps.store,
              knowledgeCheckEnforcement: resolvedDeps.knowledgeCheckEnforcement,
            },
            { namespaces }
          )
        );
      } catch (e) {
        return asError(e);
      }
    }
  );

  server.registerTool(
    'resolve_tentative',
    {
      title: 'Accept or reject tentative facts',
      description:
        'Atomically accept or reject exact tentative ground facts. Every requested claim must still be current; acceptance passes configured integrity enforcement.',
      inputSchema: {
        clauses: boundedText('Exact ground facts previously stored as tentative'),
        action: z.enum(['accept', 'reject']),
        namespace: namespaceField,
        opId: operationIdField,
        integrityMode: integrityModeField,
        integrityNamespaces: integrityNamespacesField,
        proofLimit: proofLimitField,
        maxViolations: maxViolationsField,
        entityIdentity: entityIdentityField,
        graphSelector: graphSelectorField,
      },
    },
    async ({
      clauses,
      action,
      namespace,
      opId,
      integrityMode,
      integrityNamespaces,
      proofLimit,
      maxViolations,
      entityIdentity,
      graphSelector,
    }) => {
      try {
        return asContent(
          resolveTentativeTool(
            {
              store: resolvedDeps.store,
              knowledgeCheckEnforcement: resolvedDeps.knowledgeCheckEnforcement,
            },
            {
              clauses,
              action,
              namespace,
              opId,
              integrityEnforcement: requestedIntegrity(
                resolvedDeps.integrityEnforcement,
                integrityMode,
                integrityNamespaces,
                proofLimit,
                maxViolations,
                entityIdentity,
                graphSelector
              ),
            }
          )
        );
      } catch (e) {
        return asError(e);
      }
    }
  );

  server.registerTool(
    'supersede_facts',
    {
      title: 'Supersede facts',
      description:
        "Atomically end current ground facts matching one or more patterns, preserve each as a system-managed '_until' fact, and add explicit replacement clauses. No LLM is used. Append order remains authoritative; the optional UTC timestamp is descriptive valid-time metadata.",
      inputSchema: {
        patterns: z
          .array(boundedText("A positive fact pattern, e.g. 'works_at(mira, _)'"))
          .min(1)
          .max(MAX_SUPERSEDE_PATTERNS),
        replacements: boundedText('Optional ground facts or other Datalog clauses to add')
          .optional(),
        namespace: namespaceField,
        at: validTimeInstantField,
        opId: operationIdField,
        integrityMode: integrityModeField,
        integrityNamespaces: integrityNamespacesField,
        proofLimit: proofLimitField,
        maxViolations: maxViolationsField,
        entityIdentity: entityIdentityField,
        graphSelector: graphSelectorField,
      },
    },
    async ({
      patterns,
      replacements,
      namespace,
      at,
      opId,
      integrityMode,
      integrityNamespaces,
      proofLimit,
      maxViolations,
      entityIdentity,
      graphSelector,
    }) => {
      try {
        return asContent(
          supersedeFactsTool(
            {
              store: resolvedDeps.store,
              knowledgeCheckEnforcement: resolvedDeps.knowledgeCheckEnforcement,
            },
            {
              patterns,
              replacements,
              namespace,
              at,
              opId,
              integrityEnforcement: requestedIntegrity(
                resolvedDeps.integrityEnforcement,
                integrityMode,
                integrityNamespaces,
                proofLimit,
                maxViolations,
                entityIdentity,
                graphSelector
              ),
            }
          )
        );
      } catch (e) {
        return asError(e);
      }
    }
  );

  server.registerTool(
    'search_knowledge',
    {
      title: 'Search knowledge locally',
      description:
        'Rank selected facts, rules, and policies locally using fixed lexical evidence from predicate names, atoms, authored clauses, and redacted durable source text. Returns explicit score reasons, provenance, and a query/result/clause/predicate/entity graph. This is retrieval, not semantic proof; no LLM or vector service is used.',
      inputSchema: {
        text: boundedText(),
        namespaces: namespacesField,
        limit: knowledgeSearchLimitField,
        kinds: knowledgeSearchKindsField,
        entityIdentity: entityIdentityField,
        trustMode: trustViewField,
        recordedSequence: recordedSequenceField,
      },
    },
    async ({
      text,
      namespaces,
      limit,
      kinds,
      entityIdentity,
      trustMode,
      recordedSequence,
    }) => {
      try {
        return asContent(
          searchKnowledgeTool(resolvedDeps, {
            text,
            namespaces,
            limit,
            kinds,
            entityIdentity,
            trustMode,
            recordedSequence,
          })
        );
      } catch (e) {
        return asError(e);
      }
    }
  );

  server.registerTool(
    'semantic_search_knowledge',
    {
      title: 'Search recommendation and preference memory semantically',
      description:
        'Rerank a bounded local lexical shortlist with an embedding provider. Use for recommendations, preferences, advice, and paraphrased prior context when exact lexical search is insufficient. This opt-in tool sends redacted source text only from LLM-allowed namespaces, reports provider tokens/cost, caches derived document vectors, and returns retrieval evidence—not proof or answer authority.',
      inputSchema: {
        text: boundedText(),
        namespaces: namespacesField,
        limit: knowledgeSearchLimitField,
        kinds: knowledgeSearchKindsField,
        entityIdentity: entityIdentityField,
        trustMode: trustViewField,
        recordedSequence: recordedSequenceField,
      },
    },
    async ({
      text,
      namespaces,
      limit,
      kinds,
      entityIdentity,
      trustMode,
      recordedSequence,
    }) => {
      try {
        return asContent(
          await semanticSearchKnowledgeTool(
            {
              store: resolvedDeps.store,
              embeddings,
              semanticCache,
              llmAllowedNamespaces: resolvedDeps.llmAllowedNamespaces,
              entityIdentity: resolvedDeps.entityIdentity,
              trustMode: resolvedDeps.trustMode,
            },
            {
              text,
              namespaces,
              limit,
              kinds,
              entityIdentity,
              trustMode,
              recordedSequence,
            }
          )
        );
      } catch (e) {
        return asError(e);
      }
    }
  );

  server.registerTool(
    'prepare_semantic_search',
    {
      title: 'Prepare semantic retrieval after reviewed writes',
      description:
        'Populate the bounded derived embedding cache before user-facing recommendation searches. This explicit, resumable maintenance tool reads only LLM-allowed namespaces, rejects detected secrets, reports provider usage/cost, stores no source text in the cache, and never mutates memory or establishes proof.',
      inputSchema: {
        namespaces: namespacesField,
        limit: knowledgeSearchLimitField,
        after: semanticPrepareCursorField,
        kinds: knowledgeSearchKindsField,
        entityIdentity: entityIdentityField,
        trustMode: trustViewField,
        recordedSequence: recordedSequenceField,
      },
    },
    async ({
      namespaces,
      limit,
      after,
      kinds,
      entityIdentity,
      trustMode,
      recordedSequence,
    }) => {
      try {
        return asContent(
          await prepareSemanticKnowledgeTool(
            {
              store: resolvedDeps.store,
              embeddings,
              semanticCache,
              llmAllowedNamespaces: resolvedDeps.llmAllowedNamespaces,
              entityIdentity: resolvedDeps.entityIdentity,
              trustMode: resolvedDeps.trustMode,
            },
            {
              namespaces,
              limit,
              after,
              kinds,
              entityIdentity,
              trustMode,
              recordedSequence,
            }
          )
        );
      } catch (e) {
        return asError(e);
      }
    }
  );

  server.registerTool(
    'browse_knowledge_graph',
    {
      title: 'Browse the explicit personal knowledge graph',
      description:
        'Browse a bounded entity- or predicate-seeded hypergraph derived only from explicit stored ground facts. Returns claim/entity nodes, argument edges, provenance, aliases, trust, and recorded-view metadata. Rules and inferred claims require query/explain; no graph sidecar or LLM is used.',
      inputSchema: {
        focus: browseEntityFocusField,
        predicate: browsePredicateField,
        depth: browseDepthField,
        maxClaims: browseClaimLimitField,
        namespaces: namespacesField,
        entityIdentity: entityIdentityField,
        trustMode: trustViewField,
        recordedSequence: recordedSequenceField,
      },
    },
    async ({
      focus,
      predicate,
      depth,
      maxClaims,
      namespaces,
      entityIdentity,
      trustMode,
      recordedSequence,
    }) => {
      try {
        return asContent(
          browseKnowledgeGraphTool(resolvedDeps, {
            focus,
            predicate,
            depth,
            maxClaims,
            namespaces,
            entityIdentity,
            trustMode,
            recordedSequence,
          })
        );
      } catch (e) {
        return asError(e);
      }
    }
  );

  server.registerTool(
    'connect_knowledge_graph',
    {
      title: 'Find proof-aware personal knowledge paths',
      description:
        'Find every bounded shortest path between two atom or numeric entities. Explicit ground facts are the default; includeDerived opts into bounded rule conclusions and attaches a complete sourced proof to every selected claim. Returns ordered segments, a proof/provenance graph, aliases, trust, and recorded-view metadata. A no_path result distinguishes component exhaustion from a depth-bounded search; no LLM, persisted inference, or graph sidecar is used.',
      inputSchema: {
        from: pathEndpointField,
        to: pathEndpointField,
        maxDepth: pathDepthField,
        maxPaths: pathLimitField,
        maxClaims: browseClaimLimitField,
        includeDerived: z
          .boolean()
          .optional()
          .describe('Traverse rule-derived facts only when each returned claim carries a proof'),
        namespaces: namespacesField,
        entityIdentity: entityIdentityField,
        trustMode: trustViewField,
        recordedSequence: recordedSequenceField,
      },
    },
    async ({
      from,
      to,
      maxDepth,
      maxPaths,
      maxClaims,
      includeDerived,
      namespaces,
      entityIdentity,
      trustMode,
      recordedSequence,
    }) => {
      try {
        return asContent(
          connectKnowledgeGraphTool(resolvedDeps, {
            from,
            to,
            maxDepth,
            maxPaths,
            maxClaims,
            includeDerived,
            namespaces,
            entityIdentity,
            trustMode,
            recordedSequence,
          })
        );
      } catch (e) {
        return asError(e);
      }
    }
  );

  server.registerTool(
    'run_knowledge_checks',
    {
      title: 'Run deterministic knowledge regression suite',
      description:
        'Run a bounded JSON v1 suite of named empty, nonempty, exact-row, or set-row Datalog expectations against one current or recorded view. Passing checks stay compact by default; failures include row deltas plus proof or why-not evidence. No test metadata is stored and no LLM or mutation is used.',
      inputSchema: {
        suite: z
          .string()
          .max(MAX_KNOWLEDGE_CHECK_SUITE_BYTES)
          .describe('Serialized knowledge check suite JSON'),
        namespaces: namespacesField,
        proofLimit: proofLimitField,
        entityIdentity: entityIdentityField,
        trustMode: trustViewField,
        recordedSequence: recordedSequenceField,
        includePassingEvidence: z.boolean().optional(),
      },
    },
    async ({
      suite,
      namespaces,
      proofLimit,
      entityIdentity,
      trustMode,
      recordedSequence,
      includePassingEvidence,
    }) => {
      try {
        return asContent(
          runKnowledgeChecksTool(resolvedDeps, {
            suite,
            namespaces,
            proofLimit,
            entityIdentity,
            trustMode,
            recordedSequence,
            includePassingEvidence,
          })
        );
      } catch (e) {
        return asError(e);
      }
    }
  );

  server.registerTool(
    'profile_query',
    {
      title: 'Profile deterministic query work',
      description:
        'Run a Datalog query with exact proofs and graph while returning deterministic relation lookup, index-build, and candidate-fact counters. Optional full-scan comparison reruns with indexes disabled and returns only if explanations are byte-identical. No timing, LLM, or mutation is used.',
      inputSchema: {
        query: boundedText(),
        namespaces: namespacesField,
        proofLimit: proofLimitField,
        entityIdentity: entityIdentityField,
        trustMode: trustViewField,
        graphSelector: graphSelectorField,
        recordedSequence: recordedSequenceField,
        compareFullScan: z.boolean().optional(),
      },
    },
    async ({
      query,
      namespaces,
      proofLimit,
      entityIdentity,
      trustMode,
      graphSelector,
      recordedSequence,
      compareFullScan,
    }) => {
      try {
        return asContent(
          profileKnowledgeTool(resolvedDeps, {
            query,
            namespaces,
            proofLimit,
            entityIdentity,
            trustMode,
            graphSelector,
            recordedSequence,
            compareFullScan,
          })
        );
      } catch (e) {
        return asError(e);
      }
    }
  );

  server.registerTool(
    'query',
    {
      title: 'Query',
      description:
        "Run a raw Datalog query and get variable bindings, e.g. 'works_at(X, acme)', 'score(X, S), S > 10 + 5', 'employee(X), \\+ suspended(X)', or 'count(*) as Count where works_at(Person, acme)'.",
      inputSchema: {
        query: boundedText(),
        namespaces: namespacesField,
        entityIdentity: entityIdentityField,
        trustMode: trustViewField,
        recordedSequence: recordedSequenceField,
      },
    },
    async ({ query, namespaces, entityIdentity, trustMode, recordedSequence }) => {
      try {
        return asContent(
          queryTool(resolvedDeps, {
            query,
            namespaces,
            entityIdentity,
            trustMode,
            recordedSequence,
          })
        );
      } catch (e) {
        return asError(e);
      }
    }
  );

  server.registerTool(
    'explain_query',
    {
      title: 'Explain query',
      description:
        'Run a raw Datalog query and return bindings, deterministic derivation or aggregate proofs, durable memory sources, and a query-scoped knowledge graph.',
      inputSchema: {
        query: boundedText(),
        namespaces: namespacesField,
        proofLimit: proofLimitField,
        entityIdentity: entityIdentityField,
        trustMode: trustViewField,
        graphSelector: graphSelectorField,
        recordedSequence: recordedSequenceField,
      },
    },
    async ({
      query,
      namespaces,
      proofLimit,
      entityIdentity,
      trustMode,
      graphSelector,
      recordedSequence,
    }) => {
      try {
        return asContent(
          explainQueryTool(resolvedDeps, {
            query,
            namespaces,
            proofLimit,
            entityIdentity,
            trustMode,
            graphSelector,
            recordedSequence,
          })
        );
      } catch (e) {
        return asError(e);
      }
    }
  );

  server.registerTool(
    'why_not',
    {
      title: 'Explain why a query is blocked',
      description:
        'Deterministically explain an empty Datalog query by following conjunction bindings and rule alternatives to missing facts, present negated facts, false comparisons, recursive cycles, or aggregate output mismatches. Includes sourced nearby facts and a blocker graph; no LLM is used.',
      inputSchema: {
        query: boundedText(),
        namespaces: namespacesField,
        proofLimit: proofLimitField,
        entityIdentity: entityIdentityField,
        trustMode: trustViewField,
        recordedSequence: recordedSequenceField,
        maxFailures: whyNotFailureLimitField,
        maxDiagnosticDepth: whyNotDepthField,
        maxCandidatesPerFailure: whyNotCandidateLimitField,
        maxEvidenceFacts: whyNotEvidenceLimitField,
      },
    },
    async ({
      query,
      namespaces,
      proofLimit,
      entityIdentity,
      trustMode,
      recordedSequence,
      maxFailures,
      maxDiagnosticDepth,
      maxCandidatesPerFailure,
      maxEvidenceFacts,
    }) => {
      try {
        return asContent(
          whyNotTool(resolvedDeps, {
            query,
            namespaces,
            proofLimit,
            entityIdentity,
            trustMode,
            recordedSequence,
            maxFailures,
            maxDiagnosticDepth,
            maxCandidatesPerFailure,
            maxEvidenceFacts,
          })
        );
      } catch (e) {
        return asError(e);
      }
    }
  );

  server.registerTool(
    'audit_rules',
    {
      title: 'Audit deterministic rule health',
      description:
        'Audit the selected current or recorded rule program for undefined closed-world negation, policy inputs without definitions, inert recursion, open positive inputs, inactive derived predicates, alpha-equivalent duplicates, and predicate arity overload. Findings link to an evidence topology graph; no LLM or mutation is used.',
      inputSchema: {
        namespaces: namespacesField,
        focus: topologyFocusField,
        direction: topologyDirectionField,
        entityIdentity: entityIdentityField,
        trustMode: trustViewField,
        recordedSequence: recordedSequenceField,
      },
    },
    async ({
      namespaces,
      focus,
      direction,
      entityIdentity,
      trustMode,
      recordedSequence,
    }) => {
      try {
        return asContent(
          auditRulesTool(resolvedDeps, {
            namespaces,
            focus,
            direction,
            entityIdentity,
            trustMode,
            recordedSequence,
          })
        );
      } catch (e) {
        return asError(e);
      }
    }
  );

  server.registerTool(
    'plan_query_repair',
    {
      title: 'Plan verified query repairs',
      description:
        'Search bounded why-not blockers for subset-minimal ground fact assumptions or retractions, then counterfactually verify every returned plan against the query and integrity policies on one captured baseline. Plans are proposals only; no fact, source, or journal entry is written.',
      inputSchema: {
        query: boundedText(),
        namespace: namespaceField,
        namespaces: namespacesField,
        proofLimit: proofLimitField,
        maxViolations: maxViolationsField,
        entityIdentity: entityIdentityField,
        trustMode: trustViewField,
        maxPlans: repairPlanLimitField,
        maxSteps: repairStepLimitField,
        maxSearchStates: repairSearchStateLimitField,
      },
    },
    async ({
      query,
      namespace,
      namespaces,
      proofLimit,
      maxViolations,
      entityIdentity,
      trustMode,
      maxPlans,
      maxSteps,
      maxSearchStates,
    }) => {
      try {
        return asContent(
          repairPlanTool(resolvedDeps, {
            query,
            namespace,
            namespaces,
            proofLimit,
            maxViolations,
            entityIdentity,
            trustMode,
            maxPlans,
            maxSteps,
            maxSearchStates,
          })
        );
      } catch (e) {
        return asError(e);
      }
    }
  );

  server.registerTool(
    'knowledge_topology',
    {
      title: 'Map knowledge topology',
      description:
        'Build a deterministic semantic graph of predicates, alpha-equivalent rule groups, integrity policies, positive/negative/aggregate dependencies, strata, recursive components, provenance, and open inputs. Optional focus selects complete upstream/downstream influence without an LLM or persistent graph.',
      inputSchema: {
        namespaces: namespacesField,
        focus: topologyFocusField,
        direction: topologyDirectionField,
        entityIdentity: entityIdentityField,
        trustMode: trustViewField,
        recordedSequence: recordedSequenceField,
      },
    },
    async ({
      namespaces,
      focus,
      direction,
      entityIdentity,
      trustMode,
      recordedSequence,
    }) => {
      try {
        return asContent(
          topologyTool(resolvedDeps, {
            namespaces,
            focus,
            direction,
            entityIdentity,
            trustMode,
            recordedSequence,
          })
        );
      } catch (e) {
        return asError(e);
      }
    }
  );

  server.registerTool(
    'diff_recorded_knowledge',
    {
      title: 'Diff recorded knowledge',
      description:
        'Compare two exact global journal positions as one coherent read. Returns semantic fact/rule/policy and provenance changes, topology node/edge impact, introduced/resolved integrity violations, and optional before/after query proofs. Timestamps never order the diff and no memory is changed.',
      inputSchema: {
        fromSequence: recordedDiffSequenceField,
        toSequence: recordedDiffSequenceField,
        namespaces: namespacesField,
        query: boundedText('Optional Datalog query whose result/proof impact is compared').optional(),
        proofLimit: proofLimitField,
        maxViolations: maxViolationsField,
        entityIdentity: entityIdentityField,
        trustMode: trustViewField,
      },
    },
    async ({
      fromSequence,
      toSequence,
      namespaces,
      query,
      proofLimit,
      maxViolations,
      entityIdentity,
      trustMode,
    }) => {
      try {
        return asContent(
          recordedDiffTool(resolvedDeps, {
            fromSequence,
            toSequence,
            namespaces,
            query,
            proofLimit,
            maxViolations,
            entityIdentity,
            trustMode,
          })
        );
      } catch (e) {
        return asError(e);
      }
    }
  );

  server.registerTool(
    'check_integrity',
    {
      title: 'Check knowledge integrity',
      description:
        'Evaluate every explicit headless Datalog constraint over the selected current or recorded knowledge view. Returns one policy check each; violating rows include deterministic proofs, durable sources, and query-scoped graphs. No LLM is used and no memory is changed.',
      inputSchema: {
        namespaces: namespacesField,
        proofLimit: proofLimitField,
        maxViolations: maxViolationsField,
        entityIdentity: entityIdentityField,
        trustMode: trustViewField,
        graphSelector: graphSelectorField,
        recordedSequence: recordedSequenceField,
      },
    },
    async ({
      namespaces,
      proofLimit,
      maxViolations,
      entityIdentity,
      trustMode,
      graphSelector,
      recordedSequence,
    }) => {
      try {
        return asContent(
          checkIntegrityTool(resolvedDeps, {
            namespaces,
            proofLimit,
            maxViolations,
            entityIdentity,
            trustMode,
            graphSelector,
            recordedSequence,
          })
        );
      } catch (e) {
        return asError(e);
      }
    }
  );

  server.registerTool(
    'conflict_views',
    {
      title: 'Inspect focused knowledge conflicts',
      description:
        "Group complete integrity violations by each constraint's first alpha-stable binding. Optional focus, canonical identity, recorded snapshots, proofs, durable sources, and per-cluster graph selection are supported. No LLM is used and no conflict store is persisted.",
      inputSchema: {
        focus: conflictFocusField,
        namespaces: namespacesField,
        proofLimit: proofLimitField,
        maxViolations: maxViolationsField,
        entityIdentity: entityIdentityField,
        trustMode: trustViewField,
        graphSelector: graphSelectorField,
        recordedSequence: recordedSequenceField,
      },
    },
    async ({
      focus,
      namespaces,
      proofLimit,
      maxViolations,
      entityIdentity,
      trustMode,
      graphSelector,
      recordedSequence,
    }) => {
      try {
        return asContent(
          conflictViewsTool(resolvedDeps, {
            focus,
            namespaces,
            proofLimit,
            maxViolations,
            entityIdentity,
            trustMode,
            graphSelector,
            recordedSequence,
          })
        );
      } catch (e) {
        return asError(e);
      }
    }
  );

  server.registerTool(
    'forget',
    {
      title: 'Forget',
      description:
        "Retract facts matching a pattern, e.g. 'works_at(rahul, _)', or remove an exact rule by giving it in full.",
      inputSchema: {
        pattern: boundedText(),
        namespace: namespaceField,
        opId: operationIdField,
        integrityMode: integrityModeField,
        integrityNamespaces: integrityNamespacesField,
        proofLimit: proofLimitField,
        maxViolations: maxViolationsField,
        entityIdentity: entityIdentityField,
        graphSelector: graphSelectorField,
      },
    },
    async ({
      pattern,
      namespace,
      opId,
      integrityMode,
      integrityNamespaces,
      proofLimit,
      maxViolations,
      entityIdentity,
      graphSelector,
    }) => {
      try {
        return asContent(
          forgetTool({
            store: resolvedDeps.store,
            knowledgeCheckEnforcement: resolvedDeps.knowledgeCheckEnforcement,
          }, {
            pattern,
            namespace,
            opId,
            integrityEnforcement: requestedIntegrity(
              resolvedDeps.integrityEnforcement,
              integrityMode,
              integrityNamespaces,
              proofLimit,
              maxViolations,
              entityIdentity,
              graphSelector
            ),
          })
        );
      } catch (e) {
        return asError(e);
      }
    }
  );

  server.registerTool(
    'what_if',
    {
      title: 'Simulate knowledge changes',
      description:
        'Read-only deterministic counterfactual over one current or recorded baseline. Retract/assume facts or exact alpha-equivalent rules, then compare query proofs, provenance, integrity, rule audit/topology, and an optional knowledge-suite/coverage result. Proposed rules receive hypothetical sources; no LLM is used and nothing is persisted.',
      inputSchema: {
        query: boundedText('Datalog query whose result impact should be explained'),
        assume: boundedText(
          `Ordinary ground Datalog facts to assume (maximum ${MAX_COUNTERFACTUAL_ASSUMPTIONS})`
        ).optional(),
        without: z
          .array(boundedText('One positive ground-fact pattern'))
          .max(MAX_COUNTERFACTUAL_RETRACTIONS)
          .optional(),
        assumeRules: boundedText(
          `Ordinary or aggregate Datalog rules to assume (maximum ${MAX_COUNTERFACTUAL_RULE_ADDITIONS})`
        ).optional(),
        withoutRules: boundedText(
          `Exact alpha-equivalent rules to remove (maximum ${MAX_COUNTERFACTUAL_RULE_REMOVALS})`
        ).optional(),
        checkSuite: z
          .string()
          .max(MAX_KNOWLEDGE_CHECK_SUITE_BYTES)
          .optional()
          .describe('Optional serialized JSON v1 knowledge check and coverage suite'),
        namespace: namespaceField,
        namespaces: namespacesField,
        proofLimit: proofLimitField,
        maxViolations: maxViolationsField,
        entityIdentity: entityIdentityField,
        trustMode: trustViewField,
        recordedSequence: recordedSequenceField,
      },
    },
    async ({
      query,
      assume,
      without,
      assumeRules,
      withoutRules,
      checkSuite,
      namespace,
      namespaces,
      proofLimit,
      maxViolations,
      entityIdentity,
      trustMode,
      recordedSequence,
    }) => {
      try {
        return asContent(
          whatIfTool(resolvedDeps, {
            query,
            assume,
            without,
            assumeRules,
            withoutRules,
            checkSuite,
            namespace,
            namespaces,
            proofLimit,
            maxViolations,
            entityIdentity,
            trustMode,
            recordedSequence,
          })
        );
      } catch (e) {
        return asError(e);
      }
    }
  );

  server.registerTool(
    'apply_rule_change',
    {
      title: 'Apply a reviewed digest-bound rule proposal',
      description:
        'Apply only an explicitly reviewed v1 ruleProposal emitted by what_if. Revalidates the proposal digest, exact current multi-namespace baseline, candidate rule audit, attached knowledge/coverage suite, and no-new-integrity-violations policy under one mutation lock before a crash-safe idempotent journal commit. Recorded-baseline proposals are never applicable. This is mutation authority: call only after human review.',
      inputSchema: {
        proposal: z
          .string()
          .max(MAX_RULE_CHANGE_PROPOSAL_BYTES)
          .describe('Standalone ruleProposal JSON or complete what_if JSON containing one'),
        opId: z
          .string()
          .min(1)
          .max(MAX_OPERATION_ID_BYTES)
          .describe('Caller-stable idempotency key for this reviewed application'),
        maxViolations: maxViolationsField,
      },
    },
    async ({ proposal, opId, maxViolations }) => {
      try {
        return asContent(
          applyRuleChangeProposalTool(resolvedDeps, {
            proposal,
            opId,
            maxViolations,
          })
        );
      } catch (e) {
        return asError(e);
      }
    }
  );

  server.registerTool(
    'export_knowledge_bundle',
    {
      title: 'Export content-addressed knowledge bundle',
      description:
        'Export raw portable namespace clauses, trust/identity metadata, and durable provenance as deterministic JSON with a SHA-256 content digest. Defaults to all namespaces; an optional recorded sequence exports that exact state. The bundle contains personal data and is returned verbatim without an LLM.',
      inputSchema: {
        namespaces: namespacesField,
        recordedSequence: recordedSequenceField,
      },
    },
    async ({ namespaces, recordedSequence }) => {
      try {
        const bundle = exportKnowledgeBundleTool(
          { store: resolvedDeps.store },
          { namespaces, recordedSequence }
        );
        return asRawContent(
          serializeKnowledgeBundle(bundle),
          'MCP knowledge bundle'
        );
      } catch (e) {
        return asError(e);
      }
    }
  );

  server.registerTool(
    'verify_knowledge_bundle',
    {
      title: 'Verify standalone knowledge bundle',
      description:
        'Verify bundle JSON structure, canonical clauses and ordering, durable provenance, resource bounds, recorded coordinates, and SHA-256 digest without importing or mutating knowledge.',
      inputSchema: {
        bundle: z
          .string()
          .max(MAX_KNOWLEDGE_BUNDLE_BYTES)
          .describe('Serialized rembero knowledge bundle JSON'),
      },
    },
    async ({ bundle }) => {
      try {
        return asContent(verifyKnowledgeBundleTool({ bundle }));
      } catch (e) {
        return asError(e);
      }
    }
  );

  server.registerTool(
    'checkpoint_journal',
    {
      title: 'Checkpoint journal',
      description:
        'Rotate the active append-only journal into an immutable SHA-256 verified segment and publish an exact namespace/source checkpoint without changing global recorded sequences.',
      inputSchema: {
        opId: operationIdField,
        at: validTimeInstantField,
        dryRun: z.boolean().optional(),
      },
    },
    async ({ opId, at, dryRun }) => {
      try {
        return asContent(
          checkpointJournalTool(
            { store: resolvedDeps.store },
            { opId, at, dryRun }
          )
        );
      } catch (e) {
        return asError(e);
      }
    }
  );

  server.registerTool(
    'list_checkpoints',
    {
      title: 'List journal checkpoints',
      description:
        'List validated immutable journal checkpoint artifacts and their global sequence boundaries.',
      inputSchema: {},
    },
    async () => {
      try {
        return asContent(listCheckpointsTool({ store: resolvedDeps.store }));
      } catch (e) {
        return asError(e);
      }
    }
  );

  server.registerTool(
    'history',
    {
      title: 'Memory history',
      description:
        'Show the deterministic append-order life story of facts matching one Datalog literal, including exact supersession archives and redacted provenance. No LLM is used.',
      inputSchema: {
        pattern: boundedText("One fact pattern, e.g. 'works_at(mira, _)'"),
        namespaces: namespacesField,
        limit: z.number().int().min(1).max(MAX_HISTORY_EVENTS).optional(),
      },
    },
    async ({ pattern, namespaces, limit }) => {
      try {
        return asContent(historyTool(deps, { pattern, namespaces, limit }));
      } catch (e) {
        return asError(e);
      }
    }
  );

  server.registerTool(
    'list_memories',
    {
      title: 'List memories',
      description:
        'List stored facts and rules grouped by predicate, plus explicit integrity constraints when present.',
      inputSchema: {
        namespaces: namespacesField,
        predicate: z.string().optional().describe("Filter: 'name' or 'name/arity'"),
        trustMode: trustViewField,
        recordedSequence: recordedSequenceField,
      },
    },
    async ({ namespaces, predicate, trustMode, recordedSequence }) => {
      try {
        return asContent(
          listMemoriesTool(resolvedDeps, {
            namespaces,
            predicate,
            trustMode,
            recordedSequence,
          })
        );
      } catch (e) {
        return asError(e);
      }
    }
  );

  return server;
}

export async function serveStdio(deps: PipelineDeps): Promise<void> {
  const server = createServer(deps);
  await server.connect(new StdioServerTransport());
  // stdout is the MCP channel — diagnostics must use stderr
  console.error('rembero MCP server listening on stdio');
}
