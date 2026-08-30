#!/usr/bin/env node
import { lstatSync, readFileSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { autoCaptureClaudeStop } from './autocapture/capture.js';
import { buildSessionBrief } from './autocapture/session-brief.js';
import { runInit } from './init.js';
import { backupKnowledge, restoreKnowledge } from './knowledge/backup.js';
import {
  DEFAULT_AUTO_CAPTURE_DAILY_CAP,
  MANAGED_HOOK_MARKER,
  defaultClaudeSettingsPath,
  installClaudeHook,
  removeClaudeHook,
} from './autocapture/hooks.js';
import { DEFAULT_TRANSCRIPT_TAIL_BYTES } from './autocapture/transcript.js';
import {
  createDocumentMemorgExport,
  serializeDocumentMemorgExport,
  verifyDocumentMemorgExport,
} from './document/memorg.js';
import { MAX_PROOFS_PER_ROW, serializeClause } from './engine/index.js';
import {
  entityIdentityFromEnv,
  integrityEnforcementFromEnv,
  knowledgeCheckEnforcementFromEnv,
  loadEnv,
  mcpToolProfileFromEnv,
  recallAnswerModeFromEnv,
  recallSchemaPredicateLimitFromEnv,
  validTimeModeFromEnv,
} from './env.js';
import { clientFromEnv, lazyClientFromEnv } from './llm/client.js';
import {
  rememberText,
  recallQuestion,
  type McpToolProfile,
  type RecallAnswerMode,
  type RecallRelatedKnowledgeOptions,
} from './llm/pipeline.js';
import { MAX_RECALL_SCHEMA_PREDICATES } from './llm/schema.js';
import { MAX_INTEGRITY_VIOLATIONS } from './knowledge/integrity.js';
import {
  MAX_WHY_NOT_CANDIDATES,
  MAX_WHY_NOT_DEPTH,
  MAX_WHY_NOT_EVIDENCE,
  MAX_WHY_NOT_FAILURES,
} from './knowledge/why-not.js';
import type { TopologyDirection } from './knowledge/topology.js';
import {
  MAX_REPAIR_PLANS,
  MAX_REPAIR_SEARCH_STATES,
  MAX_REPAIR_STEPS,
} from './knowledge/repair.js';
import {
  MAX_KNOWLEDGE_SEARCH_LIMIT,
  type KnowledgeSearchClauseKind,
  type KnowledgeSearchResult,
} from './knowledge/search.js';
import {
  MAX_BROWSE_GRAPH_CLAIMS,
  MAX_BROWSE_GRAPH_DEPTH,
} from './knowledge/browse.js';
import {
  MAX_KNOWLEDGE_PATH_DEPTH,
  MAX_KNOWLEDGE_PATHS,
} from './knowledge/paths.js';
import {
  MAX_KNOWLEDGE_BUNDLE_BYTES,
  serializeKnowledgeBundle,
} from './knowledge/bundle.js';
import { MAX_KNOWLEDGE_CHECK_SUITE_BYTES } from './knowledge/checks.js';
import {
  applyRuleChangeProposal,
  MAX_RULE_CHANGE_PROPOSAL_BYTES,
  RuleChangeCheckError,
} from './knowledge/rule-change.js';
import {
  applyMemoryProposal,
  MAX_MEMORY_PROPOSAL_BYTES,
  MemoryChangeCheckError,
} from './knowledge/memory-application.js';
import { KnowledgeCheckEnforcementError } from './knowledge/check-enforcement.js';
import {
  IntegrityViolationError,
  type IntegrityEnforcementOptions,
} from './knowledge/enforcement.js';
import type { EntityIdentityMode } from './knowledge/identity.js';
import {
  TrustMetadataError,
  type KnowledgeTrust,
  type TrustViewMode,
} from './knowledge/trust.js';
import {
  MAX_GRAPH_NEIGHBOR_DEPTH,
  MAX_GRAPH_NODE_ID_BYTES,
  MAX_GRAPH_RESULT_ROW,
  type ExplanationGraphSelector,
} from './knowledge/graph-navigation.js';
import { serveStdio } from './mcp/server.js';
import { captureRememberoVersion } from './ledger/remembero-version.js';
import {
  openSemanticLedgerIfSupported,
  promoteRememberoReview,
  reviewRememberoCandidate,
} from './ledger/remembero-review.js';
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
  resolveTentativeTool,
  reviewTentativeTool,
  proposeMemoryTool,
  whatIfTool,
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
  knowledgeHealthTool,
  supersedeFactsTool,
} from './mcp/tools.js';
import { embeddingClientFromEnv } from './llm/embeddings.js';
import {
  FileEmbeddingCache,
  LayeredEmbeddingCache,
  MemoryEmbeddingCache,
} from './knowledge/semantic-search.js';
import {
  MAX_HISTORY_EVENTS,
  MAX_OPERATION_ID_BYTES,
  MAX_SUPERSEDE_PATTERNS,
  MemoryStore,
  defaultRoot,
  IncompleteHistoryError,
  OperationConflictError,
  MemoryChangeStaleError,
  RuleChangeStaleError,
  type ValidTimeMode,
} from './store/store.js';
import {
  MAX_INPUT_BYTES,
  MAX_OUTPUT_BYTES,
  assertBoundedOutput,
  llmNamespaceAllowlistFromEnv,
  stringifyBoundedResult,
} from './safety.js';
import { buildSqliteExtension, openDatalogDatabase } from './sqlite/extension.js';

const USAGE = `remembero — logic-based memory for chats and agents

Usage:
  remembero serve                          Start the MCP server on stdio
  remembero serve --profile core           Register only the daily-driver tool set
  remembero serve -n proj-myapp            Default namespace for namespace-less tool calls
  remembero remember <text>                Extract facts from text and store them
  remembero propose-memory <text>         Extract and preview accepted memory without writing
  remembero apply-memory <file>            Apply one reviewed accepted-memory proposal
  remembero remember --batch               Auto-capture from Claude Stop-hook JSON on stdin
  remembero recall <question>              Answer a question from memory
  remembero recall-explain <question>      Recall with proofs, sources, and a graph
  remembero query <datalog>                Run a raw Datalog query
  remembero assert <datalog>               Store raw Datalog facts, rules, or constraints
  remembero claims                         List tentative facts awaiting review
  remembero accept <datalog>               Promote exact tentative facts to accepted
  remembero reject <datalog>               Reject exact tentative facts without accepting
  remembero supersede [datalog]            End matching facts; optionally add replacements
  remembero explain <datalog>              Query with proofs, sources, and a knowledge graph
  remembero check                          Check explicit integrity constraints with evidence
  remembero conflicts [focus]              Group conflicts by authored focus with evidence
  remembero what-if <query>                Preview fact or rule changes with deterministic impact
  remembero apply-rule-change <file>       Apply one reviewed digest-bound rule proposal
  remembero why-not <query>                Explain deterministic blockers for a query
  remembero topology [predicate]           Map rules, policies, strata, and influence
  remembero diff <from> <to>               Compare two exact recorded knowledge states
  remembero version <command>              Capture and review semantic versions
  remembero repair <query>                 Propose minimal verified fact-only query repairs
  remembero audit-rules [predicate]        Audit rule health with deterministic evidence
  remembero health                         Inspect complete deterministic knowledge health
  remembero search <text>                  Search facts, rules, policies, and sources locally
  remembero semantic-search <text>         Provider-assisted search for preferences and advice
  remembero semantic-index                 Prewarm derived vectors after reviewed writes
  remembero browse [entity]                Browse a bounded explicit personal graph
  remembero connect <from> <to>            Find bounded shortest explicit graph paths
  remembero bundle                         Export raw clauses and provenance with a digest
  remembero verify-bundle <file>           Verify a standalone knowledge bundle
  remembero document-memorg                Export the real-PDF corpus as Memorg memory
  remembero verify-document-memorg <file>  Verify a standalone Memorg memory artifact
  remembero test-knowledge <file>          Run a deterministic rule regression suite
  remembero profile <query>                Profile deterministic relation work and proofs
  remembero forget <pattern>               Retract facts matching a pattern
  remembero history <pattern>              Show a fact's deterministic life story
  remembero checkpoint                     Rotate the active journal into a verified segment
  remembero checkpoints                    List immutable journal checkpoints
  remembero list                           List stored memories
  remembero review                         Review recent auto-captured facts
  remembero init                           One-command Claude Code setup (hooks + MCP + snippet)
  remembero init-hooks                     Install the opt-in Claude capture + brief hooks
  remembero init-hooks --remove            Remove only Remembero's managed hooks
  remembero session-brief                  Print the bounded session-start memory brief
  remembero export                         Print all memories as portable Datalog
  remembero import <ns> <file>             Load clauses from a .dl file into a namespace
  remembero sqlite-build                   Compile the loadable SQLite extension
  remembero sqlite-sql <db> <rule>         Compile a Datalog rule against a SQLite database
  remembero sqlite-query <db> <program>    Execute a Datalog program against a SQLite database
  remembero sqlite-explain <db> <program>  Execute with one derivation proof per result
  remembero sqlite-plan <db> <program>     Inspect routing and schema without scanning rows

Options:
  -n, --namespace <ns>     Namespace to write to / read from (default: "default")
      --namespaces <a,b|*> Namespaces to search for health/propose-memory/recall/query/profile/what-if/why-not/topology/diff/audit-rules/search/browse/connect/test-knowledge/check/conflicts/list/claims/history
      --valid-time-mode <mode>  Supersession: delete (default) or archive_until
      --schema-predicate-limit <n>  Detailed recall predicates (default: 8; max: 256)
      --proof-limit <n>    Proof witnesses per explain result (default: 1; max: ${MAX_PROOFS_PER_ROW})
      --max-violations <n> Maximum integrity violations (default: 1000; max: ${MAX_INTEGRITY_VIOLATIONS})
      --integrity-mode <mode>  Write guard: off, strict, or no_new_violations
      --integrity-namespaces <a,b|*>  Knowledge view governed by write enforcement
      --entity-identity <mode>  Read projection: off (default) or canonical
      --trust <mode>        Writes: accepted/tentative; reads: accepted/include_tentative
      --answer-mode <mode>  Recall phrasing: natural, deterministic, or evidence
      --related           Include local discovery evidence when recall cannot answer
      --related-limit <n> Maximum related matches (default: 20; max: ${MAX_KNOWLEDGE_SEARCH_LIMIT})
      --related-kind <kind> Related fact, rule, or constraint filter; repeatable
      --pattern <datalog>  Fact pattern to end; repeat for supersede (maximum: ${MAX_SUPERSEDE_PATTERNS})
      --assume <facts>     Ground facts to add in a what-if simulation; repeatable
      --without <pattern> Ground fact pattern to remove in a what-if simulation; repeatable
      --assume-rule <rule> Rule to add in a what-if simulation; repeatable
      --without-rule <rule> Exact alpha-equivalent rule to remove; repeatable
      --check-suite <file> Evaluate a knowledge check/coverage suite before and after
      --failure-limit <n> Why-not blocker limit (default: 32; max: ${MAX_WHY_NOT_FAILURES})
      --diagnostic-depth <n> Why-not rule depth (default: 8; max: ${MAX_WHY_NOT_DEPTH})
      --candidate-limit <n> Nearby facts per blocker (default: 4; max: ${MAX_WHY_NOT_CANDIDATES})
      --evidence-limit <n> Sourced nearby facts overall (default: 16; max: ${MAX_WHY_NOT_EVIDENCE})
      --direction <mode>  Topology focus: upstream, downstream, or both (default: both)
      --query <datalog>   Optional query whose proof/result impact is included in diff
      --plan-limit <n>    Repair plans (default: 8; max: ${MAX_REPAIR_PLANS})
      --repair-steps <n>  Iterative repair depth (default: 4; max: ${MAX_REPAIR_STEPS})
      --search-states <n> Repair search states (default: 128; max: ${MAX_REPAIR_SEARCH_STATES})
      --search-limit <n>  Search results (default: 20; max: ${MAX_KNOWLEDGE_SEARCH_LIMIT})
      --kind <kind>       Search kind: fact, rule, or constraint; repeatable
      --after <cursor>    Resume semantic-index from a returned nextCursor
      --predicate <name>  Browse seed predicate name or name/arity
      --browse-depth <n>  Explicit graph depth (default: 1; max: ${MAX_BROWSE_GRAPH_DEPTH})
      --claim-limit <n>   Explicit graph claims (default: 100; max: ${MAX_BROWSE_GRAPH_CLAIMS})
      --focus-number      Interpret browse entity focus as a numeric term
      --path-depth <n>    Explicit relationship hops (default: 4; max: ${MAX_KNOWLEDGE_PATH_DEPTH})
      --path-limit <n>    Complete shortest paths (default: 3; max: ${MAX_KNOWLEDGE_PATHS})
      --from-number       Interpret the connect start as a numeric term
      --to-number         Interpret the connect end as a numeric term
      --include-derived   Let connect traverse rule-derived claims with proofs
      --include-passing-evidence  Include proofs/graphs for passing knowledge checks
      --compare-scan        Re-run profile with relation indexes disabled and prove equivalence
      --at <ISO>           Canonical UTC valid-until instant for supersede
      --dry-run            Preview checkpoint metadata without rotating journal.log
      --op-id <id>        Stable key for writes including reviewed rule application
      --as-of-sequence <n> Read the knowledge view after global journal entry n (0 = empty)
      --graph-result <n>  Export the complete support graph for result row n
      --graph-support <node-id>  Export the support closure for one graph node
      --graph-neighbors <node-id>  Export a bounded undirected neighborhood
      --graph-depth <n>   Neighborhood depth (default: 1; max: ${MAX_GRAPH_NEIGHBOR_DEPTH})
      --limit <n>          History event limit (maximum: 1000)
      --extension <path>   Path to the compiled Remembero SQLite extension
      --daily-cap <n>      Max auto-capture attempts per namespace/UTC day (default: 10)
      --tail-bytes <n>     Transcript tail bytes sent for extraction (default: 24576)
      --days <n>           Auto-capture review window (default: 7)
      --forget <n,...>     Prune numbered facts shown by review
      --settings <path>    Claude settings JSON (default: ~/.claude/settings.json)
      --json               Emit machine-readable batch/review/history output
`;

interface ParsedArgs {
  positional: string[];
  namespace?: string;
  namespaces?: string[] | '*';
  extensionPath?: string;
  batch: boolean;
  json: boolean;
  remove: boolean;
  dryRun: boolean;
  dailyCap?: string;
  tailBytes?: string;
  days?: string;
  forget?: string;
  settingsPath?: string;
  managedBy?: string;
  validTimeMode?: string;
  schemaPredicateLimit?: string;
  proofLimit?: string;
  maxViolations?: string;
  integrityMode?: string;
  integrityNamespaces?: string[] | '*';
  entityIdentity?: string;
  trust?: string;
  answerMode?: string;
  profile?: string;
  opId?: string;
  graphResult?: string;
  graphSupport?: string;
  graphNeighbors?: string;
  graphDepth?: string;
  limit?: string;
  asOfSequence?: string;
  patterns: string[];
  assumptions: string[];
  without: string[];
  assumedRules: string[];
  withoutRules: string[];
  checkSuitePath?: string;
  failureLimit?: string;
  diagnosticDepth?: string;
  candidateLimit?: string;
  evidenceLimit?: string;
  direction?: string;
  queryText?: string;
  planLimit?: string;
  repairSteps?: string;
  searchStates?: string;
  searchLimit?: string;
  searchKinds: string[];
  semanticAfter?: string;
  related: boolean;
  relatedLimit?: string;
  relatedKinds: string[];
  browsePredicate?: string;
  browseDepth?: string;
  claimLimit?: string;
  focusNumber: boolean;
  pathDepth?: string;
  pathLimit?: string;
  fromNumber: boolean;
  toNumber: boolean;
  includeDerived: boolean;
  includePassingEvidence: boolean;
  compareScan: boolean;
  at?: string;
}

function parseArgs(argv: string[]): ParsedArgs {
  const parsed: ParsedArgs = {
    positional: [],
    batch: false,
    json: false,
    remove: false,
    dryRun: false,
    patterns: [],
    assumptions: [],
    without: [],
    assumedRules: [],
    withoutRules: [],
    searchKinds: [],
    related: false,
    relatedKinds: [],
    focusNumber: false,
    fromNumber: false,
    toNumber: false,
    includeDerived: false,
    includePassingEvidence: false,
    compareScan: false,
  };
  const valueAfter = (index: number, flag: string): string => {
    const value = argv[index + 1];
    if (value === undefined) throw new Error(`${flag} requires a value`);
    return value;
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '-n' || arg === '--namespace') {
      parsed.namespace = valueAfter(i, arg);
      i += 1;
    } else if (arg === '--namespaces') {
      const value = valueAfter(i, arg);
      i += 1;
      parsed.namespaces = value === '*' ? '*' : value.split(',');
    } else if (arg === '--extension') {
      parsed.extensionPath = valueAfter(i, arg);
      i += 1;
    } else if (arg === '--batch') {
      parsed.batch = true;
    } else if (arg === '--json') {
      parsed.json = true;
    } else if (arg === '--remove') {
      parsed.remove = true;
    } else if (arg === '--dry-run') {
      parsed.dryRun = true;
    } else if (arg === '--daily-cap') {
      parsed.dailyCap = valueAfter(i, arg);
      i += 1;
    } else if (arg === '--tail-bytes') {
      parsed.tailBytes = valueAfter(i, arg);
      i += 1;
    } else if (arg === '--valid-time-mode') {
      parsed.validTimeMode = valueAfter(i, arg);
      i += 1;
    } else if (arg === '--profile') {
      parsed.profile = valueAfter(i, arg);
      i += 1;
    } else if (arg === '--schema-predicate-limit') {
      parsed.schemaPredicateLimit = valueAfter(i, arg);
      i += 1;
    } else if (arg === '--proof-limit') {
      parsed.proofLimit = valueAfter(i, arg);
      i += 1;
    } else if (arg === '--max-violations') {
      parsed.maxViolations = valueAfter(i, arg);
      i += 1;
    } else if (arg === '--integrity-mode') {
      parsed.integrityMode = valueAfter(i, arg);
      i += 1;
    } else if (arg === '--integrity-namespaces') {
      const value = valueAfter(i, arg);
      i += 1;
      parsed.integrityNamespaces = value === '*' ? '*' : value.split(',');
    } else if (arg === '--entity-identity') {
      parsed.entityIdentity = valueAfter(i, arg);
      i += 1;
    } else if (arg === '--trust') {
      parsed.trust = valueAfter(i, arg);
      i += 1;
    } else if (arg === '--answer-mode') {
      parsed.answerMode = valueAfter(i, arg);
      i += 1;
    } else if (arg === '--related') {
      parsed.related = true;
    } else if (arg === '--related-limit') {
      parsed.relatedLimit = valueAfter(i, arg);
      i += 1;
    } else if (arg === '--related-kind') {
      parsed.relatedKinds.push(valueAfter(i, arg));
      i += 1;
    } else if (arg === '--op-id') {
      parsed.opId = valueAfter(i, arg);
      i += 1;
    } else if (arg === '--pattern') {
      parsed.patterns.push(valueAfter(i, arg));
      i += 1;
    } else if (arg === '--assume') {
      parsed.assumptions.push(valueAfter(i, arg));
      i += 1;
    } else if (arg === '--without') {
      parsed.without.push(valueAfter(i, arg));
      i += 1;
    } else if (arg === '--assume-rule') {
      parsed.assumedRules.push(valueAfter(i, arg));
      i += 1;
    } else if (arg === '--without-rule') {
      parsed.withoutRules.push(valueAfter(i, arg));
      i += 1;
    } else if (arg === '--check-suite') {
      parsed.checkSuitePath = valueAfter(i, arg);
      i += 1;
    } else if (arg === '--failure-limit') {
      parsed.failureLimit = valueAfter(i, arg);
      i += 1;
    } else if (arg === '--diagnostic-depth') {
      parsed.diagnosticDepth = valueAfter(i, arg);
      i += 1;
    } else if (arg === '--candidate-limit') {
      parsed.candidateLimit = valueAfter(i, arg);
      i += 1;
    } else if (arg === '--evidence-limit') {
      parsed.evidenceLimit = valueAfter(i, arg);
      i += 1;
    } else if (arg === '--direction') {
      parsed.direction = valueAfter(i, arg);
      i += 1;
    } else if (arg === '--query') {
      parsed.queryText = valueAfter(i, arg);
      i += 1;
    } else if (arg === '--plan-limit') {
      parsed.planLimit = valueAfter(i, arg);
      i += 1;
    } else if (arg === '--repair-steps') {
      parsed.repairSteps = valueAfter(i, arg);
      i += 1;
    } else if (arg === '--search-states') {
      parsed.searchStates = valueAfter(i, arg);
      i += 1;
    } else if (arg === '--search-limit') {
      parsed.searchLimit = valueAfter(i, arg);
      i += 1;
    } else if (arg === '--kind') {
      parsed.searchKinds.push(valueAfter(i, arg));
      i += 1;
    } else if (arg === '--after') {
      parsed.semanticAfter = valueAfter(i, arg);
      i += 1;
    } else if (arg === '--predicate') {
      parsed.browsePredicate = valueAfter(i, arg);
      i += 1;
    } else if (arg === '--browse-depth') {
      parsed.browseDepth = valueAfter(i, arg);
      i += 1;
    } else if (arg === '--claim-limit') {
      parsed.claimLimit = valueAfter(i, arg);
      i += 1;
    } else if (arg === '--focus-number') {
      parsed.focusNumber = true;
    } else if (arg === '--path-depth') {
      parsed.pathDepth = valueAfter(i, arg);
      i += 1;
    } else if (arg === '--path-limit') {
      parsed.pathLimit = valueAfter(i, arg);
      i += 1;
    } else if (arg === '--from-number') {
      parsed.fromNumber = true;
    } else if (arg === '--to-number') {
      parsed.toNumber = true;
    } else if (arg === '--include-derived') {
      parsed.includeDerived = true;
    } else if (arg === '--include-passing-evidence') {
      parsed.includePassingEvidence = true;
    } else if (arg === '--compare-scan') {
      parsed.compareScan = true;
    } else if (arg === '--at') {
      parsed.at = valueAfter(i, arg);
      i += 1;
    } else if (arg === '--graph-result') {
      parsed.graphResult = valueAfter(i, arg);
      i += 1;
    } else if (arg === '--graph-support') {
      parsed.graphSupport = valueAfter(i, arg);
      i += 1;
    } else if (arg === '--graph-neighbors') {
      parsed.graphNeighbors = valueAfter(i, arg);
      i += 1;
    } else if (arg === '--graph-depth') {
      parsed.graphDepth = valueAfter(i, arg);
      i += 1;
    } else if (arg === '--limit') {
      parsed.limit = valueAfter(i, arg);
      i += 1;
    } else if (arg === '--as-of-sequence') {
      parsed.asOfSequence = valueAfter(i, arg);
      i += 1;
    } else if (arg === '--days') {
      parsed.days = valueAfter(i, arg);
      i += 1;
    } else if (arg === '--forget') {
      parsed.forget = valueAfter(i, arg);
      i += 1;
    } else if (arg === '--settings') {
      parsed.settingsPath = valueAfter(i, arg);
      i += 1;
    } else if (arg === '--managed-by') {
      parsed.managedBy = valueAfter(i, arg);
      i += 1;
    } else {
      parsed.positional.push(arg);
    }
  }
  return parsed;
}

function integerOption(value: string | undefined, fallback: number, label: string): number {
  if (value === undefined) return fallback;
  if (!/^\d+$/.test(value)) throw new Error(`${label} must be an integer`);
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed)) throw new Error(`${label} must be a safe integer`);
  return parsed;
}

function validTimeModeOption(value: string | undefined): ValidTimeMode {
  if (value === undefined) return validTimeModeFromEnv();
  if (value === 'delete' || value === 'archive_until') return value;
  throw new Error("--valid-time-mode must be 'delete' or 'archive_until'");
}

function recallSchemaPredicateLimitOption(value: string | undefined): number {
  if (value === undefined) return recallSchemaPredicateLimitFromEnv();
  const parsed = integerOption(value, 0, 'recall schema predicate limit');
  if (parsed < 1 || parsed > MAX_RECALL_SCHEMA_PREDICATES) {
    throw new Error(
      `recall schema predicate limit must be from 1 to ${MAX_RECALL_SCHEMA_PREDICATES}`
    );
  }
  return parsed;
}

function proofLimitOption(value: string | undefined): number | undefined {
  if (value === undefined) return undefined;
  const parsed = integerOption(value, 0, 'proof limit');
  if (parsed < 1 || parsed > MAX_PROOFS_PER_ROW) {
    throw new Error(`proof limit must be from 1 to ${MAX_PROOFS_PER_ROW}`);
  }
  return parsed;
}

function maxViolationsOption(value: string | undefined): number | undefined {
  if (value === undefined) return undefined;
  const parsed = integerOption(value, 0, 'maximum integrity violations');
  if (parsed < 1 || parsed > MAX_INTEGRITY_VIOLATIONS) {
    throw new Error(
      `maximum integrity violations must be from 1 to ${MAX_INTEGRITY_VIOLATIONS}`
    );
  }
  return parsed;
}

function topologyDirectionOption(
  value: string | undefined
): TopologyDirection | undefined {
  if (value === undefined) return undefined;
  if (value === 'upstream' || value === 'downstream' || value === 'both') {
    return value;
  }
  throw new Error("topology direction must be 'upstream', 'downstream', or 'both'");
}

function searchKindsOption(
  values: string[],
  flag = '--kind'
): KnowledgeSearchClauseKind[] | undefined {
  if (values.length === 0) return undefined;
  const kinds = [...new Set(values)];
  for (const kind of kinds) {
    if (kind !== 'fact' && kind !== 'rule' && kind !== 'constraint') {
      throw new Error(`${flag} must be 'fact', 'rule', or 'constraint'`);
    }
  }
  return kinds as KnowledgeSearchClauseKind[];
}

function relatedKnowledgeOption(
  args: ParsedArgs
): boolean | RecallRelatedKnowledgeOptions | undefined {
  if (!args.related && args.relatedLimit === undefined && args.relatedKinds.length === 0) {
    return undefined;
  }
  const kinds = searchKindsOption(args.relatedKinds, '--related-kind');
  let limit: number | undefined;
  if (args.relatedLimit !== undefined) {
    limit = integerOption(args.relatedLimit, 0, 'related knowledge limit');
    if (limit < 1 || limit > MAX_KNOWLEDGE_SEARCH_LIMIT) {
      throw new Error(
        `related knowledge limit must be from 1 to ${MAX_KNOWLEDGE_SEARCH_LIMIT}`
      );
    }
  }
  if (limit === undefined && kinds === undefined) return true;
  return {
    ...(limit === undefined ? {} : { limit }),
    ...(kinds === undefined ? {} : { kinds }),
  };
}

function relatedKnowledgeText(result: KnowledgeSearchResult): string {
  const lines = ['Related knowledge (discovery only; not an answer or proof):'];
  if (result.status === 'no_match') {
    lines.push('  No local lexical matches.');
  } else {
    for (const item of result.results) {
      lines.push(`  ${item.rank}. ${item.clause} (score ${item.score})`);
    }
    if (result.truncated) {
      lines.push(`  ... ${result.matchCount - result.returnedCount} more matches`);
    }
  }
  const text = lines.join('\n');
  assertBoundedOutput(text, 'CLI related knowledge');
  return text;
}

function entityIdentityOption(
  value: string | undefined
): EntityIdentityMode | false | undefined {
  if (value === undefined) return entityIdentityFromEnv();
  if (value === 'off') return false;
  if (value === 'canonical') return value;
  throw new Error("--entity-identity must be 'off' or 'canonical'");
}

function knowledgeTrustOption(value: string | undefined): KnowledgeTrust {
  if (value === undefined || value === 'accepted') return 'accepted';
  if (value === 'tentative') return value;
  throw new Error("write --trust must be 'accepted' or 'tentative'");
}

function trustViewOption(value: string | undefined): TrustViewMode {
  if (value === undefined || value === 'accepted') return 'accepted';
  if (value === 'include_tentative') return value;
  throw new Error("read --trust must be 'accepted' or 'include_tentative'");
}

function recallAnswerModeOption(
  value: string | undefined
): RecallAnswerMode {
  if (value === undefined) return recallAnswerModeFromEnv();
  if (value === 'natural' || value === 'deterministic' || value === 'evidence') {
    return value;
  }
  throw new Error("--answer-mode must be 'natural', 'deterministic', or 'evidence'");
}

function toolProfileOption(value: string | undefined): McpToolProfile {
  if (value === undefined) return mcpToolProfileFromEnv();
  if (value === 'core' || value === 'full') return value;
  throw new Error("--profile must be 'core' or 'full'");
}

function graphNodeIdOption(value: string, label: string): string {
  if (value.length === 0) throw new Error(`${label} must not be empty`);
  if (Buffer.byteLength(value, 'utf8') > MAX_GRAPH_NODE_ID_BYTES) {
    throw new Error(`${label} exceeds ${MAX_GRAPH_NODE_ID_BYTES} bytes`);
  }
  return value;
}

function operationIdOption(value: string | undefined): string | undefined {
  if (value === undefined) return undefined;
  if (value.length === 0) throw new Error('operation id must not be empty');
  if (Buffer.byteLength(value, 'utf8') > MAX_OPERATION_ID_BYTES) {
    throw new Error(`operation id exceeds ${MAX_OPERATION_ID_BYTES} bytes`);
  }
  return value;
}

function graphSelectorOption(args: ParsedArgs): ExplanationGraphSelector | undefined {
  const selected = [args.graphResult, args.graphSupport, args.graphNeighbors].filter(
    (value) => value !== undefined
  );
  if (selected.length > 1) {
    throw new Error(
      '--graph-result, --graph-support, and --graph-neighbors are mutually exclusive'
    );
  }
  if (args.graphDepth !== undefined && args.graphNeighbors === undefined) {
    throw new Error('--graph-depth requires --graph-neighbors');
  }
  if (args.graphResult !== undefined) {
    const row = integerOption(args.graphResult, 0, 'graph result row');
    if (row < 1 || row > MAX_GRAPH_RESULT_ROW) {
      throw new Error(`graph result row must be from 1 to ${MAX_GRAPH_RESULT_ROW}`);
    }
    return { kind: 'result', row };
  }
  if (args.graphSupport !== undefined) {
    return {
      kind: 'support',
      nodeId: graphNodeIdOption(args.graphSupport, 'graph support node id'),
    };
  }
  if (args.graphNeighbors !== undefined) {
    const depth = integerOption(args.graphDepth, 1, 'graph neighbor depth');
    if (depth < 1 || depth > MAX_GRAPH_NEIGHBOR_DEPTH) {
      throw new Error(`graph neighbor depth must be from 1 to ${MAX_GRAPH_NEIGHBOR_DEPTH}`);
    }
    return {
      kind: 'neighbors',
      nodeId: graphNodeIdOption(args.graphNeighbors, 'graph neighbor node id'),
      depth,
    };
  }
  return undefined;
}

function integrityEnforcementOption(
  mode: string | undefined,
  namespaces: string[] | '*' | undefined,
  fallback: IntegrityEnforcementOptions | undefined,
  proofLimit: string | undefined,
  maxViolations: string | undefined,
  graphSelector: ExplanationGraphSelector | undefined
): IntegrityEnforcementOptions | false | undefined {
  const proofLimitValue = proofLimitOption(proofLimit);
  const maxViolationsValue = maxViolationsOption(maxViolations);
  if (mode === undefined) {
    if (
      (namespaces !== undefined ||
        proofLimitValue !== undefined ||
        maxViolationsValue !== undefined ||
        graphSelector !== undefined) &&
      fallback === undefined
    ) {
      throw new Error(
        'integrity write options require --integrity-mode or REMBERO_INTEGRITY_MODE'
      );
    }
    return fallback === undefined
      ? undefined
      : {
          ...fallback,
          ...(namespaces === undefined ? {} : { namespaces }),
          ...(proofLimitValue === undefined
            ? {}
            : { maxProofsPerRow: proofLimitValue }),
          ...(maxViolationsValue === undefined ? {} : { maxViolations: maxViolationsValue }),
          ...(graphSelector === undefined ? {} : { graphSelector }),
        };
  }
  if (mode === 'off') {
    if (
      namespaces !== undefined ||
      proofLimitValue !== undefined ||
      maxViolationsValue !== undefined ||
      graphSelector !== undefined
    ) {
      throw new Error("--integrity-mode 'off' cannot use integrity write options");
    }
    return false;
  }
  if (mode !== 'strict' && mode !== 'no_new_violations') {
    throw new Error("--integrity-mode must be 'off', 'strict', or 'no_new_violations'");
  }
  return {
    mode,
    ...(namespaces === undefined ? {} : { namespaces }),
    ...(proofLimitValue === undefined ? {} : { maxProofsPerRow: proofLimitValue }),
    ...(maxViolationsValue === undefined ? {} : { maxViolations: maxViolationsValue }),
    ...(graphSelector === undefined ? {} : { graphSelector }),
  };
}

async function readStdinBounded(maxBytes = MAX_INPUT_BYTES): Promise<string> {
  const chunks: Buffer[] = [];
  let bytes = 0;
  for await (const chunk of process.stdin) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    bytes += buffer.length;
    if (bytes > maxBytes) throw new Error(`stdin exceeds ${maxBytes} bytes`);
    chunks.push(buffer);
  }
  return Buffer.concat(chunks).toString('utf8');
}

function reviewSelections(raw: string | undefined, factCount: number): number[] {
  if (raw === undefined) return [];
  const values = raw.split(',').map((value) => value.trim());
  if (values.some((value) => !/^\d+$/.test(value))) {
    throw new Error('--forget must be a comma-separated list of review numbers');
  }
  const selected = [...new Set(values.map(Number))];
  const invalid = selected.find((value) => value < 1 || value > factCount);
  if (invalid !== undefined) {
    throw new Error(`review fact number ${invalid} is outside 1..${factCount}`);
  }
  return selected;
}

const VERSION_USAGE = `Remembero semantic versions

Commands:
  remembero version capture [--label <label>] [--ref <name>] [--ledger <path>]
  remembero version list [--ledger <path>]
  remembero version inspect <digest|label|ref> [--ledger <path>]
  remembero version diff <from> <to> [--ledger <path>]
  remembero version review <candidate> [--no-document-evaluation] [--ledger <path>]
  remembero version history [ref] [--ledger <path>]
  remembero version promote <candidate> <assessment> --op-id <id> [--ref <name>]
    [--accept-review <dimension>] [--reason <text>] [--ledger <path>]
`;

interface VersionArgs {
  positional: string[];
  ledgerPath?: string;
  label?: string;
  ref: string;
  opId?: string;
  acceptReviews: string[];
  reason?: string;
  documentEvaluation: boolean;
}

function parseVersionArgs(argv: string[]): VersionArgs {
  const parsed: VersionArgs = {
    positional: [],
    ref: 'main',
    acceptReviews: [],
    documentEvaluation: true,
  };
  const valueAfter = (index: number, flag: string): string => {
    const value = argv[index + 1];
    if (value === undefined) throw new Error(`${flag} requires a value`);
    return value;
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--ledger') {
      parsed.ledgerPath = valueAfter(index, arg);
      index += 1;
    } else if (arg === '--label') {
      parsed.label = valueAfter(index, arg);
      index += 1;
    } else if (arg === '--ref') {
      parsed.ref = valueAfter(index, arg);
      index += 1;
    } else if (arg === '--op-id') {
      parsed.opId = valueAfter(index, arg);
      index += 1;
    } else if (arg === '--accept-review') {
      parsed.acceptReviews.push(valueAfter(index, arg));
      index += 1;
    } else if (arg === '--reason') {
      parsed.reason = valueAfter(index, arg);
      index += 1;
    } else if (arg === '--no-document-evaluation') {
      parsed.documentEvaluation = false;
    } else if (arg === '--help' || arg === '-h') {
      console.log(VERSION_USAGE);
      return parsed;
    } else {
      parsed.positional.push(arg);
    }
  }
  return parsed;
}

async function runVersionCommand(argv: string[]): Promise<void> {
  const [subcommand, ...rest] = argv;
  if (subcommand === undefined || subcommand === '--help' || subcommand === '-h') {
    console.log(VERSION_USAGE);
    return;
  }
  const args = parseVersionArgs(rest);
  const ledgerPath = args.ledgerPath ?? join(defaultRoot(), 'semantic.sqlite');
  const semanticAuthority = await openSemanticLedgerIfSupported(ledgerPath);
  if (semanticAuthority === undefined) {
    throw new Error(
      "semantic version commands require the node:sqlite module (Node 22 or newer)"
    );
  }
  const { database, ledger } = semanticAuthority;
  try {
    if (subcommand === 'capture') {
      if (args.positional.length !== 0) throw new Error('version capture accepts no positional arguments');
      const store = new MemoryStore();
      const parent = ledger.getRef(args.ref)?.versionDigest;
      const capture = captureRememberoVersion({
        ledger,
        store,
        ...(parent === undefined ? {} : { parents: [parent] }),
        label: args.label ?? `remembero@capture-${Date.now()}`,
        metadata: { source: 'cli' },
      });
      if (parent === undefined) {
        ledger.setRef({
          name: args.ref,
          versionDigest: capture.version.digest,
          operationId: `remembero-version-initialize-${args.ref}`,
          reason: 'Initialize semantic version ref',
        });
      }
      console.log(stringifyBoundedResult({
        version: capture.version,
        baselineVersionDigest: parent,
        recordedSnapshot: {
          sequence: capture.recordedSnapshot.sequence,
          journalEntries: capture.recordedSnapshot.journalEntries,
          namespaces: capture.recordedSnapshot.namespaces,
        },
      }, 'version capture'));
      return;
    }
    if (subcommand === 'list') {
      if (args.positional.length !== 0) throw new Error('version list accepts no positional arguments');
      console.log(stringifyBoundedResult({ refs: ledger.listRefs(), versions: ledger.listVersions() }, 'version list'));
      return;
    }
    if (subcommand === 'inspect') {
      if (args.positional.length !== 1) throw new Error('version inspect requires one reference');
      console.log(stringifyBoundedResult(ledger.resolveVersion(args.positional[0]), 'version inspect'));
      return;
    }
    if (subcommand === 'diff') {
      if (args.positional.length !== 2) throw new Error('version diff requires <from> <to>');
      const from = ledger.resolveVersion(args.positional[0]);
      const to = ledger.resolveVersion(args.positional[1]);
      console.log(stringifyBoundedResult(ledger.diffVersions(from.digest, to.digest), 'version diff'));
      return;
    }
    if (subcommand === 'review') {
      if (args.positional.length !== 1) throw new Error('version review requires one candidate');
      const candidate = ledger.resolveVersion(args.positional[0]);
      const store = new MemoryStore();
      const result = reviewRememberoCandidate({
        ledger,
        store,
        candidateVersionDigest: candidate.digest,
        baselineVersionDigest: candidate.parents[0],
        includeDocumentEvaluation: args.documentEvaluation,
      });
      console.log(stringifyBoundedResult(result, 'version review'));
      return;
    }
    if (subcommand === 'history') {
      if (args.positional.length > 1) throw new Error('version history accepts at most one ref');
      const ref = args.positional[0] ?? args.ref;
      console.log(stringifyBoundedResult({ ref, history: ledger.refHistory(ref) }, 'version history'));
      return;
    }
    if (subcommand === 'promote') {
      if (args.positional.length !== 2) throw new Error('version promote requires <candidate> <assessment>');
      if (args.opId === undefined) throw new Error('version promote requires --op-id');
      const candidate = ledger.resolveVersion(args.positional[0]);
      const current = ledger.getRef(args.ref);
      const decision = promoteRememberoReview({
        ledger,
        ref: args.ref,
        candidateVersionDigest: candidate.digest,
        assessmentDigest: args.positional[1],
        operationId: args.opId,
        expectedCurrentVersionDigest: current?.versionDigest,
        acceptedReviewDimensions: args.acceptReviews,
        reason: args.reason,
      });
      console.log(stringifyBoundedResult(decision, 'version promotion'));
      return;
    }
    throw new Error(`unknown version command '${subcommand}'`);
  } finally {
    database.close();
  }
}

async function main(): Promise<void> {
  const [command, ...rest] = process.argv.slice(2);
  if (command === undefined || command === '--help' || command === '-h') {
    console.log(USAGE);
    return;
  }
  loadEnv();
  if (command === 'version') {
    await runVersionCommand(rest);
    return;
  }
  const args = parseArgs(rest);
  if (command === 'document-memorg') {
    if (args.positional.length !== 0) {
      throw new Error('document-memorg does not accept positional arguments');
    }
    console.log(serializeDocumentMemorgExport(createDocumentMemorgExport()));
    return;
  }
  if (command === 'verify-document-memorg') {
    if (args.positional.length !== 1) {
      throw new Error('verify-document-memorg requires exactly one artifact file');
    }
    const file = resolve(args.positional[0]);
    const stat = lstatSync(file);
    if (stat.isSymbolicLink() || !stat.isFile()) {
      throw new Error('refusing non-regular Memorg artifact file');
    }
    if (stat.size > MAX_OUTPUT_BYTES) {
      throw new Error(`Memorg artifact exceeds ${MAX_OUTPUT_BYTES} bytes`);
    }
    console.log(
      stringifyBoundedResult(
        verifyDocumentMemorgExport(readFileSync(file, 'utf8')),
        'CLI result'
      )
    );
    return;
  }
  const store = new MemoryStore();
  const graphSelector = graphSelectorOption(args);
  const operationId = operationIdOption(args.opId);
  const recordedSequence = args.asOfSequence === undefined
    ? undefined
    : integerOption(args.asOfSequence, 0, 'recorded snapshot sequence');
  const entityIdentitySetting = entityIdentityOption(args.entityIdentity);
  const entityIdentity = entityIdentitySetting === false
    ? undefined
    : entityIdentitySetting;
  const writeCommand = [
    'serve',
    'remember',
    'assert',
    'accept',
    'reject',
    'supersede',
    'forget',
    'import',
    'review',
    'checkpoint',
  ].includes(command ?? '');
  const graphCommand = ['recall-explain', 'explain', 'profile', 'check', 'conflicts'].includes(command ?? '');
  if (graphSelector !== undefined && !writeCommand && !graphCommand) {
    throw new Error(
      'graph selection is available for recall-explain, explain, profile, check, conflicts, and integrity-guarded writes'
    );
  }
  if (
    operationId !== undefined &&
    !['assert', 'accept', 'reject', 'supersede', 'forget', 'import', 'checkpoint', 'apply-rule-change', 'apply-memory'].includes(command ?? '')
  ) {
    throw new Error('--op-id is available for assert, accept, reject, supersede, forget, import, checkpoint, apply-rule-change, and apply-memory');
  }
  if (
    args.trust !== undefined &&
    ![
      'serve',
      'remember',
      'recall',
      'recall-explain',
      'query',
      'assert',
      'explain',
      'check',
      'conflicts',
      'what-if',
      'why-not',
      'topology',
      'diff',
      'repair',
      'audit-rules',
      'health',
      'search',
      'browse',
      'connect',
      'test-knowledge',
      'profile',
      'list',
    ].includes(command ?? '')
  ) {
    throw new Error('--trust is unavailable for this command');
  }
  if (args.batch && args.trust !== undefined) {
    throw new Error('--trust does not apply to auto-capture batches');
  }
  if (
    args.answerMode !== undefined &&
    !['serve', 'recall', 'recall-explain'].includes(command ?? '')
  ) {
    throw new Error('--answer-mode is available only for serve, recall, or recall-explain');
  }
  if (args.profile !== undefined && command !== 'serve') {
    throw new Error('--profile is available only for serve');
  }
  if (args.patterns.length > 0 && command !== 'supersede') {
    throw new Error('--pattern is available only for supersede');
  }
  if (args.assumptions.length > 0 && command !== 'what-if') {
    throw new Error('--assume is available only for what-if');
  }
  if (args.without.length > 0 && command !== 'what-if') {
    throw new Error('--without is available only for what-if');
  }
  if (
    (args.assumedRules.length > 0 ||
      args.withoutRules.length > 0) &&
    command !== 'what-if'
  ) {
    throw new Error('rule simulation options are available only for what-if');
  }
  if (
    args.checkSuitePath !== undefined &&
    command !== 'what-if' &&
    command !== 'health' &&
    command !== 'propose-memory'
  ) {
    throw new Error('--check-suite is available only for propose-memory, what-if, or health');
  }
  if (
    [args.failureLimit, args.diagnosticDepth, args.candidateLimit, args.evidenceLimit].some(
      (value) => value !== undefined
    ) &&
    command !== 'why-not'
  ) {
    throw new Error('why-not diagnostic limits are available only for why-not');
  }
  if (
    args.direction !== undefined &&
    command !== 'topology' &&
    command !== 'audit-rules'
  ) {
    throw new Error('--direction is available only for topology or audit-rules');
  }
  if (args.queryText !== undefined && command !== 'diff') {
    throw new Error('--query is available only for diff');
  }
  if (
    [args.planLimit, args.repairSteps, args.searchStates].some(
      (value) => value !== undefined
    ) &&
    command !== 'repair'
  ) {
    throw new Error('repair search limits are available only for repair');
  }
  if (
    (args.searchLimit !== undefined || args.searchKinds.length > 0) &&
    command !== 'search' &&
    command !== 'semantic-search' &&
    command !== 'semantic-index'
  ) {
    throw new Error(
      'search limits and kinds are available only for search, semantic-search, or semantic-index'
    );
  }
  if (args.semanticAfter !== undefined && command !== 'semantic-index') {
    throw new Error('--after is available only for semantic-index');
  }
  if (
    (args.related ||
      args.relatedLimit !== undefined ||
      args.relatedKinds.length > 0) &&
    command !== 'recall' &&
    command !== 'recall-explain'
  ) {
    throw new Error(
      'related knowledge options are available only for recall or recall-explain'
    );
  }
  if (
    (args.browsePredicate !== undefined ||
      args.browseDepth !== undefined ||
      args.focusNumber) &&
    command !== 'browse'
  ) {
    throw new Error('browse options are available only for browse');
  }
  if (args.claimLimit !== undefined && command !== 'browse' && command !== 'connect') {
    throw new Error('--claim-limit is available only for browse or connect');
  }
  if (
    (args.pathDepth !== undefined ||
      args.pathLimit !== undefined ||
      args.fromNumber ||
      args.toNumber ||
      args.includeDerived) &&
    command !== 'connect'
  ) {
    throw new Error('path options are available only for connect');
  }
  if (args.includePassingEvidence && command !== 'test-knowledge') {
    throw new Error('--include-passing-evidence is available only for test-knowledge');
  }
  if (args.compareScan && command !== 'profile') {
    throw new Error('--compare-scan is available only for profile');
  }
  if (
    args.at !== undefined &&
    command !== 'supersede' &&
    command !== 'checkpoint' &&
    command !== 'propose-memory'
  ) {
    throw new Error('--at is available only for supersede, checkpoint, or propose-memory');
  }
  if (args.dryRun && command !== 'checkpoint') {
    throw new Error('--dry-run is available only for checkpoint');
  }
  if (command === 'supersede' && args.validTimeMode !== undefined) {
    throw new Error(
      '--valid-time-mode does not apply to supersede; it always preserves _until history'
    );
  }
  if (
    recordedSequence !== undefined &&
    !['health', 'recall', 'recall-explain', 'query', 'explain', 'profile', 'what-if', 'why-not', 'topology', 'audit-rules', 'search', 'semantic-search', 'semantic-index', 'browse', 'connect', 'bundle', 'test-knowledge', 'check', 'conflicts', 'list'].includes(command ?? '')
  ) {
    throw new Error(
      '--as-of-sequence is available for health, recall, recall-explain, query, explain, profile, what-if, why-not, topology, audit-rules, search, semantic-search, semantic-index, browse, connect, bundle, test-knowledge, check, conflicts, and list'
    );
  }
  const rawIntegritySetting = integrityEnforcementOption(
    args.integrityMode,
    args.integrityNamespaces,
    integrityEnforcementFromEnv(),
    writeCommand ? args.proofLimit : undefined,
    writeCommand ? args.maxViolations : undefined,
    writeCommand ? graphSelector : undefined
  );
  const integritySetting =
    rawIntegritySetting === undefined || rawIntegritySetting === false
      ? rawIntegritySetting
      : {
          ...rawIntegritySetting,
          ...(entityIdentity === undefined ? {} : { entityIdentity }),
        };
  const integrityEnforcement =
    integritySetting === false ? undefined : integritySetting;
  const knowledgeCheckEnforcement = knowledgeCheckEnforcementFromEnv();
  const llmAllowedNamespaces = llmNamespaceAllowlistFromEnv();
  const text = args.positional.join(' ');
  const namespaces = args.namespaces ?? (args.namespace ? [args.namespace] : undefined);

  switch (command) {
    case 'serve':
      {
        const semantic = await openSemanticLedgerIfSupported(
          join(defaultRoot(), 'semantic.sqlite')
        );
      await serveStdio({
        store,
        ...(semantic === undefined ? {} : { semanticLedger: semantic.ledger }),
        llm: lazyClientFromEnv(),
        llmAllowedNamespaces,
        validTimeMode: validTimeModeOption(args.validTimeMode),
        recallSchemaPredicateLimit: recallSchemaPredicateLimitOption(
          args.schemaPredicateLimit
        ),
        recallAnswerMode: recallAnswerModeOption(args.answerMode),
        integrityEnforcement: integritySetting,
        knowledgeCheckEnforcement,
        entityIdentity: entityIdentitySetting,
        trustMode: trustViewOption(args.trust),
        toolProfile: toolProfileOption(args.profile),
        ...(args.namespace === undefined ? {} : { defaultNamespace: args.namespace }),
      });
      return; // keep process alive; transport owns stdio
      }
    case 'session-brief': {
      if (args.managedBy !== undefined && args.managedBy !== MANAGED_HOOK_MARKER) {
        throw new Error('unrecognized auto-capture hook marker');
      }
      try {
        if (args.managedBy !== undefined) {
          // Drain the SessionStart hook payload so the pipe closes cleanly; the
          // brief itself is derived from the store, not from the payload.
          await readStdinBounded();
        }
        const brief = buildSessionBrief(store, args.namespace ?? 'default');
        if (brief !== '') console.log(brief);
      } catch (error) {
        if (args.managedBy === undefined) throw error;
        // A hook failure must never break session start; report and stay quiet.
        console.error(error instanceof Error ? error.message : String(error));
      }
      return;
    }
    case 'remember': {
      if (args.batch) {
        if (args.managedBy !== undefined && args.managedBy !== MANAGED_HOOK_MARKER) {
          throw new Error('unrecognized auto-capture hook marker');
        }
        const rawHookInput = await readStdinBounded();
        const result = await autoCaptureClaudeStop(
          {
            store,
            llm: lazyClientFromEnv(),
            llmAllowedNamespaces,
            integrityEnforcement: integritySetting,
            knowledgeCheckEnforcement,
            entityIdentity: entityIdentitySetting,
          },
          rawHookInput,
          {
            namespace: args.namespace,
            dailyCap: integerOption(
              args.dailyCap ?? process.env.REMBERO_AUTO_CAPTURE_DAILY_CAP,
              DEFAULT_AUTO_CAPTURE_DAILY_CAP,
              'auto-capture daily cap'
            ),
            tailBytes: integerOption(
              args.tailBytes ?? process.env.REMBERO_AUTO_CAPTURE_TAIL_BYTES,
              DEFAULT_TRANSCRIPT_TAIL_BYTES,
              'auto-capture tail bytes'
            ),
          }
        );
        if (args.json) console.log(stringifyBoundedResult(result, 'CLI result'));
        return;
      }
      const validTimeMode = validTimeModeOption(args.validTimeMode);
      const result = await rememberText(
        {
          store,
          llm: clientFromEnv(),
          llmAllowedNamespaces,
          entityIdentity: entityIdentitySetting,
          knowledgeCheckEnforcement,
        },
        text,
        args.namespace,
        {
          validTimeMode,
          integrityEnforcement: integritySetting,
          knowledgeCheckEnforcement,
          entityIdentity: entityIdentitySetting,
          trust: knowledgeTrustOption(args.trust),
        }
      );
      console.log(stringifyBoundedResult(result, 'CLI result'));
      return;
    }
    case 'propose-memory': {
      let checkSuite: string | undefined;
      if (args.checkSuitePath !== undefined) {
        const file = resolve(args.checkSuitePath);
        const stat = lstatSync(file);
        if (stat.isSymbolicLink() || !stat.isFile()) {
          throw new Error('refusing non-regular memory proposal check suite file');
        }
        if (stat.size > MAX_KNOWLEDGE_CHECK_SUITE_BYTES) {
          throw new Error(
            `memory proposal check suite exceeds ${MAX_KNOWLEDGE_CHECK_SUITE_BYTES} bytes`
          );
        }
        checkSuite = readFileSync(file, 'utf8');
      }
      const result = await proposeMemoryTool(
        {
          store,
          llm: clientFromEnv(),
          llmAllowedNamespaces,
          integrityEnforcement: integritySetting,
          knowledgeCheckEnforcement,
          entityIdentity: entityIdentitySetting,
        },
        {
          text,
          namespace: args.namespace,
          namespaces,
          validTimeMode: validTimeModeOption(args.validTimeMode),
          at: args.at,
          checkSuite,
          integrityEnforcement,
          entityIdentity,
        }
      );
      console.log(stringifyBoundedResult(result, 'CLI result'));
      return;
    }
    case 'apply-memory': {
      if (args.positional.length !== 1) {
        throw new Error('apply-memory requires exactly one proposal JSON file');
      }
      if (operationId === undefined) throw new Error('apply-memory requires --op-id');
      const file = resolve(args.positional[0]);
      const stat = lstatSync(file);
      if (stat.isSymbolicLink() || !stat.isFile()) {
        throw new Error('refusing non-regular memory proposal file');
      }
      if (stat.size > MAX_MEMORY_PROPOSAL_BYTES) {
        throw new Error(`memory proposal exceeds ${MAX_MEMORY_PROPOSAL_BYTES} bytes`);
      }
      const maxViolations = maxViolationsOption(args.maxViolations);
      const result = applyMemoryProposal(
        store,
        readFileSync(file, 'utf8'),
        {
          opId: operationId,
          ...(maxViolations === undefined ? {} : { maxViolations }),
          ...(knowledgeCheckEnforcement === undefined
            ? {}
            : { knowledgeCheckEnforcement }),
        }
      );
      console.log(stringifyBoundedResult(result, 'CLI result'));
      return;
    }
    case 'recall': {
      const relatedKnowledge = relatedKnowledgeOption(args);
      const result = await recallQuestion(
        {
          store,
          llm: clientFromEnv(),
          llmAllowedNamespaces,
          recallSchemaPredicateLimit: recallSchemaPredicateLimitOption(
            args.schemaPredicateLimit
          ),
          recallAnswerMode: recallAnswerModeOption(args.answerMode),
          entityIdentity: entityIdentitySetting,
          trustMode: trustViewOption(args.trust),
        },
        text,
        namespaces,
        {
          ...(recordedSequence === undefined ? {} : { recordedSequence }),
          ...(relatedKnowledge === undefined ? {} : { relatedKnowledge }),
        }
      );
      assertBoundedOutput(result.answer, 'CLI recall answer');
      console.log(result.answer);
      const recorded = result.recordedSnapshot === undefined
        ? ''
        : `, recorded: ${result.recordedSnapshot.sequence}/${result.recordedSnapshot.journalEntries}`;
      console.log(
        `  (status: ${result.status}, query: ${result.query ?? 'n/a'}, matches: ${result.bindings.length}, trust: ${result.trustMode ?? 'accepted'}${recorded})`
      );
      if (result.relatedKnowledge !== undefined) {
        console.log(relatedKnowledgeText(result.relatedKnowledge));
      }
      return;
    }
    case 'recall-explain': {
      const proofLimit = proofLimitOption(args.proofLimit);
      const relatedKnowledge = relatedKnowledgeOption(args);
      const result = await recallQuestion(
        {
          store,
          llm: clientFromEnv(),
          llmAllowedNamespaces,
          recallSchemaPredicateLimit: recallSchemaPredicateLimitOption(
            args.schemaPredicateLimit
          ),
          recallAnswerMode: recallAnswerModeOption(args.answerMode),
          entityIdentity: entityIdentitySetting,
          trustMode: trustViewOption(args.trust),
        },
        text,
        namespaces,
        {
          explain: true,
          ...(proofLimit === undefined ? {} : { proofLimit }),
          ...(graphSelector === undefined ? {} : { graphSelector }),
          ...(recordedSequence === undefined ? {} : { recordedSequence }),
          ...(relatedKnowledge === undefined ? {} : { relatedKnowledge }),
        }
      );
      console.log(stringifyBoundedResult(result, 'CLI result'));
      return;
    }
    case 'query': {
      const result = queryTool(
        {
          store,
          entityIdentity: entityIdentitySetting,
          trustMode: trustViewOption(args.trust),
        },
        {
          query: text,
          namespaces,
          ...(recordedSequence === undefined ? {} : { recordedSequence }),
        }
      );
      console.log(
        stringifyBoundedResult(
          recordedSequence === undefined && result.trustMode === undefined
            ? result.bindings
            : result,
          'CLI result'
        )
      );
      return;
    }
    case 'assert': {
      const result = knowledgeTrustOption(args.trust) === 'tentative'
        ? assertTentativeTool(
            { store, integrityEnforcement, knowledgeCheckEnforcement },
            { clauses: text, namespace: args.namespace, opId: operationId }
          )
        : assertFactsTool(
            { store, integrityEnforcement, knowledgeCheckEnforcement },
            { clauses: text, namespace: args.namespace, opId: operationId }
          );
      console.log(stringifyBoundedResult(result, 'CLI result'));
      return;
    }
    case 'accept':
    case 'reject': {
      const result = resolveTentativeTool(
        { store, integrityEnforcement, knowledgeCheckEnforcement },
        {
          clauses: text,
          action: command === 'accept' ? 'accept' : 'reject',
          namespace: args.namespace,
          opId: operationId,
        }
      );
      console.log(stringifyBoundedResult(result, 'CLI result'));
      return;
    }
    case 'explain': {
      const proofLimit = proofLimitOption(args.proofLimit);
      const result = explainQueryTool(
        {
          store,
          entityIdentity: entityIdentitySetting,
          trustMode: trustViewOption(args.trust),
        },
        {
          query: text,
          namespaces,
          ...(proofLimit === undefined ? {} : { proofLimit }),
          ...(graphSelector === undefined ? {} : { graphSelector }),
          ...(recordedSequence === undefined ? {} : { recordedSequence }),
        }
      );
      console.log(stringifyBoundedResult(result, 'CLI result'));
      return;
    }
    case 'profile': {
      const proofLimit = proofLimitOption(args.proofLimit);
      const result = profileKnowledgeTool(
        {
          store,
          entityIdentity: entityIdentitySetting,
          trustMode: trustViewOption(args.trust),
        },
        {
          query: text,
          namespaces,
          ...(proofLimit === undefined ? {} : { proofLimit }),
          ...(graphSelector === undefined ? {} : { graphSelector }),
          ...(recordedSequence === undefined ? {} : { recordedSequence }),
          ...(args.compareScan ? { compareFullScan: true } : {}),
        }
      );
      console.log(stringifyBoundedResult(result, 'CLI result'));
      return;
    }
    case 'what-if': {
      const proofLimit = proofLimitOption(args.proofLimit);
      const maxViolations = maxViolationsOption(args.maxViolations);
      let checkSuite: string | undefined;
      if (args.checkSuitePath !== undefined) {
        const file = resolve(args.checkSuitePath);
        const stat = lstatSync(file);
        if (stat.isSymbolicLink() || !stat.isFile()) {
          throw new Error('refusing non-regular counterfactual knowledge check suite file');
        }
        if (stat.size > MAX_KNOWLEDGE_CHECK_SUITE_BYTES) {
          throw new Error(
            `counterfactual knowledge check suite exceeds ${MAX_KNOWLEDGE_CHECK_SUITE_BYTES} bytes`
          );
        }
        checkSuite = readFileSync(file, 'utf8');
      }
      const result = whatIfTool(
        {
          store,
          entityIdentity: entityIdentitySetting,
          trustMode: trustViewOption(args.trust),
        },
        {
          query: text,
          assume: args.assumptions.join('\n'),
          without: args.without,
          assumeRules: args.assumedRules.join('\n'),
          withoutRules: args.withoutRules.join('\n'),
          ...(checkSuite === undefined ? {} : { checkSuite }),
          namespace: args.namespace,
          namespaces,
          ...(proofLimit === undefined ? {} : { proofLimit }),
          ...(maxViolations === undefined ? {} : { maxViolations }),
          ...(recordedSequence === undefined ? {} : { recordedSequence }),
        }
      );
      console.log(stringifyBoundedResult(result, 'CLI result'));
      return;
    }
    case 'apply-rule-change': {
      if (args.positional.length !== 1) {
        throw new Error('apply-rule-change requires exactly one proposal JSON file');
      }
      if (operationId === undefined) {
        throw new Error('apply-rule-change requires --op-id');
      }
      const file = resolve(args.positional[0]);
      const stat = lstatSync(file);
      if (stat.isSymbolicLink() || !stat.isFile()) {
        throw new Error('refusing non-regular rule change proposal file');
      }
      if (stat.size > MAX_RULE_CHANGE_PROPOSAL_BYTES) {
        throw new Error(
          `rule change proposal exceeds ${MAX_RULE_CHANGE_PROPOSAL_BYTES} bytes`
        );
      }
      const maxViolations = maxViolationsOption(args.maxViolations);
      const result = applyRuleChangeProposal(
        store,
        readFileSync(file, 'utf8'),
        {
          opId: operationId,
          ...(maxViolations === undefined ? {} : { maxViolations }),
          ...(knowledgeCheckEnforcement === undefined
            ? {}
            : { knowledgeCheckEnforcement }),
        }
      );
      console.log(stringifyBoundedResult(result, 'CLI result'));
      return;
    }
    case 'why-not': {
      const proofLimit = proofLimitOption(args.proofLimit);
      const result = whyNotTool(
        {
          store,
          entityIdentity: entityIdentitySetting,
          trustMode: trustViewOption(args.trust),
        },
        {
          query: text,
          namespaces,
          ...(proofLimit === undefined ? {} : { proofLimit }),
          ...(recordedSequence === undefined ? {} : { recordedSequence }),
          ...(args.failureLimit === undefined
            ? {}
            : {
                maxFailures: integerOption(
                  args.failureLimit,
                  0,
                  'why-not failure limit'
                ),
              }),
          ...(args.diagnosticDepth === undefined
            ? {}
            : {
                maxDiagnosticDepth: integerOption(
                  args.diagnosticDepth,
                  0,
                  'why-not diagnostic depth'
                ),
              }),
          ...(args.candidateLimit === undefined
            ? {}
            : {
                maxCandidatesPerFailure: integerOption(
                  args.candidateLimit,
                  0,
                  'why-not candidate limit'
                ),
              }),
          ...(args.evidenceLimit === undefined
            ? {}
            : {
                maxEvidenceFacts: integerOption(
                  args.evidenceLimit,
                  0,
                  'why-not evidence limit'
                ),
              }),
        }
      );
      console.log(stringifyBoundedResult(result, 'CLI result'));
      return;
    }
    case 'topology': {
      const result = topologyTool(
        {
          store,
          entityIdentity: entityIdentitySetting,
          trustMode: trustViewOption(args.trust),
        },
        {
          namespaces,
          ...(text.length === 0 ? {} : { focus: text }),
          ...(args.direction === undefined
            ? {}
            : { direction: topologyDirectionOption(args.direction) }),
          ...(recordedSequence === undefined ? {} : { recordedSequence }),
        }
      );
      console.log(stringifyBoundedResult(result, 'CLI result'));
      return;
    }
    case 'diff': {
      if (args.positional.length !== 2) {
        throw new Error('diff requires exactly <from-sequence> <to-sequence>');
      }
      const proofLimit = proofLimitOption(args.proofLimit);
      const maxViolations = maxViolationsOption(args.maxViolations);
      const result = recordedDiffTool(
        {
          store,
          entityIdentity: entityIdentitySetting,
          trustMode: trustViewOption(args.trust),
        },
        {
          fromSequence: integerOption(
            args.positional[0],
            0,
            'recorded diff from sequence'
          ),
          toSequence: integerOption(
            args.positional[1],
            0,
            'recorded diff to sequence'
          ),
          namespaces,
          query: args.queryText,
          ...(proofLimit === undefined ? {} : { proofLimit }),
          ...(maxViolations === undefined ? {} : { maxViolations }),
        }
      );
      console.log(stringifyBoundedResult(result, 'CLI result'));
      return;
    }
    case 'repair': {
      const proofLimit = proofLimitOption(args.proofLimit);
      const maxViolations = maxViolationsOption(args.maxViolations);
      const result = repairPlanTool(
        {
          store,
          entityIdentity: entityIdentitySetting,
          trustMode: trustViewOption(args.trust),
        },
        {
          query: text,
          namespace: args.namespace,
          namespaces,
          ...(proofLimit === undefined ? {} : { proofLimit }),
          ...(maxViolations === undefined ? {} : { maxViolations }),
          ...(args.planLimit === undefined
            ? {}
            : {
                maxPlans: integerOption(
                  args.planLimit,
                  0,
                  'repair plan limit'
                ),
              }),
          ...(args.repairSteps === undefined
            ? {}
            : {
                maxSteps: integerOption(
                  args.repairSteps,
                  0,
                  'repair step limit'
                ),
              }),
          ...(args.searchStates === undefined
            ? {}
            : {
                maxSearchStates: integerOption(
                  args.searchStates,
                  0,
                  'repair search state limit'
                ),
              }),
        }
      );
      console.log(stringifyBoundedResult(result, 'CLI result'));
      return;
    }
    case 'audit-rules': {
      const result = auditRulesTool(
        {
          store,
          entityIdentity: entityIdentitySetting,
          trustMode: trustViewOption(args.trust),
        },
        {
          namespaces,
          ...(text.length === 0 ? {} : { focus: text }),
          ...(args.direction === undefined
            ? {}
            : { direction: topologyDirectionOption(args.direction) }),
          ...(recordedSequence === undefined ? {} : { recordedSequence }),
        }
      );
      console.log(stringifyBoundedResult(result, 'CLI result'));
      if (result.warningCount > 0) process.exitCode = 2;
      return;
    }
    case 'health': {
      let checkSuite: string | undefined;
      if (args.checkSuitePath !== undefined) {
        const file = resolve(args.checkSuitePath);
        const stat = lstatSync(file);
        if (stat.isSymbolicLink() || !stat.isFile()) {
          throw new Error('refusing non-regular health knowledge check suite file');
        }
        if (stat.size > MAX_KNOWLEDGE_CHECK_SUITE_BYTES) {
          throw new Error(
            `health knowledge check suite exceeds ${MAX_KNOWLEDGE_CHECK_SUITE_BYTES} bytes`
          );
        }
        checkSuite = readFileSync(file, 'utf8');
      }
      const proofLimit = proofLimitOption(args.proofLimit);
      const maxViolations = maxViolationsOption(args.maxViolations);
      const result = knowledgeHealthTool(
        {
          store,
          entityIdentity: entityIdentitySetting,
          trustMode: trustViewOption(args.trust),
        },
        {
          namespaces,
          ...(recordedSequence === undefined ? {} : { recordedSequence }),
          ...(entityIdentity === undefined ? {} : { entityIdentity }),
          ...(args.trust === undefined
            ? {}
            : { trustMode: trustViewOption(args.trust) }),
          ...(checkSuite === undefined ? {} : { checkSuite }),
          ...(proofLimit === undefined ? {} : { proofLimit }),
          ...(maxViolations === undefined ? {} : { maxViolations }),
        }
      );
      console.log(stringifyBoundedResult(result, 'CLI result'));
      if (result.status === 'violations') process.exitCode = 3;
      else if (result.status === 'review') process.exitCode = 2;
      return;
    }
    case 'search': {
      const kinds = searchKindsOption(args.searchKinds);
      const result = searchKnowledgeTool(
        {
          store,
          entityIdentity: entityIdentitySetting,
          trustMode: trustViewOption(args.trust),
        },
        {
          text,
          namespaces,
          ...(args.searchLimit === undefined
            ? {}
            : {
                limit: integerOption(
                  args.searchLimit,
                  0,
                  'knowledge search limit'
                ),
              }),
          ...(kinds === undefined ? {} : { kinds }),
          ...(recordedSequence === undefined ? {} : { recordedSequence }),
        }
      );
      console.log(stringifyBoundedResult(result, 'CLI result'));
      return;
    }
    case 'semantic-search': {
      const kinds = searchKindsOption(args.searchKinds);
      const result = await semanticSearchKnowledgeTool(
        {
          store,
          embeddings: embeddingClientFromEnv(),
          semanticCache: new LayeredEmbeddingCache(
            new MemoryEmbeddingCache(),
            new FileEmbeddingCache(store.semanticEmbeddingCacheRoot())
          ),
          llmAllowedNamespaces,
          entityIdentity: entityIdentitySetting,
          trustMode: trustViewOption(args.trust),
        },
        {
          text,
          namespaces,
          ...(args.searchLimit === undefined
            ? {}
            : {
                limit: integerOption(
                  args.searchLimit,
                  0,
                  'semantic search limit'
                ),
              }),
          ...(kinds === undefined ? {} : { kinds }),
          ...(recordedSequence === undefined ? {} : { recordedSequence }),
        }
      );
      console.log(stringifyBoundedResult(result, 'CLI result'));
      return;
    }
    case 'semantic-index': {
      const kinds = searchKindsOption(args.searchKinds);
      const result = await prepareSemanticKnowledgeTool(
        {
          store,
          embeddings: embeddingClientFromEnv(),
          semanticCache: new LayeredEmbeddingCache(
            new MemoryEmbeddingCache(),
            new FileEmbeddingCache(store.semanticEmbeddingCacheRoot())
          ),
          llmAllowedNamespaces,
          entityIdentity: entityIdentitySetting,
          trustMode: trustViewOption(args.trust),
        },
        {
          namespaces,
          ...(args.searchLimit === undefined
            ? {}
            : {
                limit: integerOption(
                  args.searchLimit,
                  0,
                  'semantic prepare limit'
                ),
              }),
          ...(args.semanticAfter === undefined ? {} : { after: args.semanticAfter }),
          ...(kinds === undefined ? {} : { kinds }),
          ...(recordedSequence === undefined ? {} : { recordedSequence }),
        }
      );
      console.log(stringifyBoundedResult(result, 'CLI result'));
      return;
    }
    case 'browse': {
      let focus: string | number | undefined = text.length === 0 ? undefined : text;
      if (args.focusNumber) {
        if (text.length === 0) throw new Error('--focus-number requires an entity focus');
        const numeric = Number(text);
        if (!Number.isFinite(numeric)) {
          throw new Error('numeric browse focus must be finite');
        }
        focus = numeric;
      }
      const result = browseKnowledgeGraphTool(
        {
          store,
          entityIdentity: entityIdentitySetting,
          trustMode: trustViewOption(args.trust),
        },
        {
          focus,
          predicate: args.browsePredicate,
          namespaces,
          ...(args.browseDepth === undefined
            ? {}
            : {
                depth: integerOption(
                  args.browseDepth,
                  0,
                  'knowledge graph browse depth'
                ),
              }),
          ...(args.claimLimit === undefined
            ? {}
            : {
                maxClaims: integerOption(
                  args.claimLimit,
                  0,
                  'knowledge graph claim limit'
                ),
              }),
          ...(recordedSequence === undefined ? {} : { recordedSequence }),
        }
      );
      console.log(stringifyBoundedResult(result, 'CLI result'));
      return;
    }
    case 'connect': {
      if (args.positional.length !== 2) {
        throw new Error('connect requires exactly two entity arguments: <from> <to>');
      }
      let from: string | number = args.positional[0];
      let to: string | number = args.positional[1];
      if (args.fromNumber) {
        const numeric = Number(from);
        if (!Number.isFinite(numeric)) throw new Error('numeric path start must be finite');
        from = numeric;
      }
      if (args.toNumber) {
        const numeric = Number(to);
        if (!Number.isFinite(numeric)) throw new Error('numeric path end must be finite');
        to = numeric;
      }
      const result = connectKnowledgeGraphTool(
        {
          store,
          entityIdentity: entityIdentitySetting,
          trustMode: trustViewOption(args.trust),
        },
        {
          from,
          to,
          namespaces,
          ...(args.pathDepth === undefined
            ? {}
            : {
                maxDepth: integerOption(
                  args.pathDepth,
                  0,
                  'knowledge graph path depth'
                ),
              }),
          ...(args.pathLimit === undefined
            ? {}
            : {
                maxPaths: integerOption(
                  args.pathLimit,
                  0,
                  'knowledge graph path limit'
                ),
              }),
          ...(args.claimLimit === undefined
            ? {}
            : {
                maxClaims: integerOption(
                  args.claimLimit,
                  0,
                  'knowledge graph claim limit'
                ),
              }),
          ...(args.includeDerived ? { includeDerived: true } : {}),
          ...(recordedSequence === undefined ? {} : { recordedSequence }),
        }
      );
      console.log(stringifyBoundedResult(result, 'CLI result'));
      return;
    }
    case 'check': {
      const proofLimit = proofLimitOption(args.proofLimit);
      const maxViolations = maxViolationsOption(args.maxViolations);
      const result = checkIntegrityTool(
        {
          store,
          entityIdentity: entityIdentitySetting,
          trustMode: trustViewOption(args.trust),
        },
        {
          namespaces,
          ...(proofLimit === undefined ? {} : { proofLimit }),
          ...(maxViolations === undefined ? {} : { maxViolations }),
          ...(graphSelector === undefined ? {} : { graphSelector }),
          ...(recordedSequence === undefined ? {} : { recordedSequence }),
        }
      );
      console.log(stringifyBoundedResult(result, 'CLI result'));
      if (result.status === 'violations') process.exitCode = 2;
      return;
    }
    case 'conflicts': {
      const proofLimit = proofLimitOption(args.proofLimit);
      const maxViolations = maxViolationsOption(args.maxViolations);
      const result = conflictViewsTool(
        {
          store,
          entityIdentity: entityIdentitySetting,
          trustMode: trustViewOption(args.trust),
        },
        {
          ...(text.length === 0 ? {} : { focus: text }),
          namespaces,
          ...(proofLimit === undefined ? {} : { proofLimit }),
          ...(maxViolations === undefined ? {} : { maxViolations }),
          ...(graphSelector === undefined ? {} : { graphSelector }),
          ...(recordedSequence === undefined ? {} : { recordedSequence }),
        }
      );
      console.log(stringifyBoundedResult(result, 'CLI result'));
      if (result.matchingViolationCount > 0) process.exitCode = 2;
      return;
    }
    case 'forget': {
      const result = forgetTool(
        { store, integrityEnforcement, knowledgeCheckEnforcement },
        { pattern: text, namespace: args.namespace, opId: operationId }
      );
      console.log(`removed ${result.removed} clause(s)`);
      return;
    }
    case 'supersede': {
      const result = supersedeFactsTool(
        { store, integrityEnforcement, knowledgeCheckEnforcement },
        {
          patterns: args.patterns,
          replacements: text,
          namespace: args.namespace,
          at: args.at,
          opId: operationId,
        }
      );
      console.log(stringifyBoundedResult(result, 'CLI result'));
      return;
    }
    case 'history': {
      const result = historyTool(
        { store },
        {
          pattern: text,
          namespaces,
          limit: integerOption(args.limit, MAX_HISTORY_EVENTS, 'history limit'),
        }
      );
      if (args.json) {
        console.log(stringifyBoundedResult(result, 'CLI result'));
        return;
      }
      if (result.events.length === 0) {
        console.log(`no history for ${result.pattern}`);
        return;
      }
      for (const event of result.events) {
        const current = event.current ? ' [current]' : '';
        const archive = event.archivedAs ? ` -> ${event.archivedAs}` : '';
        const trust = event.trustAction ? ` [trust: ${event.trustAction}]` : '';
        console.log(
          `${event.sequence}.${event.position} ${event.ts} ${event.namespace} ${event.action}${current}${trust}: ${event.clause}${archive}`
        );
        if (event.sourceText) console.log(`  source: ${event.sourceText}`);
      }
      return;
    }
    case 'checkpoint': {
      const result = checkpointJournalTool(
        { store },
        {
          opId: operationId,
          at: args.at,
          dryRun: args.dryRun,
        }
      );
      console.log(stringifyBoundedResult(result, 'CLI result'));
      return;
    }
    case 'checkpoints': {
      console.log(
        stringifyBoundedResult(
          listCheckpointsTool({ store }),
          'CLI result'
        )
      );
      return;
    }
    case 'bundle': {
      const bundle = exportKnowledgeBundleTool(
        { store },
        {
          namespaces,
          ...(recordedSequence === undefined ? {} : { recordedSequence }),
        }
      );
      console.log(serializeKnowledgeBundle(bundle));
      return;
    }
    case 'verify-bundle': {
      if (args.positional.length !== 1) {
        throw new Error('verify-bundle requires exactly one bundle file');
      }
      const file = resolve(args.positional[0]);
      const stat = lstatSync(file);
      if (stat.isSymbolicLink() || !stat.isFile()) {
        throw new Error('refusing non-regular knowledge bundle file');
      }
      if (stat.size > MAX_KNOWLEDGE_BUNDLE_BYTES) {
        throw new Error(
          `knowledge bundle exceeds ${MAX_KNOWLEDGE_BUNDLE_BYTES} bytes`
        );
      }
      const result = verifyKnowledgeBundleTool({
        bundle: readFileSync(file, 'utf8'),
      });
      console.log(stringifyBoundedResult(result, 'CLI result'));
      return;
    }
    case 'test-knowledge': {
      if (args.positional.length !== 1) {
        throw new Error('test-knowledge requires exactly one suite file');
      }
      const file = resolve(args.positional[0]);
      const stat = lstatSync(file);
      if (stat.isSymbolicLink() || !stat.isFile()) {
        throw new Error('refusing non-regular knowledge check suite file');
      }
      if (stat.size > MAX_KNOWLEDGE_CHECK_SUITE_BYTES) {
        throw new Error(
          `knowledge check suite exceeds ${MAX_KNOWLEDGE_CHECK_SUITE_BYTES} bytes`
        );
      }
      const proofLimit = proofLimitOption(args.proofLimit);
      const result = runKnowledgeChecksTool(
        {
          store,
          entityIdentity: entityIdentitySetting,
          trustMode: trustViewOption(args.trust),
        },
        {
          suite: readFileSync(file, 'utf8'),
          namespaces,
          ...(proofLimit === undefined ? {} : { proofLimit }),
          ...(recordedSequence === undefined ? {} : { recordedSequence }),
          ...(args.includePassingEvidence
            ? { includePassingEvidence: true }
            : {}),
        }
      );
      console.log(stringifyBoundedResult(result, 'CLI result'));
      if (result.status === 'failed') process.exitCode = 2;
      return;
    }
    case 'backup': {
      const [file] = args.positional;
      if (!file) {
        console.error('usage: remembero backup <file.json>');
        process.exitCode = 1;
        return;
      }
      const summary = backupKnowledge(store, file);
      console.log(
        `backed up ${summary.clauseCount} clauses across ${summary.namespaceCount} namespaces to ${file} (sha256 ${summary.sha256.slice(0, 12)})`
      );
      return;
    }
    case 'restore': {
      const [file] = args.positional;
      if (!file) {
        console.error('usage: remembero restore <file.json>');
        process.exitCode = 1;
        return;
      }
      const result = restoreKnowledge(store, file);
      console.log(
        `restored ${result.clausesAdded} clauses into ${result.namespaces.length} namespaces from ${file} (sha256 ${result.sha256.slice(0, 12)})`
      );
      return;
    }
    case 'export': {
      for (const ns of store.listNamespaces()) {
        console.log(`% namespace: ${ns}`);
        for (const clause of store.load(ns)) console.log(serializeClause(clause));
        console.log('');
      }
      return;
    }
    case 'import': {
      const [ns, file] = args.positional;
      if (!ns || !file) {
        console.error('usage: remembero import <namespace> <file.dl>');
        process.exitCode = 1;
        return;
      }
      const size = statSync(file).size;
      if (size > MAX_INPUT_BYTES) {
        throw new Error(`import file exceeds ${MAX_INPUT_BYTES} bytes`);
      }
      const result = store.importClauses(
        ns,
        readFileSync(file, 'utf8'),
        {
          ...(operationId === undefined ? {} : { opId: operationId }),
          ...(integrityEnforcement === undefined ? {} : { integrity: integrityEnforcement }),
          ...(knowledgeCheckEnforcement === undefined
            ? {}
            : { checks: knowledgeCheckEnforcement }),
        }
      );
      console.log(`imported ${result.added.length} clause(s), ${result.duplicates} duplicate(s) skipped`);
      return;
    }
    case 'sqlite-build':
      console.log(buildSqliteExtension());
      return;
    case 'sqlite-sql':
    case 'sqlite-query':
    case 'sqlite-explain':
    case 'sqlite-plan': {
      const [databasePath, ...ruleParts] = args.positional;
      const rule = ruleParts.join(' ');
      if (!databasePath || !rule) {
        console.error(`usage: remembero ${command} <database> <datalog-program>`);
        process.exitCode = 1;
        return;
      }
      const database = await openDatalogDatabase(databasePath, {
        extensionPath: args.extensionPath,
      });
      try {
        const result = command === 'sqlite-sql'
          ? database.datalogSql(rule)
          : stringifyBoundedResult(
              command === 'sqlite-plan'
                ? database.datalogPlan(rule)
                : command === 'sqlite-explain'
                  ? database.datalogExplain(rule)
                  : database.datalogQuery(rule),
              'CLI result'
            );
        console.log(result);
      } finally {
        database.close();
      }
      return;
    }
    case 'list': {
      const result = listMemoriesTool(
        { store, trustMode: trustViewOption(args.trust) },
        {
          namespaces,
          ...(recordedSequence === undefined ? {} : { recordedSequence }),
        }
      );
      console.log(stringifyBoundedResult(result, 'CLI result'));
      return;
    }
    case 'claims': {
      const result = reviewTentativeTool(
        { store },
        { namespaces }
      );
      console.log(stringifyBoundedResult(result, 'CLI result'));
      return;
    }
    case 'review': {
      const days = integerOption(args.days, 7, 'review days');
      const review = store.reviewAutoCaptures({ days, namespace: args.namespace });
      const selectedNumbers = reviewSelections(args.forget, review.facts.length);
      if (selectedNumbers.length > 0) {
        const selectedFacts = selectedNumbers.map((number) => review.facts[number - 1]);
        const result = store.pruneAutoCaptureFacts(selectedFacts, {
          ...(integrityEnforcement === undefined
            ? {}
            : { integrity: integrityEnforcement }),
          ...(knowledgeCheckEnforcement === undefined
            ? {}
            : { checks: knowledgeCheckEnforcement }),
        });
        if (args.json) {
          console.log(
            stringifyBoundedResult(
              { ...result, selected: selectedNumbers, facts: selectedFacts },
              'CLI result'
            )
          );
        } else {
          console.log(`removed ${result.removed} auto-captured fact(s)`);
        }
        return;
      }
      if (args.json) {
        console.log(stringifyBoundedResult(review, 'CLI result'));
        return;
      }
      for (const capture of review.captures) {
        const detail = capture.reason ? ` (${capture.reason})` : '';
        console.log(
          `${capture.ts}  ${capture.namespace}  ${capture.status}${detail}  ${capture.captureId}`
        );
      }
      if (review.facts.length === 0) {
        console.log(`no auto-captured facts in the last ${days} day(s)`);
        return;
      }
      console.log('');
      review.facts.forEach((fact, index) => {
        console.log(
          `${index + 1}. ${fact.current ? '[current]' : '[removed]'} ${fact.namespace}: ${fact.clause}`
        );
      });
      console.log('\nPrune with: remembero review --forget <number,...>');
      return;
    }
    case 'init': {
      const result = runInit({
        settingsPath: resolve(args.settingsPath ?? defaultClaudeSettingsPath()),
        nodePath: process.execPath,
        cliPath: resolve(process.argv[1]),
        namespace: args.namespace ?? 'personal',
        dailyCap: integerOption(
          args.dailyCap ?? process.env.REMBERO_AUTO_CAPTURE_DAILY_CAP,
          DEFAULT_AUTO_CAPTURE_DAILY_CAP,
          'auto-capture daily cap'
        ),
        tailBytes: integerOption(
          args.tailBytes ?? process.env.REMBERO_AUTO_CAPTURE_TAIL_BYTES,
          DEFAULT_TRANSCRIPT_TAIL_BYTES,
          'auto-capture tail bytes'
        ),
      });
      console.log(
        `hooks: ${result.hooks.changed ? 'installed' : 'already current'} (${result.hooks.settingsPath})`
      );
      console.log(`mcp registration: ${result.registration.detail}`);
      if (!result.registration.ok) {
        console.log(`  ${result.registration.command.join(' ')}`);
      }
      console.log('\nAdd this to your CLAUDE.md (or system prompt):\n');
      console.log(result.claudeMdSnippet);
      if ((process.env.LLM_API_KEY ?? '') === '') {
        console.log(
          "\nNote: LLM_API_KEY is not set; natural-language remember/recall will be unavailable until it is configured (the raw query tools work without it)."
        );
      }
      return;
    }
    case 'init-hooks':
    case 'remove-hooks': {
      const settingsPath = resolve(args.settingsPath ?? defaultClaudeSettingsPath());
      const remove = command === 'remove-hooks' || args.remove;
      const result = remove
        ? removeClaudeHook({ settingsPath })
        : installClaudeHook({
            settingsPath,
            nodePath: process.execPath,
            cliPath: resolve(process.argv[1]),
            namespace: args.namespace ?? 'default',
            dailyCap: integerOption(
              args.dailyCap ?? process.env.REMBERO_AUTO_CAPTURE_DAILY_CAP,
              DEFAULT_AUTO_CAPTURE_DAILY_CAP,
              'auto-capture daily cap'
            ),
            tailBytes: integerOption(
              args.tailBytes ?? process.env.REMBERO_AUTO_CAPTURE_TAIL_BYTES,
              DEFAULT_TRANSCRIPT_TAIL_BYTES,
              'auto-capture tail bytes'
            ),
          });
      console.log(
        `${remove ? 'removed' : 'installed'} Remembero Claude hook${result.changed ? '' : ' (already current)'}: ${result.settingsPath}`
      );
      return;
    }
    default:
      console.error(USAGE);
      process.exitCode = command ? 1 : 0;
  }
}

main().catch((e: unknown) => {
  if (e instanceof TrustMetadataError) {
    console.error(stringifyBoundedResult(e.toJSON(), 'CLI trust metadata error'));
    process.exitCode = 6;
    return;
  }
  if (e instanceof IncompleteHistoryError) {
    console.error(stringifyBoundedResult(e.toJSON(), 'CLI recorded history error'));
    process.exitCode = 5;
    return;
  }
  if (e instanceof OperationConflictError) {
    console.error(stringifyBoundedResult(e.toJSON(), 'CLI operation conflict'));
    process.exitCode = 4;
    return;
  }
  if (e instanceof RuleChangeStaleError) {
    console.error(stringifyBoundedResult(e.toJSON(), 'CLI stale rule proposal'));
    process.exitCode = 7;
    return;
  }
  if (e instanceof MemoryChangeStaleError) {
    console.error(stringifyBoundedResult(e.toJSON(), 'CLI stale memory proposal'));
    process.exitCode = 7;
    return;
  }
  if (e instanceof RuleChangeCheckError) {
    console.error(stringifyBoundedResult(e.toJSON(), 'CLI rule change check failure'));
    process.exitCode = 2;
    return;
  }
  if (e instanceof MemoryChangeCheckError) {
    console.error(stringifyBoundedResult(e.toJSON(), 'CLI memory change check failure'));
    process.exitCode = 2;
    return;
  }
  if (e instanceof KnowledgeCheckEnforcementError) {
    console.error(
      stringifyBoundedResult(e.toJSON(), 'CLI knowledge check enforcement rejection')
    );
    process.exitCode = 8;
    return;
  }
  if (e instanceof IntegrityViolationError) {
    try {
      console.error(stringifyBoundedResult(e.toJSON(), 'CLI integrity rejection'));
    } catch {
      console.error(
        JSON.stringify({
          error: 'integrity_rejection_output_exceeded',
          message: 'write was rejected, but complete evidence exceeds the CLI output bound',
          mode: e.mode,
          baselineViolationCount: e.baselineViolationCount,
          blockingViolationCount: e.blockingViolations.length,
          introducedViolationCount: e.introducedViolations.length,
        })
      );
    }
    process.exitCode = 3;
    return;
  }
  console.error(e instanceof Error ? e.message : String(e));
  process.exitCode = 1;
});
