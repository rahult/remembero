import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

const projectRoot = resolve(import.meta.dirname, '..');
const directory = mkdtempSync(join(tmpdir(), 'rembero-package-smoke-'));

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: options.cwd ?? projectRoot,
    encoding: 'utf8',
    input: options.input,
    env: options.env,
  });
  if (result.status !== 0) {
    throw new Error(
      `${command} ${args.join(' ')} failed\n${result.stdout ?? ''}${result.stderr ?? ''}`
    );
  }
  return result.stdout.trim();
}

try {
  run('npm', ['pack', '--ignore-scripts', '--pack-destination', directory]);
  const archive = readdirSync(directory).find((entry) => entry.endsWith('.tgz'));
  if (!archive) throw new Error('npm pack did not produce an archive');

  run(
    'npm',
    ['install', '--ignore-scripts', '--no-audit', '--no-fund', join(directory, archive)],
    { cwd: directory }
  );
  const installedPackage = join(directory, 'node_modules', 'remembero');
  const installedPackageJson = JSON.parse(
    readFileSync(join(installedPackage, 'package.json'), 'utf8')
  );
  if (
    installedPackageJson.name !== 'remembero' ||
    installedPackageJson.bin?.remembero !== 'dist/cli.js' ||
    installedPackageJson.bin?.['remembero-web'] !== 'dist/web/server.js' ||
    installedPackageJson.bin?.rembero !== 'dist/cli.js'
  ) {
    throw new Error('packed package name or CLI aliases are incorrect');
  }
  if (
    !readFileSync(join(installedPackage, 'dist', 'web-client', 'index.html'), 'utf8').includes(
      '<div id="root"></div>'
    ) ||
    !readFileSync(join(installedPackage, 'dist', 'web', 'server.js'), 'utf8').includes(
      'Remembero web console'
    )
  ) {
    throw new Error('packed web console assets are incomplete');
  }
  run(
    process.execPath,
    [
      '--input-type=module',
      '--eval',
      "import { IncompleteHistoryError, IntegrityViolationError, MemoryStore, OperationConflictError, RememberoWebService, analyzeKnowledgeTopology, applyMemoryProposal, applyRuleChangeProposal, assertTentativeFacts, auditKnowledgeRules, browseKnowledgeGraph, canonicalizeKnowledge, checkIntegrity, connectKnowledgeGraph, createKnowledgeBundle, diffRecordedKnowledge, evaluate, evaluateQuerySpec, explainKnowledge, explainWhyNot, inspectConflicts, inspectKnowledgeHealth, isAggregateRule, materialize, parseProgram, parseQuery, parseQuerySpec, planKnowledgeRepair, profileKnowledge, proposeRememberText, recallQuestion, resolveTentativeFacts, retrieveQuestion, reviewTentativeClaims, runKnowledgeChecks, searchKnowledge, selectExplanationGraph, selectRecallSchema, serializeKnowledgeBundle, simulateKnowledge, sqliteDatalogExecutionMode, verifyKnowledgeBundle } from 'remembero'; " +
        "const rows = evaluateQuerySpec(parseProgram('item(a). item(b).'), parseQuerySpec('count(*) as Count where item(Item)')); " +
        "if (rows[0]?.Count?.value !== 2) throw new Error('public aggregate API failed'); " +
        "const projectedRows = evaluateQuerySpec(parseProgram('edge(a, x1). edge(a, x2). edge(x1, z). edge(x2, z).'), parseQuerySpec('select End where edge(a, Mid), edge(Mid, End)')); " +
        "if (projectedRows.length !== 1 || projectedRows[0]?.End?.value !== 'z' || 'Mid' in projectedRows[0]) throw new Error('public relational projection API failed'); " +
        "if (materialize(parseProgram('base(a). derived(X) :- base(X).')).filter((fact) => fact.derived).length !== 1) throw new Error('public proof-free materialization API failed'); " +
        "const arithmetic = evaluateQuerySpec(parseProgram('score(a, 20). score(b, 14).'), parseQuerySpec('score(X, S), S > 10 + 5')); " +
        "if (arithmetic.length !== 1 || arithmetic[0]?.X?.value !== 'a') throw new Error('public arithmetic API failed'); " +
        "const indexedProgram = parseProgram([...Array.from({ length: 100 }, (_, i) => `related(person_${i}, topic_${i % 7}).`), 'selected(person_99).', 'relevant(X, Y) :- selected(X), related(X, Y).'].join('\\n')); " +
        "const indexedMetrics = { relationLookups: 0, indexedRelationLookups: 0, indexFactsProcessed: 0, candidateFactsVisited: 0 }; " +
        "const indexedRows = evaluate(indexedProgram, parseQuery('relevant(X, Y)'), { metrics: indexedMetrics }); " +
        "const scannedRows = evaluate(indexedProgram, parseQuery('relevant(X, Y)'), { relationIndex: 'off' }); " +
        "if (JSON.stringify(indexedRows) !== JSON.stringify(scannedRows) || indexedMetrics.indexedRelationLookups < 1 || indexedMetrics.indexFactsProcessed !== 100) throw new Error('public relation index API failed'); " +
        "const noise = Array.from({ length: 100 }, (_, i) => `noise_${i}(value_${i}).`).join('\\n'); " +
        "const schema = selectRecallSchema(parseProgram(`${noise}\\nworks_at(mira, acme).`), 'Who employs Mira?', { predicateLimit: 4 }); " +
        "if (!schema.pruned || !schema.selectedPredicates.includes('works_at/2') || schema.summaryBytes > 24576) throw new Error('public recall schema API failed'); " +
        "const kinshipSchema = selectRecallSchema(parseProgram(`${noise}\\nparent(alice, bob). parent(bob, carol). grandparent(X, Y) :- parent(X, Z), parent(Z, Y).`), \"Who is Alice's grandchild?\", { predicateLimit: 4 }); " +
        "if (kinshipSchema.selectedPredicates[0] !== 'grandparent/2' || !kinshipSchema.summary.includes('grandparent(X, Y) :- parent(X, Z), parent(Z, Y).')) throw new Error('public inverse kinship recall ranking failed'); " +
        "const sourceRankStore = new MemoryStore('./source-rank-memory'); sourceRankStore.assert('default', 'atlas_owner(atlas, rahul).'); sourceRankStore.assert('default', 'fact_z(atlas, rust).', { sourceText: 'What technology stack does Atlas use?' }); " +
        "const sourceRanked = selectRecallSchema(sourceRankStore.clausesFor(['default']), 'What technology stack does Atlas use?', { predicateLimit: 1, sourceIndex: sourceRankStore.sourcesFor(['default']) }); " +
        "if (sourceRanked.selectedPredicates[0] !== 'fact_z/2' || sourceRanked.sourceMatchedPredicates[0] !== 'fact_z/2' || sourceRanked.summary.includes('technology stack does Atlas use')) throw new Error('public provenance-aware recall ranking failed'); " +
        "const integrity = checkIntegrity(parseProgram('active(mira). suspended(mira). :- active(X), suspended(X).')); " +
        "if (integrity.status !== 'violations' || integrity.violationCount !== 1) throw new Error('public integrity API failed'); " +
        "const conflicts = inspectConflicts(parseProgram('active(mira). suspended(mira). :- active(Person), suspended(Person).'), new Map(), { focus: 'mira' }); " +
        "if (conflicts.clusterCount !== 1 || conflicts.clusters[0]?.focus !== 'mira' || !conflicts.clusters[0]?.graph.nodes.some((node) => node.kind === 'conflict')) throw new Error('public conflict view API failed'); " +
        "const whyNot = explainWhyNot(parseProgram('employee(bob). eligible(X) :- employee(X), badge(X).'), 'eligible(bob)'); " +
        "if (whyNot.status !== 'blocked' || whyNot.failures[0]?.rules[0]?.failures[0]?.reason !== 'missing_fact') throw new Error('public why-not API failed'); " +
        "const topology = analyzeKnowledgeTopology(parseProgram('employee(alice). eligible(X) :- employee(X), badge(X).')); " +
        "if (topology.ruleCount !== 1 || topology.openInputs[0] !== 'badge/1' || !topology.graph.edges.some((edge) => edge.kind === 'defines')) throw new Error('public topology API failed'); " +
        "const ruleAudit = auditKnowledgeRules(parseProgram('employee(alice). eligible(X) :- employee(X), badge(X).')); " +
        "if (ruleAudit.status !== 'advisory' || !ruleAudit.findings.some((finding) => finding.code === 'open_positive_input')) throw new Error('public rule audit API failed'); " +
        "const healthStore = new MemoryStore('./health-memory'); healthStore.assert('default', 'base(a). derived(X) :- base(X).', { opId: 'health-source' }); const health = inspectKnowledgeHealth(healthStore); " +
        "if (health.status !== 'healthy' || health.provenance.sourceCoveragePercent !== 100 || health.rules.topology.ruleCount !== 1) throw new Error('public knowledge health API failed'); " +
        "const guardedWriteStore = new MemoryStore('./guarded-write-memory'); const writeGuard = { mode: 'strict', namespaces: ['default'], suite: { version: 1, checks: [{ name: 'forbidden absent', query: 'forbidden(a)', expect: { kind: 'empty' } }] } }; guardedWriteStore.assert('default', 'safe(a).', { checks: writeGuard }); let guardRejected = false; try { guardedWriteStore.assert('default', 'forbidden(a).', { checks: writeGuard }); } catch (error) { guardRejected = error?.code === 'knowledge_check_enforcement'; } " +
        "if (!guardRejected || guardedWriteStore.load('default').length !== 1) throw new Error('public knowledge check write enforcement failed'); " +
        "const repairStore = new MemoryStore('./repair-memory'); repairStore.assert('default', 'employee(bob). eligible(X) :- employee(X), badge(X).'); " +
        "const repair = planKnowledgeRepair(repairStore, 'eligible(bob)'); " +
        "if (repair.plans[0]?.assume[0] !== 'badge(bob).' || repairStore.clausesFor(['default']).length !== 2) throw new Error('public repair planning API failed'); " +
        "const localSearch = searchKnowledge(repairStore.clausesFor(['default']), 'eligible', repairStore.sourcesFor(['default']), { kinds: ['rule'] }); " +
        "if (localSearch.results[0]?.clause !== 'eligible(X) :- employee(X), badge(X).' || !localSearch.graph.edges.some((edge) => edge.kind === 'defines')) throw new Error('public local search API failed'); " +
        "const webStore = new MemoryStore('./web-memory'); const webService = new RememberoWebService({ store: webStore, llmConfigured: false }); webService.seedDemo(); const webBootstrap = webService.bootstrap(); const webAnswer = await webService.ask({ question: 'Who is collaborating on Atlas?', presetId: 'collaborators' }); " +
        "if (webBootstrap.counts.facts !== 12 || webBootstrap.counts.rules !== 3 || webBootstrap.health.status !== 'healthy' || webAnswer.answer !== 'Maya is collaborating on Atlas.' || webAnswer.evidence.rules.length !== 1) throw new Error('public web console service API failed'); " +
        "const browsed = browseKnowledgeGraph(repairStore.clausesFor(['default']), repairStore.sourcesFor(['default']), { focus: 'bob' }); " +
        "if (browsed.selection.selectedClaims !== 1 || !browsed.graph.nodes.some((node) => node.kind === 'claim' && node.predicate === 'employee') || browsed.graph.nodes.some((node) => node.kind === 'claim' && node.predicate === 'eligible')) throw new Error('public explicit graph browse API failed'); " +
        "const connected = connectKnowledgeGraph(parseProgram('works_at(mira, acme). works_at(rahul, acme). colleague(X, Y) :- works_at(X, C), works_at(Y, C), X != Y.'), new Map(), 'mira', 'rahul', { maxDepth: 2, includeDerived: true }); " +
        "if (connected.status !== 'connected' || connected.shortestHops !== 1 || connected.paths[0]?.segments[0]?.predicate !== 'colleague' || connected.claimProofs?.[0]?.proof.rule !== 1) throw new Error('public proof-carrying graph path API failed'); " +
        "const checkSuite = { version: 1, checks: [{ name: 'employee row', query: 'employee(X)', expect: { kind: 'rows', order: 'exact', rows: [{ X: 'bob' }] } }, { name: 'eligible absent', query: 'eligible(bob)', expect: { kind: 'empty' } }] }; " +
        "const checkResult = runKnowledgeChecks(repairStore.clausesFor(['default']), repairStore.sourcesFor(['default']), checkSuite); " +
        "if (checkResult.status !== 'passed' || checkResult.passedCount !== 2) throw new Error('public knowledge check suite API failed'); " +
        "const coverageStore = new MemoryStore('./coverage-memory'); coverageStore.assert('default', 'base(a). derived(X) :- base(X).'); " +
        "const coverageResult = runKnowledgeChecks(coverageStore.clausesFor(['default']), coverageStore.sourcesFor(['default']), { version: 1, coverage: { minimumPercent: 100 }, checks: [{ name: 'derived proof', query: 'derived(a)', expect: { kind: 'nonempty' } }] }); " +
        "if (!coverageResult.coveragePassed || coverageResult.coverage.percent !== 100 || coverageResult.coverage.rules[0]?.checkNames[0] !== 'derived proof') throw new Error('public semantic rule coverage failed'); " +
        "const profile = profileKnowledge(parseProgram('base(a). derived(X) :- base(X).'), 'derived(a)', new Map(), { compareFullScan: true }); " +
        "if (!profile.equivalent || profile.explanation.rows.length !== 1 || profile.fullScan === undefined || profile.workReduction === undefined) throw new Error('public deterministic query profile failed'); " +
        "const aggregateRules = parseProgram('member(red, alice). member(red, bob). team_size(Team, Count) :- count(*) as Count where member(Team, Person).'); " +
        "const aggregateRuleRows = evaluate(aggregateRules, parseQuery('team_size(Team, Count)')); " +
        "if (!isAggregateRule(aggregateRules[2]) || aggregateRuleRows[0]?.Count?.value !== 2) throw new Error('public aggregate rule API failed'); " +
        "const reviewStore = new MemoryStore('./review-memory'); reviewStore.assert('default', 'uses_language(atlas, rust). project_owner(atlas, rahul).'); " +
        "const reviewLlm = { responses: ['?- uses_language(atlas, Value).', '?- project_owner(atlas, Owner).'], async complete() { const value = this.responses.shift(); if (value === undefined) throw new Error('review responses exhausted'); return value; } }; " +
        "const review = await retrieveQuestion({ store: reviewStore, llm: reviewLlm }, 'Who owns Atlas?'); " +
        "if (review.query !== 'project_owner(atlas, Owner)' || review.bindings[0]?.Owner !== 'rahul' || review.queryReviews?.[0]?.outcome !== 'corrected') throw new Error('public recall disambiguation API failed'); " +
        "const negativeStore = new MemoryStore('./negative-memory'); negativeStore.assert('default', 'works_at(maya, acme).'); " +
        "const negativeLlm = { calls: 0, responses: ['?- works_at(zoe, Company).', '?- works_at(zoe, Company).'], async complete() { this.calls += 1; const value = this.responses.shift(); if (value === undefined) throw new Error('negative recall used an unexpected phrasing call'); return value; } }; " +
        "const negative = await recallQuestion({ store: negativeStore, llm: negativeLlm }, 'Where does Zoe work?', ['default'], { relatedKnowledge: { limit: 1, kinds: ['fact'] } }); " +
        "if (negative.status !== 'no_match' || negativeLlm.calls !== 2 || negative.answer !== negative.whyNot?.summary || !negative.answer.includes('Required fact works_at(zoe, Company) is missing') || negative.relatedKnowledge?.results[0]?.clause !== 'works_at(maya, acme).') throw new Error('public grounded negative recall with related knowledge failed'); " +
        "const deterministicLlm = { calls: 0, responses: ['?- works_at(maya, Company).'], async complete() { this.calls += 1; const value = this.responses.shift(); if (value === undefined) throw new Error('deterministic recall used an unexpected phrasing call'); return value; } }; " +
        "const deterministic = await recallQuestion({ store: negativeStore, llm: deterministicLlm }, 'Where does Maya work?', ['default'], { answerMode: 'deterministic' }); " +
        "if (deterministicLlm.calls !== 1 || deterministic.answerMode !== 'deterministic' || deterministic.answer !== 'Result for works_at(maya, Company): Company = acme.') throw new Error('public deterministic answer mode failed'); " +
        "const evidenceLlm = { calls: 0, responses: ['?- works_at(maya, Company).'], async complete() { this.calls += 1; return this.responses.shift(); } }; const evidenceAnswer = await recallQuestion({ store: negativeStore, llm: evidenceLlm }, 'Where does Maya work?', ['default'], { answerMode: 'evidence' }); " +
        "if (evidenceLlm.calls !== 1 || evidenceAnswer.answerMode !== 'evidence' || !evidenceAnswer.answer.includes('Sources: default/') || evidenceAnswer.explanation?.rows.length !== 1) throw new Error('public evidence answer mode failed'); " +
        "const trustStore = new MemoryStore('./trust-memory'); assertTentativeFacts(trustStore, 'default', 'status(mira, active).', { opId: 'tentative' }); " +
        "if (explainKnowledge(trustStore.clausesFor(['default']), 'status(mira, State)', trustStore.sourcesFor(['default'])).rows.length !== 0) throw new Error('tentative default isolation failed'); " +
        "const tentativeExplain = explainKnowledge(trustStore.clausesFor(['default']), 'status(mira, State)', trustStore.sourcesFor(['default']), { trustMode: 'include_tentative' }); " +
        "if (tentativeExplain.rows[0]?.proofs[0]?.trust !== 'tentative' || reviewTentativeClaims(trustStore).length !== 1) throw new Error('public tentative review API failed'); " +
        "resolveTentativeFacts(trustStore, 'default', 'status(mira, active).', 'accept', { opId: 'accepted' }); " +
        "if (explainKnowledge(trustStore.clausesFor(['default']), 'status(mira, State)', trustStore.sourcesFor(['default'])).rows[0]?.bindings?.State !== 'active') throw new Error('public tentative promotion API failed'); " +
        "const recordedDiff = diffRecordedKnowledge(trustStore, 1, 2, { query: 'status(mira, State)' }); " +
        "if (recordedDiff.clauses.added[0]?.clause !== 'status(mira, active).' || recordedDiff.queryImpact?.added[0]?.bindings?.State !== 'active') throw new Error('public recorded diff API failed'); " +
        "const simulated = simulateKnowledge(trustStore, 'status(mira, State)', { assume: 'status(mira, paused).' }); " +
        "if (!simulated.changed || simulated.resultDelta.added[0]?.bindings?.State !== 'paused' || trustStore.clausesFor(['default']).length !== 1) throw new Error('public counterfactual API failed'); " +
        "const simulatedRule = simulateKnowledge(trustStore, 'active(Person)', { assumeRules: 'active(Person) :- status(Person, active).', checkSuite: { version: 1, coverage: { minimumPercent: 100 }, checks: [{ name: 'active Mira', query: 'active(mira)', expect: { kind: 'nonempty' } }] } }); " +
        "if (simulatedRule.candidate.rows[0]?.bindings?.Person !== 'mira' || simulatedRule.ruleAuditDelta?.candidate.topology.ruleCount !== 1 || simulatedRule.checkDelta?.candidate.status !== 'passed') throw new Error('public rule counterfactual API failed'); " +
        "const ruleApplyStore = new MemoryStore('./rule-apply-memory'); ruleApplyStore.assert('default', 'base(a).', { opId: 'rule-base' }); const rulePlan = simulateKnowledge(ruleApplyStore, 'derived(X)', { assumeRules: 'derived(X) :- base(X).' }).ruleProposal; const ruleApplied = applyRuleChangeProposal(ruleApplyStore, rulePlan, { opId: 'reviewed-rule' }); " +
        "if (ruleApplied.audit.topology.ruleCount !== 1 || evaluate(ruleApplyStore.load('default'), parseQuery('derived(X)'))[0]?.X?.value !== 'a') throw new Error('public reviewed rule application API failed'); " +
        "const memoryProposalStore = new MemoryStore('./memory-proposal'); memoryProposalStore.assert('default', 'works_at(mira, acme).'); const memoryProposalLlm = { async complete() { return 'retract works_at(mira, _).\\nworks_at(mira, initech).'; } }; const memoryProposal = await proposeRememberText({ store: memoryProposalStore, llm: memoryProposalLlm }, 'Mira now works at Initech.', 'default', { checkSuite: { version: 1, checks: [{ name: 'new employer', query: 'works_at(mira, initech)', expect: { kind: 'nonempty' } }] } }); " +
        "if (memoryProposal.proposal?.removeClauses[0] !== 'works_at(mira, acme).' || memoryProposalStore.load('default').length !== 1) throw new Error('public proposal-first memory API failed'); const memoryApplied = applyMemoryProposal(memoryProposalStore, memoryProposal.proposal, { opId: 'reviewed-memory' }); " +
        "if (memoryApplied.audit.topology.factCount !== 1 || memoryApplied.checks?.status !== 'passed' || memoryProposalStore.load('default')[0]?.head.args[1]?.value !== 'initech') throw new Error('public reviewed memory application API failed'); " +
        "const checkpoint = trustStore.compactJournal({ opId: 'package-checkpoint', at: new Date('2026-08-17T02:00:00.000Z') }); " +
        "if (checkpoint.sequence !== 2 || trustStore.listJournalCheckpoints().length !== 1 || trustStore.recordedSnapshot(['default'], 1).clauses.length !== 1) throw new Error('public journal checkpoint API failed'); " +
        "const bundle = createKnowledgeBundle(trustStore); const bundleText = serializeKnowledgeBundle(bundle); const bundleVerification = verifyKnowledgeBundle(bundleText); " +
        "if (!bundleVerification.valid || bundleVerification.clauseCount !== 1 || bundleVerification.sha256 !== bundle.sha256) throw new Error('public knowledge bundle API failed'); " +
        "if (typeof IntegrityViolationError !== 'function') throw new Error('public integrity enforcement API failed'); " +
        "const identity = canonicalizeKnowledge(parseProgram(\"rembero_alias('Mira Patel', mira). rembero_entity_position(works_at, 2, 0). works_at('Mira Patel', acme).\")); " +
        "if (identity.clauses[0]?.head.args[0]?.value !== 'mira') throw new Error('public identity API failed'); " +
        "const fullGraph = explainKnowledge(parseProgram('edge(a, b). edge(b, c).'), 'edge(X, Y)'); " +
        "const selectedGraph = selectExplanationGraph(fullGraph, { kind: 'result', row: 1 }); " +
        "if (selectedGraph.rows.length !== 2 || selectedGraph.graphSelection?.selector?.row !== 1 || selectedGraph.graph.nodes.length >= fullGraph.graph.nodes.length) throw new Error('public graph navigation API failed'); " +
        "if (typeof OperationConflictError !== 'function' || typeof IncompleteHistoryError !== 'function') throw new Error('public history or operation error API failed'); " +
        "if (sqliteDatalogExecutionMode('item(X), X = X') !== 'portable' || sqliteDatalogExecutionMode('copy(X) :- item(X).') !== 'native') throw new Error('public SQLite execution mode API failed');",
    ],
    { cwd: directory }
  );
  const installedCli = join(directory, 'node_modules', 'remembero', 'dist', 'cli.js');
  const installedPrimaryCli = join(directory, 'node_modules', '.bin', 'remembero');
  const primaryHelp = run(installedPrimaryCli, ['--help'], { cwd: directory });
  if (!primaryHelp.startsWith('remembero — logic-based memory')) {
    throw new Error('primary remembero CLI executable failed');
  }
  const installedLegacyCli = join(directory, 'node_modules', '.bin', 'rembero');
  const legacyHelp = run(installedLegacyCli, ['--help'], { cwd: directory });
  if (!legacyHelp.startsWith('remembero — logic-based memory')) {
    throw new Error('legacy rembero CLI alias failed');
  }
  const installedExtractionEval = join(
    directory,
    'node_modules',
    'remembero',
    'dist',
    'evals',
    'run-extraction.js'
  );
  const extractionEvalHelp = run(
    process.execPath,
    [installedExtractionEval, '--help'],
    { cwd: directory }
  );
  if (!extractionEvalHelp.includes('Usage: npm run eval:extract')) {
    throw new Error('packaged extraction evaluation runner failed');
  }
  const claudeSettings = join(directory, 'claude', 'settings.json');
  run(
    process.execPath,
    [
      installedCli,
      'init-hooks',
      '--settings',
      claudeSettings,
      '--namespace',
      'personal',
      '--daily-cap',
      '3',
      '--tail-bytes',
      '8192',
    ],
    { cwd: directory }
  );
  const hookSettings = JSON.parse(readFileSync(claudeSettings, 'utf8'));
  const hookHandler = hookSettings.hooks?.Stop?.flatMap(({ hooks }) => hooks).find(
    ({ args }) => Array.isArray(args) && args.includes('rembero-auto-capture-v1')
  );
  if (
    hookHandler?.async !== true ||
    hookHandler.command !== process.execPath ||
    !hookHandler.args.includes('personal')
  ) {
    throw new Error('packaged auto-capture hook installation failed');
  }
  run(
    process.execPath,
    [installedCli, 'init-hooks', '--remove', '--settings', claudeSettings],
    { cwd: directory }
  );
  const removedSettings = JSON.parse(readFileSync(claudeSettings, 'utf8'));
  if (removedSettings.hooks?.Stop !== undefined) {
    throw new Error('packaged auto-capture hook removal failed');
  }
  const trustHome = join(directory, 'trust-home');
  const trustEnv = { ...process.env, REMBERO_HOME: trustHome };
  run(
    process.execPath,
    [
      installedCli,
      'assert',
      'status(mira, active).',
      '--trust',
      'tentative',
      '--op-id',
      'package-tentative',
    ],
    { cwd: directory, env: trustEnv }
  );
  const hiddenTrust = JSON.parse(
    run(process.execPath, [installedCli, 'query', 'status(mira, State)'], {
      cwd: directory,
      env: trustEnv,
    })
  );
  const includedTrust = JSON.parse(
    run(
      process.execPath,
      [
        installedCli,
        'explain',
        'status(mira, State)',
        '--trust',
        'include_tentative',
      ],
      { cwd: directory, env: trustEnv }
    )
  );
  const trustClaims = JSON.parse(
    run(process.execPath, [installedCli, 'claims'], {
      cwd: directory,
      env: trustEnv,
    })
  );
  if (
    hiddenTrust.length !== 0 ||
    includedTrust.rows[0]?.proofs[0]?.trust !== 'tentative' ||
    trustClaims.count !== 1
  ) {
    throw new Error('packaged tentative trust inspection failed');
  }
  run(
    process.execPath,
    [
      installedCli,
      'accept',
      'status(mira, active).',
      '--op-id',
      'package-accept',
    ],
    { cwd: directory, env: trustEnv }
  );
  const acceptedTrust = JSON.parse(
    run(process.execPath, [installedCli, 'query', 'status(mira, State)'], {
      cwd: directory,
      env: trustEnv,
    })
  );
  if (acceptedTrust[0]?.State !== 'active') {
    throw new Error('packaged tentative trust acceptance failed');
  }
  const recordedDiffOutput = JSON.parse(
    run(
      process.execPath,
      [
        installedCli,
        'diff',
        '1',
        '2',
        '--query',
        'status(mira, State)',
      ],
      { cwd: directory, env: trustEnv }
    )
  );
  if (
    recordedDiffOutput.clauses?.added?.[0]?.clause !== 'status(mira, active).' ||
    recordedDiffOutput.queryImpact?.added?.[0]?.bindings?.State !== 'active'
  ) {
    throw new Error('packaged recorded knowledge diff failed');
  }
  const trustTopology = JSON.parse(
    run(process.execPath, [installedCli, 'topology'], {
      cwd: directory,
      env: trustEnv,
    })
  );
  if (
    trustTopology.predicateCount !== 1 ||
    trustTopology.predicates?.[0]?.key !== 'status/2' ||
    trustTopology.factCount !== 1
  ) {
    throw new Error('packaged knowledge topology failed');
  }
  const whyNotTrust = JSON.parse(
    run(
      process.execPath,
      [installedCli, 'why-not', 'status(mira, paused)'],
      { cwd: directory, env: trustEnv }
    )
  );
  if (
    whyNotTrust.status !== 'blocked' ||
    whyNotTrust.failures?.[0]?.reason !== 'missing_fact' ||
    whyNotTrust.failures?.[0]?.nearby?.[0]?.fact !== 'status(mira, active).'
  ) {
    throw new Error('packaged why-not explanation failed');
  }
  const simulatedTrust = JSON.parse(
    run(
      process.execPath,
      [
        installedCli,
        'what-if',
        'status(mira, State)',
        '--assume',
        'status(mira, paused).',
      ],
      { cwd: directory, env: trustEnv }
    )
  );
  if (
    simulatedTrust.resultDelta?.added?.[0]?.bindings?.State !== 'paused' ||
    simulatedTrust.candidate?.rows?.length !== 2
  ) {
    throw new Error('packaged counterfactual simulation failed');
  }
  const simulatedRuleTrust = JSON.parse(
    run(
      process.execPath,
      [
        installedCli,
        'what-if',
        'active(Person)',
        '--assume-rule',
        'active(Person) :- status(Person, active).',
      ],
      { cwd: directory, env: trustEnv }
    )
  );
  if (
    simulatedRuleTrust.candidate?.rows?.[0]?.bindings?.Person !== 'mira' ||
    simulatedRuleTrust.ruleAuditDelta?.candidate?.topology?.ruleCount !== 1
  ) {
    throw new Error('packaged rule counterfactual simulation failed');
  }
  const ruleApplyHome = join(directory, 'rule-apply-home');
  const ruleApplyEnv = { ...process.env, REMBERO_HOME: ruleApplyHome };
  run(
    process.execPath,
    [installedCli, 'assert', 'base(a).', '--op-id', 'package-rule-base'],
    { cwd: directory, env: ruleApplyEnv }
  );
  const rulePreview = run(
    process.execPath,
    [
      installedCli,
      'what-if',
      'derived(X)',
      '--assume-rule',
      'derived(X) :- base(X).',
    ],
    { cwd: directory, env: ruleApplyEnv }
  );
  const ruleProposalFile = join(directory, 'rule-proposal.json');
  writeFileSync(ruleProposalFile, rulePreview);
  const ruleApplication = JSON.parse(
    run(
      process.execPath,
      [
        installedCli,
        'apply-rule-change',
        ruleProposalFile,
        '--op-id',
        'package-reviewed-rule',
      ],
      { cwd: directory, env: ruleApplyEnv }
    )
  );
  if (
    ruleApplication.audit?.topology?.ruleCount !== 1 ||
    ruleApplication.added?.length !== 1
  ) {
    throw new Error('packaged reviewed rule application failed');
  }
  const checkpointOutput = JSON.parse(
    run(
      process.execPath,
      [
        installedCli,
        'checkpoint',
        '--op-id',
        'package-checkpoint',
        '--at',
        '2026-08-17T02:00:00.000Z',
      ],
      { cwd: directory, env: trustEnv }
    )
  );
  const checkpointList = JSON.parse(
    run(process.execPath, [installedCli, 'checkpoints'], {
      cwd: directory,
      env: trustEnv,
    })
  );
  if (checkpointOutput.sequence !== 2 || checkpointList.count !== 1) {
    throw new Error('packaged journal checkpoint failed');
  }
  const bundleOutput = run(process.execPath, [installedCli, 'bundle'], {
    cwd: directory,
    env: trustEnv,
  });
  const bundleFile = join(directory, 'trust-bundle.json');
  writeFileSync(bundleFile, bundleOutput);
  const verifiedBundleOutput = JSON.parse(
    run(process.execPath, [installedCli, 'verify-bundle', bundleFile], {
      cwd: directory,
      env: trustEnv,
    })
  );
  if (
    verifiedBundleOutput.valid !== true ||
    verifiedBundleOutput.clauseCount !== 1 ||
    verifiedBundleOutput.sourceCount !== 1
  ) {
    throw new Error('packaged content-addressed knowledge bundle failed');
  }
  const repairHome = join(directory, 'repair-home');
  const repairEnv = { ...process.env, REMBERO_HOME: repairHome };
  run(
    process.execPath,
    [
      installedCli,
      'assert',
      'employee(bob). eligible(X) :- employee(X), badge(X).',
      '--op-id',
      'package-repair-baseline',
    ],
    { cwd: directory, env: repairEnv }
  );
  const repairOutput = JSON.parse(
    run(process.execPath, [installedCli, 'repair', 'eligible(bob)'], {
      cwd: directory,
      env: repairEnv,
    })
  );
  if (
    repairOutput.status !== 'repairable' ||
    repairOutput.plans?.[0]?.assume?.[0] !== 'badge(bob).'
  ) {
    throw new Error('packaged verified repair planning failed');
  }
  const profileOutput = JSON.parse(
    run(
      process.execPath,
      [installedCli, 'profile', 'eligible(bob)', '--compare-scan'],
      { cwd: directory, env: repairEnv }
    )
  );
  if (
    profileOutput.equivalent !== true ||
    profileOutput.fullScan === undefined ||
    profileOutput.workReduction === undefined
  ) {
    throw new Error('packaged deterministic query profile failed');
  }
  const repairAudit = JSON.parse(
    run(process.execPath, [installedCli, 'audit-rules'], {
      cwd: directory,
      env: repairEnv,
    })
  );
  if (
    repairAudit.status !== 'advisory' ||
    !repairAudit.findings?.some(({ code }) => code === 'open_positive_input')
  ) {
    throw new Error('packaged deterministic rule audit failed');
  }
  const repairSearch = JSON.parse(
    run(
      process.execPath,
      [installedCli, 'search', 'eligible', '--kind', 'rule'],
      { cwd: directory, env: repairEnv }
    )
  );
  if (
    repairSearch.status !== 'matches' ||
    repairSearch.results?.[0]?.clause !==
      'eligible(X) :- employee(X), badge(X).'
  ) {
    throw new Error('packaged deterministic local search failed');
  }
  const repairBrowse = JSON.parse(
    run(process.execPath, [installedCli, 'browse', 'bob'], {
      cwd: directory,
      env: repairEnv,
    })
  );
  if (
    repairBrowse.selection?.selectedClaims !== 1 ||
    !repairBrowse.graph?.nodes?.some(
      ({ kind, predicate }) => kind === 'claim' && predicate === 'employee'
    )
  ) {
    throw new Error('packaged explicit knowledge graph browse failed');
  }
  const checkFile = join(directory, 'repair-checks.json');
  writeFileSync(
    checkFile,
    JSON.stringify({
      version: 1,
      checks: [
        {
          name: 'employee row',
          query: 'employee(X)',
          expect: { kind: 'rows', order: 'exact', rows: [{ X: 'bob' }] },
        },
        {
          name: 'eligible absent',
          query: 'eligible(bob)',
          expect: { kind: 'empty' },
        },
      ],
    })
  );
  const checkOutput = JSON.parse(
    run(process.execPath, [installedCli, 'test-knowledge', checkFile], {
      cwd: directory,
      env: repairEnv,
    })
  );
  if (checkOutput.status !== 'passed' || checkOutput.passedCount !== 2) {
    throw new Error('packaged knowledge regression suite failed');
  }
  const extensionPath = run(process.execPath, [installedCli, 'sqlite-build'], {
    cwd: directory,
  });
  if (!extensionPath) throw new Error('packaged sqlite-build returned no extension path');

  const databasePath = join(directory, 'world.db');
  run(
    'sqlite3',
    [databasePath],
    {
      input:
        "CREATE TABLE works_at(person TEXT, company TEXT);" +
        "INSERT INTO works_at VALUES ('alice','acme'),('bob','acme');" +
        "CREATE TABLE edge(source TEXT, target TEXT);" +
        "INSERT INTO edge VALUES ('a','b'),('b','c');" +
        "CREATE TABLE employee(person TEXT);" +
        "INSERT INTO employee VALUES ('bob'),('alice');" +
        "CREATE TABLE suspended(person TEXT);" +
        "INSERT INTO suspended VALUES ('bob');" +
        "CREATE TABLE score(person TEXT, points INTEGER);" +
        "INSERT INTO score VALUES ('bob',14),('alice',20);" +
        "CREATE TABLE baseline(team TEXT, points INTEGER);" +
        "INSERT INTO baseline VALUES ('team',10);" +
        "CREATE TABLE member(team TEXT, person TEXT);" +
        "INSERT INTO member VALUES ('red','alice'),('red','bob'),('blue','carol');",
    }
  );
  const rememberoDatabaseEnv = {
    ...process.env,
    REMBERO_TEST_DATABASE: databasePath,
  };
  run(
    process.execPath,
    [
      '--input-type=module',
      '--eval',
      "import { captureKnowledgeVersion, createSemanticLedger, diffKnowledgeVersions, openRememberoDatabase, serializeClause } from 'remembero'; " +
        "const db = await openRememberoDatabase(process.env.REMBERO_TEST_DATABASE); " +
        "try { " +
          "if (db.prepare('SELECT COUNT(*) AS count FROM works_at').get()?.count !== 2) throw new Error('native SQLite query failed'); " +
          "db.memory.assert('default', 'status(mira, active).', { opId: 'package-sqlite-memory' }); " +
          "const ledger = createSemanticLedger(db); " +
          "db.memory.assert('ledger', 'base(a).', { opId: 'package-ledger-v1' }); " +
          "const ledgerV1 = captureKnowledgeVersion(ledger, db.memory, { namespaces: ['ledger'], label: 'package@1' }); " +
          "db.memory.assert('ledger', 'derived(X) :- base(X).', { opId: 'package-ledger-v2' }); " +
          "const ledgerV2 = captureKnowledgeVersion(ledger, db.memory, { namespaces: ['ledger'], parents: [ledgerV1.version.digest], label: 'package@2' }); " +
          "const ledgerImpact = diffKnowledgeVersions(ledger, db.memory, ledgerV1.version.digest, ledgerV2.version.digest, { query: 'derived(a)' }); " +
          "if (ledger.resolveVersion('package@2').digest !== ledgerV2.version.digest || ledgerImpact.queryImpact?.added.length !== 1) throw new Error('public semantic ledger API failed'); " +
          "db.exec('BEGIN'); " +
          "db.prepare('INSERT INTO works_at VALUES (?, ?)').run('temporary', 'rollback'); " +
          "db.memory.assert('default', 'status(mira, paused).', { opId: 'package-sqlite-rollback' }); " +
          "db.exec('ROLLBACK'); " +
          "if (db.prepare(\"SELECT COUNT(*) AS count FROM works_at WHERE person = 'temporary'\").get()?.count !== 0) throw new Error('SQLite rollback failed'); " +
          "if (db.memory.load('default').map(serializeClause).join('') !== 'status(mira, active).') throw new Error('SQLite-backed memory rollback failed'); " +
          "if (db.datalogQuery('available(X) :- employee(X), \\\\+ suspended(X).')[0]?.X !== 'alice') throw new Error('enhanced Datalog query failed'); " +
          "let blocked = false; try { db.enableLoadExtension(true); } catch { blocked = true; } " +
          "if (!blocked) throw new Error('extension loading was re-enabled'); " +
        "} finally { db.close(); } " +
        "const reopened = await openRememberoDatabase(process.env.REMBERO_TEST_DATABASE); " +
        "try { if (reopened.memory.load('default').map(serializeClause).join('') !== 'status(mira, active).') throw new Error('SQLite-backed memory did not reopen'); } finally { reopened.close(); }",
    ],
    { cwd: directory, env: rememberoDatabaseEnv }
  );
  const output = run(
    process.execPath,
    [
      installedCli,
      'sqlite-query',
      databasePath,
      'colleague(X, Y) :- works_at(X, C), works_at(Y, C), X != Y.',
    ],
    { cwd: directory }
  );
  const rows = JSON.parse(output);
  if (rows.length !== 2 || rows[0].X !== 'alice' || rows[1].X !== 'bob') {
    throw new Error(`unexpected packaged query result: ${output}`);
  }
  const sqlitePlanOutput = run(
    process.execPath,
    [
      installedCli,
      'sqlite-plan',
      databasePath,
      'colleague(X, Y) :- works_at(X, C), works_at(Y, C), X != Y.',
    ],
    { cwd: directory }
  );
  const sqlitePlan = JSON.parse(sqlitePlanOutput);
  if (
    sqlitePlan.mode !== 'native' ||
    sqlitePlan.scansData !== false ||
    sqlitePlan.baseRelations?.[0]?.predicate !== 'works_at' ||
    typeof sqlitePlan.nativeSql !== 'string'
  ) {
    throw new Error(`unexpected packaged SQLite Datalog plan: ${sqlitePlanOutput}`);
  }

  const recursiveProgram =
    'path(X, Y) :- edge(X, Y).\n' +
    'path(X, Y) :- edge(X, Z), path(Z, Y).';
  const recursiveOutput = run(
    process.execPath,
    [installedCli, 'sqlite-query', databasePath, recursiveProgram],
    { cwd: directory }
  );
  const recursiveRows = JSON.parse(recursiveOutput);
  if (
    recursiveRows.length !== 3 ||
    !recursiveRows.some((row) => row.X === 'a' && row.Y === 'c')
  ) {
    throw new Error(`unexpected packaged recursive result: ${recursiveOutput}`);
  }
  const explainOutput = run(
    process.execPath,
    [installedCli, 'sqlite-explain', databasePath, recursiveProgram],
    { cwd: directory }
  );
  const explanations = JSON.parse(explainOutput);
  const recursiveProof = explanations.find(
    ({ row }) => row.X === 'a' && row.Y === 'c'
  )?.proof;
  if (recursiveProof?.rule !== 2 || recursiveProof.because?.length !== 2) {
    throw new Error(`unexpected packaged explanation: ${explainOutput}`);
  }

  const advancedProgram =
    'answer(X) :- available(X), score(X, S), baseline(team, B), S > B + 5.\n' +
    'available(X) :- employee(X), \\+ suspended(X).';
  const advancedOutput = run(
    process.execPath,
    [installedCli, 'sqlite-query', databasePath, advancedProgram],
    { cwd: directory }
  );
  const advancedRows = JSON.parse(advancedOutput);
  if (advancedRows.length !== 1 || advancedRows[0].X !== 'alice') {
    throw new Error(`unexpected packaged advanced query result: ${advancedOutput}`);
  }
  const sqliteAggregateOutput = run(
    process.execPath,
    [installedCli, 'sqlite-explain', databasePath, 'count(*) as Count where employee(Person)'],
    { cwd: directory }
  );
  const sqliteAggregate = JSON.parse(sqliteAggregateOutput);
  if (sqliteAggregate[0]?.row?.Count !== 2 || sqliteAggregate[0]?.proof?.aggregated !== true) {
    throw new Error(`unexpected packaged aggregate explanation: ${sqliteAggregateOutput}`);
  }
  const sqliteAggregateRuleOutput = run(
    process.execPath,
    [
      installedCli,
      'sqlite-explain',
      databasePath,
      'team_size(Team, Count) :- count(*) as Count where member(Team, Person).',
    ],
    { cwd: directory }
  );
  const sqliteAggregateRules = JSON.parse(sqliteAggregateRuleOutput);
  if (
    sqliteAggregateRules.length !== 2 ||
    sqliteAggregateRules[1]?.row?.Count !== 2 ||
    sqliteAggregateRules[1]?.proof?.aggregate?.contributors?.length !== 2
  ) {
    throw new Error(`unexpected packaged aggregate rule: ${sqliteAggregateRuleOutput}`);
  }

  const memoryFile = join(directory, 'personal.dl');
  const memoryHome = join(directory, 'personal-home');
  writeFileSync(
    memoryFile,
    'parent(a, b). parent(b, c). ancestor(X, Y) :- parent(X, Y). ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y). ' +
      'employee(alice). employee(bob). suspended(bob). available(X) :- employee(X), \\+ suspended(X). ' +
      'score(alice, 20). score(bob, 14). baseline(team, 10).\n' +
      'member(red, alice). member(red, bob). team_size(Team, Count) :- count(*) as Count where member(Team, Person).\n'
  );
  const memoryEnv = { ...process.env, REMBERO_HOME: memoryHome };
  const temporalHome = join(directory, 'temporal-home');
  const temporalEnv = { ...process.env, REMBERO_HOME: temporalHome };
  run(
    process.execPath,
    [
      installedCli,
      'assert',
      'works_at(mira, acme).',
      '--op-id',
      'before',
    ],
    { cwd: directory, env: temporalEnv }
  );
  const supersedeArgs = [
    installedCli,
    'supersede',
    'works_at(mira, initech).',
    '--pattern',
    'works_at(mira, _)',
    '--at',
    '2026-08-16T16:59:00.000Z',
    '--op-id',
    'after',
  ];
  const supersedeOutput = run(process.execPath, supersedeArgs, {
    cwd: directory,
    env: temporalEnv,
  });
  const supersedeReplay = run(process.execPath, supersedeArgs, {
    cwd: directory,
    env: temporalEnv,
  });
  const superseded = JSON.parse(supersedeOutput);
  if (
    supersedeReplay !== supersedeOutput ||
    superseded.retracted !== 1 ||
    superseded.archived[0] !==
      "works_at_until(mira, acme, '2026-08-16T16:59:00.000Z')."
  ) {
    throw new Error(`unexpected packaged supersede result: ${supersedeOutput}`);
  }
  const temporalOutput = run(
    process.execPath,
    [installedCli, 'history', 'works_at(mira, _)', '--json'],
    { cwd: directory, env: temporalEnv }
  );
  const temporalHistory = JSON.parse(temporalOutput);
  if (
    temporalHistory.events.length !== 3 ||
    !temporalHistory.events.some(({ action }) => action === 'superseded') ||
    !temporalHistory.events.some(({ clause, current }) =>
      clause === 'works_at(mira, initech).' && current === true
    )
  ) {
    throw new Error(`unexpected packaged temporal history: ${temporalOutput}`);
  }
  const recordedOutput = run(
    process.execPath,
    [installedCli, 'query', 'works_at(mira, Company)', '--as-of-sequence', '1'],
    { cwd: directory, env: temporalEnv }
  );
  const recorded = JSON.parse(recordedOutput);
  if (
    recorded.bindings[0]?.Company !== 'acme' ||
    recorded.recordedSnapshot?.sequence !== 1 ||
    recorded.recordedSnapshot?.journalEntries !== 2
  ) {
    throw new Error(`unexpected packaged recorded snapshot: ${recordedOutput}`);
  }
  const importOutput = run(
    process.execPath,
    [installedCli, 'import', 'default', memoryFile, '--op-id', 'package-import'],
    { cwd: directory, env: memoryEnv }
  );
  const importReplay = run(
    process.execPath,
    [installedCli, 'import', 'default', memoryFile, '--op-id', 'package-import'],
    { cwd: directory, env: memoryEnv }
  );
  if (importReplay !== importOutput) {
    throw new Error(`unexpected packaged import replay: ${importReplay}`);
  }
  const aggregateRuleOutput = run(
    process.execPath,
    [installedCli, 'explain', 'team_size(red, Count)'],
    { cwd: directory, env: memoryEnv }
  );
  const aggregateRule = JSON.parse(aggregateRuleOutput);
  if (
    aggregateRule.rows[0]?.bindings?.Count !== '2' ||
    aggregateRule.rows[0]?.proofs?.[0]?.aggregate?.contributors?.length !== 2 ||
    !aggregateRule.graph.nodes.some(({ kind }) => kind === 'aggregate')
  ) {
    throw new Error(`unexpected packaged aggregate rule explanation: ${aggregateRuleOutput}`);
  }
  const retryAssert = run(
    process.execPath,
    [installedCli, 'assert', 'package_retry(value).', '--op-id', 'package-retry'],
    { cwd: directory, env: memoryEnv }
  );
  const retryReplay = run(
    process.execPath,
    [installedCli, 'assert', 'package_retry(value).', '--op-id', 'package-retry'],
    { cwd: directory, env: memoryEnv }
  );
  if (retryReplay !== retryAssert || JSON.parse(retryReplay).added[0] !== 'package_retry(value).') {
    throw new Error(`unexpected packaged operation replay: ${retryReplay}`);
  }
  const graphOutput = run(process.execPath, [installedCli, 'explain', 'ancestor(a, Y)'], {
    cwd: directory,
    env: memoryEnv,
  });
  const graph = JSON.parse(graphOutput);
  if (
    graph.rows.length !== 2 ||
    !graph.rows.some(({ bindings }) => bindings.Y === 'c') ||
    !graph.graph.nodes.some(({ kind }) => kind === 'claim')
  ) {
    throw new Error(`unexpected packaged personal graph: ${graphOutput}`);
  }
  const selectedGraphOutput = run(
    process.execPath,
    [installedCli, 'explain', 'ancestor(a, Y)', '--graph-result', '1'],
    { cwd: directory, env: memoryEnv }
  );
  const selectedGraph = JSON.parse(selectedGraphOutput);
  if (
    selectedGraph.rows.length !== 2 ||
    selectedGraph.graphSelection?.selector?.row !== 1 ||
    selectedGraph.graph.nodes.length >= graph.graph.nodes.length
  ) {
    throw new Error(`unexpected packaged graph selection: ${selectedGraphOutput}`);
  }
  const absenceOutput = run(process.execPath, [installedCli, 'explain', 'available(X)'], {
    cwd: directory,
    env: memoryEnv,
  });
  const absenceGraph = JSON.parse(absenceOutput);
  if (
    absenceGraph.rows.length !== 1 ||
    absenceGraph.rows[0].bindings.X !== 'alice' ||
    !absenceGraph.graph.nodes.some(
      ({ kind, predicate }) => kind === 'absence' && predicate === 'suspended'
    )
  ) {
    throw new Error(`unexpected packaged negation graph: ${absenceOutput}`);
  }
  const aggregateOutput = run(
    process.execPath,
    [installedCli, 'query', 'count(*) as Count where employee(Person)'],
    { cwd: directory, env: memoryEnv }
  );
  const aggregateRows = JSON.parse(aggregateOutput);
  if (aggregateRows.length !== 1 || aggregateRows[0].Count !== '2') {
    throw new Error(`unexpected packaged aggregate result: ${aggregateOutput}`);
  }
  const arithmeticOutput = run(
    process.execPath,
    [
      installedCli,
      'query',
      'score(Person, Points), baseline(team, Base), Points > Base + 5',
    ],
    { cwd: directory, env: memoryEnv }
  );
  const arithmeticRows = JSON.parse(arithmeticOutput);
  if (arithmeticRows.length !== 1 || arithmeticRows[0].Person !== 'alice') {
    throw new Error(`unexpected packaged arithmetic result: ${arithmeticOutput}`);
  }
  const aggregateExplainOutput = run(
    process.execPath,
    [installedCli, 'explain', 'max(Person) as Last where employee(Person)'],
    { cwd: directory, env: memoryEnv }
  );
  const aggregateGraph = JSON.parse(aggregateExplainOutput);
  if (
    aggregateGraph.rows[0]?.bindings.Last !== 'bob' ||
    !aggregateGraph.graph.nodes.some(
      ({ kind, op, contributorCount }) =>
        kind === 'aggregate' && op === 'max' && contributorCount === 2
    ) ||
    aggregateGraph.graph.edges.filter(({ kind }) => kind === 'witness').length !== 1
  ) {
    throw new Error(`unexpected packaged aggregate explanation: ${aggregateExplainOutput}`);
  }
  console.log(
    'packed install, generic semantic ledger, real-use-case local web console, deterministic related-knowledge recall fallback, compact deterministic evidence recall answers, all-writer knowledge check enforcement, regression-gated reviewed personal memory, immutable deterministic personal knowledge health, explicit deterministic relational projection, digest-bound reviewed personal memory application, proposal-first accepted personal memory extraction, digest-bound reviewed rule change application, proposal-only deterministic rule change impact, proof-carrying derived personal knowledge paths, deterministic explicit personal knowledge paths, exact personal knowledge extraction evaluation, verified cross-model recall ranking, indexed-versus-scan deterministic query profiling, transaction-safe schema-only SQLite Datalog planning, enforced semantic rule coverage, portable deterministic knowledge regression suites, bounded provenance-aware recall ranking, verified content-addressed portable knowledge bundle, bounded explicit personal knowledge graph browse, deterministic local knowledge search with evidence graph, deterministic positive answer mode, grounded negative recall without model phrasing, deterministic rule health audit, verified repair planning, exact recorded knowledge diff, deterministic knowledge topology, deterministic why-not explanations, deterministic counterfactual impact, immutable journal checkpoints, reviewable knowledge trust, reusable aggregate rules, non-empty recall disambiguation, focused conflict views, deterministic relation indexing, explicit temporal corrections, recorded-time snapshots, retry-safe writes, graph navigation, explicit entity identity, deterministic recall pruning, safe auto-capture hook lifecycle, temporal history, native recursion, personal proofs, atomic integrity enforcement, stratified negation, scalar aggregation, arithmetic filters, and explanation graph passed'
  );
} finally {
  rmSync(directory, { recursive: true, force: true });
}
