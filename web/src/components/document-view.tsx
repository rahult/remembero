import type {
  DocumentClaim,
  DocumentCorpusEvaluation,
  DocumentEvaluationSummary,
  DocumentListItem,
  DocumentPage,
  DocumentProofStep,
  DocumentRegion,
  DocumentShowcaseResponse,
  LiveOcrEvidenceSnapshot,
  ProductShipEvidence,
} from '../api';
import {
  AskIcon,
  CheckCircleIcon,
  DocumentEvidenceIcon,
  DocumentIcon,
  ProofIcon,
  RefreshIcon,
} from '../icons';

type EvidenceMode = 'regions' | 'claims';

interface DocumentViewProps {
  documents: DocumentListItem[];
  data: DocumentShowcaseResponse | null;
  loading: boolean;
  error: string | null;
  pendingDocumentId: string | null;
  action: 'idle' | 'asking' | 'parsing';
  mobile: boolean;
  evidenceMode: EvidenceMode;
  selectedQuestionId: string | null;
  selectedPageId: string | null;
  selectedRegionId: string | null;
  selectedClaimId: string | null;
  selectedSourceId: string | null;
  onSelectDocument: (documentId: string) => void;
  onSelectEvidenceMode: (mode: EvidenceMode) => void;
  onSelectQuestion: (questionId: string) => void;
  onAskQuestion: () => void;
  onParse: () => void;
  onRetry: () => void;
  onSelectPage: (pageId: string) => void;
  onSelectRegion: (regionId: string) => void;
  onSelectClaim: (claimId: string) => void;
  onSelectSource: (sourceId: string) => void;
}

function pageById(data: DocumentShowcaseResponse | null, pageId: string | null): DocumentPage | null {
  if (data === null) return null;
  return data.document.pages.find((page) => page.id === pageId) ?? data.document.pages[0] ?? null;
}

function regionById(
  data: DocumentShowcaseResponse | null,
  regionId: string | null
): DocumentRegion | null {
  if (data === null || regionId === null) return null;
  for (const page of data.document.pages) {
    const region = page.regions.find((entry) => entry.id === regionId);
    if (region !== undefined) return region;
  }
  return null;
}

function claimById(data: DocumentShowcaseResponse | null, claimId: string | null): DocumentClaim | null {
  if (data === null || claimId === null) return null;
  return data.document.claims.find((claim) => claim.id === claimId) ?? null;
}

function proofStepById(
  data: DocumentShowcaseResponse | null,
  sourceId: string | null
): DocumentProofStep | null {
  if (data === null || sourceId === null) return null;
  return data.proof.steps.find((step) => step.id === sourceId) ?? null;
}

function claimsForPage(data: DocumentShowcaseResponse, pageId: string): DocumentClaim[] {
  return data.document.claims.filter((claim) => claim.pageId === pageId);
}

function claimsForRegion(data: DocumentShowcaseResponse, regionId: string): DocumentClaim[] {
  return data.document.claims.filter((claim) => claim.regionId === regionId);
}

function regionKindLabel(region: DocumentRegion): string {
  return region.kind === 'table_row' ? 'table row' : region.kind.replaceAll('_', ' ');
}

function pageRegionCount(page: DocumentPage): string {
  return `${page.regions.length} block${page.regions.length === 1 ? '' : 's'}`;
}

function countLabel(value: number, singular: string, plural = `${singular}s`): string {
  return `${value} ${value === 1 ? singular : plural}`;
}

function badgeTone(label?: string): 'accepted' | 'proposed' | 'supported' | 'neutral' {
  const lower = label?.toLowerCase() ?? '';
  if (lower.includes('accepted')) return 'accepted';
  if (lower.includes('proposed')) return 'proposed';
  if (lower.includes('supported')) return 'supported';
  return 'neutral';
}

function verdictTone(status: 'pass' | 'fail'): 'accepted' | 'proposed' {
  return status === 'pass' ? 'accepted' : 'proposed';
}

function metricSummary(
  label: string,
  value: { percent: number; passed: number; total: number }
): string {
  return `${label} ${value.percent}%`;
}

function regionStyle(page: DocumentPage, region: DocumentRegion) {
  return {
    left: `${(region.bbox.x / page.width) * 100}%`,
    top: `${(region.bbox.y / page.height) * 100}%`,
    width: `${(region.bbox.width / page.width) * 100}%`,
    height: `${(region.bbox.height / page.height) * 100}%`,
  };
}

function DocumentLoadingState({
  loading,
  error,
  onRetry,
}: {
  loading: boolean;
  error: string | null;
  onRetry: () => void;
}) {
  if (loading) {
    return (
      <div className="loading-panel">
        <p>Loading reviewed real-document evidence…</p>
      </div>
    );
  }

  if (error !== null) {
    return (
      <div className="loading-panel">
        <p>{error}</p>
        <button className="secondary-button" type="button" onClick={onRetry}>
          Retry document
        </button>
      </div>
    );
  }

  return (
    <div className="empty-panel">
      <p>No document showcase is available yet.</p>
      <button className="secondary-button" type="button" onClick={onRetry}>
        Load document
      </button>
    </div>
  );
}

function DocumentEvaluationPanel({
  evaluation,
  corpusEvaluation,
  liveOcrEvidence,
  shipEvidence,
}: {
  evaluation: DocumentEvaluationSummary | undefined;
  corpusEvaluation: DocumentCorpusEvaluation | undefined;
  liveOcrEvidence: LiveOcrEvidenceSnapshot | undefined;
  shipEvidence: ProductShipEvidence | undefined;
}) {
  if (
    evaluation === undefined &&
    corpusEvaluation === undefined &&
    liveOcrEvidence === undefined &&
    shipEvidence === undefined
  ) return null;

  return (
    <section className="document-evaluation-grid" aria-label="Deterministic showcase evaluation">
      {evaluation !== undefined ? (
        <article className="document-evaluation-card">
          <div className="document-evaluation-card__header">
            <div>
              <h2>Selected document scorecard</h2>
              <p>Parse, recall, proof grounding, abstention, and idempotency from reviewed source-backed rules.</p>
            </div>
            <span className={`document-badge document-badge--${verdictTone(evaluation.status)}`}>
              {evaluation.status === 'pass' ? 'All checks passing' : 'Check failures'}
            </span>
          </div>

          <div className="document-evaluation-metrics">
            {[
              metricSummary('Parse', evaluation.metrics.parseCoverage),
              metricSummary('Status', evaluation.metrics.statusAccuracy),
              metricSummary('Answer', evaluation.metrics.answerAccuracy),
              metricSummary('Recall', evaluation.metrics.sourceRecall),
              metricSummary('Proof', evaluation.metrics.proofGrounding),
              metricSummary('Abstention', evaluation.metrics.abstentionCorrectness),
              metricSummary('Idempotency', evaluation.metrics.idempotency),
            ].map((item) => (
              <span key={item} className="document-stat-chip">
                {item}
              </span>
            ))}
          </div>

          <div className="document-selection-summary document-selection-summary--evaluation">
            <span>
              Parse latency: <strong>{evaluation.latencyMs.parseMs} ms</strong>
            </span>
            <span>
              Avg question latency: <strong>{evaluation.latencyMs.averageQuestionMs} ms</strong>
            </span>
            <span>
              Max question latency: <strong>{evaluation.latencyMs.maxQuestionMs} ms</strong>
            </span>
          </div>

          <div className="document-list">
            {evaluation.checks.map((check, index) => (
              <div key={check.questionId} className="document-list-item document-list-item--static">
                <div className="document-list-item__index">{index + 1}</div>
                <div className="document-list-item__body">
                  <div className="document-list-item__header">
                    <strong>{check.label}</strong>
                    <span
                      className={`document-badge document-badge--${check.actualStatus === 'answered' ? 'supported' : 'neutral'}`}
                    >
                      {check.actualStatus}
                    </span>
                  </div>
                  <p>{check.question}</p>
                  <div className="document-list-item__meta">
                    <span>{check.statusPass ? 'status ok' : 'status mismatch'}</span>
                    <span>{check.answerPass ? 'answer ok' : 'answer mismatch'}</span>
                    <span>{check.sourceRecallPass ? 'recall ok' : 'recall mismatch'}</span>
                    <span>{check.proofGroundingPass ? 'proof grounded' : 'proof mismatch'}</span>
                    <span>{check.abstentionPass ? 'abstention ok' : 'abstention mismatch'}</span>
                    <span>{check.latencyMs} ms</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </article>
      ) : null}

      {corpusEvaluation !== undefined ? (
        <article className="document-evaluation-card document-evaluation-card--corpus">
          <div className="document-evaluation-card__header">
            <div>
              <h2>Corpus aggregate</h2>
              <p>
                {corpusEvaluation.documentCount} documents · {corpusEvaluation.questionCount} deterministic questions
              </p>
            </div>
            <span className={`document-badge document-badge--${verdictTone(corpusEvaluation.status)}`}>
              {corpusEvaluation.status === 'pass' ? 'Corpus passing' : 'Corpus drift'}
            </span>
          </div>

          <div className="document-evaluation-metrics">
            {[
              metricSummary('Parse', corpusEvaluation.metrics.parseCoverage),
              metricSummary('Status', corpusEvaluation.metrics.statusAccuracy),
              metricSummary('Answer', corpusEvaluation.metrics.answerAccuracy),
              metricSummary('Recall', corpusEvaluation.metrics.sourceRecall),
              metricSummary('Proof', corpusEvaluation.metrics.proofGrounding),
              metricSummary('Abstention', corpusEvaluation.metrics.abstentionCorrectness),
              metricSummary('Idempotency', corpusEvaluation.metrics.idempotency),
            ].map((item) => (
              <span key={item} className="document-stat-chip">
                {item}
              </span>
            ))}
          </div>

          <div className="document-selection-summary document-selection-summary--evaluation">
            <span>
              Total parse latency: <strong>{corpusEvaluation.latencyMs.totalParseMs} ms</strong>
            </span>
            <span>
              Avg question latency: <strong>{corpusEvaluation.latencyMs.averageQuestionMs} ms</strong>
            </span>
            <span>
              Max question latency: <strong>{corpusEvaluation.latencyMs.maxQuestionMs} ms</strong>
            </span>
            <span>
              Total runtime: <strong>{corpusEvaluation.latencyMs.totalMs} ms</strong>
            </span>
          </div>
        </article>
      ) : null}

      {shipEvidence !== undefined ? (
        <article className="document-evaluation-card document-evaluation-card--models">
          <div className="document-evaluation-card__header">
            <div>
              <h2>Model, token &amp; cost checkpoint</h2>
              <p>Same grounded recall and exact extraction suites across four live models.</p>
            </div>
            <span className="document-badge document-badge--accepted">Local alpha ready</span>
          </div>

          <div className="document-model-default">
            <span className="document-eyebrow">Shipping default</span>
            <strong>{shipEvidence.defaultModel}</strong>
            <p>
              {shipEvidence.testFiles} test files · {shipEvidence.passingTests} passing · deterministic document path {shipEvidence.deterministicDocumentAccuracyPercent}% at 0 tokens / $0
            </p>
          </div>

          <div className="document-model-table" role="table" aria-label="Frontier model comparison">
            {shipEvidence.models.map((model) => (
              <div className="document-model-row" role="row" key={model.model}>
                <div role="cell">
                  <strong>{model.model}</strong>
                  <span className={`document-badge document-badge--${model.role === 'default' ? 'accepted' : 'neutral'}`}>
                    {model.role}
                  </span>
                </div>
                <span role="cell">Recall <strong>{model.recallAccuracyPercent}%</strong></span>
                <span role="cell">Write <strong>{model.extractionAccuracyPercent}%</strong></span>
                <span role="cell"><strong>{(model.recallTokens + model.extractionTokens).toLocaleString('en-US')}</strong> tokens</span>
                <span role="cell"><strong>${(model.recallCostUsd + model.extractionCostUsd).toFixed(6)}</strong></span>
                <span role="cell"><strong>{((model.recallDurationMs + model.extractionDurationMs) / 1000).toFixed(1)} s</strong></span>
              </div>
            ))}
          </div>

          <div className="document-boundary">
            <DocumentEvidenceIcon size={18} />
            <p>{shipEvidence.boundary}</p>
          </div>
        </article>
      ) : null}

      {liveOcrEvidence !== undefined ? (
        <article className="document-evaluation-card document-evaluation-card--live">
          <div className="document-evaluation-card__header">
            <div>
              <h2>Live Unlimited-OCR evidence</h2>
              <p>
                {liveOcrEvidence.completedDocuments}/{liveOcrEvidence.documentCount} real public PDF pages · {liveOcrEvidence.model} · {liveOcrEvidence.mode}
              </p>
            </div>
            <span className={`document-badge document-badge--${liveOcrEvidence.status === 'blocked' ? 'neutral' : verdictTone(liveOcrEvidence.status)}`}>
              {liveOcrEvidence.status === 'pass'
                ? 'Live corpus passing'
                : liveOcrEvidence.status === 'blocked'
                  ? 'Provider quota blocked'
                  : 'Live corpus drift'}
            </span>
          </div>

          {liveOcrEvidence.status === 'blocked' ? (
            <div className="document-provider-blocked" role="status">
              <strong>No model-quality score was produced.</strong>
              <p>{liveOcrEvidence.operationalMessage}</p>
              <span>
                Attempted {liveOcrEvidence.documentCount} real pages · {liveOcrEvidence.errorDocuments ?? liveOcrEvidence.documentCount} provider errors · frozen {new Date(liveOcrEvidence.generatedAt).toLocaleString('en-AU')}
              </span>
            </div>
          ) : (
            <>
              <div className="document-evaluation-metrics">
                {[
                  metricSummary('Fields', liveOcrEvidence.requiredFieldRecall),
                  metricSummary('Order recall', liveOcrEvidence.readingOrderRecall),
                  metricSummary('Order', liveOcrEvidence.readingOrderOrder),
                  metricSummary('Grounding', liveOcrEvidence.groundingCoordinateCoverage),
                  metricSummary('Tables', liveOcrEvidence.tableDetection),
                  ...(liveOcrEvidence.normalizedSimilarityPercent === undefined
                    ? []
                    : [`Similarity ${liveOcrEvidence.normalizedSimilarityPercent}%`]),
                ].map((item) => (
                  <span key={item} className="document-stat-chip">{item}</span>
                ))}
              </div>

              <div className="document-selection-summary document-selection-summary--evaluation">
                <span>Total live runtime: <strong>{(liveOcrEvidence.totalLatencyMs / 1000).toFixed(1)} s</strong></span>
                <span>Average page: <strong>{(liveOcrEvidence.averageDocumentLatencyMs / 1000).toFixed(1)} s</strong></span>
                <span>Maximum page: <strong>{(liveOcrEvidence.maximumDocumentLatencyMs / 1000).toFixed(1)} s</strong></span>
                <span>Frozen: <strong>{new Date(liveOcrEvidence.generatedAt).toLocaleString('en-AU')}</strong></span>
              </div>
            </>
          )}

          <div className="document-boundary">
            <DocumentEvidenceIcon size={18} />
            <p>{liveOcrEvidence.authorityBoundary}</p>
          </div>
        </article>
      ) : null}
    </section>
  );
}

function DocumentPagePreview({
  page,
  pages,
  selectedRegionId,
  selectedSourceId,
  onSelectPage,
  onSelectRegion,
}: {
  page: DocumentPage;
  pages: DocumentPage[];
  selectedRegionId: string | null;
  selectedSourceId: string | null;
  onSelectPage: (pageId: string) => void;
  onSelectRegion: (regionId: string) => void;
}) {
  return (
    <section className="document-panel document-panel--preview">
      <div className="document-panel__header">
        <div>
          <h2>Document preview</h2>
          <p>{page.label} · {pageRegionCount(page)}</p>
        </div>
        <div className="document-page-tabs" role="tablist" aria-label="Pages">
          {pages.map((entry) => (
            <button
              key={entry.id}
              className={`document-page-tab${entry.id === page.id ? ' document-page-tab--active' : ''}`}
              type="button"
              role="tab"
              aria-selected={entry.id === page.id}
              onClick={() => onSelectPage(entry.id)}
            >
              {entry.label}
            </button>
          ))}
        </div>
      </div>

      <div className="document-sheet">
        <div className="document-sheet__paper" style={{ aspectRatio: `${page.width} / ${page.height}` }}>
          <img
            className="document-sheet__source"
            src={page.imageUrl}
            alt={`Rendered ${page.label}`}
            draggable={false}
          />
          {page.regions.map((region) => {
            const selected = selectedRegionId === region.id;
            const sourced = selectedSourceId !== null && selected;
            return (
              <button
                key={region.id}
                className={`document-region${selected ? ' document-region--active' : ''}${sourced ? ' document-region--sourced' : ''}`}
                style={regionStyle(page, region)}
                type="button"
                aria-label={`${region.label}: ${region.text}`}
                onClick={() => onSelectRegion(region.id)}
              >
                <span className="document-region__anchor">{region.order}</span>
                <span className="document-region__text">{region.text}</span>
              </button>
            );
          })}
          <div className="document-sheet__footer">
            <span>{page.label}</span>
          </div>
        </div>
      </div>

      <div className="document-thumbnail-row" aria-label="Page thumbnails">
        {pages.map((entry) => (
          <button
            key={entry.id}
            className={`document-thumbnail${entry.id === page.id ? ' document-thumbnail--active' : ''}`}
            type="button"
            onClick={() => onSelectPage(entry.id)}
          >
            <span className="document-thumbnail__page">{entry.label}</span>
            <span className="document-thumbnail__meta">{pageRegionCount(entry)}</span>
          </button>
        ))}
      </div>
    </section>
  );
}

function ParsedEvidencePanel({
  data,
  page,
  evidenceMode,
  selectedRegionId,
  selectedClaimId,
  onSelectEvidenceMode,
  onSelectRegion,
  onSelectClaim,
}: {
  data: DocumentShowcaseResponse;
  page: DocumentPage;
  evidenceMode: EvidenceMode;
  selectedRegionId: string | null;
  selectedClaimId: string | null;
  onSelectEvidenceMode: (mode: EvidenceMode) => void;
  onSelectRegion: (regionId: string) => void;
  onSelectClaim: (claimId: string) => void;
}) {
  const pageClaims = claimsForPage(data, page.id);

  return (
    <section className="document-panel document-panel--evidence">
      <div className="document-panel__header">
        <div>
          <h2>Parsed evidence</h2>
          <p>Select a raw region or reviewed extraction from {page.label.toLowerCase()}.</p>
        </div>
      </div>

      <div className="document-tab-strip" role="tablist" aria-label="Evidence mode">
        <button
          className={`document-tab${evidenceMode === 'regions' ? ' document-tab--active' : ''}`}
          type="button"
          role="tab"
          aria-selected={evidenceMode === 'regions'}
          onClick={() => onSelectEvidenceMode('regions')}
        >
          Text blocks
          <span>{page.regions.length}</span>
        </button>
        <button
          className={`document-tab${evidenceMode === 'claims' ? ' document-tab--active' : ''}`}
          type="button"
          role="tab"
          aria-selected={evidenceMode === 'claims'}
          onClick={() => onSelectEvidenceMode('claims')}
        >
          Extracted claims
          <span>{pageClaims.length}</span>
        </button>
      </div>

      {evidenceMode === 'regions' ? (
        <>
          <p className="document-note">
            Text-layer blocks are atomic source regions with stable page and coordinate anchors.
          </p>
          <div className="document-list">
            {page.regions.map((region) => {
              const linkedClaims = claimsForRegion(data, region.id);
              return (
                <button
                  key={region.id}
                  className={`document-list-item${selectedRegionId === region.id ? ' document-list-item--active' : ''}`}
                  type="button"
                  onClick={() => onSelectRegion(region.id)}
                >
                  <div className="document-list-item__index">{region.order}</div>
                  <div className="document-list-item__body">
                    <div className="document-list-item__header">
                      <strong>{region.text}</strong>
                      <span>{region.label}</span>
                    </div>
                    <div className="document-list-item__meta">
                      <span>{page.label}</span>
                      <span>{regionKindLabel(region)}</span>
                      <span>{linkedClaims.length} linked claim{linkedClaims.length === 1 ? '' : 's'}</span>
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        </>
      ) : (
        <>
          <p className="document-note">
            Proposed claims stay outside proof. Accepted facts are the only document-backed inputs to recall.
          </p>
          <div className="document-list">
            {pageClaims.map((claim, index) => (
              <button
                key={claim.id}
                className={`document-list-item${selectedClaimId === claim.id ? ' document-list-item--active' : ''}`}
                type="button"
                onClick={() => onSelectClaim(claim.id)}
              >
                <div className="document-list-item__index">{index + 1}</div>
                <div className="document-list-item__body">
                  <div className="document-list-item__header">
                    <strong><code>{claim.clause}</code></strong>
                    <span className={`document-badge document-badge--${badgeTone(claim.reviewLabel)}`}>
                      {claim.reviewLabel}
                    </span>
                  </div>
                  <p>{claim.summary}</p>
                  <div className="document-list-item__meta">
                    <span>{page.label}</span>
                    <span>{page.regions.find((region) => region.id === claim.regionId)?.label ?? claim.regionId}</span>
                  </div>
                </div>
              </button>
            ))}
          </div>
        </>
      )}

      <div className="document-boundary">
        <DocumentEvidenceIcon size={18} />
        <p>
          Parsed material is evidence, not authority. Reviewed facts in the local knowledge base
          power proofs.
        </p>
      </div>
    </section>
  );
}

function RecallAndProofPanel({
  data,
  selectedQuestionId,
  selectedSourceId,
  mobile,
  action,
  onSelectQuestion,
  onAskQuestion,
  onSelectSource,
}: {
  data: DocumentShowcaseResponse;
  selectedQuestionId: string | null;
  selectedSourceId: string | null;
  mobile: boolean;
  action: 'idle' | 'asking' | 'parsing';
  onSelectQuestion: (questionId: string) => void;
  onAskQuestion: () => void;
  onSelectSource: (sourceId: string) => void;
}) {
  const { proof, questions, defaultQuestionId } = data;
  const activeQuestionId = selectedQuestionId ?? proof.questionId ?? defaultQuestionId;
  const hasPendingQuestion = activeQuestionId !== proof.questionId;

  return (
    <section className="document-panel document-panel--proof">
      <div className="document-panel__header">
        <div>
          <h2>Recall &amp; proof</h2>
          <p>Accepted facts and reviewed rules are the only proof-bearing inputs.</p>
        </div>
      </div>

      <div className="document-question-strip" role="tablist" aria-label="Guided questions">
        {questions.map((question) => (
          <button
            key={question.id}
            className={`document-question-chip${activeQuestionId === question.id ? ' document-question-chip--active' : ''}`}
            type="button"
            role="tab"
            aria-selected={activeQuestionId === question.id}
            onClick={() => onSelectQuestion(question.id)}
          >
            {question.label}
          </button>
        ))}
      </div>

      <div className="document-proof-toolbar">
        <button className="primary-button" type="button" onClick={onAskQuestion} disabled={action !== 'idle'}>
          <AskIcon size={18} />
          {action === 'asking' ? 'Asking…' : 'Ask from this document'}
        </button>
        {hasPendingQuestion ? (
          <p className="document-note">This will replace the current proof.</p>
        ) : null}
      </div>

      <div className={`document-answer-card${proof.status === 'unsupported' ? ' document-answer-card--unsupported' : ''}`}>
        <div className="document-answer-card__section">
          <span className="document-eyebrow">Question</span>
          <p>{proof.question}</p>
        </div>
        <div className="document-answer-card__section">
          <span className="document-eyebrow">Answer</span>
          <p className="document-answer-card__answer">{proof.answer}</p>
          {proof.status === 'unsupported' ? (
            <p className="document-answer-card__boundary">
              Related evidence below is context only. It is not accepted proof.
            </p>
          ) : null}
        </div>
        <div className="document-answer-card__section">
          <span className="document-eyebrow">Canonical query</span>
          <code className="document-query">{proof.query}</code>
        </div>
      </div>

      {proof.status === 'answered' ? (
        <>
          <div className="document-proof-chain">
            {proof.steps.map((step, index) =>
              step.kind === 'conclusion' ? (
                <div key={step.id} className="document-conclusion-card">
                  <CheckCircleIcon size={22} />
                  <div>
                    <strong>{step.label}</strong>
                    <p>{step.detail}</p>
                  </div>
                </div>
              ) : (
                <div
                  key={step.id}
                  className={`document-proof-step${selectedSourceId === step.id ? ' document-proof-step--active' : ''}`}
                >
                  <div className="document-proof-step__index">{index + 1}</div>
                  <div className="document-proof-step__body">
                    <div className="document-proof-step__header">
                      <code>{step.clause ?? step.label}</code>
                      <span className={`document-badge document-badge--${badgeTone(step.badge)}`}>
                        {step.badge}
                      </span>
                    </div>
                    {step.regionId !== undefined ? (
                      <button
                        className={`document-proof-source${selectedSourceId === step.id ? ' document-proof-source--active' : ''}`}
                        type="button"
                        onClick={() => onSelectSource(step.id)}
                      >
                        <span>{step.pageNumber !== undefined ? `Page ${step.pageNumber}` : 'Source'}</span>
                        <strong>{step.anchorLabel}</strong>
                        <p>{step.detail}</p>
                      </button>
                    ) : (
                      <div className="document-proof-rule">
                        <ProofIcon size={18} />
                        <p>{step.detail}</p>
                      </div>
                    )}
                  </div>
                </div>
              )
            )}
          </div>
          {proof.sources.length > 0 ? (
            <div className="document-related-evidence">
              <div className="document-related-evidence__header">
                <h3>Supporting evidence</h3>
                {mobile ? null : <DocumentIcon size={18} />}
              </div>
              <div className="document-list">
                {proof.sources.map((item, index) => (
                  <div key={item.id} className="document-list-item document-list-item--static">
                    <div className="document-list-item__index">{index + 1}</div>
                    <div className="document-list-item__body">
                      <div className="document-list-item__header">
                        <strong>{item.clause ?? item.label}</strong>
                        <span className={`document-badge document-badge--${badgeTone(item.badge)}`}>
                          {item.badge ?? 'Evidence'}
                        </span>
                      </div>
                      <p>{item.detail}</p>
                      <div className="document-list-item__meta">
                        <span>{item.pageNumber !== undefined ? `Page ${item.pageNumber}` : 'No page anchor'}</span>
                        <span>{item.anchorLabel ?? 'Supporting evidence'}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : null}
        </>
      ) : (
        <div className="document-related-evidence">
          <div className="document-related-evidence__header">
            <h3>Related evidence</h3>
            {mobile ? null : <DocumentIcon size={18} />}
          </div>
          <div className="document-list">
            {proof.relatedEvidence.map((item, index) => (
              <div key={item.id} className="document-list-item document-list-item--static">
                <div className="document-list-item__index">{index + 1}</div>
                <div className="document-list-item__body">
                  <div className="document-list-item__header">
                    <strong>{item.clause ?? item.label}</strong>
                    <span className={`document-badge document-badge--${badgeTone(item.badge)}`}>
                      {item.badge ?? 'Evidence'}
                    </span>
                  </div>
                  <p>{item.detail}</p>
                  <div className="document-list-item__meta">
                    <span>{item.pageNumber !== undefined ? `Page ${item.pageNumber}` : 'No page anchor'}</span>
                    <span>{item.anchorLabel ?? 'Related evidence'}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

export function DocumentView({
  documents,
  data,
  loading,
  error,
  pendingDocumentId,
  action,
  mobile,
  evidenceMode,
  selectedQuestionId,
  selectedPageId,
  selectedRegionId,
  selectedClaimId,
  selectedSourceId,
  onSelectDocument,
  onSelectEvidenceMode,
  onSelectQuestion,
  onAskQuestion,
  onParse,
  onRetry,
  onSelectPage,
  onSelectRegion,
  onSelectClaim,
  onSelectSource,
}: DocumentViewProps) {
  if (data === null) {
    return <DocumentLoadingState loading={loading} error={error} onRetry={onRetry} />;
  }

  const page = pageById(data, selectedPageId);
  const selectedRegion = regionById(data, selectedRegionId);
  const selectedClaim = claimById(data, selectedClaimId);
  const selectedTraceSource = proofStepById(data, selectedSourceId);

  if (page === null) {
    return <DocumentLoadingState loading={loading} error={error} onRetry={onRetry} />;
  }

  const activeDocument =
    documents.find((item) => item.id === data.document.id) ??
    documents[0] ?? {
      id: data.document.id,
      fileName: data.document.fileName,
      title: data.document.title,
      kindLabel: data.document.kindLabel,
      pageCount: data.parse.pageCount,
      questionCount: data.questions.length,
      acceptedClaimCount: data.parse.acceptedClaimCount,
      proposedClaimCount: data.parse.proposedClaimCount,
      supportedQuestionCount: data.questions.length,
      unsupportedQuestionCount: 0,
    };

  return (
    <div className="document-workspace">
      <section className="view-header document-header">
        <div>
          <h1>Document intelligence</h1>
          <p className="muted-copy">
            Real-PDF parse, recall, and proof desk. Downloaded bytes and extracted regions remain
            evidence; only human-accepted facts and reviewed rules can enter a proof.
          </p>
        </div>

        <div className="document-selector" aria-label="Document selector">
          {documents.map((item) => {
            const active = item.id === data.document.id;
            const pending = pendingDocumentId === item.id;
            return (
              <button
                key={item.id}
                className={`document-selector-card${active ? ' document-selector-card--active' : ''}${pending ? ' document-selector-card--pending' : ''}`}
                type="button"
                onClick={() => onSelectDocument(item.id)}
                disabled={pending}
              >
                <div className="document-selector-card__header">
                  <span className="document-badge document-badge--neutral">{item.kindLabel}</span>
                  <span className="document-selector-card__meta">
                    {countLabel(item.pageCount, 'page')} · {countLabel(item.questionCount, 'question')}
                  </span>
                </div>
                <strong>{item.title}</strong>
                <p>{item.fileName}</p>
              </button>
            );
          })}
        </div>

        <div className="document-header__controls">
          <div className="document-file-chip">
            <DocumentIcon size={18} />
            <span>{activeDocument.title}</span>
            <small>{activeDocument.fileName}</small>
          </div>
          <div className="document-stat-chip document-stat-chip--healthy">
            {data.parse.pageCoveragePercent}% page · {data.parse.acceptedClaimCoveragePercent}% claim coverage
          </div>
          <div className="document-stat-chip">
            {activeDocument.kindLabel} · {data.parse.acceptedClaimCount} accepted · {data.parse.proposedClaimCount} proposed
          </div>
          <a
            className="secondary-button document-source-link"
            href={data.document.source.url}
            target="_blank"
            rel="noreferrer"
          >
            Original PDF
          </a>
          {data.memorgExport !== undefined ? (
            <a
              className="secondary-button document-source-link"
              href={data.memorgExport.downloadUrl}
              download="document-intelligence.memorg.json"
            >
              Memorg memory
            </a>
          ) : null}
          <button className="secondary-button" type="button" onClick={onParse} disabled={action !== 'idle'}>
            <RefreshIcon size={18} />
            {action === 'parsing' ? 'Re-parsing…' : 'Re-run reviewed parse'}
          </button>
        </div>
      </section>

      <section className="document-provenance" aria-label="Document provenance">
        <div>
          <span className="document-eyebrow">Publisher</span>
          <strong>{data.document.source.publisher}</strong>
        </div>
        <div>
          <span className="document-eyebrow">Original PDF</span>
          <code title={data.document.source.sha256}>{data.document.source.sha256.slice(0, 16)}…</code>
        </div>
        <div>
          <span className="document-eyebrow">Rendered page</span>
          <code title={page.imageSha256}>{page.imageSha256.slice(0, 16)}…</code>
        </div>
        <div>
          <span className="document-eyebrow">Coverage</span>
          <strong>{page.label} of {data.document.source.pdfPageCount}</strong>
        </div>
        <p>{data.document.source.rightsNote}</p>
      </section>

      {data.memorgExport !== undefined ? (
        <section className="document-memorg" aria-label="Memorg export">
          <div>
            <span className="document-eyebrow">Portable external memory</span>
            <strong>Memorg memory ready</strong>
            <p>
              {data.memorgExport.itemCount} parent-first memory items · Memorg {data.memorgExport.targetVersion} · accepted and proposed authority preserved
            </p>
          </div>
          <code title={data.memorgExport.sha256}>{data.memorgExport.sha256.slice(0, 20)}…</code>
        </section>
      ) : null}

      <div className="document-selection-summary">
        <span>
          Current document: <strong>{activeDocument.title}</strong>
        </span>
        <span>
          Current page: <strong>{page.label}</strong>
        </span>
        <span>
          Region: <strong>{selectedRegion?.label ?? 'None'}</strong>
        </span>
        <span>
          Claim: <strong>{selectedClaim?.reviewLabel ?? 'None selected'}</strong>
        </span>
        <span>
          Proof source: <strong>{selectedTraceSource?.anchorLabel ?? 'None selected'}</strong>
        </span>
        <span>
          Parse: <strong>{data.parse.status} · {data.parse.seededCount} new · {data.parse.duplicateCount} unchanged{data.parse.fixtureDigest.length > 0 ? ` · ${data.parse.fixtureDigest.slice(0, 8)}` : ''}</strong>
        </span>
      </div>

      <div className="document-columns">
        <DocumentPagePreview
          page={page}
          pages={data.document.pages}
          selectedRegionId={selectedRegionId}
          selectedSourceId={selectedSourceId}
          onSelectPage={onSelectPage}
          onSelectRegion={onSelectRegion}
        />
        <ParsedEvidencePanel
          data={data}
          page={page}
          evidenceMode={evidenceMode}
          selectedRegionId={selectedRegionId}
          selectedClaimId={selectedClaimId}
          onSelectEvidenceMode={onSelectEvidenceMode}
          onSelectRegion={onSelectRegion}
          onSelectClaim={onSelectClaim}
        />
        <RecallAndProofPanel
          data={data}
          selectedQuestionId={selectedQuestionId}
          selectedSourceId={selectedSourceId}
          mobile={mobile}
          action={action}
          onSelectQuestion={onSelectQuestion}
          onAskQuestion={onAskQuestion}
          onSelectSource={onSelectSource}
        />
      </div>

      <DocumentEvaluationPanel
        evaluation={data.evaluation ?? activeDocument.evaluation}
        corpusEvaluation={data.corpusEvaluation}
        liveOcrEvidence={data.liveOcrEvidence}
        shipEvidence={data.shipEvidence}
      />
    </div>
  );
}
