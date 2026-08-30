import { useMemo, useState } from 'react';
import type {
  SemanticVersionReview,
  SemanticVersionView,
  SemanticVersionWorkspace,
} from '../api';
import { CheckCircleIcon, DatabaseIcon, GitBranchIcon, RefreshIcon } from '../icons';

interface VersionsViewProps {
  workspace: SemanticVersionWorkspace | null;
  selectedDigest: string | null;
  review: SemanticVersionReview | null;
  loading: boolean;
  error: string | null;
  onSelect: (digest: string) => void;
  onCapture: () => void;
  onReview: () => void;
  onPromote: () => void;
}

const columns: Array<{ status: SemanticVersionView['status']; label: string }> = [
  { status: 'baseline', label: 'Baseline' },
  { status: 'candidate', label: 'Candidates' },
  { status: 'review', label: 'Review' },
  { status: 'blocked', label: 'Blocked' },
  { status: 'promoted', label: 'Promoted' },
];

function shortDigest(value: string): string {
  return value.length > 16 ? `${value.slice(0, 12)}…${value.slice(-4)}` : value;
}

function VersionCard({ version, selected, onSelect }: { version: SemanticVersionView; selected: boolean; onSelect: () => void }) {
  return (
    <button className={`version-card${selected ? ' version-card--selected' : ''}`} type="button" onClick={onSelect}>
      <div className="version-card__topline">
        <span className={`pill${version.status === 'blocked' ? ' pill--amber' : ''}`}>{version.status}</span>
        <code>{shortDigest(version.digest)}</code>
      </div>
      <strong>{version.labels[0] ?? 'Unnamed semantic version'}</strong>
      <span>{version.memberKeys.length} members · {version.edgeCount} edges · {version.contractCount} contracts</span>
    </button>
  );
}

export function VersionsView({
  workspace,
  selectedDigest,
  review,
  loading,
  error,
  onSelect,
  onCapture,
  onReview,
  onPromote,
}: VersionsViewProps) {
  const selected = workspace?.versions.find((version) => version.digest === selectedDigest) ?? null;
  const reviewDimensions = useMemo(
    () => review?.assessment.checks.filter((check) => check.status === 'review').map((check) => check.dimension) ?? [],
    [review]
  );
  const [showEvidence, setShowEvidence] = useState(false);

  return (
    <div className="versions-view">
      <section className="view-header versions-view__header">
        <div>
          <h1>Versions</h1>
          <p>Review the exact meaning, evidence, and promotion history of Remembero changes.</p>
        </div>
        <button className="primary-button" type="button" onClick={onCapture} disabled={loading}>
          <DatabaseIcon size={18} />
          {loading ? 'Capturing…' : 'Capture current state'}
        </button>
      </section>

      {error ? <p className="inline-error">{error}</p> : null}

      <section className="version-ref-strip" aria-label="Semantic refs">
        {(workspace?.refs ?? []).map((ref) => (
          <div key={ref.name} className="version-ref">
            <GitBranchIcon size={17} />
            <span>{ref.name}</span>
            <code>{shortDigest(ref.versionDigest)}</code>
          </div>
        ))}
        {(workspace?.refs ?? []).length === 0 ? <span className="empty-panel">No semantic refs yet.</span> : null}
      </section>

      <section className="version-board" aria-label="Semantic version board">
        {columns.map((column) => {
          const versions = (workspace?.versions ?? []).filter((version) => version.status === column.status);
          return (
            <section className="version-column" key={column.status}>
              <header><span className="stage-dot" />{column.label}<span className="count">{versions.length}</span></header>
              <div className="version-column__list">
                {versions.map((version) => (
                  <VersionCard
                    key={version.digest}
                    version={version}
                    selected={version.digest === selectedDigest}
                    onSelect={() => onSelect(version.digest)}
                  />
                ))}
                {versions.length === 0 ? <p className="empty-column">No versions.</p> : null}
              </div>
            </section>
          );
        })}
      </section>

      {selected ? (
        <section className="version-detail panel-section">
          <div className="panel-section__header">
            <div>
              <h2>{selected.labels[0] ?? 'Semantic version'}</h2>
              <code>{selected.digest}</code>
            </div>
            <span className="pill">{selected.status}</span>
          </div>
          <div className="version-detail__metrics">
            <span>{selected.memberKeys.length} members</span>
            <span>{selected.edgeCount} typed edges</span>
            <span>{selected.contractCount} contracts</span>
            <span>{new Date(selected.createdAt).toLocaleString('en-AU')}</span>
          </div>
          <div className="version-detail__members">
            {selected.memberKeys.map((member) => <span key={member} className="pill">{member}</span>)}
          </div>
          <div className="drawer-form__actions">
            <button className="secondary-button" type="button" onClick={onReview} disabled={loading}>
              <RefreshIcon size={18} />
              {loading ? 'Reviewing…' : 'Run deterministic review'}
            </button>
            {review && review.candidateVersionDigest === selected.digest && reviewDimensions.length > 0 ? (
              <button className="primary-button" type="button" onClick={onPromote} disabled={loading || review.assessment.checks.some((check) => check.status === 'fail' || check.status === 'blocked')}>
                <CheckCircleIcon size={18} />
                Accept review and promote
              </button>
            ) : null}
          </div>
          {review && review.candidateVersionDigest === selected.digest ? (
            <div className="version-review">
              <h3>Compatibility review</h3>
              {review.assessment.checks.map((check) => (
                <div key={check.dimension} className="version-check">
                  <span className={`pill${check.status === 'fail' || check.status === 'blocked' ? ' pill--amber' : ''}`}>{check.status}</span>
                  <div><strong>{check.dimension}</strong><p>{check.summary}</p></div>
                </div>
              ))}
              <button className="text-button" type="button" onClick={() => setShowEvidence((value) => !value)}>
                {showEvidence ? 'Hide evidence' : 'Show evidence'} ({review.evidence.length})
              </button>
              {showEvidence ? review.evidence.map((evidence) => (
                <div key={evidence.digest} className="version-evidence">
                  <strong>{evidence.kind}</strong><span>{evidence.status} · {evidence.evaluator ?? 'deterministic'}</span><code>{shortDigest(evidence.digest)}</code>
                </div>
              )) : null}
            </div>
          ) : <p className="muted-copy">Run the review to attach deterministic evidence and compatibility checks.</p>}
        </section>
      ) : null}
    </div>
  );
}

