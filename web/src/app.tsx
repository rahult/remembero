import { startTransition, useDeferredValue, useEffect, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import type {
  AskPreset,
  AskResponse,
  BootstrapResponse,
  DocumentListItem,
  DocumentProofResult,
  DocumentShowcaseResponse,
  GraphResponse,
  NavigationView,
  SearchKind,
  SearchResponse,
} from './api';
import {
  ApiError,
  askDocumentQuestion,
  createMemory,
  getDocument,
  getBootstrap,
  getGraph,
  parseDocument,
  seedDemo,
  searchKnowledge,
  askMemory,
} from './api';
import {
  AskIcon,
  ArrowUpRightIcon,
  ChevronDownIcon,
  ChevronRightIcon,
  CloseIcon,
  DatabaseIcon,
  DocumentIcon,
  FolderIcon,
  GearIcon,
  GraphIcon,
  KnowledgeIcon,
  MenuIcon,
  PlusIcon,
  RulesIcon,
  SearchLineIcon,
  ShieldIcon,
  SparkIcon,
} from './icons';
import { AskView } from './components/ask-view';
import { GraphView } from './components/graph-view';
import { GraphCanvas } from './components/graph-canvas';
import { KnowledgeView } from './components/knowledge-view';
import { RulesView } from './components/rules-view';
import { DocumentView } from './components/document-view';

type DocumentEvidenceMode = 'regions' | 'claims';

interface DocumentSelectionState {
  questionId: string | null;
  pageId: string | null;
  regionId: string | null;
  claimId: string | null;
  sourceId: string | null;
  evidenceMode: DocumentEvidenceMode;
}

const EMPTY_DOCUMENT_SELECTION: DocumentSelectionState = {
  questionId: null,
  pageId: null,
  regionId: null,
  claimId: null,
  sourceId: null,
  evidenceMode: 'regions',
};

const NAV_ITEMS: Array<{
  id: NavigationView;
  label: string;
  icon: typeof AskIcon;
}> = [
  { id: 'ask', label: 'Ask', icon: AskIcon },
  { id: 'documents', label: 'Documents', icon: DocumentIcon },
  { id: 'knowledge', label: 'Knowledge', icon: KnowledgeIcon },
  { id: 'graph', label: 'Graph', icon: GraphIcon },
  { id: 'rules', label: 'Rules', icon: RulesIcon },
];

function firstDocumentPageId(data: DocumentShowcaseResponse | null): string | null {
  return data?.document.pages[0]?.id ?? null;
}

function firstRegionForPage(
  data: DocumentShowcaseResponse | null,
  pageId: string | null
): string | null {
  if (data === null || pageId === null) return null;
  return data.document.pages.find((page) => page.id === pageId)?.regions[0]?.id ?? null;
}

function pageIdForRegion(
  data: DocumentShowcaseResponse | null,
  regionId: string | null
): string | null {
  if (data === null || regionId === null) return null;
  for (const page of data.document.pages) {
    if (page.regions.some((region) => region.id === regionId)) return page.id;
  }
  return null;
}

function claimIdForRegion(
  data: DocumentShowcaseResponse | null,
  regionId: string | null
): string | null {
  if (data === null || regionId === null) return null;
  return data.document.claims.find((claim) => claim.regionId === regionId)?.id ?? null;
}

function claimPageId(
  data: DocumentShowcaseResponse | null,
  claimId: string | null
): string | null {
  if (data === null || claimId === null) return null;
  return data.document.claims.find((claim) => claim.id === claimId)?.pageId ?? null;
}

function claimRegionId(
  data: DocumentShowcaseResponse | null,
  claimId: string | null
): string | null {
  if (data === null || claimId === null) return null;
  return data.document.claims.find((claim) => claim.id === claimId)?.regionId ?? null;
}

function firstGroundedTrace(proof: DocumentProofResult): { id: string; regionId: string } | null {
  for (const step of proof.steps) {
    if (step.regionId !== undefined) return { id: step.id, regionId: step.regionId };
  }
  return null;
}

function firstRelatedAnchor(proof: DocumentProofResult): { regionId: string } | null {
  const evidence = proof.relatedEvidence.find((item) => item.regionId !== undefined);
  return evidence?.regionId === undefined ? null : { regionId: evidence.regionId };
}

function traceById(
  data: DocumentShowcaseResponse | null,
  sourceId: string | null
) {
  if (data === null || sourceId === null) return null;
  return data.proof.steps.find((step) => step.id === sourceId) ?? null;
}

function claimIdForTrace(
  data: DocumentShowcaseResponse | null,
  sourceId: string | null
): string | null {
  const trace = traceById(data, sourceId);
  if (trace?.regionId === undefined) return null;
  const normalizedClause = trace.clause?.replace(/\.$/, '');
  return (
    data?.document.claims.find(
      (claim) =>
        claim.regionId === trace.regionId &&
        claim.clause.replace(/\.$/, '') === normalizedClause
    )?.id ??
    claimIdForRegion(data, trace.regionId)
  );
}

function questionIdForSnapshot(data: DocumentShowcaseResponse): string {
  return data.proof.questionId || data.defaultQuestionId;
}

function selectionFromSnapshot(data: DocumentShowcaseResponse): DocumentSelectionState {
  const grounded = firstGroundedTrace(data.proof);
  const related = grounded === null ? firstRelatedAnchor(data.proof) : null;
  const pageId =
    grounded === null
      ? related === null
        ? firstDocumentPageId(data)
        : pageIdForRegion(data, related.regionId)
      : pageIdForRegion(data, grounded.regionId);
  const regionId =
    grounded?.regionId ??
    related?.regionId ??
    firstRegionForPage(data, pageId);
  const claimId =
    grounded === null
      ? claimIdForRegion(data, regionId)
      : (claimIdForTrace(data, grounded.id) ?? claimIdForRegion(data, grounded.regionId));

  return {
    questionId: questionIdForSnapshot(data),
    pageId,
    regionId,
    claimId,
    sourceId: grounded?.id ?? null,
    evidenceMode: grounded === null ? 'regions' : 'claims',
  };
}

function normalizeError(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  if (error instanceof Error) return error.message;
  return 'An unexpected error occurred.';
}

function useMobileViewport(): boolean {
  const [mobile, setMobile] = useState(() =>
    typeof window !== 'undefined' ? window.innerWidth < 960 : false
  );

  useEffect(() => {
    const onResize = () => setMobile(window.innerWidth < 960);
    window.addEventListener('resize', onResize, { passive: true });
    return () => window.removeEventListener('resize', onResize);
  }, []);

  return mobile;
}

function RightRail({
  bootstrap,
  graph,
  onSelectNode,
}: {
  bootstrap: BootstrapResponse | null;
  graph: GraphResponse | null;
  onSelectNode: (label: string) => void;
}) {
  const memoryPulse = bootstrap?.memoryPulse;
  const currentGraph = graph?.graph ?? bootstrap?.graph ?? {
    focus: null,
    nodes: [],
    links: [],
    relationships: [],
  };

  return (
    <div className="right-rail">
      <section className="rail-card">
        <div className="panel-section__header">
          <h3>Memory pulse</h3>
        </div>
        {memoryPulse ? (
          <>
            <div className="pulse-grid">
              <div>
                <DatabaseIcon size={28} />
                <strong>{memoryPulse.factCount}</strong>
                <span>facts</span>
              </div>
              <div>
                <RulesIcon size={28} />
                <strong>{memoryPulse.ruleCount}</strong>
                <span>rules</span>
              </div>
              <div>
                <ShieldIcon size={28} />
                <strong>{memoryPulse.sourceCoveragePercent}%</strong>
                <span>sourced</span>
              </div>
            </div>
            <div className="health-row">
              <span>Health</span>
              <strong>
                {memoryPulse.healthLabel}
                <span className={`status-dot status-dot--${memoryPulse.healthTone}`} />
              </strong>
            </div>
          </>
        ) : (
          <p className="muted-copy">Waiting for bootstrap health data.</p>
        )}
      </section>
      <section className="rail-card">
        <div className="panel-section__header">
          <h3>Knowledge graph</h3>
          <ArrowUpRightIcon size={18} />
        </div>
        <GraphCanvas graph={currentGraph} compact onSelectNode={onSelectNode} />
        <p className="rail-note">Click a node to explore related facts.</p>
      </section>
    </div>
  );
}

function MemoryDrawer({
  open,
  status,
  error,
  onClose,
  onSubmit,
}: {
  open: boolean;
  status: 'idle' | 'saving';
  error: string | null;
  onClose: () => void;
  onSubmit: (payload: {
    subject: string;
    predicate: string;
    object: string;
    sourceText: string;
  }) => Promise<void>;
}) {
  const [subject, setSubject] = useState('');
  const [predicate, setPredicate] = useState('');
  const [object, setObject] = useState('');
  const [sourceText, setSourceText] = useState('');
  const firstFieldRef = useRef<HTMLInputElement | null>(null);
  const returnFocusRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!open) return;
    returnFocusRef.current = document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null;
    firstFieldRef.current?.focus();
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    document.body.style.overflow = 'hidden';
    window.addEventListener('keydown', onKeyDown);
    return () => {
      document.body.style.overflow = '';
      window.removeEventListener('keydown', onKeyDown);
      returnFocusRef.current?.focus();
    };
  }, [onClose, open]);

  if (!open) return null;

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    await onSubmit({ subject, predicate, object, sourceText });
    setSubject('');
    setPredicate('');
    setObject('');
    setSourceText('');
  }

  return (
    <div className="drawer-shell" role="presentation">
      <button
        className="drawer-shell__scrim"
        type="button"
        aria-label="Close add memory drawer"
        onClick={onClose}
      />
      <section
        className="drawer-panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby="memory-drawer-title"
      >
        <div className="drawer-panel__header">
          <div>
            <p className="drawer-panel__kicker">Structured capture</p>
            <h2 id="memory-drawer-title">Add memory</h2>
          </div>
          <button className="icon-button icon-button--ghost" type="button" onClick={onClose}>
            <CloseIcon size={22} />
          </button>
        </div>
        <form className="drawer-form" onSubmit={handleSubmit}>
          <label>
            <span>Subject</span>
            <input
              ref={firstFieldRef}
              value={subject}
              onChange={(event) => setSubject(event.target.value)}
              placeholder="maya"
              required
            />
          </label>
          <label>
            <span>Predicate</span>
            <input
              value={predicate}
              onChange={(event) => setPredicate(event.target.value)}
              placeholder="project_contributor"
              required
            />
          </label>
          <label>
            <span>Object</span>
            <input
              value={object}
              onChange={(event) => setObject(event.target.value)}
              placeholder="atlas"
              required
            />
          </label>
          <label>
            <span>Source text</span>
            <textarea
              value={sourceText}
              onChange={(event) => setSourceText(event.target.value)}
              placeholder="Atlas planning session noted Maya as a project contributor."
              rows={5}
              required
            />
          </label>
          {error ? <p className="inline-error">{error}</p> : null}
          <div className="drawer-form__actions">
            <button className="secondary-button" type="button" onClick={onClose}>
              Cancel
            </button>
            <button className="primary-button" type="submit" disabled={status === 'saving'}>
              <PlusIcon size={18} />
              {status === 'saving' ? 'Saving…' : 'Store memory'}
            </button>
          </div>
        </form>
      </section>
    </div>
  );
}

export function App() {
  const mobile = useMobileViewport();
  const [activeView, setActiveView] = useState<NavigationView>('ask');
  const [bootstrap, setBootstrap] = useState<BootstrapResponse | null>(null);
  const [bootstrapError, setBootstrapError] = useState<string | null>(null);
  const [bootstrapLoading, setBootstrapLoading] = useState(true);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const [question, setQuestion] = useState('Who is collaborating on Atlas?');
  const [selectedPresetId, setSelectedPresetId] = useState<string | null>(null);
  const [askResult, setAskResult] = useState<AskResponse | null>(null);
  const [askLoading, setAskLoading] = useState(false);
  const [askError, setAskError] = useState<string | null>(null);

  const [knowledgeQuery, setKnowledgeQuery] = useState('');
  const [knowledgeKind, setKnowledgeKind] = useState<SearchKind | 'all'>('all');
  const [knowledgeResult, setKnowledgeResult] = useState<SearchResponse | null>(null);
  const [knowledgeLoading, setKnowledgeLoading] = useState(false);
  const [knowledgeError, setKnowledgeError] = useState<string | null>(null);

  const [graphResponse, setGraphResponse] = useState<GraphResponse | null>(null);
  const [graphFocusInput, setGraphFocusInput] = useState('');
  const [graphLoading, setGraphLoading] = useState(false);
  const [graphError, setGraphError] = useState<string | null>(null);

  const [drawerOpen, setDrawerOpen] = useState(false);
  const [memoryStatus, setMemoryStatus] = useState<'idle' | 'saving'>('idle');
  const [memoryError, setMemoryError] = useState<string | null>(null);
  const [documentCatalog, setDocumentCatalog] = useState<DocumentListItem[]>([]);
  const [documentData, setDocumentData] = useState<DocumentShowcaseResponse | null>(null);
  const [documentLoading, setDocumentLoading] = useState(false);
  const [documentError, setDocumentError] = useState<string | null>(null);
  const [documentPendingId, setDocumentPendingId] = useState<string | null>(null);
  const [documentAction, setDocumentAction] = useState<'idle' | 'asking' | 'parsing'>('idle');
  const [documentSelection, setDocumentSelection] =
    useState<DocumentSelectionState>(EMPTY_DOCUMENT_SELECTION);

  const deferredKnowledgeQuery = useDeferredValue(knowledgeQuery);

  async function loadBootstrap() {
    setBootstrapLoading(true);
    try {
      const data = await getBootstrap();
      startTransition(() => {
        setBootstrap(data);
        setBootstrapError(null);
        setGraphResponse((current) =>
          current ?? {
            focus: data.graph.focus,
            graph: data.graph,
          }
        );
        if (graphFocusInput.length === 0 && data.graph.focus) {
          setGraphFocusInput(data.graph.focus);
        }
      });
    } catch (error) {
      startTransition(() => {
        setBootstrapError(normalizeError(error));
      });
    } finally {
      setBootstrapLoading(false);
    }
  }

  useEffect(() => {
    void loadBootstrap();
  }, []);

  useEffect(() => {
    if (
      activeView !== 'documents' ||
      (documentData !== null && documentCatalog.length > 0) ||
      documentLoading
    ) {
      return;
    }
    void loadDocument();
  }, [activeView, documentCatalog.length, documentData, documentLoading]);

  useEffect(() => {
    if (activeView !== 'knowledge') return;
    const text = deferredKnowledgeQuery.trim();
    if (text.length === 0) {
      setKnowledgeLoading(false);
      setKnowledgeError(null);
      setKnowledgeResult(null);
      return;
    }

    const handle = window.setTimeout(async () => {
      setKnowledgeLoading(true);
      try {
        const result = await searchKnowledge({
          text,
          ...(knowledgeKind === 'all' ? {} : { kinds: [knowledgeKind] }),
        });
        startTransition(() => {
          setKnowledgeResult(result);
          setKnowledgeError(null);
        });
      } catch (error) {
        startTransition(() => {
          setKnowledgeError(normalizeError(error));
        });
      } finally {
        setKnowledgeLoading(false);
      }
    }, 220);

    return () => window.clearTimeout(handle);
  }, [activeView, deferredKnowledgeQuery, knowledgeKind]);

  async function handleAsk() {
    if (question.trim().length === 0) return;
    setAskLoading(true);
    try {
      const result = await askMemory({
        question: question.trim(),
        ...(selectedPresetId === null ? {} : { presetId: selectedPresetId }),
      });
      startTransition(() => {
        setAskResult(result);
        setAskError(null);
        if (result.graph.nodes.length > 0) {
          setGraphResponse({
            focus: result.graph.focus,
            graph: result.graph,
          });
          if (result.graph.focus) setGraphFocusInput(result.graph.focus);
        }
      });
    } catch (error) {
      startTransition(() => {
        setAskError(normalizeError(error));
      });
    } finally {
      setAskLoading(false);
    }
  }

  async function handleGraphExplore(focusValue?: string) {
    const focus = (focusValue ?? graphFocusInput).trim();
    setGraphLoading(true);
    try {
      const result = await getGraph(focus.length > 0 ? focus : undefined);
      startTransition(() => {
        setGraphResponse(result);
        setGraphError(null);
        if (result.focus) setGraphFocusInput(result.focus);
      });
    } catch (error) {
      startTransition(() => {
        setGraphError(normalizeError(error));
      });
    } finally {
      setGraphLoading(false);
    }
  }

  async function handleSeed() {
    try {
      await seedDemo();
      await loadBootstrap();
    } catch (error) {
      setBootstrapError(normalizeError(error));
    }
  }

  async function loadDocument(documentId?: string) {
    setDocumentLoading(true);
    setDocumentPendingId(documentId ?? null);
    try {
      const data = await getDocument(documentId);
      startTransition(() => {
        setDocumentCatalog(data.documents);
        setDocumentData(data);
        setDocumentError(null);
        setDocumentSelection(selectionFromSnapshot(data));
      });
    } catch (error) {
      startTransition(() => {
        setDocumentError(normalizeError(error));
      });
    } finally {
      setDocumentPendingId(null);
      setDocumentLoading(false);
    }
  }

  async function handleParseDocument() {
    if (documentData === null) return;
    setDocumentAction('parsing');
    try {
      const result = await parseDocument({ documentId: documentData.document.id });
      startTransition(() => {
        setDocumentCatalog(result.documents);
        setDocumentData(result);
        setDocumentSelection(selectionFromSnapshot(result));
        setDocumentError(null);
      });
    } catch (error) {
      startTransition(() => {
        setDocumentError(normalizeError(error));
      });
    } finally {
      setDocumentAction('idle');
    }
  }

  async function handleAskDocumentQuestion() {
    if (documentData === null) return;
    const questionId = documentSelection.questionId ?? documentData.defaultQuestionId;
    setDocumentAction('asking');
    try {
      const proof = await askDocumentQuestion({
        documentId: documentData.document.id,
        questionId,
      });
      const nextData: DocumentShowcaseResponse = { ...documentData, proof };
      startTransition(() => {
        setDocumentData(nextData);
        setDocumentError(null);
        setDocumentSelection(selectionFromSnapshot(nextData));
      });
    } catch (error) {
      startTransition(() => {
        setDocumentError(normalizeError(error));
      });
    } finally {
      setDocumentAction('idle');
    }
  }

  async function handleMemorySubmit(payload: {
    subject: string;
    predicate: string;
    object: string;
    sourceText: string;
  }) {
    setMemoryStatus('saving');
    try {
      await createMemory(payload);
      startTransition(() => {
        setDrawerOpen(false);
        setMemoryError(null);
      });
      await loadBootstrap();
    } catch (error) {
      startTransition(() => {
        setMemoryError(normalizeError(error));
      });
      return;
    } finally {
      setMemoryStatus('idle');
    }
  }

  function selectPreset(preset: AskPreset) {
    setQuestion(preset.question);
    setSelectedPresetId(preset.id);
  }

  function selectGraphNode(label: string) {
    setGraphFocusInput(label);
    setActiveView('graph');
    void handleGraphExplore(label);
  }

  function navigateTo(view: NavigationView) {
    setActiveView(view);
    setMobileMenuOpen(false);
    if (view === 'documents' && documentCatalog.length === 0 && documentData === null && !documentLoading) {
      void loadDocument();
    }
    if (view === 'graph') {
      void handleGraphExplore(graphFocusInput.trim() || 'atlas');
    }
  }

  function selectDocument(documentId: string) {
    if (documentLoading || documentPendingId === documentId || documentData?.document.id === documentId) {
      return;
    }
    void loadDocument(documentId);
  }

  function selectDocumentPage(pageId: string) {
    if (documentData === null) return;
    const firstRegionId = firstRegionForPage(documentData, pageId);
    const currentSource = traceById(documentData, documentSelection.sourceId);
    const keepSource =
      currentSource?.pageNumber ===
      documentData.document.pages.find((page) => page.id === pageId)?.pageNumber;
    const nextRegionId = keepSource ? currentSource?.regionId ?? firstRegionId : firstRegionId;
    startTransition(() => {
      setDocumentSelection((current) => ({
        ...current,
        pageId,
        regionId: nextRegionId,
        claimId:
          keepSource && currentSource !== null
            ? claimIdForTrace(documentData, current.sourceId) ??
              claimIdForRegion(documentData, nextRegionId)
            : claimPageId(documentData, current.claimId) === pageId
              ? current.claimId
              : claimIdForRegion(documentData, nextRegionId),
        sourceId: keepSource ? current.sourceId : null,
      }));
    });
  }

  function selectDocumentRegion(regionId: string) {
    if (documentData === null) return;
    startTransition(() => {
      setDocumentSelection((current) => ({
        ...current,
        pageId: pageIdForRegion(documentData, regionId),
        regionId,
        claimId: claimIdForRegion(documentData, regionId),
        sourceId:
          traceById(documentData, current.sourceId)?.regionId === regionId ? current.sourceId : null,
        evidenceMode: 'regions',
      }));
    });
  }

  function selectDocumentClaim(claimId: string) {
    if (documentData === null) return;
    const regionId = claimRegionId(documentData, claimId);
    startTransition(() => {
      setDocumentSelection((current) => ({
        ...current,
        claimId,
        pageId: pageIdForRegion(documentData, regionId),
        regionId,
        sourceId:
          documentData.proof.steps.find((step) => {
            if (step.regionId !== regionId) return false;
            const clause = documentData.document.claims.find((claim) => claim.id === claimId)?.clause;
            return step.clause?.replace(/\.$/, '') === clause?.replace(/\.$/, '');
          })?.id ?? null,
        evidenceMode: 'claims',
      }));
    });
  }

  function selectDocumentSource(sourceId: string) {
    if (documentData === null) return;
    const trace = traceById(documentData, sourceId);
    if (trace?.regionId === undefined) return;
    const regionId = trace.regionId;
    startTransition(() => {
      setDocumentSelection((current) => ({
        ...current,
        sourceId,
        pageId: pageIdForRegion(documentData, regionId),
        regionId,
        claimId: claimIdForTrace(documentData, sourceId) ?? claimIdForRegion(documentData, regionId),
        evidenceMode: 'claims',
      }));
    });
  }

  const topBar = mobile ? (
    <header className="mobile-topbar">
      <div className="brandmark">remembero</div>
      <button className="nav-button nav-button--profile" type="button">
        <FolderIcon size={24} />
        <span>{bootstrap?.profile.personaLabel ?? 'Personal'}</span>
        <ChevronDownIcon size={18} />
      </button>
      <button
        className="icon-button icon-button--light"
        type="button"
        aria-label="Open navigation"
        onClick={() => setMobileMenuOpen(true)}
      >
        <MenuIcon size={28} />
      </button>
    </header>
  ) : (
    <header className="desktop-topbar">
      <button className="nav-button nav-button--profile" type="button">
        <FolderIcon size={20} />
        <span>{bootstrap?.profile.personaLabel ?? 'Personal'}</span>
        <ChevronDownIcon size={16} />
      </button>
      <div className="desktop-topbar__actions">
        <button className="icon-button icon-button--ghost" type="button" aria-label="Search">
          <SearchLineIcon size={20} />
        </button>
        <button className="icon-button icon-button--ghost" type="button" aria-label="Signals">
          <SparkIcon size={20} />
        </button>
        <button className="icon-button icon-button--ghost" type="button" aria-label="Settings">
          <GearIcon size={20} />
        </button>
      </div>
    </header>
  );

  const navigation = (
    <nav className={mobile ? 'mobile-nav-sheet__nav' : 'sidebar__nav'} aria-label="Primary">
      {NAV_ITEMS.map((item) => {
        const Icon = item.icon;
        return (
          <button
            key={item.id}
            className={`nav-link${activeView === item.id ? ' nav-link--active' : ''}`}
            type="button"
            onClick={() => {
              navigateTo(item.id);
            }}
          >
            <Icon size={24} />
            <span>{item.label}</span>
          </button>
        );
      })}
    </nav>
  );

  return (
    <div className="app-shell">
      {!mobile ? (
        <aside className="sidebar">
          <div className="brandmark">remembero</div>
          {navigation}
          <div className="sidebar__footer">
            <button className="nav-button nav-button--footer" type="button">
              <FolderIcon size={20} />
              <span>{bootstrap?.profile.personaLabel ?? 'Personal'}</span>
              <ChevronDownIcon size={16} />
            </button>
            <div className="sidebar-status">
              <div>
                <DatabaseIcon size={20} />
                <span>{bootstrap?.profile.storageLabel ?? 'Local-first'}</span>
              </div>
              <span className="status-dot status-dot--healthy" />
            </div>
            <button className="nav-button nav-button--footer" type="button">
              <ChevronRightIcon size={16} />
              <span>Collapse</span>
            </button>
          </div>
        </aside>
      ) : null}

      <div className="workspace-shell">
        {topBar}
        {mobile && mobileMenuOpen ? (
          <div className="mobile-nav-sheet" role="dialog" aria-modal="true">
            <button
              className="mobile-nav-sheet__scrim"
              type="button"
              aria-label="Close navigation"
              onClick={() => setMobileMenuOpen(false)}
            />
            <div className="mobile-nav-sheet__panel">
              <div className="mobile-nav-sheet__header">
                <div className="brandmark">remembero</div>
                <button
                  className="icon-button icon-button--ghost"
                  type="button"
                  onClick={() => setMobileMenuOpen(false)}
                >
                  <CloseIcon size={22} />
                </button>
              </div>
              {navigation}
            </div>
          </div>
        ) : null}

        <div className={`workspace${!mobile && activeView === 'ask' ? ' workspace--with-rail' : ''}`}>
          <main className="workspace__main">
            {bootstrapLoading ? (
              <div className="loading-panel">
                <p>Loading local-first memory workspace…</p>
              </div>
            ) : bootstrapError ? (
              <div className="loading-panel">
                <p>{bootstrapError}</p>
                <button className="secondary-button" type="button" onClick={() => void loadBootstrap()}>
                  Retry bootstrap
                </button>
              </div>
            ) : null}

            {!bootstrapLoading && !bootstrapError ? (
              <>
                {activeView === 'ask' ? (
                  <AskView
                    bootstrap={bootstrap}
                    memoryPulse={bootstrap?.memoryPulse ?? null}
                    question={question}
                    selectedPresetId={selectedPresetId}
                    result={askResult}
                    loading={askLoading}
                    error={askError}
                    mobile={mobile}
                    onQuestionChange={(value) => {
                      setQuestion(value);
                      if (selectedPresetId !== null) setSelectedPresetId(null);
                    }}
                    onSelectPreset={selectPreset}
                    onAsk={() => void handleAsk()}
                    onOpenDrawer={() => {
                      setDrawerOpen(true);
                      setMemoryError(null);
                    }}
                    onOpenKnowledge={() => setActiveView('knowledge')}
                    onSeed={() => void handleSeed()}
                  />
                ) : null}

                {activeView === 'knowledge' ? (
                  <KnowledgeView
                    bootstrap={bootstrap}
                    query={knowledgeQuery}
                    kind={knowledgeKind}
                    result={knowledgeResult}
                    loading={knowledgeLoading}
                    error={knowledgeError}
                    onQueryChange={setKnowledgeQuery}
                    onKindChange={setKnowledgeKind}
                  />
                ) : null}

                {activeView === 'documents' ? (
                  <DocumentView
                    documents={documentCatalog}
                    data={documentData}
                    loading={documentLoading}
                    error={documentError}
                    pendingDocumentId={documentPendingId}
                    action={documentAction}
                    mobile={mobile}
                    evidenceMode={documentSelection.evidenceMode}
                    selectedQuestionId={documentSelection.questionId}
                    selectedPageId={documentSelection.pageId}
                    selectedRegionId={documentSelection.regionId}
                    selectedClaimId={documentSelection.claimId}
                    selectedSourceId={documentSelection.sourceId}
                    onSelectDocument={selectDocument}
                    onSelectEvidenceMode={(mode) =>
                      setDocumentSelection((current) => ({ ...current, evidenceMode: mode }))
                    }
                    onSelectQuestion={(questionId) =>
                      setDocumentSelection((current) => ({ ...current, questionId }))
                    }
                    onAskQuestion={() => void handleAskDocumentQuestion()}
                    onParse={() => void handleParseDocument()}
                    onRetry={() => void loadDocument()}
                    onSelectPage={selectDocumentPage}
                    onSelectRegion={selectDocumentRegion}
                    onSelectClaim={selectDocumentClaim}
                    onSelectSource={selectDocumentSource}
                  />
                ) : null}

                {activeView === 'graph' ? (
                  <GraphView
                    graphResponse={graphResponse}
                    focusInput={graphFocusInput}
                    loading={graphLoading}
                    error={graphError}
                    onFocusChange={setGraphFocusInput}
                    onExplore={() => void handleGraphExplore()}
                    onSelectNode={selectGraphNode}
                  />
                ) : null}

                {activeView === 'rules' ? (
                  <RulesView bootstrap={bootstrap} onSeed={() => void handleSeed()} />
                ) : null}
              </>
            ) : null}
          </main>

          {!mobile && activeView === 'ask' ? (
            <aside className="workspace__rail">
              <RightRail
                bootstrap={bootstrap}
                graph={graphResponse}
                onSelectNode={selectGraphNode}
              />
            </aside>
          ) : null}
        </div>

        {mobile ? (
          <nav className="bottom-nav" aria-label="Bottom navigation">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              return (
                <button
                  key={item.id}
                  className={`bottom-nav__link${activeView === item.id ? ' bottom-nav__link--active' : ''}`}
                  type="button"
                  onClick={() => navigateTo(item.id)}
                >
                  <Icon size={26} />
                  <span>{item.label}</span>
                </button>
              );
            })}
          </nav>
        ) : null}
      </div>

      <MemoryDrawer
        open={drawerOpen}
        status={memoryStatus}
        error={memoryError}
        onClose={() => setDrawerOpen(false)}
        onSubmit={handleMemorySubmit}
      />
    </div>
  );
}
