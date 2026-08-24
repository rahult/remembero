"use client";

/* eslint-disable @next/next/no-html-link-for-pages -- GitHub Pages needs document navigation, not RSC prefetch. */

import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent,
  type PointerEvent as ReactPointerEvent,
} from "react";
import {
  CONSTRAINT_EXAMPLE,
  DEFAULT_SQL,
  INITIAL_LINEAGE,
  INSERT_DEFAULTS,
  PRESETS,
  SAMPLE_SETUP_SQL,
  eventTime,
  formatCell,
  quoteIdentifier,
  type DemoPreset,
  type LineageEvent,
} from "../lib/ide-demo";
import {
  BrowserDatalogDatabase,
  openBrowserDatalogDatabase,
  type BrowserDatalogExplanation,
  type BrowserDatalogProof,
  type BrowserSqliteRuntimeInfo,
  type SqliteRow,
  type SqliteScalar,
} from "../lib/sqlite-wasm";
import {
  CheckIcon,
  ChevronIcon,
  CloseIcon,
  DatabaseIcon,
  PlayIcon,
  ProofIcon,
  ResetIcon,
} from "./ide-icons";
import { ProofGraph } from "./proof-graph";

interface TableColumn {
  name: string;
  type: string;
  notNull: boolean;
  primaryKey: boolean;
}

interface TableSnapshot {
  name: string;
  columns: TableColumn[];
  rows: SqliteRow[];
}

interface Evaluation {
  compiledSql: string;
  rows: Array<Record<string, string | number>>;
  explanations: BrowserDatalogExplanation[];
}

interface RuntimeMetrics {
  bootMs: number | null;
  ruleMs: number | null;
  insertMs: number | null;
  sqlMs: number | null;
}

interface WorkspaceLayout {
  schema: number;
  inspector: number;
}

interface ResultColumnConfig {
  visible: boolean;
  width: number;
}

type WorkspacePane = keyof WorkspaceLayout;

type MobilePane = "data" | "query" | "proof" | "graph";

const github = "https://github.com/rahult/remembero";

function stringValue(value: SqliteScalar | undefined): string {
  return value === undefined || value === null ? "" : String(value);
}

function timestampedEvent(
  kind: LineageEvent["kind"],
  target: string,
  detail: string,
): LineageEvent {
  return {
    id: `${kind}-${target}-${Date.now()}-${Math.random().toString(16).slice(2)}`,
    kind,
    target,
    detail,
    timestamp: eventTime(),
  };
}

function describeError(value: unknown): string {
  return value instanceof Error ? value.message : String(value);
}

async function readTables(database: BrowserDatalogDatabase): Promise<TableSnapshot[]> {
  const names = await database.exec(`
    SELECT name
      FROM sqlite_master
     WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
     ORDER BY name
  `);
  return Promise.all(
    names.flatMap((row) =>
      typeof row.name !== "string"
        ? []
        : [
            (async (): Promise<TableSnapshot> => {
              const name = row.name as string;
              const [columnRows, dataRows] = await Promise.all([
                database.exec(`PRAGMA table_info(${quoteIdentifier(name)})`),
                database.exec(`SELECT * FROM ${quoteIdentifier(name)} ORDER BY rowid`),
              ]);
              return {
                name,
                columns: columnRows.map((column) => ({
                  name: stringValue(column.name),
                  type: stringValue(column.type) || "ANY",
                  notNull: column.notnull === 1,
                  primaryKey: column.pk === 1,
                })),
                rows: dataRows,
              };
            })(),
          ],
    ),
  );
}

async function evaluateProgram(
  database: BrowserDatalogDatabase,
  program: string,
): Promise<Evaluation> {
  const [rows, explanations] = await Promise.all([
    database.datalogQuery(program),
    database.datalogExplain(program),
  ]);
  let compiledSql: string;
  try {
    compiledSql = await database.datalogSql(program);
  } catch {
    compiledSql =
      "Recursive programs run in Remembero's native fixpoint evaluator; there is no single compiled SQL statement.";
  }
  return { compiledSql, rows, explanations };
}

function collectFactProofs(
  proof: BrowserDatalogProof | null,
  output: BrowserDatalogProof[] = [],
): BrowserDatalogProof[] {
  if (!proof) return output;
  if (proof.rule === undefined) output.push(proof);
  for (const child of proof.because ?? []) collectFactProofs(child, output);
  return output;
}

function displayAnswer(preset: DemoPreset, row: Record<string, string | number>): string {
  switch (preset.id) {
    case "follow_up":
      return `${row.Person} needs a follow-up on ${row.Project}`;
    case "collaborators":
      return `${row.Person} collaborates on ${row.Project}`;
    case "recursive_paths":
      return `${row.X} can reach ${row.Y}`;
    case "support_escalation":
      return `${row.Customer} needs escalation for ${row.Ticket}`;
    case "release_readiness":
      return `${row.Service} is ready to ship`;
    case "access_control":
      return `${row.Person} can read ${row.Document}`;
  }
}

function editorPosition(value: string, offset: number): { line: number; column: number } {
  const before = value.slice(0, offset).split("\n");
  return { line: before.length, column: before.at(-1)!.length + 1 };
}

function programHead(value: string): string {
  return value.trim().match(/^([a-z][a-z0-9_]*)\s*\(/)?.[1] ?? "query";
}

function formatMetric(value: number | null): string {
  if (value === null) return "—";
  const precision = value < 10 ? 2 : 1;
  return `${Math.max(value, 0.01).toFixed(precision)} ms`;
}

export function SqliteIde() {
  const databaseRef = useRef<BrowserDatalogDatabase | null>(null);
  const operationRef = useRef(0);
  const [runtime, setRuntime] = useState<BrowserSqliteRuntimeInfo | null>(null);
  const [phase, setPhase] = useState<"loading" | "ready" | "running" | "error">(
    "loading",
  );
  const [error, setError] = useState<string | null>(null);
  const [tables, setTables] = useState<TableSnapshot[]>([]);
  const [selectedTable, setSelectedTable] = useState("status");
  const [insertTable, setInsertTable] = useState("status");
  const [insertValues, setInsertValues] = useState<Record<string, SqliteScalar>>(
    () => ({ ...INSERT_DEFAULTS.status }),
  );
  const [sqlText, setSqlText] = useState(DEFAULT_SQL);
  const [sqlRows, setSqlRows] = useState<SqliteRow[]>([]);
  const [preset, setPreset] = useState<DemoPreset>(PRESETS[0]);
  const [program, setProgram] = useState(PRESETS[0].program);
  const [evaluation, setEvaluation] = useState<Evaluation>({
    compiledSql: "",
    rows: [],
    explanations: [],
  });
  const [selectedResult, setSelectedResult] = useState(0);
  const [selectedProofId, setSelectedProofId] = useState<string | null>("proof");
  const [lineage, setLineage] = useState<LineageEvent[]>(() => [
    ...INITIAL_LINEAGE,
  ]);
  const [guidanceOpen, setGuidanceOpen] = useState(true);
  const [proofVisited, setProofVisited] = useState(false);
  const [mobilePane, setMobilePane] = useState<MobilePane>("data");
  const [cursor, setCursor] = useState({ line: 1, column: 1 });
  const [metrics, setMetrics] = useState<RuntimeMetrics>({
    bootMs: null,
    ruleMs: null,
    insertMs: null,
    sqlMs: null,
  });
  const [workspaceLayout, setWorkspaceLayout] = useState<WorkspaceLayout>({
    schema: 248,
    inspector: 340,
  });
  const [schemaVisible, setSchemaVisible] = useState(true);
  const [inspectorVisible, setInspectorVisible] = useState(true);
  const [resultColumnConfig, setResultColumnConfig] = useState<
    Record<string, ResultColumnConfig>
  >({});
  const workspaceResizeRef = useRef<{
    pane: WorkspacePane;
    startX: number;
    startWidth: number;
  } | null>(null);
  const resultResizeRef = useRef<{
    column: string;
    startX: number;
    startWidth: number;
  } | null>(null);

  const selectedSnapshot = useMemo(
    () => tables.find((table) => table.name === selectedTable) ?? tables[0] ?? null,
    [selectedTable, tables],
  );
  const insertSnapshot = useMemo(
    () => tables.find((table) => table.name === insertTable) ?? null,
    [insertTable, tables],
  );
  const selectedExplanation = evaluation.explanations[selectedResult] ?? null;
  const selectedRow = evaluation.rows[selectedResult] ?? null;
  const factProofs = useMemo(
    () => collectFactProofs(selectedExplanation?.proof ?? null),
    [selectedExplanation],
  );

  const applyEvaluation = (next: Evaluation) => {
    setEvaluation(next);
    setSelectedResult(0);
    setSelectedProofId("proof");
  };

  useEffect(() => {
    let active = true;
    let opened: BrowserDatalogDatabase | null = null;

    const boot = async () => {
      const bootOperation = operationRef.current;
      try {
        const bootStarted = performance.now();
        opened = await openBrowserDatalogDatabase();
        const bootMs = performance.now() - bootStarted;
        if (!active) {
          await opened.close();
          return;
        }
        databaseRef.current = opened;
        await opened.exec(SAMPLE_SETUP_SQL);
        const initialRule = (async () => {
          const started = performance.now();
          const value = await evaluateProgram(opened!, PRESETS[0].program);
          return { value, durationMs: performance.now() - started };
        })();
        const [nextTables, nextSqlRows, nextEvaluation] = await Promise.all([
          readTables(opened),
          opened.exec(DEFAULT_SQL),
          initialRule,
        ]);
        if (!active || operationRef.current !== bootOperation) return;
        setRuntime(opened.runtime);
        setTables(nextTables);
        setSqlRows(nextSqlRows);
        applyEvaluation(nextEvaluation.value);
        setMetrics({
          bootMs,
          ruleMs: nextEvaluation.durationMs,
          insertMs: null,
          sqlMs: null,
        });
        setLineage((current) => [
          ...current,
          timestampedEvent(
            "DATALOG",
            "needs_follow_up",
            `2 derived rows · ${formatMetric(nextEvaluation.durationMs)}`,
          ),
        ]);
        setPhase("ready");
      } catch (value) {
        if (!active) return;
        setError(describeError(value));
        setPhase("error");
      }
    };

    void boot();
    return () => {
      active = false;
      databaseRef.current = null;
      if (opened) void opened.close();
    };
  }, []);

  const refreshTables = async (database: BrowserDatalogDatabase) => {
    const next = await readTables(database);
    setTables(next);
    return next;
  };

  const runProgram = async (
    database: BrowserDatalogDatabase,
    source: string,
    moveToProof = false,
  ) => {
    setPhase("running");
    setError(null);
    try {
      const started = performance.now();
      const next = await evaluateProgram(database, source);
      const durationMs = performance.now() - started;
      applyEvaluation(next);
      setMetrics((current) => ({ ...current, ruleMs: durationMs }));
      setLineage((current) => [
        ...current,
        timestampedEvent(
          "DATALOG",
          programHead(source),
          `${next.rows.length} derived rows · ${formatMetric(durationMs)}`,
        ),
      ]);
      setPhase("ready");
      if (moveToProof) setMobilePane("proof");
    } catch (value) {
      setError(describeError(value));
      setPhase("error");
    }
  };

  const runSql = async () => {
    const database = databaseRef.current;
    if (!database) return;
    setPhase("running");
    setError(null);
    try {
      const started = performance.now();
      const rows = await database.exec(sqlText);
      setSqlRows(rows);
      await refreshTables(database);
      const durationMs = performance.now() - started;
      setMetrics((current) => ({ ...current, sqlMs: durationMs }));
      setLineage((current) => [
        ...current,
        timestampedEvent(
          "SQL",
          "scratchpad",
          `${rows.length ? `${rows.length} rows` : "statement complete"} · ${formatMetric(durationMs)}`,
        ),
      ]);
      await runProgram(database, program);
    } catch (value) {
      setError(describeError(value));
      setPhase("error");
    }
  };

  const runDatalog = async () => {
    const database = databaseRef.current;
    if (!database) return;
    await runProgram(database, program, true);
  };

  const insertRow = async () => {
    const database = databaseRef.current;
    if (!database || !insertSnapshot) return;
    const columns = insertSnapshot.columns.map((column) => column.name);
    setPhase("running");
    setError(null);
    try {
      const started = performance.now();
      await database.exec(
        `INSERT INTO ${quoteIdentifier(insertSnapshot.name)} (${columns
          .map(quoteIdentifier)
          .join(", ")}) VALUES (${columns.map(() => "?").join(", ")})`,
        columns.map((column) => insertValues[column] ?? null),
      );
      await refreshTables(database);
      const durationMs = performance.now() - started;
      setMetrics((current) => ({ ...current, insertMs: durationMs }));
      setSelectedTable(insertSnapshot.name);
      setLineage((current) => [
        ...current,
        timestampedEvent(
          "INSERT",
          insertSnapshot.name,
          `${columns
            .map((column) => formatCell(insertValues[column] ?? null))
            .join(", ")} · ${formatMetric(durationMs)}`,
        ),
      ]);
      await runProgram(database, program);
      setPhase("ready");
    } catch (value) {
      setError(describeError(value));
      setPhase("error");
    }
  };

  const resetSample = async () => {
    const database = databaseRef.current;
    if (!database) return;
    setPhase("running");
    setError(null);
    try {
      await database.exec(SAMPLE_SETUP_SQL);
      const resetRule = (async () => {
        const started = performance.now();
        const value = await evaluateProgram(database, PRESETS[0].program);
        return { value, durationMs: performance.now() - started };
      })();
      const [nextTables, nextSqlRows, nextEvaluation] = await Promise.all([
        readTables(database),
        database.exec(DEFAULT_SQL),
        resetRule,
      ]);
      setTables(nextTables);
      setSqlRows(nextSqlRows);
      setPreset(PRESETS[0]);
      setProgram(PRESETS[0].program);
      setSelectedTable("status");
      setInsertTable("status");
      setInsertValues({ ...INSERT_DEFAULTS.status });
      applyEvaluation(nextEvaluation.value);
      setMetrics((current) => ({
        ...current,
        ruleMs: nextEvaluation.durationMs,
        insertMs: null,
        sqlMs: null,
      }));
      setLineage([
        ...INITIAL_LINEAGE,
        timestampedEvent(
          "RESET",
          "Atlas.db",
          `sample restored · ${formatMetric(nextEvaluation.durationMs)}`,
        ),
      ]);
      setProofVisited(false);
      setGuidanceOpen(true);
      setPhase("ready");
    } catch (value) {
      setError(describeError(value));
      setPhase("error");
    }
  };

  const choosePreset = async (id: DemoPreset["id"]) => {
    const next = PRESETS.find((item) => item.id === id)!;
    setPreset(next);
    setProgram(next.program);
    const database = databaseRef.current;
    if (!database) return;
    const operation = ++operationRef.current;
    setPhase("running");
    setError(null);
    try {
      await database.exec(SAMPLE_SETUP_SQL);
      if (next.setupSql) await database.exec(next.setupSql);
      const nextTables = await refreshTables(database);
      if (operationRef.current !== operation) return;
      const focusTable = nextTables.find((table) => table.name === next.focusTable) ?? nextTables[0];
      if (focusTable) {
        setSelectedTable(focusTable.name);
        setInsertTable(focusTable.name);
        setInsertValues(
          Object.fromEntries(
            focusTable.columns.map((column) => [
              column.name,
              INSERT_DEFAULTS[focusTable.name]?.[column.name] ?? "",
            ]),
          ),
        );
      }
      setSqlRows([]);
      setLineage([
        ...INITIAL_LINEAGE,
        timestampedEvent(
          "RESET",
          "Atlas.db",
          `${next.category} scenario loaded`,
        ),
      ]);
      await runProgram(database, next.program, true);
    } catch (value) {
      setError(describeError(value));
      setPhase("error");
    }
  };

  const chooseInsertTable = (name: string) => {
    setInsertTable(name);
    const table = tables.find((item) => item.name === name);
    setInsertValues(
      Object.fromEntries(
        (table?.columns ?? []).map((column) => [
          column.name,
          INSERT_DEFAULTS[name]?.[column.name] ?? "",
        ]),
      ),
    );
  };

  const handleShortcut = (
    event: KeyboardEvent<HTMLTextAreaElement>,
    action: () => Promise<void>,
  ) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      void action();
    } else if (event.key === "Escape" && error) {
      setError(null);
    }
  };

  const navigate = (pane: MobilePane) => {
    setMobilePane(pane);
    requestAnimationFrame(() => {
      const desktopTarget = {
        data: ".table-band",
        query: ".query-workspace",
        proof: ".proof-panel",
        graph: ".graph-panel",
      }[pane];
      document.querySelector<HTMLElement>(desktopTarget)?.focus({ preventScroll: true });
    });
  };

  const selectProof = (id: string) => {
    setSelectedProofId(id);
    setProofVisited(true);
  };

  const startWorkspaceResize = (
    pane: WorkspacePane,
    event: ReactPointerEvent<HTMLElement>,
  ) => {
    event.preventDefault();
    workspaceResizeRef.current = {
      pane,
      startX: event.clientX,
      startWidth: workspaceLayout[pane],
    };
    document.body.classList.add("is-resizing-columns");

    const stop = () => {
      workspaceResizeRef.current = null;
      document.body.classList.remove("is-resizing-columns");
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
      window.removeEventListener("pointercancel", stop);
    };
    const move = (moveEvent: globalThis.PointerEvent) => {
      const current = workspaceResizeRef.current;
      if (!current) return;
      const delta = moveEvent.clientX - current.startX;
      const min = current.pane === "schema" ? 180 : 260;
      const max = current.pane === "schema" ? 420 : 520;
      const nextWidth = Math.min(max, Math.max(min, current.startWidth + delta));
      setWorkspaceLayout((currentLayout) => ({
        ...currentLayout,
        [current.pane]: nextWidth,
      }));
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop);
    window.addEventListener("pointercancel", stop);
  };

  const nudgeWorkspaceResize = (
    pane: WorkspacePane,
    event: KeyboardEvent<HTMLElement>,
  ) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const direction = event.key === "ArrowRight" ? 1 : -1;
    const min = pane === "schema" ? 180 : 260;
    const max = pane === "schema" ? 420 : 520;
    setWorkspaceLayout((current) => ({
      ...current,
      [pane]: Math.min(max, Math.max(min, current[pane] + direction * 16)),
    }));
  };

  const startResultColumnResize = (
    column: string,
    event: ReactPointerEvent<HTMLButtonElement>,
  ) => {
    event.preventDefault();
    event.stopPropagation();
    resultResizeRef.current = {
      column,
      startX: event.clientX,
      startWidth: resultColumnConfig[column]?.width ?? 140,
    };
    document.body.classList.add("is-resizing-columns");

    const stop = () => {
      resultResizeRef.current = null;
      document.body.classList.remove("is-resizing-columns");
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
      window.removeEventListener("pointercancel", stop);
    };
    const move = (moveEvent: globalThis.PointerEvent) => {
      const current = resultResizeRef.current;
      if (!current) return;
      const nextWidth = Math.min(
        360,
        Math.max(96, current.startWidth + moveEvent.clientX - current.startX),
      );
      setResultColumnConfig((currentConfig) => ({
        ...currentConfig,
        [current.column]: {
          visible: currentConfig[current.column]?.visible ?? true,
          width: nextWidth,
        },
      }));
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop);
    window.addEventListener("pointercancel", stop);
  };

  const busy = phase === "loading" || phase === "running";
  const resultColumns = evaluation.rows[0] ? Object.keys(evaluation.rows[0]) : [];
  const visibleResultColumns = resultColumns.filter(
    (column) => resultColumnConfig[column]?.visible !== false,
  );
  const resultGridTemplate = visibleResultColumns.length
    ? visibleResultColumns
        .map((column) => `${resultColumnConfig[column]?.width ?? 140}px`)
        .join(" ")
    : "minmax(120px, 1fr)";
  const guidanceSteps = [
    { title: "Add data", body: "Insert or change an ordinary SQLite row.", done: tables.length > 0 },
    { title: "Inspect SQLite", body: "Select a table and read the stored rows.", done: Boolean(selectedSnapshot) },
    { title: "Run a prepared rule", body: "Start with the question; inspect the rule when ready.", done: evaluation.rows.length > 0 },
    { title: "Trace why", body: "Select a proof or graph node to verify the answer.", done: proofVisited },
  ];

  return (
    <section
      className="ide-root"
      id="playground"
      aria-label="Remembero SQLite and Datalog IDE"
      data-mobile-pane={mobilePane}
    >
      <header className="ide-topbar">
        <a className="ide-brand" href="/" aria-label="Remembero main site">
          remembero
        </a>
        <strong className="ide-product-name">SQLite + Datalog IDE</strong>
        <span className="ide-runtime-line" aria-live="polite">
          {runtime
            ? `SQLite ${runtime.sqliteVersion} · Remembero extension · WebAssembly · browser local`
            : phase === "error"
              ? "SQLite runtime could not start"
              : "Starting SQLite WebAssembly…"}
        </span>
        <nav className="ide-top-actions" aria-label="IDE actions">
          <div className="ide-layout-actions" aria-label="Workspace columns">
            <button
              type="button"
              className={`ide-layout-toggle${schemaVisible ? " active" : ""}`}
              aria-pressed={schemaVisible}
              onClick={() => setSchemaVisible((visible) => !visible)}
              title={`${schemaVisible ? "Hide" : "Show"} schema column`}
            >
              <DatabaseIcon width="14" height="14" />
              <span>Schema</span>
            </button>
            <button
              type="button"
              className={`ide-layout-toggle${inspectorVisible ? " active" : ""}`}
              aria-pressed={inspectorVisible}
              onClick={() => setInspectorVisible((visible) => !visible)}
              title={`${inspectorVisible ? "Hide" : "Show"} proof column`}
            >
              <ProofIcon width="14" height="14" />
              <span>Proof</span>
            </button>
          </div>
          <a href="/">Home</a>
          <a href={`${github}#readme`}>Docs</a>
          <a href={github}>GitHub</a>
          <button
            type="button"
            className="ide-icon-button"
            onClick={() => void resetSample()}
            disabled={busy}
            aria-label="Reset sample database"
            title="Reset sample database"
          >
            <ResetIcon />
          </button>
        </nav>
      </header>

      <div className="ide-mobile-tabs" role="tablist" aria-label="IDE panes">
        {(["data", "query", "proof", "graph"] as const).map((pane) => (
          <button
            key={pane}
            type="button"
            role="tab"
            aria-selected={mobilePane === pane}
            onClick={() => navigate(pane)}
          >
            {pane.slice(0, 1).toUpperCase() + pane.slice(1)}
          </button>
        ))}
      </div>

      <div
        className="ide-body"
        data-schema-collapsed={!schemaVisible ? "true" : undefined}
        data-inspector-collapsed={!inspectorVisible ? "true" : undefined}
        style={
          {
            "--schema-width": schemaVisible ? `${workspaceLayout.schema}px` : "0px",
            "--inspector-width": inspectorVisible
              ? `${workspaceLayout.inspector}px`
              : "0px",
          } as CSSProperties
        }
      >
        <aside className="ide-left" data-pane="data" tabIndex={-1}>
          {guidanceOpen ? (
            <section className="guide-panel" aria-labelledby="guide-title">
              <div className="ide-panel-heading compact">
                <h2 id="guide-title">Get to the proof</h2>
                <button type="button" onClick={() => setGuidanceOpen(false)}>
                  Skip guidance
                </button>
              </div>
              <ol>
                {guidanceSteps.map((step, index) => (
                  <li key={step.title} data-done={step.done}>
                    <span className="guide-marker" aria-hidden="true">
                      {step.done ? <CheckIcon /> : index + 1}
                    </span>
                    <div>
                      <strong>{step.title}</strong>
                      <span>{step.body}</span>
                    </div>
                  </li>
                ))}
              </ol>
            </section>
          ) : (
            <button className="guide-resume" type="button" onClick={() => setGuidanceOpen(true)}>
              Guide me to a proof
            </button>
          )}

          <section className="schema-panel" aria-labelledby="schema-title">
            <div className="ide-panel-heading">
              <div>
                <span>Atlas.db</span>
                <h2 id="schema-title">SQLite schema</h2>
              </div>
              <span>{tables.length} tables</span>
            </div>
            <div className="schema-tree">
              {tables.map((table) => (
                <button
                  type="button"
                  key={table.name}
                  className={selectedTable === table.name ? "selected" : ""}
                  aria-pressed={selectedTable === table.name}
                  onClick={() => setSelectedTable(table.name)}
                >
                  <span className="schema-table-name">
                    <DatabaseIcon width="16" height="16" />
                    {table.name}
                  </span>
                  <span>{table.rows.length} rows</span>
                  {selectedTable === table.name ? (
                    <ul>
                      {table.columns.map((column) => (
                        <li key={column.name}>
                          <code>{column.name}</code>
                          <span>{column.type}</span>
                        </li>
                      ))}
                    </ul>
                  ) : null}
                </button>
              ))}
            </div>
          </section>
        </aside>

        <div className="ide-center">
          <section className="insert-band" data-pane="data" tabIndex={-1}>
            <div className="ide-panel-heading horizontal">
              <div>
                <span>Writes to the same browser-local database</span>
                <h2>Insert SQLite row</h2>
              </div>
              <span>Session only · nothing uploaded</span>
            </div>
            <form
              className="insert-form"
              onSubmit={(event) => {
                event.preventDefault();
                void insertRow();
              }}
            >
              <label>
                <span>Table</span>
                <select
                  value={insertTable}
                  onChange={(event) => chooseInsertTable(event.target.value)}
                  disabled={busy}
                >
                  {tables.map((table) => (
                    <option key={table.name}>{table.name}</option>
                  ))}
                </select>
              </label>
              {(insertSnapshot?.columns ?? []).map((column) => (
                <label key={column.name}>
                  <span>{column.name}</span>
                  <input
                    value={stringValue(insertValues[column.name])}
                    onChange={(event) =>
                      setInsertValues((current) => ({
                        ...current,
                        [column.name]: event.target.value,
                      }))
                    }
                    disabled={busy}
                  />
                </label>
              ))}
              <button
                className="ide-primary"
                type="button"
                onClick={() => void insertRow()}
                disabled={busy || !insertSnapshot}
              >
                <PlayIcon width="17" height="17" />
                Insert row
              </button>
            </form>
          </section>

          {error ? (
            <div className="ide-error" role="alert">
              <div>
                <strong>The database rejected that action.</strong>
                <span>{error}</span>
              </div>
              <button type="button" onClick={() => setError(null)} aria-label="Dismiss error">
                <CloseIcon />
              </button>
            </div>
          ) : null}

          <section className="table-band" data-pane="data" tabIndex={-1} aria-labelledby="tables-title">
            <div className="ide-panel-heading horizontal">
              <div>
                <span>Stored rows, not derived answers</span>
                <h2 id="tables-title">SQLite tables</h2>
              </div>
              <span>Select a table to inspect its schema</span>
            </div>
            <div className="table-strip">
              {tables.map((table) => (
                <button
                  type="button"
                  key={table.name}
                  className={`mini-table${selectedTable === table.name ? " selected" : ""}`}
                  onClick={() => setSelectedTable(table.name)}
                >
                  <strong>{table.name}</strong>
                  <span className="mini-table-columns">
                    {table.columns.map((column) => column.name).join(" · ")}
                  </span>
                  {table.rows.slice(0, 2).map((row, index) => (
                    <span className="mini-table-row" key={`${table.name}-${index}`}>
                      {table.columns.map((column) => formatCell(row[column.name] ?? null)).join(" · ")}
                    </span>
                  ))}
                  <small>{table.rows.length} rows</small>
                </button>
              ))}
            </div>
          </section>

          <section className="query-workspace" data-pane="query" tabIndex={-1}>
            <div className="query-editor-panel">
              <div className="ide-panel-heading horizontal">
                <div>
                  <span>Prepared question</span>
                  <h2>{preset.title}</h2>
                </div>
                <select
                  aria-label="Choose prepared rule"
                  value={preset.id}
                  onChange={(event) => void choosePreset(event.target.value as DemoPreset["id"])}
                  disabled={busy}
                >
                  {PRESETS.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.title}
                    </option>
                  ))}
                </select>
              </div>
              <p>{preset.description}</p>
              <p
                className="scenario-purpose"
                aria-label={`${preset.category}: ${preset.useCase}`}
              >
                <strong>{preset.category}</strong>
                <span>· {preset.useCase}</span>
              </p>
              <label className="code-editor-label">
                <span>Datalog rule executed by the SQLite extension</span>
                <textarea
                  value={program}
                  onChange={(event) => setProgram(event.target.value)}
                  onKeyDown={(event) => handleShortcut(event, runDatalog)}
                  onSelect={(event) =>
                    setCursor(editorPosition(program, event.currentTarget.selectionStart))
                  }
                  spellCheck={false}
                  aria-label="Datalog program"
                />
              </label>
              <div className="editor-actions">
                <button className="ide-primary" type="button" onClick={() => void runDatalog()} disabled={busy}>
                  <PlayIcon width="17" height="17" />
                  Run prepared rule
                </button>
                <span>Ctrl/⌘ + Enter</span>
                <strong>{evaluation.rows.length} results</strong>
              </div>

              <div className="advanced-row">
                <details>
                  <summary>
                    SQL scratchpad <ChevronIcon />
                  </summary>
                  <label className="code-editor-label compact">
                    <span>Inspect or change the session database directly</span>
                    <textarea
                      value={sqlText}
                      onChange={(event) => setSqlText(event.target.value)}
                      onKeyDown={(event) => handleShortcut(event, runSql)}
                      spellCheck={false}
                      aria-label="SQL scratchpad"
                    />
                  </label>
                  <button className="ide-secondary" type="button" onClick={() => void runSql()} disabled={busy}>
                    Run SQL
                  </button>
                  <p className="sql-run-summary" aria-live="polite">
                    {sqlRows.length
                      ? `${sqlRows.length} SQL row${sqlRows.length === 1 ? "" : "s"} returned`
                      : "No SQL result rows yet"}
                  </p>
                </details>
                <details>
                  <summary>
                    Integrity constraints <ChevronIcon />
                  </summary>
                  <p>
                    Governed-memory constraints are intentionally a Remembero policy layer,
                    not part of this SQLite bridge. The IDE shows the boundary instead of
                    pretending the native extension executes it.
                  </p>
                  <code>{CONSTRAINT_EXAMPLE}</code>
                  <a href={`${github}/blob/main/docs/INTEGRITY-CONSTRAINTS.md`}>Open constraint docs</a>
                </details>
              </div>
            </div>

            <div className="result-panel" aria-labelledby="result-title">
              <div className="ide-panel-heading horizontal">
                <div>
                  <span>Derived, never stored</span>
                  <h2 id="result-title">Result</h2>
                </div>
                <div className="result-panel-actions">
                  <details className="result-columns-menu">
                    <summary>
                      Columns <span>{visibleResultColumns.length}/{resultColumns.length}</span>
                      <ChevronIcon />
                    </summary>
                    <div className="result-columns-menu-body">
                      <span>Show and resize result columns</span>
                      {resultColumns.map((column) => {
                        const visible = resultColumnConfig[column]?.visible !== false;
                        return (
                          <label key={column}>
                            <input
                              type="checkbox"
                              checked={visible}
                              disabled={visible && visibleResultColumns.length === 1}
                              onChange={() => {
                                if (!visible && visibleResultColumns.length === 0) return;
                                setResultColumnConfig((current) => ({
                                  ...current,
                                  [column]: {
                                    visible: !visible,
                                    width: current[column]?.width ?? 140,
                                  },
                                }));
                              }}
                            />
                            <code>{column}</code>
                          </label>
                        );
                      })}
                    </div>
                  </details>
                  <span>{phase === "running" ? "Running…" : `${evaluation.rows.length} rows`}</span>
                </div>
              </div>
              {selectedRow ? (
                <>
                  <p className="human-answer">{displayAnswer(preset, selectedRow)}</p>
                  <div
                    className="result-grid"
                    role="listbox"
                    aria-label="Datalog result rows"
                    style={{ "--result-grid-template": resultGridTemplate } as CSSProperties}
                  >
                    <div className="result-grid-header">
                      {visibleResultColumns.map((column) => (
                        <span className="result-column-header" key={column}>
                          <span>{column}</span>
                          <button
                            type="button"
                            className="result-column-resizer"
                            aria-label={`Resize ${column} result column`}
                            onPointerDown={(event) => startResultColumnResize(column, event)}
                          />
                        </span>
                      ))}
                    </div>
                    {evaluation.rows.map((row, index) => (
                      <button
                        type="button"
                        role="option"
                        aria-selected={selectedResult === index}
                        className={selectedResult === index ? "selected" : ""}
                        key={JSON.stringify(row)}
                        style={{ "--result-grid-template": resultGridTemplate } as CSSProperties}
                        onClick={() => {
                          setSelectedResult(index);
                          setSelectedProofId("proof");
                          setProofVisited(true);
                          if (window.matchMedia("(max-width: 900px)").matches) setMobilePane("proof");
                        }}
                      >
                        {visibleResultColumns.map((column) => (
                          <code key={column}>{row[column]}</code>
                        ))}
                      </button>
                    ))}
                  </div>
                </>
              ) : (
                <div className="ide-empty">
                  <strong>No derived rows</strong>
                  <span>The rule is valid, but the current SQLite rows do not support an answer.</span>
                </div>
              )}
            </div>
          </section>

          <section className="lineage-band" data-pane="data" aria-labelledby="lineage-title">
            <div className="ide-panel-heading horizontal compact">
              <h2 id="lineage-title">Lineage</h2>
              <span>oldest → newest</span>
            </div>
            <ol>
              {lineage.slice(-7).map((event) => (
                <li key={event.id}>
                  <span className="lineage-dot" />
                  <time>{event.timestamp}</time>
                  {" "}
                  <span className="lineage-event-main">
                    <strong>{event.kind}</strong>
                    {" "}
                    <code>{event.target}</code>
                  </span>
                  {" "}
                  <small>{event.detail}</small>
                </li>
              ))}
            </ol>
          </section>

          <section className="console-band" data-pane="query" aria-label="Native runtime console">
            <div className="native-console">
              <div className="ide-panel-heading compact">
                <h2>Native console</h2>
                <span>{runtime ? "current browser · seeded case" : phase}</span>
              </div>
              <div className="runtime-metrics" aria-label="Current browser performance">
                <div>
                  <strong>{formatMetric(metrics.bootMs)}</strong>
                  <span>SQLite + Wasm boot</span>
                </div>
                <div>
                  <strong>{formatMetric(metrics.ruleMs)}</strong>
                  <span>rule + proof</span>
                </div>
                <div>
                  <strong>{formatMetric(metrics.insertMs)}</strong>
                  <span>last insert</span>
                </div>
                <div>
                  <strong>{formatMetric(metrics.sqlMs)}</strong>
                  <span>last SQL</span>
                </div>
              </div>
              <code>&gt; sqlite3_auto_extension(sqlite3_rembero_init)</code>
              <strong>{runtime?.extensionLoaded ? "Remembero extension linked into SQLite" : "Loading extension…"}</strong>
              <code>&gt; SELECT sqlite_version();</code>
              <span>{runtime?.sqliteVersion ?? "…"}</span>
            </div>
            <div className="raw-output">
              <details>
                <summary>
                  Raw JSON <ChevronIcon />
                </summary>
                <pre>{JSON.stringify({ rows: evaluation.rows, proof: selectedExplanation }, null, 2)}</pre>
              </details>
              <details>
                <summary>
                  Build details <ChevronIcon />
                </summary>
                <dl>
                  <div><dt>SQLite</dt><dd>{runtime?.sqliteVersion ?? "loading"}</dd></div>
                  <div><dt>Extension</dt><dd>{runtime?.manifest.extension.registration ?? "loading"}</dd></div>
                  <div><dt>Wasm SHA-256</dt><dd><code>{runtime?.manifest.artifacts["sqlite3.wasm"]?.sha256 ?? "loading"}</code></dd></div>
                  <div><dt>Source SHA3-256</dt><dd><code>{runtime?.manifest.sqlite.sourceSha3_256 ?? "loading"}</code></dd></div>
                </dl>
              </details>
              <details>
                <summary>
                  Compiled SQL <ChevronIcon />
                </summary>
                <pre>{evaluation.compiledSql || "Run a rule to inspect its compiled SQL."}</pre>
              </details>
            </div>
          </section>
        </div>

        <aside className="ide-inspector">
          <section className="proof-panel" data-pane="proof" tabIndex={-1} aria-labelledby="proof-title">
            <div className="ide-panel-heading horizontal">
              <div>
                <span>Exact derivation</span>
                <h2 id="proof-title">Why this is true</h2>
              </div>
              <span>{factProofs.length} SQLite facts</span>
            </div>
            {selectedExplanation ? (
              <ol className="proof-ladder" aria-live="polite">
                <li className={selectedProofId?.startsWith("proof.") ? "selected" : ""}>
                  <button type="button" onClick={() => selectProof("proof.0")}>
                    <span>1</span>
                    <div>
                      <strong>Facts from SQLite</strong>
                      {factProofs.map((fact) => (
                        <code key={`${fact.predicate}-${fact.values.join("-")}`}>
                          {fact.predicate}({fact.values.join(", ")})
                        </code>
                      ))}
                    </div>
                  </button>
                </li>
                <li className={selectedProofId === "rule" ? "selected" : ""}>
                  <button type="button" onClick={() => selectProof("rule")}>
                    <span>2</span>
                    <div>
                      <strong>Rule applied</strong>
                      <code>{program}</code>
                    </div>
                  </button>
                </li>
                <li className={selectedProofId === "proof" ? "selected" : ""}>
                  <button type="button" onClick={() => selectProof("proof")}>
                    <span>3</span>
                    <div>
                      <strong>Derived answer</strong>
                      <code>
                        {selectedExplanation.proof.predicate}(
                        {selectedExplanation.proof.values.join(", ")})
                      </code>
                    </div>
                  </button>
                </li>
              </ol>
            ) : (
              <div className="ide-empty compact">
                <strong>No proof yet</strong>
                <span>Run a prepared rule to inspect its support.</span>
              </div>
            )}
          </section>

          <section className="graph-panel" data-pane="graph" tabIndex={-1} aria-labelledby="graph-title">
            <div className="ide-panel-heading horizontal">
              <div>
                <span>Projection of this result</span>
                <h2 id="graph-title">Query graph</h2>
              </div>
              <span>not a second store</span>
            </div>
            <ProofGraph
              proof={selectedExplanation?.proof ?? null}
              selectedId={selectedProofId}
              onSelect={selectProof}
            />
          </section>
        </aside>

        <button
          type="button"
          className="ide-resize-handle ide-resize-schema"
          aria-label="Resize schema column"
          onPointerDown={(event) => startWorkspaceResize("schema", event)}
          onKeyDown={(event) => nudgeWorkspaceResize("schema", event)}
        />
        <button
          type="button"
          className="ide-resize-handle ide-resize-inspector"
          aria-label="Resize proof column"
          onPointerDown={(event) => startWorkspaceResize("inspector", event)}
          onKeyDown={(event) => nudgeWorkspaceResize("inspector", event)}
        />
      </div>

      <footer className="ide-statusbar">
        <span className={runtime?.extensionLoaded ? "runtime-ok" : "runtime-waiting"}>
          {runtime?.extensionLoaded ? "Extension linked into SQLite" : "Starting SQLite…"}
        </span>
        <span>Session only · nothing uploaded</span>
        <span>Rule + proof {formatMetric(metrics.ruleMs)}</span>
        <span className="status-spacer" />
        <span>Line {cursor.line}, Col {cursor.column}</span>
        <span>Ctrl/⌘ + Enter to run</span>
      </footer>

    </section>
  );
}
