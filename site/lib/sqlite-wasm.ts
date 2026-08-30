export type SqliteScalar = string | number | null;
export type SqliteRow = Record<string, SqliteScalar>;

export interface BrowserDatalogProof {
  predicate: string;
  values?: Array<string | number>;
  rule?: number;
  because?: BrowserDatalogProof[];
  /** Absence step from stratified negation: the pattern proven to match nothing. */
  negated?: true;
  pattern?: Array<string | number | null>;
}

/** Display terms for any proof step, including verified-absence steps. */
export function proofStepValues(proof: BrowserDatalogProof): string[] {
  if (proof.values !== undefined) return proof.values.map(String);
  return (proof.pattern ?? []).map((value) => (value === null ? "_" : String(value)));
}

/** Display predicate for a proof step; absences read as `not pred`. */
export function proofStepTitle(proof: BrowserDatalogProof): string {
  return proof.negated === true || "negated" in proof
    ? `not ${proof.predicate}`
    : proof.predicate;
}

export interface BrowserDatalogExplanation {
  row: Record<string, string | number>;
  proof: BrowserDatalogProof;
}

export interface BrowserSqliteManifest {
  format: "rembero.sqlite-wasm.v1";
  generatedAt: string;
  sqlite: {
    version: string;
    sourceUrl: string;
    sourceSha3_256: string;
  };
  toolchain: { emsdkImage: string };
  extension: {
    registration: string;
    dynamicallyLoaded: false;
    sources: Record<string, { sha256: string }>;
  };
  artifacts: Record<string, { bytes: number; sha256: string }>;
}

export interface BrowserSqliteRuntimeInfo {
  sqliteVersion: string;
  extensionFunctions: string[];
  extensionLoaded: boolean;
  loadableExtensionsOmitted: boolean;
  manifest: BrowserSqliteManifest;
}

interface WorkerResponse {
  type: string;
  result?: {
    resultRows?: SqliteRow[];
    [key: string]: unknown;
  };
  message?: string;
  errorClass?: string;
}

type WorkerPromiser = (
  type: string | { type: string; args?: Record<string, unknown> },
  args?: Record<string, unknown>,
) => Promise<WorkerResponse>;

type WorkerPromiserFactory = (config: {
  worker: Worker;
  onunhandled?: (event: MessageEvent) => void;
}) => Promise<WorkerPromiser>;

const EXTENSION_FUNCTIONS = ["datalog_explain", "datalog_query", "datalog_sql"];
const WORKER_BOOT_TIMEOUT_MS = 15_000;

function assetUrl(name: string): string {
  return new URL(`/sqlite-wasm/${name}`, window.location.origin).href;
}

function errorFromResponse(response: unknown): Error {
  if (response instanceof Error) return response;
  if (response && typeof response === "object") {
    const value = response as WorkerResponse;
    const resultMessage =
      value.result && typeof value.result.message === "string"
        ? value.result.message
        : undefined;
    return new Error(
      value.message ?? resultMessage ?? value.errorClass ?? "SQLite worker request failed",
    );
  }
  return new Error(String(response));
}

async function waitForWorkerPromiser(
  factory: WorkerPromiserFactory,
  worker: Worker,
): Promise<WorkerPromiser> {
  let timeoutId: ReturnType<typeof setTimeout> | undefined;
  let rejectWorkerFailure: ((reason: Error) => void) | undefined;
  const workerFailure = new Promise<never>((_, reject) => {
    rejectWorkerFailure = reject;
  });
  const onError = (event: ErrorEvent) => {
    rejectWorkerFailure?.(
      new Error(
        `SQLite worker failed to load${event.message ? `: ${event.message}` : ""}`,
      ),
    );
  };
  const onMessageError = () => {
    rejectWorkerFailure?.(new Error("SQLite worker returned an unreadable message"));
  };
  worker.addEventListener("error", onError, { once: true });
  worker.addEventListener("messageerror", onMessageError, { once: true });

  const timedOut = new Promise<never>((_, reject) => {
    timeoutId = setTimeout(
      () => reject(new Error("SQLite worker did not become ready within 15 seconds")),
      WORKER_BOOT_TIMEOUT_MS,
    );
  });

  try {
    return await Promise.race([factory({ worker }), workerFailure, timedOut]);
  } finally {
    if (timeoutId !== undefined) clearTimeout(timeoutId);
    worker.removeEventListener("error", onError);
    worker.removeEventListener("messageerror", onMessageError);
  }
}

import {
  portableExecutionMode,
  runPortableExplain,
  runPortableQuery,
} from "./portable-datalog";

function parseJsonArray<T>(value: SqliteScalar, label: string): T[] {
  if (typeof value !== "string") throw new Error(`${label} did not return JSON text`);
  const parsed: unknown = JSON.parse(value);
  if (!Array.isArray(parsed)) throw new Error(`${label} did not return a JSON array`);
  return parsed as T[];
}

export class BrowserDatalogDatabase {
  readonly runtime: BrowserSqliteRuntimeInfo;
  #promiser: WorkerPromiser;
  #worker: Worker;
  #closed = false;

  constructor(
    promiser: WorkerPromiser,
    worker: Worker,
    runtime: BrowserSqliteRuntimeInfo,
  ) {
    this.#promiser = promiser;
    this.#worker = worker;
    this.runtime = runtime;
  }

  async exec(sql: string, bind: SqliteScalar[] = []): Promise<SqliteRow[]> {
    if (this.#closed) throw new Error("SQLite browser database is closed");
    try {
      const response = await this.#promiser("exec", {
        sql,
        bind,
        rowMode: "object",
        resultRows: [],
      });
      return response.result?.resultRows ?? [];
    } catch (error) {
      throw errorFromResponse(error);
    }
  }

  async datalogSql(program: string): Promise<string> {
    const rows = await this.exec("SELECT datalog_sql(?) AS value", [program]);
    const value = rows[0]?.value;
    if (typeof value !== "string") throw new Error("datalog_sql returned no SQL text");
    return value;
  }

  async datalogQuery(program: string): Promise<Array<Record<string, string | number>>> {
    if (portableExecutionMode(program) === "portable") {
      return runPortableQuery((sql) => this.exec(sql), program);
    }
    const rows = await this.exec("SELECT datalog_query(?) AS value", [program]);
    return parseJsonArray<Record<string, string | number>>(
      rows[0]?.value ?? null,
      "datalog_query",
    );
  }

  async datalogExplain(program: string): Promise<BrowserDatalogExplanation[]> {
    if (portableExecutionMode(program) === "portable") {
      const explanations = await runPortableExplain((sql) => this.exec(sql), program);
      return explanations as unknown as BrowserDatalogExplanation[];
    }
    const rows = await this.exec("SELECT datalog_explain(?) AS value", [program]);
    return parseJsonArray<BrowserDatalogExplanation>(
      rows[0]?.value ?? null,
      "datalog_explain",
    );
  }

  async close(): Promise<void> {
    if (this.#closed) return;
    this.#closed = true;
    try {
      await this.#promiser("close", {});
    } finally {
      this.#worker.terminate();
    }
  }
}

export async function openBrowserDatalogDatabase(): Promise<BrowserDatalogDatabase> {
  const [module, manifestResponse] = await Promise.all([
    import(/* @vite-ignore */ assetUrl("sqlite3-worker1-promiser.mjs")) as Promise<{
      default: WorkerPromiserFactory;
    }>,
    fetch(assetUrl("manifest.json"), { cache: "no-store" }),
  ]);
  if (!manifestResponse.ok) {
    throw new Error(`SQLite Wasm manifest failed with HTTP ${manifestResponse.status}`);
  }
  const manifest = (await manifestResponse.json()) as BrowserSqliteManifest;
  if (manifest.format !== "rembero.sqlite-wasm.v1") {
    throw new Error("Unsupported SQLite Wasm manifest format");
  }

  const worker = new Worker(assetUrl("sqlite3-worker1.mjs"), { type: "module" });
  try {
    const promiser = await waitForWorkerPromiser(module.default, worker);
    await promiser("open", { filename: ":memory:" });
    const execWorker = async (sql: string): Promise<SqliteRow[]> => {
      const response = await promiser("exec", {
        sql,
        rowMode: "object",
        resultRows: [],
      });
      return response.result?.resultRows ?? [];
    };
    const [versionRows, functionRows] = await Promise.all([
      execWorker(`
        SELECT sqlite_version() AS sqliteVersion,
               sqlite_compileoption_used('OMIT_LOAD_EXTENSION') AS loadableExtensionsOmitted
      `),
      execWorker(`
        SELECT name
          FROM pragma_function_list
         WHERE name IN ('datalog_sql', 'datalog_query', 'datalog_explain')
         ORDER BY name
      `),
    ]);
    const versionRow = versionRows[0];
    const sqliteVersion = versionRow?.sqliteVersion;
    if (typeof sqliteVersion !== "string") throw new Error("SQLite worker returned no version");
    const extensionFunctions = functionRows
      .flatMap((row) => (typeof row.name === "string" ? [row.name] : []))
      .filter((name) => EXTENSION_FUNCTIONS.includes(name));
    const runtime: BrowserSqliteRuntimeInfo = {
      sqliteVersion,
      extensionFunctions,
      extensionLoaded: EXTENSION_FUNCTIONS.every((name) =>
        extensionFunctions.includes(name),
      ),
      loadableExtensionsOmitted: versionRow?.loadableExtensionsOmitted === 1,
      manifest,
    };
    if (runtime.sqliteVersion !== manifest.sqlite.version) {
      throw new Error(
        `SQLite Wasm version ${runtime.sqliteVersion} does not match manifest ${manifest.sqlite.version}`,
      );
    }
    if (!runtime.extensionLoaded) throw new Error("Remembero SQLite extension is not registered");
    return new BrowserDatalogDatabase(promiser, worker, runtime);
  } catch (error) {
    worker.terminate();
    throw errorFromResponse(error);
  }
}
