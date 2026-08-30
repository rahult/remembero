"use client";

/* eslint-disable @next/next/no-html-link-for-pages -- GitHub Pages-style export needs plain document links. */

import { useEffect, useMemo, useRef, useState, type JSX } from "react";
import styles from "./chat-memory-lab.module.css";
import {
  CHAT_MEMORY_SCENARIOS,
  runChatMemoryScenario,
  type ChatMemoryScenarioId,
} from "@/lib/chat-memory-lab";
import {
  WEB_LLM_MODEL_ID,
  WEB_LLM_MODEL_LABEL,
  WEB_LLM_VRAM_MB,
  describeBrowserModelResult,
  loadWebLlm,
  promptLoadedWebLlm,
  requestLoadedWebLlmToolCall,
  type BrowserModelMode,
  type WebLlmProgress,
} from "@/lib/browser-language-model";
import {
  CHAT_MEMORY_SQLITE_SETUP,
  CHAT_MEMORY_SQLITE_VERIFY,
  chatToolDefinition,
  executeChatTool,
  parseChatToolCall,
  simulatedChatToolCall,
  verifyChatMemorySeed,
  type ChatToolDefinition,
  type ChatToolLane,
  type ParsedChatToolCall,
} from "@/lib/chat-memory-tools";
import {
  openBrowserDatalogDatabase,
  type BrowserDatalogDatabase,
} from "@/lib/sqlite-wasm";

const GITHUB = "https://github.com/rahult/remembero";
const ANSWER_SYSTEM =
  "Answer the user using only TOOL_RESULT. Return one concise natural-language answer. " +
  "Never answer with only yes or no; name the subject and cite concrete supporting values. " +
  "Do not invent values or repeat SQL or Datalog syntax.";
const FLOW_STEPS = [
  {
    key: "ask",
    label: "Ask",
    detail: "Give both agents the same question and shared SQLite database.",
  },
  {
    key: "call",
    label: "Model calls",
    detail: "Hermes emits a native function call before it can answer.",
  },
  {
    key: "execute",
    label: "SQLite executes",
    detail: "One tool returns raw SQL rows; Remembero returns bindings and proof.",
  },
  {
    key: "answer",
    label: "Answer",
    detail: "The tool result goes back to the same local model for synthesis.",
  },
] as const;

interface LaneTrace {
  callRaw: string;
  call: ParsedChatToolCall | null;
  command: string;
  result: string;
  callDurationMs: number | null;
  toolDurationMs: number | null;
  answerRaw: string;
  finalPrompt: string;
  status: "idle" | "generated" | "simulated" | "invalid" | "error";
}

function emptyLaneTrace(): LaneTrace {
  return {
    callRaw: "Not run yet",
    call: null,
    command: "Not executed",
    result: "Not executed",
    callDurationMs: null,
    toolDurationMs: null,
    answerRaw: "Not run yet",
    finalPrompt: "Not run yet",
    status: "idle",
  };
}

function prettyJson(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

function describeError(value: unknown): string {
  return value instanceof Error ? value.message : "Unknown tool-loop error";
}

function finalAnswerPrompt(
  question: string,
  call: ParsedChatToolCall,
  result: string,
): string {
  return [
    `QUESTION: ${question}`,
    `TOOL_CALL: ${prettyJson(call)}`,
    "TOOL_RESULT:",
    result,
  ].join("\n");
}

function nativeToolPrompt(
  question: string,
  definition: ChatToolDefinition,
): string {
  return [
    `Call the ${definition.name} function exactly once with query "${question}".`,
    "Do not rename the function, invent another function, or write SQL or Datalog yourself.",
    "After the tool result is returned, answer the question using only that evidence and cite its concrete values.",
    `Question: ${question}`,
  ].join("\n");
}

function formatDuration(value: number | null): string {
  if (value === null) return "not run";
  return `${Math.max(value, 0.001).toFixed(3)} ms`;
}

function matchesAnswerContract(
  value: string,
  expected: string,
  scenarioId: ChatMemoryScenarioId,
): boolean {
  const normalize = (text: string) =>
    text.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
  const normalized = normalize(value);
  if (normalized === normalize(expected)) return true;
  const includesAll = (...terms: string[]) =>
    terms.every((term) => normalized.includes(normalize(term)));

  switch (scenarioId) {
    case "root-blocker":
      return includesAll("procurement", "freeze");
    case "write-gate":
      return (
        (normalized.includes("refus") || normalized.includes("reject") ||
          normalized.includes("cannot") || normalized.includes("blocked")) &&
        normalized.includes("vendor security review")
      );
    case "unknown-preference":
      return (
        includesAll("jordan", "preference") &&
        (normalized.includes("ask") ||
          normalized.includes("not stored") ||
          normalized.includes("missing"))
      );
    case "why-not":
      return (
        includesAll("orchard") &&
        (normalized.includes("not blocked") ||
          normalized.includes("no review slot") ||
          normalized.includes("no slot") ||
          normalized.includes("failing premise") ||
          normalized.includes("cannot be derived") ||
          normalized.includes("underivable"))
      );
  }
}

function CalendarIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M7 2v3M17 2v3M3.75 9.25h16.5M5.25 5.5h13.5a1.75 1.75 0 0 1 1.75 1.75v11.5a1.75 1.75 0 0 1-1.75 1.75H5.25A1.75 1.75 0 0 1 3.5 18.75V7.25A1.75 1.75 0 0 1 5.25 5.5Z" />
    </svg>
  );
}

function PersonIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 12a4 4 0 1 0-4-4 4 4 0 0 0 4 4Zm0 2.5c-3.59 0-6.5 2.24-6.5 5v.5h13v-.5c0-2.76-2.91-5-6.5-5Z" />
    </svg>
  );
}

function QuestionIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 17h.01M9.1 9.4a2.9 2.9 0 1 1 5.15 1.86c-.57.58-1.2.96-1.8 1.46-.74.61-1.2 1.26-1.2 2.28v.5M12 21a9 9 0 1 0-9-9 9 9 0 0 0 9 9Z" />
    </svg>
  );
}

function PlayIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M8 6.5v11l9-5.5Z" />
    </svg>
  );
}

function ShieldIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 2.5 4.5 5.5v5.09c0 4.6 3.2 8.9 7.5 9.91 4.3-1.01 7.5-5.31 7.5-9.91V5.5Zm-1.2 12.3-2.8-2.8 1.06-1.06 1.74 1.73 4.14-4.13L16 9.6Z" />
    </svg>
  );
}

const SCENARIO_ICONS = {
  "root-blocker": CalendarIcon,
  "write-gate": ShieldIcon,
  "unknown-preference": PersonIcon,
  "why-not": QuestionIcon,
} satisfies Record<ChatMemoryScenarioId, () => JSX.Element>;

export function ChatMemoryLab() {
  const [scenarioId, setScenarioId] =
    useState<ChatMemoryScenarioId>("root-blocker");
  const [hasRun, setHasRun] = useState(false);
  const [running, setRunning] = useState(false);
  const [modelMode, setModelMode] = useState<BrowserModelMode>("simulated");
  const [modelDiagnostic, setModelDiagnostic] = useState(
    "SQLite data ready · WebLLM model not loaded",
  );
  const [webLlmStatus, setWebLlmStatus] = useState<
    "idle" | "loading" | "ready" | "error"
  >("idle");
  const [webLlmProgress, setWebLlmProgress] = useState<WebLlmProgress>({
    progress: 0,
    text: `On demand · about ${(WEB_LLM_VRAM_MB / 1000).toFixed(1)} GB VRAM · weights cached by the browser`,
  });
  const databaseRef = useRef<BrowserDatalogDatabase | null>(null);
  const [databaseStatus, setDatabaseStatus] = useState<
    "starting" | "ready" | "error"
  >("starting");
  const [databaseDiagnostic, setDatabaseDiagnostic] = useState(
    "Starting SQLite WebAssembly…",
  );
  const [databaseBootMs, setDatabaseBootMs] = useState<number | null>(null);
  const [databaseSeedSummary, setDatabaseSeedSummary] = useState("not verified");
  const [baselineTrace, setBaselineTrace] = useState<LaneTrace>(emptyLaneTrace);
  const [memoryTrace, setMemoryTrace] = useState<LaneTrace>(emptyLaneTrace);
  const [answerContract, setAnswerContract] = useState("Awaiting model output");

  const scenario = CHAT_MEMORY_SCENARIOS.find((entry) => entry.id === scenarioId)!;
  const comparison = useMemo(() => runChatMemoryScenario(scenarioId), [scenarioId]);
  const [baselineAnswer, setBaselineAnswer] = useState(scenario.baselineAnswer);
  const [memoryAnswer, setMemoryAnswer] = useState(comparison.answer);
  const dataTool = useMemo(
    () => chatToolDefinition("data", scenario.question),
    [scenario.question],
  );
  const rememberoTool = useMemo(
    () => chatToolDefinition("remembero", scenario.question),
    [scenario.question],
  );

  const stats = [
    `${comparison.factCount} recalled ${comparison.factCount === 1 ? "fact" : "facts"}`,
    `${comparison.ruleCount} ${comparison.ruleCount === 1 ? "rule" : "rules"}`,
    comparison.absenceCount > 0
      ? `${comparison.absenceCount} absence ${comparison.absenceCount === 1 ? "check" : "checks"}`
      : undefined,
  ].filter(Boolean) as string[];

  useEffect(() => {
    let active = true;
    let opened: BrowserDatalogDatabase | null = null;
    const boot = async () => {
      try {
        const started = performance.now();
        opened = await openBrowserDatalogDatabase();
        await opened.exec(CHAT_MEMORY_SQLITE_SETUP);
        const seed = await verifyChatMemorySeed(opened);
        if (!active) {
          await opened.close();
          return;
        }
        databaseRef.current = opened;
        setDatabaseBootMs(performance.now() - started);
        setDatabaseSeedSummary(
          `${seed.tableCount} tables · ${seed.rowCount} rows verified`,
        );
        setDatabaseDiagnostic(
          `SQLite ${opened.runtime.sqliteVersion} · ${seed.tableCount} tables · ${seed.rowCount} rows verified · Remembero extension linked`,
        );
        setDatabaseStatus("ready");
      } catch (error) {
        if (!active) return;
        setDatabaseDiagnostic(
          `SQLite WebAssembly failed to start: ${describeError(error)}`,
        );
        setDatabaseStatus("error");
      }
    };
    void boot();
    return () => {
      active = false;
      databaseRef.current = null;
      if (opened) void opened.close();
    };
  }, []);

  async function runLane(
    lane: ChatToolLane,
    definition: ChatToolDefinition,
    fallbackAnswer: string,
  ): Promise<{
    trace: LaneTrace;
    answer: string;
    diagnostic: string;
    runtime: BrowserModelMode;
  }> {
    const database = databaseRef.current;
    if (database === null) {
      throw new Error("SQLite tool runtime is not ready");
    }

    const callResult =
      webLlmStatus === "ready"
        ? await requestLoadedWebLlmToolCall(
          nativeToolPrompt(scenario.question, definition),
            {
              type: "function",
              function: definition,
            },
          )
        : {
            status: "fallback" as const,
            reason: "not_ready" as const,
            stage: "create" as const,
            runtime: "webllm" as const,
          };
    const generatedCall = callResult.status === "generated";
    const callRaw = generatedCall
      ? callResult.raw
        : prettyJson(
            simulatedChatToolCall(definition, lane, scenarioId, scenario.question),
          );
    const call = generatedCall
      ? parseChatToolCall(
          prettyJson({
            name: callResult.call.function.name,
            arguments: callResult.call.function.arguments,
          }),
          definition,
          lane,
          scenarioId,
          scenario.question,
        )
      : simulatedChatToolCall(definition, lane, scenarioId, scenario.question);

    if (call === null) {
      return {
        trace: {
          ...emptyLaneTrace(),
          callRaw,
          callDurationMs: callResult.status === "generated" ? callResult.durationMs : null,
          answerRaw: "Tool call rejected; no SQLite query executed.",
          finalPrompt: "Not sent because validation rejected the tool call.",
          status: "invalid",
        },
        answer: "Tool call rejected; no database query was executed.",
        diagnostic: "Native tool call failed validation",
        runtime: generatedCall ? "webllm" : "simulated",
      };
    }

    let execution: Awaited<ReturnType<typeof executeChatTool>>;
    try {
      execution = await executeChatTool(database, call);
    } catch (error) {
      const message = describeError(error);
      return {
        trace: {
          ...emptyLaneTrace(),
          callRaw,
          call,
          callDurationMs:
            callResult.status === "generated" ? callResult.durationMs : null,
          command: "Tool execution failed closed",
          result: message,
          answerRaw: "No answer generated because the tool failed.",
          finalPrompt: "Not sent because the tool did not return evidence.",
          status: "error",
        },
        answer: `Tool execution failed: ${message}`,
        diagnostic: message,
        runtime: generatedCall ? "webllm" : "simulated",
      };
    }
    const resultText = prettyJson(execution.result);
    const finalUserPrompt = finalAnswerPrompt(scenario.question, call, resultText);
    const displayedFinalPrompt = `SYSTEM:\n${ANSWER_SYSTEM}\n\nUSER:\n${finalUserPrompt}`;
    if (!generatedCall) {
      return {
        trace: {
          callRaw,
          call,
          command: execution.command,
          result: resultText,
          callDurationMs: null,
          toolDurationMs: execution.durationMs,
          answerRaw: "Deterministic simulator fixture",
          finalPrompt: displayedFinalPrompt,
          status: "simulated",
        },
        answer: fallbackAnswer,
        diagnostic: "SQLite queried · deterministic tool-call simulator used",
        runtime: "simulated",
      };
    }

    const answerResult = await promptLoadedWebLlm(
      ANSWER_SYSTEM,
      finalUserPrompt,
    );
    const answerRaw =
      answerResult.status === "generated"
        ? answerResult.text.trim()
        : "Model answer generation failed; deterministic contract supplied the display answer.";
    return {
      trace: {
        callRaw,
        call,
        command: execution.command,
        result: resultText,
        callDurationMs: callResult.durationMs,
        toolDurationMs: execution.durationMs,
        answerRaw,
        finalPrompt: displayedFinalPrompt,
        status: answerResult.status === "generated" ? "generated" : "error",
      },
      answer: answerResult.status === "generated" ? answerRaw : fallbackAnswer,
      diagnostic: describeBrowserModelResult(answerResult),
      runtime: answerResult.status === "generated" ? "webllm" : "simulated",
    };
  }

  async function handleRunComparison() {
    setRunning(true);
    try {
      const dataRun = await runLane(
        "data",
        dataTool,
        scenario.baselineAnswer,
      );
      const rememberoRun = await runLane(
        "remembero",
        rememberoTool,
        comparison.answer,
      );
      const baselinePassed = matchesAnswerContract(
        dataRun.answer,
        comparison.answer,
        scenarioId,
      );
      const memoryPassed = matchesAnswerContract(
        rememberoRun.answer,
        comparison.answer,
        scenarioId,
      );
      setBaselineTrace(dataRun.trace);
      setMemoryTrace(rememberoRun.trace);
      setBaselineAnswer(
        dataRun.trace.status === "invalid" || dataRun.trace.status === "error"
          ? dataRun.answer
          : baselinePassed
            ? dataRun.answer
            : comparison.answer,
      );
      setMemoryAnswer(
        rememberoRun.trace.status === "invalid" || rememberoRun.trace.status === "error"
          ? rememberoRun.answer
          : memoryPassed
            ? rememberoRun.answer
            : comparison.answer,
      );
      const contractLabel = (trace: LaneTrace, passed: boolean) =>
        trace.status === "simulated"
          ? "fixture"
          : trace.status === "invalid" || trace.status === "error"
            ? "blocked"
            : passed
              ? "passed"
              : "corrected";
      setAnswerContract(
        `data ${contractLabel(dataRun.trace, baselinePassed)} · remembero ${contractLabel(rememberoRun.trace, memoryPassed)}`,
      );
      setModelMode(rememberoRun.runtime);
      setModelDiagnostic(rememberoRun.diagnostic);
      setHasRun(true);
    } catch (error) {
      const message = describeError(error);
      const trace = {
        ...emptyLaneTrace(),
        result: message,
        answerRaw: "No answer generated because the tool loop failed.",
        status: "error" as const,
      };
      setBaselineTrace(trace);
      setMemoryTrace(trace);
      setBaselineAnswer(`Tool loop failed: ${message}`);
      setMemoryAnswer(`Tool loop failed: ${message}`);
      setAnswerContract("Tool loop failed closed · inspect error trace");
    } finally {
      setRunning(false);
    }
  }

  async function handleLoadWebLlm() {
    setWebLlmStatus("loading");
    setWebLlmProgress({ progress: 0, text: "Preparing WebLLM download" });
    const result = await loadWebLlm(setWebLlmProgress);
    if (result.status === "ready") {
      setWebLlmStatus("ready");
      setModelMode("webllm");
      setModelDiagnostic(
        `${result.modelId} ready · loaded in ${(result.loadMs / 1000).toFixed(1)} s`,
      );
      setWebLlmProgress({ progress: 1, text: "WebLLM ready for local inference" });
      return;
    }

    const text =
      result.reason === "webgpu_unsupported"
        ? "WebGPU is unavailable in this browser"
        : "WebLLM model loading failed";
    setWebLlmStatus("error");
    setModelMode("simulated");
    setModelDiagnostic(text);
    setWebLlmProgress({ progress: 0, text });
  }

  return (
    <main className={styles.page}>
      <header className={styles.header}>
        <a className={styles.brand} href="/">
          remembero
        </a>
        <nav className={styles.nav} aria-label="Lab navigation">
          <a href="/labs/chat-memory" aria-current="page">
            Labs
          </a>
          <a href="/playground">Playground</a>
          <a href="/guides/agent-harness">Guide</a>
          <a href={GITHUB}>GitHub</a>
        </nav>
        <a className={styles.backLink} href="/">
          <span aria-hidden="true">←</span> Back home
        </a>
      </header>

      <section className={styles.hero}>
        <div className={styles.heroCopy}>
          <h1>Same database. Same model. Different powers.</h1>
          <p>
            Both lanes share one browser-local SQLite database and the same model.
            The only difference is what the tool can return — and these four cases
            pick questions where that difference is structural: SQL hands back rows,
            silence, or a silently corrupted state; Remembero hands back a checkable
            proof, a named missing premise, or a refused contradiction.
          </p>
        </div>
        <div className={styles.heroStatus} aria-label="Current lab status">
          <div>
            <span>Model mode</span>
            <strong>
              {modelMode === "webllm"
                ? WEB_LLM_MODEL_ID
                : modelMode === "browser"
                  ? "Browser local model"
                  : modelDiagnostic}
            </strong>
          </div>
          <div>
            <span>Recall depth</span>
            <strong>{stats.join(" · ")}</strong>
          </div>
          <div>
            <span>Data source</span>
            <strong>{databaseDiagnostic}</strong>
          </div>
          <div>
            <span>SQLite boot</span>
            <strong>{formatDuration(databaseBootMs)} · shared in-memory database</strong>
          </div>
          <div>
            <span>Tool execution</span>
            <strong>
              SQL {formatDuration(baselineTrace.toolDurationMs)} · Remembero {formatDuration(memoryTrace.toolDurationMs)}
            </strong>
          </div>
          <div>
            <span>Answer contract</span>
            <strong>{answerContract}</strong>
          </div>
        </div>
      </section>

      <section className={styles.toolbar} aria-label="Lab controls">
        <div className={styles.selector} aria-labelledby="scenario-legend">
          <div className={styles.selectorLabel} id="scenario-legend">
            Scenario
          </div>
          <div className={styles.selectorRail} role="radiogroup" aria-labelledby="scenario-legend">
            {CHAT_MEMORY_SCENARIOS.map((entry) => {
              const Icon = SCENARIO_ICONS[entry.id];
              const checked = entry.id === scenarioId;
              return (
                <label key={entry.id} className={styles.radioCard}>
                  <span className={styles.srOnly}>Select a comparison scenario</span>
                  <input
                    type="radio"
                    name="scenario"
                    value={entry.id}
                    checked={checked}
                    disabled={running}
                    onChange={() => {
                      const nextComparison = runChatMemoryScenario(entry.id);
                      setScenarioId(entry.id);
                      setHasRun(false);
                      setBaselineAnswer(entry.baselineAnswer);
                      setMemoryAnswer(nextComparison.answer);
                      setBaselineTrace(emptyLaneTrace());
                      setMemoryTrace(emptyLaneTrace());
                      setAnswerContract("Awaiting model output");
                      if (webLlmStatus !== "ready") {
                        setModelMode("simulated");
                        setModelDiagnostic(
                          "SQLite data ready · WebLLM model not loaded",
                        );
                      }
                    }}
                  />
                  <span className={styles.radioVisual} data-checked={checked}>
                    <span className={styles.radioIcon}>
                      <Icon />
                    </span>
                    <span className={styles.radioCopy}>
                      <strong>{entry.label}</strong>
                    </span>
                    <span className={styles.radioMark} aria-hidden="true">
                      {checked ? "●" : ""}
                    </span>
                  </span>
                </label>
              );
            })}
          </div>
        </div>
        <div className={styles.actions}>
          <button
            className={styles.secondaryButton}
            type="button"
            onClick={() => void handleLoadWebLlm()}
            disabled={webLlmStatus === "loading" || webLlmStatus === "ready"}
          >
            {webLlmStatus === "loading"
              ? `Loading ${Math.round(webLlmProgress.progress * 100)}%`
              : webLlmStatus === "ready"
                ? "7B native-tools model ready"
                : `Load ${WEB_LLM_MODEL_LABEL}`}
          </button>
          <button
            className={styles.primaryButton}
            type="button"
            onClick={() => void handleRunComparison()}
            disabled={running || databaseStatus !== "ready"}
          >
            <PlayIcon />
            {running
              ? "Running model → tool → model…"
              : databaseStatus === "ready"
                ? "Run tool loop"
                : "Starting SQLite…"}
          </button>
          <span className={styles.loadStatus} aria-live="polite">
            {webLlmProgress.text}
          </span>
        </div>
      </section>

      <section className={styles.flowStrip} aria-label="How the lab works">
        {FLOW_STEPS.map((step) => (
          <article key={step.key}>
            <strong>{step.label}</strong>
            <p>{step.detail}</p>
          </article>
        ))}
      </section>

      <section className={styles.ledgerStrip} aria-label="Developer evidence ledger">
        <article className={styles.ledgerCard}>
          <span>Data-only agent</span>
          <strong>{dataTool.name} → prepared SQL · raw rows</strong>
          <pre className={styles.ledgerPacket}>
            {`MODEL PROMPT:\n${nativeToolPrompt(scenario.question, dataTool)}\n\nTOOL SCHEMA:\n${prettyJson(dataTool)}`}
          </pre>
          <dl className={styles.modelOutputList}>
            <div>
              <dt>Model tool call · {formatDuration(baselineTrace.callDurationMs)}</dt>
              <dd>{baselineTrace.callRaw}</dd>
            </div>
            <div>
              <dt>Validated call</dt>
              <dd>
                {baselineTrace.call === null
                  ? "Not validated"
                  : baselineTrace.call.normalizedName
                    ? `Provider label “${baselineTrace.call.originalName}” canonicalized to the sole allowlisted Query tool`
                    : "Query accepted without normalization"}
              </dd>
            </div>
            <div>
              <dt>Executed SQL</dt>
              <dd>
                {databaseStatus === "ready" && baselineTrace.command === "Not executed"
                  ? "SQLite ready · waiting for the model tool call"
                  : baselineTrace.command}
              </dd>
            </div>
            <div>
              <dt>Tool result</dt>
              <dd>
                {databaseStatus === "ready" && baselineTrace.result === "Not executed"
                  ? "Seed data loaded · run the tool loop to query it"
                  : baselineTrace.result}
              </dd>
            </div>
            <div>
              <dt>Final answer prompt</dt>
              <dd>{baselineTrace.finalPrompt}</dd>
            </div>
          </dl>
        </article>
        <article className={styles.ledgerCard}>
          <span>Remembero agent</span>
          <strong>{rememberoTool.name} → Remembero · bindings + proof</strong>
          <pre className={styles.ledgerPacket}>
            {`MODEL PROMPT:\n${nativeToolPrompt(scenario.question, rememberoTool)}\n\nTOOL SCHEMA:\n${prettyJson(rememberoTool)}`}
          </pre>
          <dl className={styles.modelOutputList}>
            <div>
              <dt>Model tool call · {formatDuration(memoryTrace.callDurationMs)}</dt>
              <dd>{memoryTrace.callRaw}</dd>
            </div>
            <div>
              <dt>Validated call</dt>
              <dd>
                {memoryTrace.call === null
                  ? "Not validated"
                  : memoryTrace.call.normalizedName
                    ? `Provider label “${memoryTrace.call.originalName}” canonicalized to the sole allowlisted Query tool`
                    : "Query accepted without normalization"}
              </dd>
            </div>
            <div>
              <dt>Executed Datalog</dt>
              <dd>
                {databaseStatus === "ready" && memoryTrace.command === "Not executed"
                  ? "Remembero ready · waiting for the model tool call"
                  : memoryTrace.command}
              </dd>
            </div>
            <div>
              <dt>Tool result</dt>
              <dd>
                {databaseStatus === "ready" && memoryTrace.result === "Not executed"
                  ? "Seed data loaded · run the tool loop to derive bindings and proof"
                  : memoryTrace.result}
              </dd>
            </div>
            <div>
              <dt>Final answer prompt</dt>
              <dd>{memoryTrace.finalPrompt}</dd>
            </div>
          </dl>
        </article>
        <article className={styles.ledgerCard}>
          <span>Shared SQLite</span>
          <strong>{databaseDiagnostic}</strong>
          <ul className={styles.ledgerList}>
            <li>One in-memory SQLite database for both agents</li>
            <li>Startup probe: {databaseSeedSummary}</li>
            <li>Data Query adapter returns prepared, read-only SQL rows</li>
            <li>Remembero Query adapter uses datalog_query + datalog_explain</li>
            <li>No database snapshot is pasted into either model prompt</li>
          </ul>
          <pre className={styles.ledgerPacket}>{CHAT_MEMORY_SQLITE_VERIFY}</pre>
        </article>
        <article className={styles.ledgerCard}>
          <span>Answer + proof contract</span>
          <strong>{comparison.query}</strong>
          <dl className={styles.modelOutputList}>
            <div>
              <dt>Data-only final raw</dt>
              <dd>{baselineTrace.answerRaw}</dd>
            </div>
            <div>
              <dt>Remembero final raw</dt>
              <dd>{memoryTrace.answerRaw}</dd>
            </div>
          </dl>
          <strong>{answerContract}</strong>
          <ol className={styles.ledgerProof}>
            {hasRun &&
              comparison.proofTrail.map((step) => <li key={step}>{step}</li>)}
            {!hasRun && <li>Proof output appears after the recall run.</li>}
          </ol>
        </article>
      </section>

      <section className={styles.stage}>
        <article className={styles.comparePanel} aria-labelledby="data-only-title">
          <header className={styles.panelHeader}>
            <div>
              <h2 id="data-only-title">Data only</h2>
              <p>Question → model tool call → prepared SQL → raw rows</p>
            </div>
            <span className={styles.panelBadge}>Plugin off</span>
          </header>

          <div className={styles.exchange}>
            <span className={styles.exchangeLabel}>User</span>
            <p className={styles.userBubble}>{scenario.question}</p>
          </div>

          <div className={styles.exchange} aria-live="polite">
            <span className={styles.exchangeLabel}>Answer</span>
            <p className={styles.answerBubble}>
              {hasRun ? baselineAnswer : "Run the comparison to see the baseline answer."}
            </p>
          </div>
        </article>

        <article className={styles.comparePanel} aria-labelledby="with-memory-title">
          <header className={styles.panelHeader}>
            <div>
              <h2 id="with-memory-title">With Remembero</h2>
              <p>Question → model tool call → Datalog → bindings + proof</p>
            </div>
            <span className={styles.panelBadge}>Memory packet</span>
          </header>

          <div className={styles.exchange}>
            <span className={styles.exchangeLabel}>User</span>
            <p className={styles.userBubble}>{scenario.question}</p>
          </div>

          <div className={styles.exchange} aria-live="polite">
            <span className={styles.exchangeLabel}>Answer</span>
            <p className={styles.serifAnswer}>
              {hasRun
                ? memoryAnswer
                : "Run the comparison to execute recall, proof, and answer synthesis."}
            </p>
          </div>
        </article>

      </section>

      <section className={styles.boundary}>
        <p>
          {modelMode === "webllm"
            ? `${WEB_LLM_MODEL_ID} generated native WebLLM tool_calls and final text locally. The app validated each call before executing SQLite; no source rows were embedded in the prompt.`
            : modelMode === "browser"
              ? "This browser’s local language model generated the tool calls and final text; the app validated every call before SQLite execution."
              : `Model calls use a labeled simulator (${modelDiagnostic}); SQLite, SQL, Remembero evaluation, and proof execution are real and browser-local.`}
        </p>
        <a className={styles.guideCta} href="/guides/agent-harness">
          Add this loop to your agent harness →
        </a>
      </section>
    </main>
  );
}
