import type { ChatMemoryScenarioId } from "./chat-memory-lab";
import type {
  BrowserDatalogDatabase,
  BrowserDatalogExplanation,
  SqliteRow,
} from "./sqlite-wasm";

export type ChatToolLane = "data" | "remembero";

export interface ChatToolDefinition {
  name: "Query";
  description: string;
  parameters: {
    type: "object";
    properties: {
      query: {
        type: "string";
        enum: readonly [string];
      };
    };
    required: readonly ["query"];
    additionalProperties: false;
  };
}

export interface ParsedChatToolCall {
  name: "Query";
  arguments: { query: string };
  lane: ChatToolLane;
  scenarioId: ChatMemoryScenarioId;
  originalName: string;
  normalizedName: boolean;
}

export interface ChatToolExecution {
  command: string;
  result: {
    rows: SqliteRow[] | Array<Record<string, string | number>>;
    explanations?: BrowserDatalogExplanation[];
    refused?: boolean;
    verdict?: string;
    premises?: Array<{
      literal: string;
      holds: boolean;
      rows: Array<Record<string, string | number>>;
    }>;
  };
  durationMs: number;
}

export interface ChatMemorySeedVerification {
  tableCount: number;
  rowCount: number;
}

export const CHAT_MEMORY_SQLITE_SETUP = `
  DROP TABLE IF EXISTS status;
  DROP TABLE IF EXISTS blocker;
  DROP TABLE IF EXISTS waits_on;
  DROP TABLE IF EXISTS prefers_meeting;
  DROP TABLE IF EXISTS pending_meeting;
  DROP TABLE IF EXISTS review_slot;

  CREATE TABLE status(project TEXT NOT NULL, state TEXT NOT NULL);
  CREATE TABLE blocker(project TEXT NOT NULL, blocker TEXT NOT NULL);
  CREATE TABLE waits_on(item TEXT NOT NULL, upstream TEXT NOT NULL);
  CREATE TABLE prefers_meeting(person TEXT NOT NULL, window TEXT NOT NULL);
  CREATE TABLE pending_meeting(person TEXT NOT NULL, meeting TEXT NOT NULL);
  CREATE TABLE review_slot(project TEXT NOT NULL, day TEXT NOT NULL, window TEXT NOT NULL);

  INSERT INTO status VALUES ('atlas', 'blocked'), ('orchard', 'active');
  INSERT INTO blocker VALUES ('atlas', 'vendor_security_review');
  INSERT INTO waits_on VALUES
    ('atlas', 'vendor_security_review'),
    ('vendor_security_review', 'legal_signoff'),
    ('legal_signoff', 'procurement_freeze');
  INSERT INTO prefers_meeting VALUES ('maya', 'morning');
  INSERT INTO pending_meeting VALUES ('jordan', 'roadmap_sync');
  INSERT INTO review_slot VALUES ('atlas', 'tuesday', 'morning');
`;

export const CHAT_MEMORY_SQLITE_VERIFY = `SELECT
  (SELECT COUNT(*)
     FROM sqlite_schema
    WHERE type = 'table'
      AND name IN (
        'status',
        'blocker',
        'waits_on',
        'prefers_meeting',
        'pending_meeting',
        'review_slot'
      )) AS tableCount,
  (SELECT COUNT(*) FROM status)
    + (SELECT COUNT(*) FROM blocker)
    + (SELECT COUNT(*) FROM waits_on)
    + (SELECT COUNT(*) FROM prefers_meeting)
    + (SELECT COUNT(*) FROM pending_meeting)
    + (SELECT COUNT(*) FROM review_slot) AS rowCount`;

export async function verifyChatMemorySeed(
  database: BrowserDatalogDatabase,
): Promise<ChatMemorySeedVerification> {
  const [row] = await database.exec(CHAT_MEMORY_SQLITE_VERIFY);
  const tableCount = row?.tableCount;
  const rowCount = row?.rowCount;
  if (typeof tableCount !== "number" || typeof rowCount !== "number") {
    throw new Error("SQLite seed verification returned no counts");
  }
  if (tableCount !== 6 || rowCount !== 9) {
    throw new Error(
      `SQLite seed verification expected 6 tables and 9 rows, received ${tableCount} tables and ${rowCount} rows`,
    );
  }
  return { tableCount, rowCount };
}

const SQL_BY_CASE: Record<ChatMemoryScenarioId, string> = {
  "root-blocker": `WITH RECURSIVE chain(item, upstream) AS (
  SELECT item, upstream FROM waits_on WHERE item = 'atlas'
  UNION
  SELECT w.item, w.upstream FROM waits_on AS w
  JOIN chain AS c ON w.item = c.upstream
)
SELECT upstream AS dependency FROM chain`,
  "write-gate": `SAVEPOINT lab_demo;
UPDATE status SET state = 'active' WHERE project = 'atlas';
SELECT s.project, s.state, r.day, r.window
  FROM status AS s
  JOIN review_slot AS r ON r.project = s.project
 WHERE s.project = 'atlas';
ROLLBACK TO lab_demo; -- keeps the lab repeatable; the UPDATE itself raised no objection`,
  "unknown-preference": `SELECT m.person, m.meeting, p.window AS stored_preference
FROM pending_meeting AS m
LEFT JOIN prefers_meeting AS p ON p.person = m.person
WHERE m.person = 'jordan'`,
  "why-not": `SELECT s.project, r.day, r.window, b.blocker
FROM status AS s
JOIN review_slot AS r ON r.project = s.project
JOIN blocker AS b ON b.project = s.project
JOIN prefers_meeting AS p ON p.person = 'maya' AND p.window = r.window
WHERE s.project = 'orchard' AND s.state = 'blocked'`,
};

const DATALOG_BY_CASE: Record<ChatMemoryScenarioId, string> = {
  "root-blocker": `root_blocker(Root) :- reaches(atlas, Root), \\+ waits_on(Root, _).
reaches(P, X) :- waits_on(P, X).
reaches(P, X) :- reaches(P, M), waits_on(M, X).`,
  "write-gate": `violation(P, B) :- proposed_status(P, active), blocker(P, B).`,
  "unknown-preference": `missing_preference(Person) :-
  pending_meeting(Person, _),
  \\+ prefers_meeting(Person, _).`,
  "why-not": `premise checks for schedule_review(orchard, Day, Window, Blocker)`,
};

/** Body literals of the schedule rule, checked one at a time for orchard. */
export const WHY_NOT_PREMISES: ReadonlyArray<{
  literal: string;
  program: string;
}> = [
  {
    literal: "review_slot(orchard, Day, Window)",
    program: `orchard_slot(Day, Window) :- review_slot(orchard, Day, Window).`,
  },
  {
    literal: "status(orchard, blocked)",
    program: `orchard_blocked(State) :- status(orchard, State), State = blocked.`,
  },
  {
    literal: "blocker(orchard, Blocker)",
    program: `orchard_blocker(Blocker) :- blocker(orchard, Blocker).`,
  },
  {
    literal: "prefers_meeting(maya, Window)",
    program: `maya_window(Window) :- prefers_meeting(maya, Window).`,
  },
];

export function chatToolDefinition(
  lane: ChatToolLane,
  question: string,
): ChatToolDefinition {
  return {
    name: "Query",
    description:
      lane === "data"
        ? "Run the prepared read-only SQL query for the active case and return raw SQLite rows."
        : "Evaluate the prepared Remembero relation for the active case and return bindings plus proof from SQLite.",
    parameters: {
      type: "object",
      properties: {
        query: { type: "string", enum: [question] },
      },
      required: ["query"],
      additionalProperties: false,
    },
  };
}

export function parseChatToolCall(
  raw: string,
  expected: ChatToolDefinition,
  lane: ChatToolLane,
  scenarioId: ChatMemoryScenarioId,
  question: string,
): ParsedChatToolCall | null {
  try {
    const parsed = JSON.parse(raw) as {
      name?: unknown;
      tool?: unknown;
      arguments?: unknown;
      args?: unknown;
    };
    const name = parsed.name ?? parsed.tool;
    let argumentsValue = parsed.arguments ?? parsed.args;
    if (typeof argumentsValue === "string") {
      argumentsValue = JSON.parse(argumentsValue) as unknown;
    }
    if (
      typeof name !== "string" ||
      name.trim().length === 0 ||
      typeof argumentsValue !== "object" ||
      argumentsValue === null ||
      !("query" in argumentsValue) ||
      (argumentsValue as { query?: unknown }).query !== question
    ) {
      return null;
    }
    return {
      name: expected.name,
      arguments: { query: question },
      lane,
      scenarioId,
      originalName: name,
      normalizedName: name.toLowerCase() !== expected.name.toLowerCase(),
    };
  } catch {
    return null;
  }
}

export function simulatedChatToolCall(
  definition: ChatToolDefinition,
  lane: ChatToolLane,
  scenarioId: ChatMemoryScenarioId,
  question: string,
): ParsedChatToolCall {
  return {
    name: definition.name,
    arguments: { query: question },
    lane,
    scenarioId,
    originalName: definition.name,
    normalizedName: false,
  };
}

export async function executeChatTool(
  database: BrowserDatalogDatabase,
  call: ParsedChatToolCall,
): Promise<ChatToolExecution> {
  const started = performance.now();
  if (call.lane === "data") {
    const command = SQL_BY_CASE[call.scenarioId];
    if (call.scenarioId === "write-gate") {
      await database.exec("SAVEPOINT lab_demo");
      try {
        await database.exec(
          "UPDATE status SET state = 'active' WHERE project = 'atlas'",
        );
        const rows = await database.exec(
          `SELECT s.project, s.state, r.day, r.window
             FROM status AS s
             JOIN review_slot AS r ON r.project = s.project
            WHERE s.project = 'atlas'`,
        );
        return {
          command,
          result: { rows },
          durationMs: performance.now() - started,
        };
      } finally {
        await database.exec("ROLLBACK TO lab_demo");
        await database.exec("RELEASE lab_demo");
      }
    }
    const rows = await database.exec(command);
    return {
      command,
      result: { rows },
      durationMs: performance.now() - started,
    };
  }

  const command = DATALOG_BY_CASE[call.scenarioId];
  if (call.scenarioId === "write-gate") {
    await database.exec("SAVEPOINT remembero_gate");
    try {
      await database.exec(
        "CREATE TABLE proposed_status(project TEXT NOT NULL, state TEXT NOT NULL)",
      );
      await database.exec("INSERT INTO proposed_status VALUES ('atlas', 'active')");
      const violations = await database.datalogQuery(command);
      return {
        command,
        result: {
          rows: violations,
          refused: violations.length > 0,
          verdict:
            violations.length > 0
              ? "Write refused: the proposed status contradicts a recorded blocker. Transaction rolled back; stored knowledge unchanged."
              : "No violation derived; the write would have been accepted.",
        },
        durationMs: performance.now() - started,
      };
    } finally {
      await database.exec("ROLLBACK TO remembero_gate");
      await database.exec("RELEASE remembero_gate");
    }
  }
  if (call.scenarioId === "why-not") {
    const premises = [];
    for (const premise of WHY_NOT_PREMISES) {
      const rows = await database.datalogQuery(premise.program);
      premises.push({
        literal: premise.literal,
        holds: rows.length > 0,
        rows,
      });
    }
    return {
      command: WHY_NOT_PREMISES.map((premise) => premise.program).join("\n"),
      result: {
        rows: [],
        premises,
        verdict:
          "schedule_review(orchard, …) is underivable; the failing premises above name the missing conditions.",
      },
      durationMs: performance.now() - started,
    };
  }
  const [rows, explanations] = await Promise.all([
    database.datalogQuery(command),
    database.datalogExplain(command),
  ]);
  return {
    command,
    result: { rows, explanations },
    durationMs: performance.now() - started,
  };
}
