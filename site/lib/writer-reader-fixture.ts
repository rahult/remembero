/*
 * Writer–reader lab: raw text in, proven answers out — on genuinely messy
 * history.
 *
 * The fixture is six months of chat for one person, containing the shapes
 * that make agent memory hard: a three-deep supersession chain (dentist), a
 * flip-back with a changed price (dashboard tool), an effective-dated role
 * handover, a contract that ends without renewal, relative dates, and one
 * question whose answer was never said at all.
 *
 * Execution boundary, same as the other labs: the writer's claim extraction
 * and the reader's answers are REPLAYS of recorded runs, stamped as such.
 * Everything between them — resolution, dating, supersession intervals, the
 * current-value view, retrieval scoring, and the notes block — is real
 * deterministic code running in the visitor's browser. No model weights are served, fetched, or
 * executed; nothing is stored; nothing leaves the tab.
 */

export const QUESTION_DATE = "2026-06-20";

export interface SessionFixture {
  id: string;
  date: string;
  label: string;
  text: string;
}

export const SESSIONS: SessionFixture[] = [
  {
    id: "s1",
    date: "2026-01-12",
    label: "Chat · January",
    text: "We signed Norsk Dental as a client yesterday. I'm the account lead. Their contract runs twelve months and is worth $84,000.",
  },
  {
    id: "s2",
    date: "2026-02-03",
    label: "Chat · February",
    text: "My dentist moved clinics — I now see Dr Okafor at Riverside; it was Dr Bhatt before. Also finished the inventory audit in three days; it usually takes ten.",
  },
  {
    id: "s3",
    date: "2026-03-09",
    label: "Chat · March",
    text: "We migrated the dashboard off Metabase to Lightdash last Friday. The Metabase license was $1,200 a year; Lightdash is $4,800.",
  },
  {
    id: "s4",
    date: "2026-04-18",
    label: "Chat · April",
    text: "Told the team yesterday I'm stepping off the Norsk Dental account — Priya takes over as lead from May.",
  },
  {
    id: "s5",
    date: "2026-05-06",
    label: "Chat · May",
    text: "The Norsk Dental renewal fell through: the contract ends 31 May and will not renew. Shame — it was the $84,000 one.",
  },
  {
    id: "s6",
    date: "2026-06-14",
    label: "Chat · June",
    text: "Started seeing Dr Liu at Riverside instead — Dr Okafor retired. And we dropped Lightdash; we're back on Metabase, at $1,500 this time.",
  },
];

/* ------------------------------------------------------------------ */
/* Act 1 — the writer: replayed claims, then real deterministic rules  */
/* ------------------------------------------------------------------ */

export type ClaimKind = "state" | "correction" | "end";

export interface RawClaim {
  id: string;
  session: string;
  date: string;
  subject: string;
  predicate: string;
  object: string;
  kind: ClaimKind;
  quote: string;
}

/*
 * What the writer model returned for the six sessions — replayed, not
 * executed here. Each row is the writer's own reading of one sentence:
 * subject, predicate, object, the sentence it came from, and — where the
 * sentence is a change — whether it states, corrects, or ends a value.
 */
export const WRITER_CLAIMS: readonly RawClaim[] = [
  { id: "c1", session: "s1", date: "2026-01-11", subject: "norsk_dental", predicate: "client_of", object: "us", kind: "state", quote: "We signed Norsk Dental as a client yesterday." },
  { id: "c2", session: "s1", date: "2026-01-12", subject: "norsk_dental", predicate: "account_lead", object: "you", kind: "state", quote: "I'm the account lead." },
  { id: "c3", session: "s1", date: "2026-01-12", subject: "norsk_dental", predicate: "contract_value", object: "$84,000 / 12 months", kind: "state", quote: "Their contract runs twelve months and is worth $84,000." },
  { id: "c4", session: "s2", date: "2026-02-03", subject: "you", predicate: "dentist", object: "Dr Okafor (Riverside)", kind: "correction", quote: "My dentist moved clinics — I now see Dr Okafor at Riverside; it was Dr Bhatt before." },
  { id: "c5", session: "s2", date: "2026-02-03", subject: "you", predicate: "audit_duration", object: "3 days (usually 10)", kind: "state", quote: "Finished the inventory audit in three days; it usually takes ten." },
  { id: "c6", session: "s3", date: "2026-03-06", subject: "company", predicate: "dashboard_tool", object: "Lightdash", kind: "correction", quote: "We migrated the dashboard off Metabase to Lightdash last Friday." },
  { id: "c7", session: "s3", date: "2026-03-09", subject: "company", predicate: "dashboard_tool_price", object: "Lightdash $4,800/yr (was Metabase $1,200/yr)", kind: "state", quote: "The Metabase license was $1,200 a year; Lightdash is $4,800." },
  { id: "c8", session: "s4", date: "2026-05-01", subject: "norsk_dental", predicate: "account_lead", object: "priya", kind: "correction", quote: "I'm stepping off the Norsk Dental account — Priya takes over as lead from May." },
  { id: "c9", session: "s5", date: "2026-05-31", subject: "norsk_dental", predicate: "client_of", object: "ended, no renewal", kind: "end", quote: "The contract ends 31 May and will not renew." },
  { id: "c10", session: "s6", date: "2026-06-14", subject: "you", predicate: "dentist", object: "Dr Liu (Riverside)", kind: "correction", quote: "Started seeing Dr Liu at Riverside instead — Dr Okafor retired." },
  { id: "c11", session: "s6", date: "2026-06-14", subject: "company", predicate: "dashboard_tool", object: "Metabase", kind: "correction", quote: "We dropped Lightdash; we're back on Metabase." },
  { id: "c12", session: "s6", date: "2026-06-14", subject: "company", predicate: "dashboard_tool_price", object: "Metabase $1,500/yr", kind: "correction", quote: "We're back on Metabase, at $1,500 this time." },
];

export interface TimelineRow {
  key: string;
  subject: string;
  predicate: string;
  value: string;
  from: string;
  to: string | null;
  quote: string;
  session: string;
  status: "current" | "superseded" | "ended";
}

function parseIso(iso: string): Date {
  const [y, m, d] = iso.split("-").map(Number);
  return new Date(Date.UTC(y, m - 1, d));
}

function toIso(date: Date): string {
  return date.toISOString().slice(0, 10);
}

function diffDays(from: string, to: string): number {
  return Math.round((parseIso(to).getTime() - parseIso(from).getTime()) / 86_400_000);
}

function humanizeDays(days: number): string {
  if (days < 14) return `${days} days`;
  if (days < 70) return `~${Math.max(1, Math.round(days / 7))} weeks`;
  return `~${Math.max(1, Math.round(days / 30))} months`;
}

/*
 * Deterministic resolution: group the writer's claims by subject+predicate,
 * order them in time, and give every value a validity interval. A
 * correction closes the previous value the day the new one takes effect; an
 * `end` closes without a successor. Identical repeated claims collapse.
 * This is code, not the model — run it twice, get the same table.
 */
export function buildTimelines(claims: readonly RawClaim[]): TimelineRow[] {
  const groups = new Map<string, RawClaim[]>();
  for (const claim of claims) {
    const key = `${claim.subject} · ${claim.predicate}`;
    const group = groups.get(key) ?? [];
    group.push(claim);
    groups.set(key, group);
  }

  const rows: TimelineRow[] = [];
  function rowFrom(claim: RawClaim, to: string | null, status: TimelineRow["status"]): TimelineRow {
    return {
      key: `${claim.id}`,
      subject: claim.subject,
      predicate: claim.predicate,
      value: claim.object,
      from: claim.date,
      to,
      quote: claim.quote,
      session: claim.session,
      status,
    };
  }
  for (const group of groups.values()) {
    const ordered = [...group].sort((a, b) => a.date.localeCompare(b.date));
    let open: RawClaim | null = null;
    for (const claim of ordered) {
      if (open !== null) {
        const previousEnd = claim.date < open.date ? open.date : claim.date;
        rows.push(rowFrom(open, previousEnd, claim.kind === "end" ? "ended" : "superseded"));
      }
      if (claim.kind === "end") {
        open = null;
        continue;
      }
      open = claim;
    }
    if (open !== null) rows.push(rowFrom(open, null, "current"));
  }

  return rows.sort((a, b) => `${a.subject}${a.predicate}${a.from}`.localeCompare(`${b.subject}${b.predicate}${b.from}`));
}

/* ------------------------------------------------------------------ */
/* Act 2 — the reader: live retrieval + notes, replayed answers        */
/* ------------------------------------------------------------------ */

const STOP_WORDS = new Set([
  "i", "my", "me", "the", "a", "an", "to", "and", "for", "in", "on", "of",
  "did", "do", "would", "how", "much", "when", "what", "is", "was", "it",
  "at", "after", "long", "if", "now", "so", "we", "as", "before", "them",
  "who", "s", "still", "see", "seeing",
]);

function normalize(word: string): string {
  if (word.length > 4 && word.endsWith("ing")) return word.slice(0, -3);
  if (word.length > 4 && word.endsWith("ed")) return word.slice(0, -2);
  if (word.length > 3 && word.endsWith("s")) return word.slice(0, -1);
  return word;
}

function contentWords(text: string): string[] {
  const raw = text.toLowerCase().match(/[a-z0-9$][a-z0-9$,]*/g) ?? [];
  return raw.filter((w) => !STOP_WORDS.has(w)).map(normalize);
}

export interface RetrievedClaim {
  row: TimelineRow;
  score: number;
  reason: string;
}

export function retrieveForQuestion(question: string, rows: readonly TimelineRow[]): RetrievedClaim[] {
  const wanted = new Set(contentWords(question));
  const scored = rows.map((row) => {
    const haystack = contentWords(`${row.subject} ${row.predicate} ${row.value} ${row.quote}`);
    let score = 0;
    for (const word of wanted) if (haystack.includes(word)) score += 1;
    return { row, score, reason: score > 0 ? `${score} content-word match${score === 1 ? "" : "es"}` : "no direct match" };
  });
  return scored
    .sort((a, b) => b.score - a.score || b.row.from.localeCompare(a.row.from))
    .slice(0, 6);
}

export function buildReaderNotes(question: string, rows: readonly TimelineRow[]): string {
  const lines: string[] = ["STRUCTURED-EVIDENCE NOTES — computed by code", `as of ${QUESTION_DATE} (the question date)`, ""];
  const retrieved = retrieveForQuestion(question, rows).filter((item) => item.score > 0);
  const seenKeys = new Set<string>();

  for (const { row } of retrieved) {
    const key = `${row.subject} · ${row.predicate}`;
    if (seenKeys.has(key)) continue;
    seenKeys.add(key);
    const family = rows
      .filter((candidate) => candidate.subject === row.subject && candidate.predicate === row.predicate)
      .sort((a, b) => a.from.localeCompare(b.from));

    const current = family.find((candidate) => candidate.to === null && candidate.status !== "ended");
    if (current !== undefined) {
      const days = diffDays(current.from, QUESTION_DATE);
      lines.push(`${key}: ${current.value}`);
      lines.push(`    current since ${current.from} (${days} days, ${humanizeDays(days)})`);
    } else {
      const last = family[family.length - 1];
      lines.push(`${key}: no current value — last state ended ${last?.to ?? "unknown"}`);
    }
    const history = family.filter((candidate) => candidate.to !== null);
    if (history.length > 0) {
      const chain = history
        .map((candidate) => `${candidate.value} (${candidate.from} → ${candidate.to})`)
        .join("; ");
      lines.push(`    earlier: ${chain}`);
    }
    lines.push("");
  }

  if (seenKeys.size === 0) {
    lines.push("No claim in memory matches this question's content words.");
    lines.push("Nothing in the six sessions states it, and no rule derives it.");
  }
  return lines.join("\n").trimEnd();
}

export interface ReaderArm {
  label: string;
  answer: string;
  verdict: "miss" | "hit" | "unknown-honest";
  verdictNote: string;
}

export interface QuestionFixture {
  id: string;
  kind: string;
  kindLabel: string;
  question: string;
  gold: string;
  hardBecause: string;
  rawSessionsArm: ReaderArm;
  structuredArm: ReaderArm;
}

export const QUESTIONS: QuestionFixture[] = [
  {
    id: "chain",
    kind: "update-chain",
    kindLabel: "a three-deep update chain",
    question: "Who is my dentist now — and who did I see immediately before them?",
    gold: "Dr Liu now; Dr Okafor immediately before",
    hardBecause: "Three values for one fact across five months. Raw retrieval surfaces the longest-lived or first-matching one; “immediately before” needs the ordered chain, not the loudest row.",
    rawSessionsArm: {
      label: "Reader over raw chat — replay",
      answer: "You see Dr Okafor at Riverside — you mentioned moving clinics. Before that it was Dr Bhatt.",
      verdict: "miss",
      verdictNote: "The June session naming Dr Liu ranked below February's longer, more detailed sentence about the move. The reader had every session and still answered one step stale.",
    },
    structuredArm: {
      label: "Reader over structured evidence — replay",
      answer: "Dr Liu, since 14 June. Immediately before: Dr Okafor (3 February → 14 June). The February note also names Dr Bhatt before Okafor — kept as a source quote, not a dated row, because the history never dates that era.",
      verdict: "hit",
      verdictNote: "“Now” and “immediately before” are lookups in a dated, ordered chain. The chain holds only what the history dates — Dr Bhatt stays attached to his sentence instead of being invented an era.",
    },
  },
  {
    id: "flip",
    kind: "flip-back",
    kindLabel: "a flip-back with a changed price",
    question: "What are we paying for the dashboard tool now, and what did we pay before the switch back?",
    gold: "Metabase $1,500/yr now; Lightdash $4,800/yr before",
    hardBecause: "The current tool is one the company previously left. Three prices for two products across four months; the oldest price shares its product name with the current one.",
    rawSessionsArm: {
      label: "Reader over raw chat — replay",
      answer: "Metabase — $1,200 a year. Lightdash was $4,800 before you moved back.",
      verdict: "miss",
      verdictNote: "“Metabase” matched January's $1,200 price as strongly as June's $1,500 — same name, different era. The reader conflated them.",
    },
    structuredArm: {
      label: "Reader over structured evidence — replay",
      answer: "Metabase at $1,500/year since 14 June. Before the switch back, Lightdash at $4,800/year. The earlier $1,200 Metabase price ended in March.",
      verdict: "hit",
      verdictNote: "Each price carries its own validity interval; same-name prices from different eras cannot collide.",
    },
  },
  {
    id: "role",
    kind: "effective-dated",
    kindLabel: "an effective-dated handover and an ending",
    question: "Who leads the Norsk Dental account today — and is the account still active?",
    gold: "No lead today: the account ended 31 May; Priya had taken over in May",
    hardBecause: "Two facts interlock: the lead changes with an effective date, and the account itself ends on a stated date. Answering “today” requires both, in order.",
    rawSessionsArm: {
      label: "Reader over raw chat — replay",
      answer: "You lead the Norsk Dental account — you said you're the account lead.",
      verdict: "miss",
      verdictNote: "January's confident “I'm the account lead” outranked April's handover and May's ending for directness. True then; five months stale now.",
    },
    structuredArm: {
      label: "Reader over structured evidence — replay",
      answer: "Neither — the account is no longer active: the contract ended 31 May with no renewal. Priya had become lead effective May, so the lead question is moot today.",
      verdict: "hit",
      verdictNote: "The end closes the account's interval; code marks it ended, and the notes lead with the current state — inactive.",
    },
  },
  {
    id: "abstain",
    kind: "abstention",
    kindLabel: "a question never answered anywhere",
    question: "How many people are on Priya's team?",
    gold: "Not in memory",
    hardBecause: "The name Priya appears in the history, so retrieval returns real text — none of it contains the answer. The hard part is refusing to answer with material that merely sounds adjacent.",
    rawSessionsArm: {
      label: "Reader over raw chat — replay",
      answer: "Priya's team has four people.",
      verdict: "miss",
      verdictNote: "The reader had the name and a confident tone, and invented the number. Nothing in six months of chat mentions her team.",
    },
    structuredArm: {
      label: "Reader over structured evidence — replay",
      answer: "Not in memory — the history records Priya's role on the account, never the size of her team.",
      verdict: "unknown-honest",
      verdictNote: "The notes surface Priya's role because the name matches — adjacent, not answering. No team-size claim exists, so the honest state is named instead of a number being invented.",
    },
  },
];

export const WRITER_REPLAY_NOTE =
  "The claim rows are a replay of a recorded writer run over this fixture. What happens to them next — grouping, dating, interval building — is deterministic code in your browser.";

export const READER_REPLAY_NOTE =
  "Both reader answers are replays of recorded arms: same question, same underlying history, raw chats versus structured evidence. No model executes on this page.";

export const SCOPE_NOTE =
  "The fixture's six sessions compress the shapes that dominate our benchmark misses: update chains, flip-backs, effective-dated roles, lifecycles that end, and questions whose answers were never said.";
