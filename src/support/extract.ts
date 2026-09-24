/**
 * Extraction: the LLM is a sensor. One prompt per span (a chat turn, a comment, a policy
 * line) asks for candidate claims in the registered schema, each with a verbatim quote; the
 * admission gate (claims.ts) then decides what enters the fact base. The model never sees
 * the rules, never sees other spans, and cannot smuggle a claim past the gate — an invented
 * actor or a fact assembled from two turns comes back `rejected` with its reason.
 *
 * The gate's rejection list is kept, not hidden: it is the false-admit/over-reject evidence
 * the gate challenge set measures.
 */

import type { OpenRouterClient } from '../llm/client.js';
import { PREDICATES, admitAll, type ClaimInput } from './claims.js';
import type { Source, TicketInput } from './sources.js';

const FIELD_PREDICATES: Record<string, string> = {
  opened: 'opened',
  first_response: 'first_response',
  priority: 'priority',
  customer: 'customer_of',
  tier: 'tier',
  monthly_fee: 'monthly_fee',
  calendar: 'calendar_of',
};

/**
 * Deterministic claims from structured ticket fields — the system-of-record path. No model:
 * the field name names the predicate, the field value is the argument, and the same gate
 * still checks both against the span (which carries the record's identity context).
 */
export function claimsFromTicket(ticket: TicketInput): ClaimInput[] {
  const inputs: ClaimInput[] = [];
  for (const [name, value] of Object.entries(ticket.fields)) {
    const predicate = FIELD_PREDICATES[name];
    if (predicate === undefined || value === undefined || value === '') continue;
    const args: (string | number)[] =
      predicate === 'customer_of' ? [ticket.id, String(value)]
        : predicate === 'tier' || predicate === 'monthly_fee' ? [String(ticket.customer ?? ''), predicate === 'monthly_fee' ? Number(value) : String(value)]
        : [ticket.id, typeof value === 'number' ? value : String(value)];
    inputs.push({ scope: 'case', predicate, args, sourceId: ticket.id, spanId: `field.${name}`, trust: 'system' });
  }
  return inputs;
}

export function extractionPrompt(): string {
  const registry = Object.entries(PREDICATES)
    .map(([name, spec]) => `- ${name}(${spec.args.join(', ')}) — ${spec.note} [${spec.scope}]`)
    .join('\n');
  return `You extract facts from one span of a support record into a fixed predicate schema. You extract claims only; you never draw conclusions and never state anything negative or absent.

For the span you are given, output one JSON object per line, no prose, no code fence:
{"predicate": "<name>", "args": [<typed arguments>], "quote": "<the exact words of this span that state the fact>"}

Argument types: id and word are lowercase strings; date is YYYY-MM-DD (or full ISO timestamp); money, minutes, hours and percent are numbers.
Every argument must be stated in the span itself. If a value is not literally on the span, do not invent it. Output nothing if the span states none of the predicates.

Schema:
${registry}`;
}

export interface ExtractionReport {
  inputs: ClaimInput[];
  admitted: import('./claims.js').Claim[];
  admittedCount: number;
  rejectedCount: number;
  rejections: Array<{ predicate: string; reason: string }>;
  /** Spans whose extraction call failed — a gap that must be visible, never silent. */
  failedSpans: number;
  lastError?: string;
}

/** Extract from every span of the case sources, gate each claim, and report both sides. */
export async function extractClaims(
  client: OpenRouterClient,
  sources: Source[],
  opts: { concurrency?: number; kinds?: Array<'field' | 'turn' | 'comment'> } = {},
): Promise<ExtractionReport> {
  const kinds = new Set(opts.kinds ?? ['turn', 'comment']);
  const spans = sources.flatMap((s) => s.spans.filter((sp) => kinds.has(sp.kind)));
  const inputs: ClaimInput[] = [];
  let failedSpans = 0;
  let lastError: string | undefined;
  const { mapConcurrent } = await import('../evals/map-concurrent.js');
  await mapConcurrent(spans, opts.concurrency ?? 8, async (span) => {
    try {
      const reply = await client.completeWithUsage([
        { role: 'system', content: extractionPrompt() },
        { role: 'user', content: `source ${span.sourceId} span ${span.spanId}${span.actor ? ` (actor ${span.actor})` : ''}${span.at ? ` at ${span.at}` : ''}:\n\n${span.text}` },
      ]);
      for (const line of reply.content.split('\n')) {
        const trimmed = line.trim().replace(/^,/, '');
        if (!trimmed.startsWith('{')) continue;
        try {
          const parsed = JSON.parse(trimmed) as { predicate?: string; args?: (string | number)[]; quote?: string };
          if (!parsed.predicate || !Array.isArray(parsed.args)) continue;
          inputs.push({
            scope: 'case',
            predicate: parsed.predicate,
            args: parsed.args,
            sourceId: span.sourceId,
            spanId: span.spanId,
            trust: 'model',
            at: span.at,
          });
        } catch {
          // a malformed line is not a claim; it is noise the gate never sees
        }
      }
    } catch (error) {
      // an unreachable span is an extraction gap the run reports, never hides
      failedSpans += 1;
      lastError = error instanceof Error ? error.message.slice(0, 200) : String(error).slice(0, 200);
    }
  });
  const report = admitAll(inputs, sources);
  return {
    inputs,
    admitted: report.admitted,
    admittedCount: report.admitted.length,
    rejectedCount: report.rejected.length,
    rejections: report.rejected.map((r) => ({ predicate: r.predicate, reason: r.reason })),
    failedSpans,
    lastError,
  };
}
