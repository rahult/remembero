/**
 * The source model: everything the stack reads — a ticket, a chat transcript, a long policy
 * document — reduces to the same unit, an ordered span with an optional actor and timestamp.
 * The claim gate (claims.ts) admits a claim only against the single span it cites, so the
 * span model is what makes "long document" and "small ticket" the same problem: a page line
 * and a chat turn are both just spans.
 *
 * Regime handling follows the plan's split: long POLICY text is indexed and windowed (a
 * policy enters the engine once, as a hand-gated pack); long CASES (a 200-message thread)
 * are extracted whole, because a thread is small enough to extract and too paraphrastic to
 * retrieve over. Both paths produce spans, so the layers above cannot tell them apart.
 */

export type SpanKind = 'field' | 'turn' | 'comment' | 'policy-line';

export interface Span {
  sourceId: string;
  spanId: string;
  kind: SpanKind;
  text: string;
  /** Who said or owns this span: the agent, the requester, the ticket system. */
  actor?: string;
  /** ISO instant the span happened or took effect. Missing => spans are `unanchored`. */
  at?: string;
}

export interface Source {
  sourceId: string;
  kind: 'ticket' | 'chat' | 'policy';
  spans: Span[];
}

let spanCounter = 0;
function nextSpanId(prefix: string): string {
  spanCounter += 1;
  return `${prefix}-${String(spanCounter).padStart(4, '0')}`;
}

/** A structured ticket: fields become `field` spans quoted verbatim from the field value. */
export interface TicketInput {
  id: string;
  /** The record's identity context, rendered into every field span so a system claim's
   *  subject is on the span it cites — the gate then applies to system claims unchanged. */
  customer?: string;
  fields: Record<string, string | number | undefined>;
  comments?: Array<{ actor: string; at?: string; text: string; spanId?: string }>;
}

export function ticketSource(input: TicketInput): Source {
  const context = input.customer ? `${input.id} customer ${input.customer}` : input.id;
  const spans: Span[] = Object.entries(input.fields).flatMap(([name, value]) => {
    if (value === undefined || value === '') return [];
    return [{
      sourceId: input.id,
      spanId: `field.${name}`,
      kind: 'field' as const,
      text: `${context} ${name}: ${String(value)}`,
      actor: 'ticket-system',
    }];
  });
  for (const c of input.comments ?? []) {
    spans.push({
      sourceId: input.id,
      spanId: c.spanId ?? nextSpanId('comment'),
      kind: 'comment',
      text: `${context} ${c.text}`,
      actor: c.actor,
      at: c.at,
    });
  }
  return { sourceId: input.id, kind: 'ticket', spans };
}

/** A chat transcript: one span per turn, in order, each with speaker and timestamp. */
export interface ChatInput {
  id: string;
  turns: Array<{ actor: string; at?: string; text: string }>;
}

export function chatSource(input: ChatInput): Source {
  return {
    sourceId: input.id,
    kind: 'chat',
    spans: input.turns.map((t, i) => ({
      sourceId: input.id,
      spanId: `turn-${String(i + 1).padStart(3, '0')}`,
      kind: 'turn' as const,
      text: t.text,
      actor: t.actor,
      at: t.at,
    })),
  };
}

/** A policy document: one span per line, line numbers kept so pack claims can cite offsets. */
export function policySource(id: string, text: string): Source {
  const lines = text.split('\n').filter((l) => l.trim().length > 0);
  return {
    sourceId: id,
    kind: 'policy',
    spans: lines.map((line, i) => ({
      sourceId: id,
      spanId: `p${String(i + 1).padStart(3, '0')}`,
      kind: 'policy-line' as const,
      text: line.trim(),
    })),
  };
}

export function findSpan(sources: Source[], sourceId: string, spanId: string): Span | undefined {
  return sources.find((s) => s.sourceId === sourceId)?.spans.find((sp) => sp.spanId === spanId);
}
