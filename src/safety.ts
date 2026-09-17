export const MAX_INPUT_BYTES = 64 * 1024;
export const MAX_OUTPUT_BYTES = 16 * 1024 * 1024;
export const MAX_NAMESPACE_COUNT = 32;
export const REDACTED_SOURCE = '[sensitive source omitted]';
/** What replaces one matched secret when only the span is hidden. */
export const REDACTED_SPAN = '[redacted]';

// Detecting a credential passed as an argument needs only the opening paren;
// masking needs the arguments too, or the value survives in the clear.
const SENSITIVE_CALL_PATTERN =
  /\b(?:api[_ -]?key|password|passwd|secret|access[_ -]?token|refresh[_ -]?token|account[_ -]?number|credit[_ -]?card)\s*\(/i;
const SENSITIVE_TEXT_PATTERNS = [
  /\b(?:api[_ -]?key|password|passwd|secret|access[_ -]?token|refresh[_ -]?token|account[_ -]?number|credit[_ -]?card)\b["']?\s*(?:is|=|:)\s*["']?\S+/i,
  SENSITIVE_CALL_PATTERN,
  /\b(?:my|your|the)\s+(?:api[_ -]?key|password|passwd|secret|access[_ -]?token|refresh[_ -]?token)\s+(?=\S*[0-9._~+/=-])\S{6,}/i,
  /\b(?:bearer\s+)[a-z0-9._~+/=-]{8,}/i,
  /\b(?:sk|gh[pousr])[-_][a-z0-9_-]{8,}/i,
];

// Timestamps, build IDs, and session stamps are 13+ digit runs too; only a
// Luhn-valid run in the card-number length range is treated as sensitive.
const CARD_CANDIDATE_PATTERN = /\b(?:\d[ -]*?){13,19}\b/g;

function luhnValid(digits: string): boolean {
  let sum = 0;
  let double = false;
  for (let index = digits.length - 1; index >= 0; index -= 1) {
    let digit = digits.charCodeAt(index) - 48;
    if (double) {
      digit *= 2;
      if (digit > 9) digit -= 9;
    }
    sum += digit;
    double = !double;
  }
  return sum % 10 === 0;
}

function containsCardNumber(value: string): boolean {
  for (const candidate of value.match(CARD_CANDIDATE_PATTERN) ?? []) {
    const digits = candidate.replace(/[ -]/g, '');
    if (digits.length >= 13 && digits.length <= 19 && luhnValid(digits)) return true;
  }
  return false;
}

export function assertBoundedInput(value: string, label: string): void {
  const bytes = Buffer.byteLength(value, 'utf8');
  if (bytes > MAX_INPUT_BYTES) {
    throw new Error(`${label} exceeds ${MAX_INPUT_BYTES} bytes`);
  }
}

export function normalizeUnicodeScalarText(value: string): string {
  return value.replace(/[\uD800-\uDFFF]/g, (unit, offset) => {
    const code = unit.charCodeAt(0);
    if (code >= 0xD800 && code <= 0xDBFF) {
      const next = value.charCodeAt(offset + 1);
      if (next >= 0xDC00 && next <= 0xDFFF) return unit;
    } else {
      const previous = value.charCodeAt(offset - 1);
      if (previous >= 0xD800 && previous <= 0xDBFF) return unit;
    }
    return '\uFFFD';
  });
}

export function assertBoundedOutput(
  value: string,
  label = 'output',
  maxBytes = MAX_OUTPUT_BYTES
): void {
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 0) {
    throw new Error('output byte limit must be a non-negative safe integer');
  }
  const bytes = Buffer.byteLength(value, 'utf8');
  if (bytes > maxBytes) throw new Error(`${label} exceeds ${maxBytes} bytes`);
}

export function stringifyBoundedResult(
  value: unknown,
  label = 'result',
  maxBytes = MAX_OUTPUT_BYTES
): string {
  const text = JSON.stringify(
    value,
    (_key, current: unknown) => {
      if (typeof current === 'number' && !Number.isFinite(current)) {
        throw new Error(`${label} contains a non-finite number`);
      }
      return current;
    },
    2
  );
  if (text === undefined) throw new Error(`${label} is not JSON serializable`);
  assertBoundedOutput(text, label, maxBytes);
  return text;
}

export function containsSensitiveText(value: string): boolean {
  return (
    SENSITIVE_TEXT_PATTERNS.some((pattern) => pattern.test(value)) ||
    containsCardNumber(value)
  );
}

/** A per-call global copy: a shared global regex would carry `lastIndex` between calls. */
function globalCopy(pattern: RegExp): RegExp {
  return pattern.flags.includes('g')
    ? new RegExp(pattern.source, pattern.flags)
    : new RegExp(pattern.source, `${pattern.flags}g`);
}

/**
 * How far past the opening paren a credential call's arguments may run. A turn is
 * often a whole formatted code block, so the span has to cross newlines — stopping
 * at the first one left `password(\n 'hunter2'\n)` readable — but it must not
 * swallow a whole turn either, or masking is no better than throwing the passage
 * away. 512 characters covers a realistically formatted call, roughly ten lines or
 * one long token.
 *
 * The cap cannot be a silent truncation. It consumes the credential word itself,
 * so whatever survives past it is context-free and matches no pattern:
 * `containsSensitiveText` on the result is false even though the value is right
 * there. A span that stops for any reason other than its own closing paren is
 * therefore reported as `truncated`, and the session store keeps nothing of such a
 * turn.
 */
const MAX_CALL_SPAN_CHARS = 512;

/** Why a call span ended. Anything but `closed` means the masking is inconclusive. */
type CallSpanStop = 'closed' | 'blank-line' | 'cap' | 'end-of-text';

/** True when the line beginning at `start` holds nothing but whitespace. */
function isBlankLineAt(value: string, start: number): boolean {
  for (let index = start; index < value.length; index += 1) {
    const character = value[index];
    if (character === '\n') return true;
    if (character !== ' ' && character !== '\t' && character !== '\r') {
      return false;
    }
  }
  return false;
}

/**
 * Where a credential call's arguments end: the matching close paren, counting
 * nesting; failing that a blank line, which ends the code block and lets the prose
 * after it survive; failing that `MAX_CALL_SPAN_CHARS`. A regex cannot do this —
 * `[^)]*\)` stops at the first close paren of a nested call and matches nothing at
 * all when there is no close paren, which left the secret in the clear.
 */
function callSpan(
  value: string,
  afterOpenParen: number,
): { end: number; stop: CallSpanStop } {
  const capped = afterOpenParen + MAX_CALL_SPAN_CHARS;
  const limit = Math.min(value.length, capped);
  let depth = 1;
  for (let index = afterOpenParen; index < limit; index += 1) {
    const character = value[index];
    if (character === '(') depth += 1;
    else if (character === ')') {
      depth -= 1;
      if (depth === 0) return { end: index + 1, stop: 'closed' };
    } else if (character === '\n' && isBlankLineAt(value, index + 1)) {
      return { end: index, stop: 'blank-line' };
    }
  }
  return { end: limit, stop: limit === capped ? 'cap' : 'end-of-text' };
}

/** Mask `password(...)` and friends, arguments and all. */
function maskCallSpans(value: string): {
  text: string;
  masked: number;
  truncated: boolean;
} {
  const finder = globalCopy(SENSITIVE_CALL_PATTERN);
  let text = '';
  let cursor = 0;
  let masked = 0;
  let truncated = false;
  let match = finder.exec(value);
  while (match !== null) {
    // A nested credential call inside a span already masked adds nothing.
    if (match.index >= cursor) {
      const span = callSpan(value, match.index + match[0].length);
      text += value.slice(cursor, match.index) + REDACTED_SPAN;
      masked += 1;
      if (span.stop !== 'closed') truncated = true;
      cursor = span.end;
      finder.lastIndex = span.end;
    }
    match = finder.exec(value);
  }
  return { text: text + value.slice(cursor), masked, truncated };
}

/**
 * Replace each sensitive span with `[redacted]` and keep the rest of the text.
 * `redactSensitiveText` throws the whole passage away, which is right for a fact's
 * source line; a stored conversation turn is mostly ordinary text, so the session
 * store masks the secret and keeps what is readable. The widest span goes first,
 * so the call form takes its arguments with it and the assignment form
 * ("api key = sk-...") consumes the bare token inside it: one masked span each.
 *
 * `truncated` says a span had to stop before its own closing paren, so part of a
 * secret may still be in `text` even though nothing in it looks sensitive any more.
 * A caller that cannot judge the text itself — the session store — must throw the
 * whole passage away when this is set; `masked` alone does not tell it that.
 */
export function maskSensitiveSpans(value: string): {
  text: string;
  masked: number;
  truncated: boolean;
} {
  const calls = maskCallSpans(value);
  let text = calls.text;
  let masked = calls.masked;
  for (const pattern of SENSITIVE_TEXT_PATTERNS) {
    text = text.replace(globalCopy(pattern), () => {
      masked += 1;
      return REDACTED_SPAN;
    });
  }
  text = text.replace(globalCopy(CARD_CANDIDATE_PATTERN), (candidate) => {
    const digits = candidate.replace(/[ -]/g, '');
    if (digits.length < 13 || digits.length > 19 || !luhnValid(digits)) {
      return candidate;
    }
    masked += 1;
    return REDACTED_SPAN;
  });
  return { text, masked, truncated: calls.truncated };
}

export function redactSensitiveText(value: string): { text: string; redacted: boolean } {
  return containsSensitiveText(value)
    ? { text: REDACTED_SOURCE, redacted: true }
    : { text: value, redacted: false };
}

export function assertSafeForExternalLlm(value: string, label: string): void {
  assertBoundedInput(value, label);
  if (containsSensitiveText(value)) {
    throw new Error(`refusing to send sensitive ${label} to the external LLM`);
  }
}

export function assertNamespaceCount(namespaces: string[] | '*'): void {
  if (namespaces !== '*' && namespaces.length > MAX_NAMESPACE_COUNT) {
    throw new Error(`namespace list exceeds ${MAX_NAMESPACE_COUNT} entries`);
  }
}

export function llmNamespaceAllowlistFromEnv(
  env: NodeJS.ProcessEnv = process.env
): ReadonlySet<string> | undefined {
  const configured = env.REMBERO_LLM_ALLOWED_NAMESPACES;
  if (configured === undefined) return undefined;
  return new Set(
    configured
      .split(',')
      .map((namespace) => namespace.trim())
      .filter(Boolean)
  );
}
