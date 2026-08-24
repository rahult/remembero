import { readFileSync } from 'node:fs';
import { writeFileSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { basename } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  UNLIMITED_OCR_PROMPT,
  parseUnlimitedOcrGrounding,
  type UnlimitedOcrGroundingBlock,
} from '../document/unlimited-ocr.js';

export const DEFAULT_UNLIMITED_OCR_SPACE_URL = 'https://baidu-unlimited-ocr.hf.space';
export const DEFAULT_UNLIMITED_OCR_LIVE_MODE = 'gundam';
export const UNLIMITED_OCR_LIVE_MAX_IMAGE_BYTES = 20 * 1024 * 1024;
export const UNLIMITED_OCR_LIVE_MAX_TEXT_CHARS = 1 * 1024 * 1024;
export const UNLIMITED_OCR_LIVE_DEFAULT_TIMEOUT_MS = 180_000;
export const UNLIMITED_OCR_LIVE_MAX_SSE_EVENTS = 20_000;
export const UNLIMITED_OCR_LIVE_SAVE_MARKER = '===============save results:===============';

export interface UnlimitedOcrLiveDocument {
  id: string;
  kind: string;
  image: string;
  referenceText?: string;
  expectedReadingOrder: string[];
  requiredFields: string[];
  expectedTable: boolean;
}

export interface UnlimitedOcrLiveCorpus {
  version: number;
  documents: UnlimitedOcrLiveDocument[];
}

export interface UnlimitedOcrLiveMetric {
  passed: number;
  total: number;
  percent: number;
}

export interface UnlimitedOcrLiveScalarMetric {
  value: number;
  percent: number;
}

export interface UnlimitedOcrLiveLatency {
  uploadMs: number;
  requestMs: number;
  streamMs: number;
  totalMs: number;
}

export interface UnlimitedOcrLiveGateConfig {
  requiredFieldRecallMin: number;
  readingOrderRecallMin: number;
  readingOrderOrderMin: number;
  groundingCoordinateCoverageMin: number;
  normalizedSimilarityMin: number;
}

export interface UnlimitedOcrLiveCheckResult {
  requiredFieldRecall: boolean;
  readingOrderRecall: boolean;
  readingOrderOrder: boolean;
  groundingCoordinateCoverage: boolean;
  normalizedSimilarity: boolean | null;
  tableDetection: boolean;
}

export interface UnlimitedOcrLiveDocumentResult {
  documentId: string;
  kind: string;
  imagePath: string;
  status: 'pass' | 'fail' | 'error';
  rawGroundedText: string;
  finalText: string;
  markdown: string;
  blocks: UnlimitedOcrGroundingBlock[];
  groundingParseError?: string;
  requiredFieldRecall: UnlimitedOcrLiveMetric;
  readingOrderRecall: UnlimitedOcrLiveMetric;
  readingOrderOrder: UnlimitedOcrLiveMetric;
  normalizedSimilarity?: UnlimitedOcrLiveScalarMetric;
  groundingCoordinateCoverage: UnlimitedOcrLiveMetric;
  tableDetection: {
    expected: boolean;
    actual: boolean;
    pass: boolean;
  };
  latencyMs: UnlimitedOcrLiveLatency;
  checks: UnlimitedOcrLiveCheckResult;
  missingRequiredFields: string[];
  missingReadingOrderAnchors: string[];
  outOfOrderAnchors: string[];
  groundedRequiredFields: string[];
  error?: string;
}

export interface UnlimitedOcrLiveAggregate {
  documentCount: number;
  completedDocuments: number;
  errorDocuments: number;
  status: 'pass' | 'fail';
  requiredFieldRecall: UnlimitedOcrLiveMetric;
  readingOrderRecall: UnlimitedOcrLiveMetric;
  readingOrderOrder: UnlimitedOcrLiveMetric;
  groundingCoordinateCoverage: UnlimitedOcrLiveMetric;
  tableDetection: UnlimitedOcrLiveMetric;
  normalizedSimilarity?: {
    average: number;
    min: number;
    docs: number;
    percent: number;
  };
  latencyMs: {
    totalUploadMs: number;
    totalRequestMs: number;
    totalStreamMs: number;
    totalMs: number;
    averageDocumentMs: number;
    maxDocumentMs: number;
  };
  checks: UnlimitedOcrLiveCheckResult & { noErrors: boolean };
}

export interface UnlimitedOcrLiveReport {
  generatedAt: string;
  spaceUrl: string;
  mode: string;
  prompt: string;
  selectedDocumentCount: number;
  availableDocumentCount: number;
  selectedDocumentIds: string[];
  gates: UnlimitedOcrLiveGateConfig;
  documents: UnlimitedOcrLiveDocumentResult[];
  aggregate: UnlimitedOcrLiveAggregate;
}

export interface UnlimitedOcrLiveEvalOptions {
  allowedSpaceHosts?: string[];
  spaceUrl?: string;
  mode?: string;
  prompt?: string;
  timeoutMs?: number;
  maxImageBytes?: number;
  hfToken?: string;
  documentIds?: string[];
  fetchFn?: typeof fetch;
}

export interface UnlimitedOcrLiveCliOptions extends UnlimitedOcrLiveEvalOptions {
  allowFailedOutput?: boolean;
  json?: boolean;
  check?: boolean;
  output?: string;
  help?: boolean;
}

interface GradioFileData {
  path: string;
  url: string | null;
  size: number | null;
  orig_name: string;
  mime_type: string;
  is_stream: boolean;
  meta: { _type: 'gradio.FileData' };
}

interface UnlimitedOcrLiveRunResult {
  rawGroundedText: string;
  finalText: string;
  blocks: UnlimitedOcrGroundingBlock[];
  markdown: string;
  groundingParseError?: string;
  latencyMs: UnlimitedOcrLiveLatency;
}

interface AnchorObservation {
  anchor: string;
  position: number | null;
}

function finiteInteger(
  value: number | undefined,
  name: string,
  minimum: number,
  maximum: number
): number | undefined {
  if (value === undefined) return undefined;
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new Error(`${name} must be an integer from ${minimum} to ${maximum}`);
  }
  return value;
}

function normalizeSpaceUrl(value: string, allowedSpaceHosts: readonly string[]): string {
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    throw new Error('Unlimited-OCR live spaceUrl must be a valid URL');
  }
  if (url.protocol !== 'https:') {
    throw new Error('Unlimited-OCR live spaceUrl must use https');
  }
  if (url.username !== '' || url.password !== '') {
    throw new Error('Unlimited-OCR live spaceUrl must not contain credentials');
  }
  if (url.search !== '' || url.hash !== '' || (url.pathname !== '' && url.pathname !== '/')) {
    throw new Error('Unlimited-OCR live spaceUrl must be an origin without path, query, or fragment');
  }
  const allowed = new Set(allowedSpaceHosts.map((host) => host.toLowerCase()));
  if (!allowed.has(url.hostname.toLowerCase())) {
    throw new Error(`Unlimited-OCR live space host '${url.hostname}' is not allowlisted`);
  }
  return url.origin;
}

function metric(passed: number, total: number): UnlimitedOcrLiveMetric {
  return {
    passed,
    total,
    percent: total === 0 ? 100 : Number(((passed / total) * 100).toFixed(1)),
  };
}

function scalarMetric(value: number): UnlimitedOcrLiveScalarMetric {
  return {
    value: Number(value.toFixed(4)),
    percent: Number((value * 100).toFixed(1)),
  };
}

function normalizeText(value: string): string {
  return value
    .normalize('NFKC')
    .toLowerCase()
    .replace(/<[^>]*>/g, ' ')
    .replace(/[^\p{L}\p{N}]+/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function scoreThreshold(actual: number, minimum: number): boolean {
  return actual + 1e-9 >= minimum;
}

function combineAbortSignals(timeoutMs: number, signal?: AbortSignal) {
  const controller = new AbortController();
  const timer = setTimeout(() => {
    controller.abort(new Error(`Unlimited-OCR live request timed out after ${timeoutMs}ms`));
  }, timeoutMs);
  const onAbort = () => {
    controller.abort(signal?.reason ?? new Error('Unlimited-OCR live request aborted'));
  };
  if (signal !== undefined) {
    if (signal.aborted) onAbort();
    else signal.addEventListener('abort', onAbort, { once: true });
  }
  return {
    signal: controller.signal,
    cleanup() {
      clearTimeout(timer);
      if (signal !== undefined) signal.removeEventListener('abort', onAbort);
    },
  };
}

function responseErrorDetail(body: string): string | undefined {
  const compact = body.replace(/\s+/g, ' ').trim();
  return compact === '' ? undefined : compact.slice(0, 500);
}

function parseJson<T>(text: string, label: string): T {
  try {
    return JSON.parse(text) as T;
  } catch (error) {
    throw new Error(`${label} returned invalid JSON: ${String(error)}`);
  }
}

function requireString(value: unknown, label: string): string {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new Error(`${label} must be a non-empty string`);
  }
  return value;
}

function cleanRawGroundedText(rawGroundedText: string): string {
  const markerIndex = rawGroundedText.indexOf(UNLIMITED_OCR_LIVE_SAVE_MARKER);
  const cleaned = markerIndex >= 0 ? rawGroundedText.slice(0, markerIndex) : rawGroundedText;
  return cleaned.trim();
}

function validCoordinates(block: UnlimitedOcrGroundingBlock): boolean {
  const [x1, y1, x2, y2] = block.coordinates;
  return (
    Number.isFinite(x1) &&
    Number.isFinite(y1) &&
    Number.isFinite(x2) &&
    Number.isFinite(y2) &&
    x2 > x1 &&
    y2 > y1
  );
}

function containsNormalized(haystack: string, needle: string): boolean {
  return normalizeText(haystack).includes(normalizeText(needle));
}

function locateAnchors(text: string, anchors: readonly string[]): AnchorObservation[] {
  const normalizedText = normalizeText(text);
  return anchors.map((anchor) => {
    const normalizedAnchor = normalizeText(anchor);
    const position = normalizedAnchor === '' ? -1 : normalizedText.indexOf(normalizedAnchor);
    return {
      anchor,
      position: position < 0 ? null : position,
    };
  });
}

function orderedAnchorStats(text: string, anchors: readonly string[]) {
  const normalizedText = normalizeText(text);
  let searchStart = 0;
  let orderedPairsPassed = 0;
  let foundInOrder = 0;
  let sawOrderedAnchor = false;
  const outOfOrderAnchors: string[] = [];
  for (const anchor of anchors) {
    const normalizedAnchor = normalizeText(anchor);
    if (normalizedAnchor === '') continue;
    const orderedPosition = normalizedText.indexOf(normalizedAnchor, searchStart);
    if (orderedPosition < 0) {
      const globalPosition = normalizedText.indexOf(normalizedAnchor);
      if (globalPosition >= 0 && sawOrderedAnchor) outOfOrderAnchors.push(anchor);
      continue;
    }
    if (sawOrderedAnchor) orderedPairsPassed += 1;
    foundInOrder += 1;
    sawOrderedAnchor = true;
    searchStart = orderedPosition + normalizedAnchor.length;
  }
  return {
    foundInOrder,
    orderedPairsPassed,
    orderedPairsTotal: Math.max(foundInOrder - 1, 0),
    outOfOrderAnchors,
  };
}

function tableDetected(text: string): boolean {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line !== '');
  if (
    lines.some((line) => {
      const parts = line.split('|').map((part) => part.trim()).filter(Boolean);
      return parts.length >= 3;
    })
  ) {
    return true;
  }
  return lines.some((line) => /description/i.test(line)) &&
    lines.some((line) => /quantity/i.test(line)) &&
    lines.some((line) => /amount/i.test(line));
}

function normalizedLevenshteinSimilarity(left: string, right: string): number {
  const source = normalizeText(left);
  const target = normalizeText(right);
  if (source === target) return 1;
  if (source.length === 0 || target.length === 0) return 0;
  let previous = Array.from({ length: target.length + 1 }, (_, index) => index);
  let current = new Array<number>(target.length + 1).fill(0);
  for (let i = 1; i <= source.length; i += 1) {
    current[0] = i;
    for (let j = 1; j <= target.length; j += 1) {
      const cost = source[i - 1] === target[j - 1] ? 0 : 1;
      current[j] = Math.min(
        previous[j] + 1,
        current[j - 1] + 1,
        previous[j - 1] + cost
      );
    }
    [previous, current] = [current, previous];
  }
  const distance = previous[target.length];
  const denominator = Math.max(source.length, target.length);
  return denominator === 0 ? 1 : 1 - distance / denominator;
}

function parseSseFrame(frame: string): { event: string; data: string } | null {
  const trimmed = frame.trim();
  if (trimmed === '') return null;
  let event = 'message';
  const dataLines: string[] = [];
  for (const line of trimmed.split(/\r?\n/)) {
    if (line.startsWith(':')) continue;
    if (line.startsWith('event:')) {
      event = line.slice('event:'.length).trim();
      continue;
    }
    if (line.startsWith('data:')) {
      dataLines.push(line.slice('data:'.length).trimStart());
    }
  }
  return { event, data: dataLines.join('\n') };
}

function nextSseBoundary(buffer: string): { index: number; length: number } | null {
  const lf = buffer.indexOf('\n\n');
  const crlf = buffer.indexOf('\r\n\r\n');
  if (lf < 0 && crlf < 0) return null;
  if (crlf >= 0 && (lf < 0 || crlf < lf)) return { index: crlf, length: 4 };
  return { index: lf, length: 2 };
}

function defaultGates(): UnlimitedOcrLiveGateConfig {
  return {
    requiredFieldRecallMin: 0.95,
    readingOrderRecallMin: 0.9,
    readingOrderOrderMin: 0.9,
    groundingCoordinateCoverageMin: 0.7,
    normalizedSimilarityMin: 0.85,
  };
}

function corpusPath(): URL {
  return new URL('../../benchmarks/document-ocr/real-ground-truth.json', import.meta.url);
}

function fixturePath(relativePath: string): string {
  return fileURLToPath(
    new URL(`../../benchmarks/document-ocr/${relativePath}`, import.meta.url)
  );
}

export function loadUnlimitedOcrLiveCorpus(): UnlimitedOcrLiveCorpus {
  const raw = readFileSync(corpusPath(), 'utf8');
  const payload = parseJson<UnlimitedOcrLiveCorpus>(raw, 'document OCR ground truth');
  if (!Array.isArray(payload.documents) || payload.documents.length === 0) {
    throw new Error('document OCR ground truth must contain at least one document');
  }
  return payload;
}

export class UnlimitedOcrLiveClient {
  readonly spaceUrl: string;
  readonly mode: string;
  readonly prompt: string;
  readonly timeoutMs: number;
  readonly maxImageBytes: number;
  readonly hfToken?: string;

  constructor(
    options: UnlimitedOcrLiveEvalOptions = {},
    private readonly fetchFn: typeof fetch = fetch
  ) {
    this.spaceUrl = normalizeSpaceUrl(
      options.spaceUrl ?? DEFAULT_UNLIMITED_OCR_SPACE_URL,
      [new URL(DEFAULT_UNLIMITED_OCR_SPACE_URL).hostname, ...(options.allowedSpaceHosts ?? [])]
    );
    this.mode = options.mode ?? DEFAULT_UNLIMITED_OCR_LIVE_MODE;
    if (this.mode !== 'gundam' && this.mode !== 'base') {
      throw new Error("Unlimited-OCR live mode must be 'gundam' or 'base'");
    }
    this.prompt = options.prompt ?? UNLIMITED_OCR_PROMPT.replace(/^<image>/, '');
    this.timeoutMs =
      finiteInteger(
        options.timeoutMs,
        'Unlimited-OCR live timeoutMs',
        1_000,
        30 * 60 * 1000
      ) ?? UNLIMITED_OCR_LIVE_DEFAULT_TIMEOUT_MS;
    this.maxImageBytes =
      finiteInteger(
        options.maxImageBytes,
        'Unlimited-OCR live maxImageBytes',
        1,
        100 * 1024 * 1024
      ) ?? UNLIMITED_OCR_LIVE_MAX_IMAGE_BYTES;
    this.hfToken =
      options.hfToken !== undefined && options.hfToken.trim() !== ''
        ? options.hfToken.trim()
        : undefined;
  }

  async parseImage(
    imagePath: string,
    options: { signal?: AbortSignal } = {}
  ): Promise<UnlimitedOcrLiveRunResult> {
    const { signal, cleanup } = combineAbortSignals(this.timeoutMs, options.signal);
    try {
      const totalStart = performance.now();
      const file = await this.uploadFile(imagePath, signal);
      const uploadMs = performance.now() - totalStart;
      const requestStart = performance.now();
      const eventId = await this.startRun(file, signal);
      const requestMs = performance.now() - requestStart;
      const streamStart = performance.now();
      const stream = await this.readEventStream(eventId, signal);
      const streamMs = performance.now() - streamStart;
      const totalMs = performance.now() - totalStart;

      const cleanedRaw = cleanRawGroundedText(stream.rawGroundedText);
      let markdown = stream.finalText;
      let blocks: UnlimitedOcrGroundingBlock[] = [];
      let groundingParseError: string | undefined;
      if (cleanedRaw !== '') {
        try {
          const parsed = parseUnlimitedOcrGrounding(cleanedRaw);
          markdown = parsed.markdown;
          blocks = parsed.blocks;
        } catch (error) {
          groundingParseError = String(error instanceof Error ? error.message : error);
        }
      }

      return {
        rawGroundedText: cleanedRaw,
        finalText: stream.finalText,
        markdown,
        blocks,
        groundingParseError,
        latencyMs: {
          uploadMs: Number(uploadMs.toFixed(3)),
          requestMs: Number(requestMs.toFixed(3)),
          streamMs: Number(streamMs.toFixed(3)),
          totalMs: Number(totalMs.toFixed(3)),
        },
      };
    } catch (error) {
      if (signal.aborted && signal.reason instanceof Error) {
        throw signal.reason;
      }
      throw error;
    } finally {
      cleanup();
    }
  }

  private async uploadFile(imagePath: string, signal: AbortSignal): Promise<GradioFileData> {
    const fileBytes = readFileSync(imagePath);
    if (fileBytes.byteLength < 1 || fileBytes.byteLength > this.maxImageBytes) {
      throw new Error(
        `Unlimited-OCR live input bytes must be between 1 and ${this.maxImageBytes}`
      );
    }
    const body = new FormData();
    body.append(
      'files',
      new Blob([fileBytes], { type: mimeTypeForPath(imagePath) }),
      basename(imagePath) || 'document.png'
    );
    const response = await this.fetchFn(`${this.spaceUrl}/gradio_api/upload`, {
      method: 'POST',
      headers: this.requestHeaders(),
      body,
      signal,
    });
    const text = await response.text();
    if (!response.ok) {
      throw new Error(
        `Unlimited-OCR live upload failed with status ${response.status}` +
          formatDetailSuffix(text)
      );
    }
    const payload = parseJson<unknown>(text, 'Unlimited-OCR live upload');
    if (!Array.isArray(payload) || payload.length !== 1 || typeof payload[0] !== 'string') {
      throw new Error('Unlimited-OCR live upload returned an unexpected payload');
    }
    const uploadedPath = requireString(payload[0], 'Unlimited-OCR live upload path');
    return {
      path: uploadedPath,
      url: null,
      size: fileBytes.byteLength,
      orig_name: basename(imagePath) || 'document.png',
      mime_type: mimeTypeForPath(imagePath),
      is_stream: false,
      meta: { _type: 'gradio.FileData' },
    };
  }

  private async startRun(file: GradioFileData, signal: AbortSignal): Promise<string> {
    const response = await this.fetchFn(`${this.spaceUrl}/gradio_api/call/v2/run_ocr`, {
      method: 'POST',
      headers: {
        ...this.requestHeaders(),
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image_path: file,
        mode: this.mode,
        prompt: this.prompt,
      }),
      signal,
    });
    const text = await response.text();
    if (!response.ok) {
      throw new Error(
        `Unlimited-OCR live run start failed with status ${response.status}` +
          formatDetailSuffix(text)
      );
    }
    const payload = parseJson<{ event_id?: unknown }>(text, 'Unlimited-OCR live run start');
    return requireString(payload.event_id, 'Unlimited-OCR live event_id');
  }

  private async readEventStream(
    eventId: string,
    signal: AbortSignal
  ): Promise<{ rawGroundedText: string; finalText: string }> {
    const response = await this.fetchFn(`${this.spaceUrl}/gradio_api/call/run_ocr/${eventId}`, {
      method: 'GET',
      headers: {
        ...this.requestHeaders(),
        Accept: 'text/event-stream',
      },
      signal,
    });
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(
        `Unlimited-OCR live event stream failed with status ${response.status}` +
          formatDetailSuffix(detail)
      );
    }
    if (response.body === null) {
      throw new Error('Unlimited-OCR live event stream returned no body');
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let eventCount = 0;
    let lastRawGroundedText = '';
    let finalText = '';
    let sawTerminalEvent = false;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      if (buffer.length > UNLIMITED_OCR_LIVE_MAX_TEXT_CHARS * 2) {
        throw new Error('Unlimited-OCR live event stream exceeded the buffer limit');
      }
      let boundary = nextSseBoundary(buffer);
      while (boundary !== null) {
        const frame = buffer.slice(0, boundary.index);
        buffer = buffer.slice(boundary.index + boundary.length);
        const parsed = parseSseFrame(frame);
        if (parsed !== null) {
          eventCount += 1;
          if (eventCount > UNLIMITED_OCR_LIVE_MAX_SSE_EVENTS) {
            throw new Error(
              `Unlimited-OCR live event stream exceeded ${UNLIMITED_OCR_LIVE_MAX_SSE_EVENTS} events`
            );
          }
          if (parsed.event === 'error') {
            const detail = responseErrorDetail(parsed.data) ?? 'unknown provider error';
            throw new Error(`Unlimited-OCR live provider error: ${detail}`);
          }
          if (parsed.data !== '') {
            const payload = parseJson<Array<{ text?: unknown; done?: unknown }>>(
              parsed.data,
              `Unlimited-OCR live SSE ${parsed.event}`
            );
            const item = payload[0] ?? {};
            if (typeof item.text === 'string') {
              if (item.text.length > UNLIMITED_OCR_LIVE_MAX_TEXT_CHARS) {
                throw new Error('Unlimited-OCR live text exceeded the output limit');
              }
              if (item.done === true) finalText = item.text;
              else lastRawGroundedText = item.text;
            }
          }
          if (parsed.event === 'complete') sawTerminalEvent = true;
        }
        boundary = nextSseBoundary(buffer);
      }
    }
    buffer += decoder.decode();
    const trailing = parseSseFrame(buffer);
    if (trailing !== null) {
      if (trailing.event === 'error') {
        const detail = responseErrorDetail(trailing.data) ?? 'unknown provider error';
        throw new Error(`Unlimited-OCR live provider error: ${detail}`);
      }
      if (trailing.data !== '') {
        const payload = parseJson<Array<{ text?: unknown; done?: unknown }>>(
          trailing.data,
          `Unlimited-OCR live SSE ${trailing.event}`
        );
        const item = payload[0] ?? {};
        if (typeof item.text === 'string') {
          if (item.done === true) finalText = item.text;
          else lastRawGroundedText = item.text;
        }
      }
      if (trailing.event === 'complete') sawTerminalEvent = true;
    }
    if (finalText === '') {
      finalText = lastRawGroundedText;
    }
    if (finalText === '') {
      throw new Error('Unlimited-OCR live event stream completed without OCR text');
    }
    if (!sawTerminalEvent && lastRawGroundedText === '') {
      throw new Error('Unlimited-OCR live event stream ended before any OCR payload arrived');
    }
    return {
      rawGroundedText: lastRawGroundedText,
      finalText,
    };
  }

  private requestHeaders(): HeadersInit | undefined {
    return this.hfToken === undefined
      ? undefined
      : { Authorization: `Bearer ${this.hfToken}` };
  }
}

function formatDetailSuffix(detailBody: string): string {
  const detail = responseErrorDetail(detailBody);
  return detail === undefined ? '' : `: ${detail}`;
}

function mimeTypeForPath(path: string): string {
  if (path.endsWith('.png')) return 'image/png';
  if (path.endsWith('.jpg') || path.endsWith('.jpeg')) return 'image/jpeg';
  if (path.endsWith('.webp')) return 'image/webp';
  if (path.endsWith('.pdf')) return 'application/pdf';
  return 'application/octet-stream';
}

export function scoreUnlimitedOcrLiveDocument(
  document: UnlimitedOcrLiveDocument,
  run: UnlimitedOcrLiveRunResult,
  gates: UnlimitedOcrLiveGateConfig,
  imagePath: string
): UnlimitedOcrLiveDocumentResult {
  const comparisonText = run.finalText === '' ? run.markdown : run.finalText;
  const requiredMatches = document.requiredFields.filter((field) =>
    containsNormalized(comparisonText, field)
  );
  const missingRequiredFields = document.requiredFields.filter(
    (field) => !requiredMatches.includes(field)
  );

  const anchors = locateAnchors(comparisonText, document.expectedReadingOrder);
  const foundAnchors = anchors.filter((anchor) => anchor.position !== null);
  const missingReadingOrderAnchors = anchors
    .filter((anchor) => anchor.position === null)
    .map((anchor) => anchor.anchor);
  const orderedStats = orderedAnchorStats(comparisonText, document.expectedReadingOrder);

  const groundedRequiredFields = document.requiredFields.filter((field) =>
    run.blocks.some((block) => validCoordinates(block) && containsNormalized(block.text, field))
  );
  const normalizedSimilarity =
    document.referenceText === undefined
      ? undefined
      : scalarMetric(normalizedLevenshteinSimilarity(comparisonText, document.referenceText));
  const actualTable = tableDetected(comparisonText);

  const requiredFieldRecall = metric(requiredMatches.length, document.requiredFields.length);
  const readingOrderRecall = metric(foundAnchors.length, document.expectedReadingOrder.length);
  const readingOrderOrder = metric(
    orderedStats.orderedPairsPassed,
    orderedStats.orderedPairsTotal
  );
  const groundingCoordinateCoverage = metric(
    groundedRequiredFields.length,
    document.requiredFields.length
  );

  const checks: UnlimitedOcrLiveCheckResult = {
    requiredFieldRecall: scoreThreshold(
      requiredMatches.length / Math.max(document.requiredFields.length, 1),
      gates.requiredFieldRecallMin
    ),
    readingOrderRecall: scoreThreshold(
      foundAnchors.length / Math.max(document.expectedReadingOrder.length, 1),
      gates.readingOrderRecallMin
    ),
    readingOrderOrder: scoreThreshold(
      orderedStats.orderedPairsTotal === 0
        ? 1
        : orderedStats.orderedPairsPassed / orderedStats.orderedPairsTotal,
      gates.readingOrderOrderMin
    ),
    groundingCoordinateCoverage: scoreThreshold(
      groundedRequiredFields.length / Math.max(document.requiredFields.length, 1),
      gates.groundingCoordinateCoverageMin
    ),
    normalizedSimilarity:
      normalizedSimilarity === undefined
        ? null
        : scoreThreshold(
            normalizedSimilarity.value,
            gates.normalizedSimilarityMin
          ),
    tableDetection: actualTable === document.expectedTable,
  };

  const status =
    run.finalText === ''
      ? 'error'
      : checks.requiredFieldRecall &&
          checks.readingOrderRecall &&
          checks.readingOrderOrder &&
          checks.groundingCoordinateCoverage &&
          checks.tableDetection &&
          (checks.normalizedSimilarity ?? true)
        ? 'pass'
        : 'fail';

  return {
    documentId: document.id,
    kind: document.kind,
    imagePath,
    status,
    rawGroundedText: run.rawGroundedText,
    finalText: run.finalText,
    markdown: run.markdown,
    blocks: run.blocks,
    groundingParseError: run.groundingParseError,
    requiredFieldRecall,
    readingOrderRecall,
    readingOrderOrder,
    normalizedSimilarity,
    groundingCoordinateCoverage,
    tableDetection: {
      expected: document.expectedTable,
      actual: actualTable,
      pass: checks.tableDetection,
    },
    latencyMs: run.latencyMs,
    checks,
    missingRequiredFields,
    missingReadingOrderAnchors,
    outOfOrderAnchors: Array.from(new Set(orderedStats.outOfOrderAnchors)),
    groundedRequiredFields,
  };
}

export async function evaluateUnlimitedOcrLive(
  options: UnlimitedOcrLiveEvalOptions = {}
): Promise<UnlimitedOcrLiveReport> {
  const corpus = loadUnlimitedOcrLiveCorpus();
  const selectedDocuments = selectUnlimitedOcrLiveDocuments(corpus, options.documentIds);
  const gates = defaultGates();
  const client = new UnlimitedOcrLiveClient(options, options.fetchFn);
  const documents: UnlimitedOcrLiveDocumentResult[] = [];
  for (const document of selectedDocuments) {
    const fixtureImagePath = fixturePath(document.image);
    try {
      const run = await client.parseImage(fixtureImagePath);
      documents.push(scoreUnlimitedOcrLiveDocument(document, run, gates, document.image));
    } catch (error) {
      documents.push({
        documentId: document.id,
        kind: document.kind,
        imagePath: document.image,
        status: 'error',
        rawGroundedText: '',
        finalText: '',
        markdown: '',
        blocks: [],
        requiredFieldRecall: metric(0, document.requiredFields.length),
        readingOrderRecall: metric(0, document.expectedReadingOrder.length),
        readingOrderOrder: metric(0, Math.max(document.expectedReadingOrder.length - 1, 0)),
        groundingCoordinateCoverage: metric(0, document.requiredFields.length),
        tableDetection: {
          expected: document.expectedTable,
          actual: false,
          pass: !document.expectedTable,
        },
        latencyMs: { uploadMs: 0, requestMs: 0, streamMs: 0, totalMs: 0 },
        checks: {
          requiredFieldRecall: false,
          readingOrderRecall: false,
          readingOrderOrder: false,
          groundingCoordinateCoverage: false,
          normalizedSimilarity: document.referenceText === undefined ? null : false,
          tableDetection: !document.expectedTable,
        },
        missingRequiredFields: [...document.requiredFields],
        missingReadingOrderAnchors: [...document.expectedReadingOrder],
        outOfOrderAnchors: [],
        groundedRequiredFields: [],
        error: String(error instanceof Error ? error.message : error),
      });
    }
  }

  const completedDocuments = documents.filter((document) => document.status !== 'error');
  const similarityDocs = completedDocuments.filter(
    (document): document is UnlimitedOcrLiveDocumentResult & { normalizedSimilarity: UnlimitedOcrLiveScalarMetric } =>
      document.normalizedSimilarity !== undefined
  );
  const aggregateSimilarity =
    similarityDocs.length === 0
      ? undefined
      : {
          average: Number(
            (
              similarityDocs.reduce(
                (sum, document) => sum + document.normalizedSimilarity.value,
                0
              ) / similarityDocs.length
            ).toFixed(4)
          ),
          min: Number(
            Math.min(...similarityDocs.map((document) => document.normalizedSimilarity.value)).toFixed(4)
          ),
          docs: similarityDocs.length,
          percent: Number(
            (
              (similarityDocs.reduce(
                (sum, document) => sum + document.normalizedSimilarity.value,
                0
              ) /
                similarityDocs.length) *
              100
            ).toFixed(1)
          ),
        };

  const aggregate: UnlimitedOcrLiveAggregate = {
    documentCount: documents.length,
    completedDocuments: completedDocuments.length,
    errorDocuments: documents.filter((document) => document.status === 'error').length,
    status: 'fail',
    requiredFieldRecall: metric(
      completedDocuments.reduce((sum, document) => sum + document.requiredFieldRecall.passed, 0),
      completedDocuments.reduce((sum, document) => sum + document.requiredFieldRecall.total, 0)
    ),
    readingOrderRecall: metric(
      completedDocuments.reduce((sum, document) => sum + document.readingOrderRecall.passed, 0),
      completedDocuments.reduce((sum, document) => sum + document.readingOrderRecall.total, 0)
    ),
    readingOrderOrder: metric(
      completedDocuments.reduce((sum, document) => sum + document.readingOrderOrder.passed, 0),
      completedDocuments.reduce((sum, document) => sum + document.readingOrderOrder.total, 0)
    ),
    groundingCoordinateCoverage: metric(
      completedDocuments.reduce((sum, document) => sum + document.groundingCoordinateCoverage.passed, 0),
      completedDocuments.reduce((sum, document) => sum + document.groundingCoordinateCoverage.total, 0)
    ),
    tableDetection: metric(
      completedDocuments.filter((document) => document.tableDetection.pass).length,
      completedDocuments.length
    ),
    normalizedSimilarity: aggregateSimilarity,
    latencyMs: {
      totalUploadMs: Number(
        completedDocuments.reduce((sum, document) => sum + document.latencyMs.uploadMs, 0).toFixed(3)
      ),
      totalRequestMs: Number(
        completedDocuments.reduce((sum, document) => sum + document.latencyMs.requestMs, 0).toFixed(3)
      ),
      totalStreamMs: Number(
        completedDocuments.reduce((sum, document) => sum + document.latencyMs.streamMs, 0).toFixed(3)
      ),
      totalMs: Number(
        completedDocuments.reduce((sum, document) => sum + document.latencyMs.totalMs, 0).toFixed(3)
      ),
      averageDocumentMs: Number(
        (
          completedDocuments.reduce((sum, document) => sum + document.latencyMs.totalMs, 0) /
          Math.max(completedDocuments.length, 1)
        ).toFixed(3)
      ),
      maxDocumentMs: Number(
        Math.max(...completedDocuments.map((document) => document.latencyMs.totalMs), 0).toFixed(3)
      ),
    },
    checks: {
      requiredFieldRecall: scoreThreshold(
        completedDocuments.length === 0
          ? 1
          : completedDocuments.reduce((sum, document) => sum + document.requiredFieldRecall.passed, 0) /
              Math.max(
                completedDocuments.reduce((sum, document) => sum + document.requiredFieldRecall.total, 0),
                1
              ),
        gates.requiredFieldRecallMin
      ),
      readingOrderRecall: scoreThreshold(
        completedDocuments.length === 0
          ? 1
          : completedDocuments.reduce((sum, document) => sum + document.readingOrderRecall.passed, 0) /
              Math.max(
                completedDocuments.reduce((sum, document) => sum + document.readingOrderRecall.total, 0),
                1
              ),
        gates.readingOrderRecallMin
      ),
      readingOrderOrder: scoreThreshold(
        completedDocuments.length === 0
          ? 1
          : completedDocuments.reduce((sum, document) => sum + document.readingOrderOrder.passed, 0) /
              Math.max(
                completedDocuments.reduce((sum, document) => sum + document.readingOrderOrder.total, 0),
                1
              ),
        gates.readingOrderOrderMin
      ),
      groundingCoordinateCoverage: scoreThreshold(
        completedDocuments.length === 0
          ? 1
          : completedDocuments.reduce((sum, document) => sum + document.groundingCoordinateCoverage.passed, 0) /
              Math.max(
                completedDocuments.reduce((sum, document) => sum + document.groundingCoordinateCoverage.total, 0),
                1
              ),
        gates.groundingCoordinateCoverageMin
      ),
      normalizedSimilarity:
        aggregateSimilarity === undefined
          ? null
          : scoreThreshold(aggregateSimilarity.average, gates.normalizedSimilarityMin),
      tableDetection: completedDocuments.every((document) => document.tableDetection.pass),
      noErrors: documents.every((document) => document.status !== 'error'),
    },
  };

  aggregate.status =
    aggregate.checks.requiredFieldRecall &&
    aggregate.checks.readingOrderRecall &&
    aggregate.checks.readingOrderOrder &&
    aggregate.checks.groundingCoordinateCoverage &&
    aggregate.checks.tableDetection &&
    aggregate.checks.noErrors &&
    (aggregate.checks.normalizedSimilarity ?? true)
      ? 'pass'
      : 'fail';

  return {
    generatedAt: new Date().toISOString(),
    spaceUrl: client.spaceUrl,
    mode: client.mode,
    prompt: client.prompt,
    selectedDocumentCount: selectedDocuments.length,
    availableDocumentCount: corpus.documents.length,
    selectedDocumentIds: selectedDocuments.map((document) => document.id),
    gates,
    documents,
    aggregate,
  };
}

export function formatUnlimitedOcrLiveReport(report: UnlimitedOcrLiveReport): string {
  const lines = [
    `Unlimited-OCR live eval (${report.spaceUrl}, mode=${report.mode})`,
    `status: ${report.aggregate.status}`,
    `documents: ${report.aggregate.completedDocuments}/${report.aggregate.documentCount} completed`,
    `selected: ${report.selectedDocumentCount}/${report.availableDocumentCount} (${report.selectedDocumentIds.join(', ')})`,
    `required-field recall: ${report.aggregate.requiredFieldRecall.passed}/${report.aggregate.requiredFieldRecall.total} (${report.aggregate.requiredFieldRecall.percent}%)`,
    `reading-order recall: ${report.aggregate.readingOrderRecall.passed}/${report.aggregate.readingOrderRecall.total} (${report.aggregate.readingOrderRecall.percent}%)`,
    `reading-order order: ${report.aggregate.readingOrderOrder.passed}/${report.aggregate.readingOrderOrder.total} (${report.aggregate.readingOrderOrder.percent}%)`,
    `grounding coverage: ${report.aggregate.groundingCoordinateCoverage.passed}/${report.aggregate.groundingCoordinateCoverage.total} (${report.aggregate.groundingCoordinateCoverage.percent}%)`,
    `table detection: ${report.aggregate.tableDetection.passed}/${report.aggregate.tableDetection.total} (${report.aggregate.tableDetection.percent}%)`,
  ];
  if (report.aggregate.normalizedSimilarity !== undefined) {
    lines.push(
      `normalized similarity: avg ${report.aggregate.normalizedSimilarity.average} (${report.aggregate.normalizedSimilarity.percent}%), min ${report.aggregate.normalizedSimilarity.min}`
    );
  }
  lines.push(`latency total: ${report.aggregate.latencyMs.totalMs}ms`);
  for (const document of report.documents) {
    lines.push('');
    lines.push(
      `${document.documentId}: ${document.status} | fields ${document.requiredFieldRecall.percent}% | order ${document.readingOrderOrder.percent}% | grounding ${document.groundingCoordinateCoverage.percent}% | latency ${document.latencyMs.totalMs}ms`
    );
    if (document.error !== undefined) lines.push(`  error: ${document.error}`);
    if (document.missingRequiredFields.length > 0) {
      lines.push(`  missing fields: ${document.missingRequiredFields.join(', ')}`);
    }
    if (document.missingReadingOrderAnchors.length > 0) {
      lines.push(`  missing anchors: ${document.missingReadingOrderAnchors.join(', ')}`);
    }
    if (document.outOfOrderAnchors.length > 0) {
      lines.push(`  out-of-order anchors: ${document.outOfOrderAnchors.join(', ')}`);
    }
    if (document.groundingParseError !== undefined) {
      lines.push(`  grounding parse error: ${document.groundingParseError}`);
    }
  }
  return lines.join('\n');
}

export function parseUnlimitedOcrLiveCliArgs(argv: readonly string[]): UnlimitedOcrLiveCliOptions {
  const options: UnlimitedOcrLiveCliOptions = { documentIds: [] };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--help' || arg === '-h') {
      options.help = true;
      continue;
    }
    if (arg === '--json') {
      options.json = true;
      continue;
    }
    if (arg === '--check') {
      options.check = true;
      continue;
    }
    if (arg === '--allow-failed-output') {
      options.allowFailedOutput = true;
      continue;
    }
    if (arg === '--space-url') {
      options.spaceUrl = argv[++index];
      continue;
    }
    if (arg === '--mode') {
      options.mode = argv[++index];
      continue;
    }
    if (arg === '--prompt') {
      options.prompt = argv[++index];
      continue;
    }
    if (arg === '--output') {
      options.output = argv[++index];
      continue;
    }
    if (arg === '--document') {
      const value = argv[++index];
      if (value === undefined || value.trim() === '') {
        throw new Error('--document requires a value');
      }
      options.documentIds?.push(value);
      continue;
    }
    if (arg === '--timeout-ms') {
      options.timeoutMs = Number(argv[++index]);
      continue;
    }
    throw new Error(`unknown argument: ${arg}`);
  }
  if (options.spaceUrl !== undefined && options.spaceUrl.trim() === '') {
    throw new Error('--space-url requires a value');
  }
  if (options.mode !== undefined && options.mode.trim() === '') {
    throw new Error('--mode requires a value');
  }
  if (options.prompt !== undefined && options.prompt.trim() === '') {
    throw new Error('--prompt requires a value');
  }
  if (options.output !== undefined && options.output.trim() === '') {
    throw new Error('--output requires a value');
  }
  if (options.documentIds?.length === 0) {
    delete options.documentIds;
  }
  return options;
}

export function formatUnlimitedOcrLiveHelp(): string {
  return [
    'Usage: node dist/evals/run-unlimited-ocr-live.js [options]',
    '',
    'Options:',
    '  --space-url <url>   Override the Hugging Face Space URL',
    '  --mode <mode>       OCR mode to request (default: gundam)',
    '  --prompt <text>     Prompt suffix to send after <image>',
    '  --timeout-ms <ms>   Request timeout in milliseconds',
    '  --document <id>     Limit evaluation to one document id; repeatable',
    '  --json              Print the JSON report',
    '  --check             Exit non-zero when aggregate gates fail',
    '  --output <path>     Write the rendered report to a file',
    '  --allow-failed-output  Permit a failed run to replace --output',
    '  --help, -h          Show this help text',
    '',
    'Environment:',
    '  HF_TOKEN            Optional Hugging Face token for authenticated quota',
  ].join('\n');
}

function selectUnlimitedOcrLiveDocuments(
  corpus: UnlimitedOcrLiveCorpus,
  documentIds: readonly string[] | undefined
): UnlimitedOcrLiveDocument[] {
  if (documentIds === undefined || documentIds.length === 0) return corpus.documents;
  const available = new Map(corpus.documents.map((document) => [document.id, document]));
  const selected: UnlimitedOcrLiveDocument[] = [];
  const invalidIds: string[] = [];
  for (const documentId of documentIds) {
    const document = available.get(documentId);
    if (document === undefined) {
      invalidIds.push(documentId);
      continue;
    }
    selected.push(document);
  }
  if (invalidIds.length > 0) {
    throw new Error(
      `unknown document id(s): ${invalidIds.join(', ')}. Available: ${corpus.documents.map((document) => document.id).join(', ')}`
    );
  }
  return selected;
}

export function renderUnlimitedOcrLiveOutput(
  report: UnlimitedOcrLiveReport,
  options: Pick<UnlimitedOcrLiveCliOptions, 'json'>
): string {
  return options.json
    ? `${JSON.stringify(report, null, 2)}\n`
    : `${formatUnlimitedOcrLiveReport(report)}\n`;
}

export function maybeWriteUnlimitedOcrLiveOutput(
  outputPath: string | undefined,
  content: string
): void {
  if (outputPath === undefined) return;
  writeFileSync(outputPath, content, 'utf8');
}
