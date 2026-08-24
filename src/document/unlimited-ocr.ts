const OPEN_REF = '<|ref|>';
const CLOSE_REF = '<|/ref|>';
const OPEN_DET = '<|det|>';
const CLOSE_DET = '<|/det|>';

export const DEFAULT_UNLIMITED_OCR_MODEL = 'baidu/Unlimited-OCR';
export const UNLIMITED_OCR_PROMPT = '<image>document parsing.';
export const UNLIMITED_OCR_MAX_TOKENS = 8_192;
export const UNLIMITED_OCR_SINGLE_PAGE_WINDOW = 128;
export const UNLIMITED_OCR_MULTI_PAGE_WINDOW = 1_024;
export const UNLIMITED_OCR_NGRAM_SIZE = 35;
export const UNLIMITED_OCR_MAX_IMAGES = 20;
export const UNLIMITED_OCR_MAX_IMAGE_BYTES = 10 * 1024 * 1024;
export const UNLIMITED_OCR_MAX_TOTAL_IMAGE_BYTES = 40 * 1024 * 1024;
export const UNLIMITED_OCR_MAX_RESPONSE_BYTES = 1 * 1024 * 1024;
export const UNLIMITED_OCR_MAX_OUTPUT_CHARS = 512 * 1024;
export const UNLIMITED_OCR_MAX_GROUNDING_BLOCKS = 10_000;
export const UNLIMITED_OCR_DEFAULT_TIMEOUT_MS = 3_600_000;

export interface UnlimitedOcrImageInput {
  dataUrl?: string;
  bytes?: Uint8Array | ArrayBuffer;
  mimeType?: string;
}

export interface UnlimitedOcrGroundingBlock {
  category: string;
  coordinates: [number, number, number, number];
  text: string;
}

export interface UnlimitedOcrParseResult {
  rawText: string;
  markdown: string;
  blocks: UnlimitedOcrGroundingBlock[];
  model: string;
}

export interface UnlimitedOcrClientOptions {
  allowPrivateNetwork?: boolean;
  apiKey?: string;
  baseUrl: string;
  model?: string;
  timeoutMs?: number;
  maxImages?: number;
  maxImageBytes?: number;
  maxTotalImageBytes?: number;
  maxResponseBytes?: number;
  maxOutputChars?: number;
  maxGroundingBlocks?: number;
}

interface NormalizedImage {
  dataUrl: string;
  byteLength: number;
}

interface ParsedGrounding {
  markdown: string;
  blocks: UnlimitedOcrGroundingBlock[];
}

interface CombinedSignal {
  signal: AbortSignal;
  cleanup(): void;
}

interface PendingBlock {
  category: string;
  coordinates: [number, number, number, number];
  text: string;
}

function isPrivateHostname(hostname: string): boolean {
  const normalized = hostname.toLowerCase().replace(/^\[|\]$/g, '');
  if (normalized === 'localhost' || normalized.endsWith('.localhost')) return true;
  if (normalized === '::1' || normalized.startsWith('fc') || normalized.startsWith('fd')) {
    return true;
  }
  if (normalized.startsWith('fe80:')) return true;
  if (/^127\./.test(normalized) || /^10\./.test(normalized)) return true;
  if (/^192\.168\./.test(normalized) || /^169\.254\./.test(normalized)) return true;
  return /^172\.(1[6-9]|2\d|3[01])\./.test(normalized);
}

function normalizeBaseUrl(value: string, allowPrivateNetwork: boolean): string {
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    throw new Error('Unlimited-OCR baseUrl must be a valid URL');
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    throw new Error('Unlimited-OCR baseUrl must use http or https');
  }
  if (url.username !== '' || url.password !== '') {
    throw new Error('Unlimited-OCR baseUrl must not contain credentials');
  }
  if (url.search !== '' || url.hash !== '') {
    throw new Error('Unlimited-OCR baseUrl must not contain a query or fragment');
  }
  const privateNetwork = isPrivateHostname(url.hostname);
  if (privateNetwork && !allowPrivateNetwork) {
    throw new Error('Unlimited-OCR private-network baseUrl requires allowPrivateNetwork');
  }
  if (url.protocol === 'http:' && !privateNetwork) {
    throw new Error('Unlimited-OCR public baseUrl must use https');
  }
  const normalizedPath = url.pathname.replace(/\/$/, '');
  if (normalizedPath !== '/v1') {
    throw new Error('Unlimited-OCR baseUrl path must be /v1');
  }
  url.pathname = '/v1';
  return url.toString().replace(/\/$/, '');
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

function byteLengthForDataUrl(dataUrl: string): number {
  const match = /^data:([^;,]+)?(?:;charset=[^;,]+)?;base64,([A-Za-z0-9+/=]+)$/.exec(dataUrl);
  if (match === null) {
    throw new Error('Unlimited-OCR images must be base64 data URLs');
  }
  return Buffer.from(match[2], 'base64').byteLength;
}

function normalizeImageInput(input: UnlimitedOcrImageInput, maxImageBytes: number): NormalizedImage {
  if (typeof input.dataUrl === 'string') {
    const byteLength = byteLengthForDataUrl(input.dataUrl);
    if (byteLength < 1 || byteLength > maxImageBytes) {
      throw new Error(
        `Unlimited-OCR image bytes must be between 1 and ${maxImageBytes}`
      );
    }
    return { dataUrl: input.dataUrl, byteLength };
  }
  if (input.bytes === undefined) {
    throw new Error('Unlimited-OCR image input requires either dataUrl or bytes');
  }
  if (typeof input.mimeType !== 'string' || !/^image\/[A-Za-z0-9.+-]+$/.test(input.mimeType)) {
    throw new Error('Unlimited-OCR byte inputs require an image/* mimeType');
  }
  const bytes =
    input.bytes instanceof Uint8Array ? input.bytes : new Uint8Array(input.bytes);
  if (bytes.byteLength < 1 || bytes.byteLength > maxImageBytes) {
    throw new Error(`Unlimited-OCR image bytes must be between 1 and ${maxImageBytes}`);
  }
  return {
    dataUrl: `data:${input.mimeType};base64,${Buffer.from(bytes).toString('base64')}`,
    byteLength: bytes.byteLength,
  };
}

function providerErrorDetail(body: string): string | undefined {
  if (Buffer.byteLength(body) > 16 * 1024) return undefined;
  try {
    const payload = JSON.parse(body) as { error?: { message?: unknown } };
    const message = payload.error?.message;
    if (typeof message !== 'string') return undefined;
    const trimmed = message.replace(/\s+/g, ' ').trim();
    return trimmed === '' ? undefined : trimmed.slice(0, 500);
  } catch {
    return undefined;
  }
}

function parseDetMetadata(meta: string): Omit<PendingBlock, 'text'> {
  const match =
    /^\s*([^\[\]\r\n]+?)\s*\[\s*(-?(?:\d+(?:\.\d+)?|\.\d+))\s*,\s*(-?(?:\d+(?:\.\d+)?|\.\d+))\s*,\s*(-?(?:\d+(?:\.\d+)?|\.\d+))\s*,\s*(-?(?:\d+(?:\.\d+)?|\.\d+))\s*\]\s*$/.exec(
      meta
    );
  if (match === null) {
    throw new Error('Unlimited-OCR grounding metadata was malformed');
  }
  const category = match[1].trim();
  if (category === '') {
    throw new Error('Unlimited-OCR grounding category was empty');
  }
  const coordinates = match.slice(2).map((value) => Number(value));
  if (!coordinates.every((value) => Number.isFinite(value))) {
    throw new Error('Unlimited-OCR grounding coordinates must be finite numbers');
  }
  return {
    category,
    coordinates: coordinates as [number, number, number, number],
  };
}

function nextTokenIndex(raw: string, cursor: number): number {
  const candidates = [
    raw.indexOf(OPEN_REF, cursor),
    raw.indexOf(CLOSE_REF, cursor),
    raw.indexOf(OPEN_DET, cursor),
    raw.indexOf(CLOSE_DET, cursor),
  ].filter((value) => value >= 0);
  return candidates.length === 0 ? -1 : Math.min(...candidates);
}

function flushPending(
  pending: PendingBlock | null,
  blocks: UnlimitedOcrGroundingBlock[],
  maxBlocks: number
): PendingBlock | null {
  if (pending === null) return null;
  const text = pending.text.trim();
  if (text === '') {
    throw new Error('Unlimited-OCR grounding block had no text');
  }
  if (blocks.length >= maxBlocks) {
    throw new Error(`Unlimited-OCR grounding exceeded ${maxBlocks} blocks`);
  }
  blocks.push({
    category: pending.category,
    coordinates: pending.coordinates,
    text,
  });
  return null;
}

export function parseUnlimitedOcrGrounding(
  rawText: string,
  options: {
    maxOutputChars?: number;
    maxGroundingBlocks?: number;
  } = {}
): ParsedGrounding {
  if (typeof rawText !== 'string') {
    throw new Error('Unlimited-OCR output must be a string');
  }
  const maxOutputChars = options.maxOutputChars ?? UNLIMITED_OCR_MAX_OUTPUT_CHARS;
  const maxGroundingBlocks =
    options.maxGroundingBlocks ?? UNLIMITED_OCR_MAX_GROUNDING_BLOCKS;
  finiteInteger(maxOutputChars, 'Unlimited-OCR max output chars', 1, 4 * 1024 * 1024);
  finiteInteger(
    maxGroundingBlocks,
    'Unlimited-OCR max grounding blocks',
    1,
    100_000
  );
  if (rawText.length > maxOutputChars) {
    throw new Error(`Unlimited-OCR output exceeded ${maxOutputChars} characters`);
  }
  let cursor = 0;
  let markdown = '';
  let inRef = false;
  let pending: PendingBlock | null = null;
  const blocks: UnlimitedOcrGroundingBlock[] = [];
  while (cursor < rawText.length) {
    if (rawText.startsWith(OPEN_REF, cursor)) {
      if (inRef) {
        throw new Error('Unlimited-OCR grounding had nested <|ref|> tags');
      }
      cursor += OPEN_REF.length;
      inRef = true;
      continue;
    }
    if (rawText.startsWith(CLOSE_REF, cursor)) {
      if (!inRef) {
        throw new Error('Unlimited-OCR grounding closed <|/ref|> without <|ref|>');
      }
      pending = flushPending(pending, blocks, maxGroundingBlocks);
      cursor += CLOSE_REF.length;
      inRef = false;
      continue;
    }
    if (rawText.startsWith(OPEN_DET, cursor)) {
      pending = flushPending(pending, blocks, maxGroundingBlocks);
      const closeIndex = rawText.indexOf(CLOSE_DET, cursor + OPEN_DET.length);
      if (closeIndex < 0) {
        throw new Error('Unlimited-OCR grounding had an unterminated <|det|> tag');
      }
      pending = {
        ...parseDetMetadata(rawText.slice(cursor + OPEN_DET.length, closeIndex)),
        text: '',
      };
      cursor = closeIndex + CLOSE_DET.length;
      continue;
    }
    if (rawText.startsWith(CLOSE_DET, cursor)) {
      throw new Error('Unlimited-OCR grounding closed <|/det|> without <|det|>');
    }
    const next = nextTokenIndex(rawText, cursor);
    const text = rawText.slice(cursor, next < 0 ? rawText.length : next);
    markdown += text;
    if (pending !== null) pending.text += text;
    cursor = next < 0 ? rawText.length : next;
  }
  if (inRef) {
    throw new Error('Unlimited-OCR grounding had an unterminated <|ref|> tag');
  }
  pending = flushPending(pending, blocks, maxGroundingBlocks);
  return { markdown, blocks };
}

function combineAbortSignals(timeoutMs: number, signal?: AbortSignal): CombinedSignal {
  const controller = new AbortController();
  const timer = setTimeout(() => {
    controller.abort(new Error(`Unlimited-OCR request timed out after ${timeoutMs}ms`));
  }, timeoutMs);
  const onAbort = () => {
    controller.abort(signal?.reason ?? new Error('Unlimited-OCR request aborted'));
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

export class UnlimitedOcrClient {
  readonly baseUrl: string;
  readonly model: string;
  readonly timeoutMs: number;
  readonly maxImages: number;
  readonly maxImageBytes: number;
  readonly maxTotalImageBytes: number;
  readonly maxResponseBytes: number;
  readonly maxOutputChars: number;
  readonly maxGroundingBlocks: number;

  constructor(
    private readonly options: UnlimitedOcrClientOptions,
    private readonly fetchFn: typeof fetch = fetch
  ) {
    const normalizedBaseUrl = normalizeBaseUrl(
      options.baseUrl,
      options.allowPrivateNetwork === true
    );
    this.baseUrl = normalizedBaseUrl;
    this.model = options.model ?? DEFAULT_UNLIMITED_OCR_MODEL;
    this.timeoutMs =
      finiteInteger(
        options.timeoutMs,
        'Unlimited-OCR timeoutMs',
        1_000,
        24 * 60 * 60 * 1000
      ) ?? UNLIMITED_OCR_DEFAULT_TIMEOUT_MS;
    this.maxImages =
      finiteInteger(options.maxImages, 'Unlimited-OCR maxImages', 1, 128) ??
      UNLIMITED_OCR_MAX_IMAGES;
    this.maxImageBytes =
      finiteInteger(
        options.maxImageBytes,
        'Unlimited-OCR maxImageBytes',
        1,
        64 * 1024 * 1024
      ) ?? UNLIMITED_OCR_MAX_IMAGE_BYTES;
    this.maxTotalImageBytes =
      finiteInteger(
        options.maxTotalImageBytes,
        'Unlimited-OCR maxTotalImageBytes',
        1,
        256 * 1024 * 1024
      ) ?? UNLIMITED_OCR_MAX_TOTAL_IMAGE_BYTES;
    this.maxResponseBytes =
      finiteInteger(
        options.maxResponseBytes,
        'Unlimited-OCR maxResponseBytes',
        1,
        16 * 1024 * 1024
      ) ?? UNLIMITED_OCR_MAX_RESPONSE_BYTES;
    this.maxOutputChars =
      finiteInteger(
        options.maxOutputChars,
        'Unlimited-OCR maxOutputChars',
        1,
        4 * 1024 * 1024
      ) ?? UNLIMITED_OCR_MAX_OUTPUT_CHARS;
    this.maxGroundingBlocks =
      finiteInteger(
        options.maxGroundingBlocks,
        'Unlimited-OCR maxGroundingBlocks',
        1,
        100_000
      ) ?? UNLIMITED_OCR_MAX_GROUNDING_BLOCKS;
  }

  async parse(
    images: readonly UnlimitedOcrImageInput[],
    options: {
      signal?: AbortSignal;
      maxTokens?: number;
    } = {}
  ): Promise<UnlimitedOcrParseResult> {
    if (!Array.isArray(images) || images.length === 0) {
      throw new Error('Unlimited-OCR parse requires at least one image');
    }
    if (images.length > this.maxImages) {
      throw new Error(`Unlimited-OCR input exceeded ${this.maxImages} images`);
    }
    const maxTokens =
      finiteInteger(options.maxTokens, 'Unlimited-OCR maxTokens', 1, 32_768) ??
      UNLIMITED_OCR_MAX_TOKENS;
    const normalized = images.map((image) => normalizeImageInput(image, this.maxImageBytes));
    const totalBytes = normalized.reduce((sum, image) => sum + image.byteLength, 0);
    if (totalBytes > this.maxTotalImageBytes) {
      throw new Error(
        `Unlimited-OCR total image bytes exceeded ${this.maxTotalImageBytes}`
      );
    }
    const windowSize =
      normalized.length > 1
        ? UNLIMITED_OCR_MULTI_PAGE_WINDOW
        : UNLIMITED_OCR_SINGLE_PAGE_WINDOW;
    const { signal, cleanup } = combineAbortSignals(this.timeoutMs, options.signal);
    try {
      const response = await this.fetchFn(`${this.baseUrl}/chat/completions`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${this.options.apiKey ?? 'EMPTY'}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: this.model,
          messages: [
            {
              role: 'user',
              content: [
                { type: 'text', text: UNLIMITED_OCR_PROMPT },
                ...normalized.map((image) => ({
                  type: 'image_url',
                  image_url: { url: image.dataUrl },
                })),
              ],
            },
          ],
          max_tokens: maxTokens,
          temperature: 0,
          skip_special_tokens: false,
          vllm_xargs: {
            ngram_size: UNLIMITED_OCR_NGRAM_SIZE,
            window_size: windowSize,
          },
        }),
        signal,
      });
      if (!response.ok) {
        const detail = providerErrorDetail(await response.text());
        throw new Error(
          `Unlimited-OCR request failed with status ${response.status}` +
          (detail === undefined ? '' : `: ${detail}`)
        );
      }
      const contentLength = response.headers.get('content-length');
      if (
        contentLength !== null &&
        Number.isFinite(Number(contentLength)) &&
        Number(contentLength) > this.maxResponseBytes
      ) {
        throw new Error(`Unlimited-OCR response exceeded ${this.maxResponseBytes} bytes`);
      }
      const body = await response.text();
      if (Buffer.byteLength(body) > this.maxResponseBytes) {
        throw new Error(`Unlimited-OCR response exceeded ${this.maxResponseBytes} bytes`);
      }
      const payload = JSON.parse(body) as {
        model?: unknown;
        choices?: Array<{
          message?: { content?: unknown };
        }>;
      };
      const rawText = payload.choices?.[0]?.message?.content;
      if (typeof rawText !== 'string') {
        throw new Error('Unlimited-OCR response had no message content');
      }
      const parsed = parseUnlimitedOcrGrounding(rawText, {
        maxOutputChars: this.maxOutputChars,
        maxGroundingBlocks: this.maxGroundingBlocks,
      });
      return {
        rawText,
        markdown: parsed.markdown,
        blocks: parsed.blocks,
        model: typeof payload.model === 'string' ? payload.model : this.model,
      };
    } catch (error) {
      if (signal.aborted && /timed out/i.test(String(signal.reason))) {
        throw signal.reason instanceof Error
          ? signal.reason
          : new Error(String(signal.reason));
      }
      throw error;
    } finally {
      cleanup();
    }
  }
}
