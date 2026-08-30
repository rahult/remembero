#!/usr/bin/env node
import { createReadStream, existsSync, statSync } from 'node:fs';
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { dirname, extname, join, resolve, sep } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { loadEnv } from '../env.js';
import { lazyClientFromEnv } from '../llm/client.js';
import { MAX_INPUT_BYTES, stringifyBoundedResult } from '../safety.js';
import { defaultRoot, MemoryStore } from '../store/store.js';
import { openSemanticLedgerIfSupported } from '../ledger/remembero-review.js';
import { RemberoWebService, WebServiceError } from './service.js';

const LOOPBACK_HOSTS = new Set(['127.0.0.1', 'localhost', '::1']);
const MIME_TYPES: Record<string, string> = {
  '.css': 'text/css; charset=utf-8',
  '.html': 'text/html; charset=utf-8',
  '.ico': 'image/x-icon',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.map': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.webp': 'image/webp',
};

export interface StartWebServerOptions {
  dev?: boolean;
  host?: string;
  port?: number;
  root?: string;
  namespace?: string;
  seedDemo?: boolean;
  /** Demo mode: a seeded fictional sandbox instead of the real memory root. */
  demo?: boolean;
}

export interface ResolvedWebConfig {
  demo: boolean;
  root: string;
  namespace: string | undefined;
  seedDemo: boolean;
}

/**
 * The console shows the user's real memory by default; the fictional demo
 * workspace and its `.rembero-web` sandbox are explicit opt-ins.
 */
export function resolveWebConfig(
  options: StartWebServerOptions = {},
  env: NodeJS.ProcessEnv = process.env
): ResolvedWebConfig {
  const demo =
    options.demo ?? (options.seedDemo === true || env.REMBERO_WEB_DEMO === 'true');
  const root = options.root ?? env.REMBERO_WEB_ROOT ?? (demo ? resolve('.rembero-web') : defaultRoot());
  const namespace =
    options.namespace ?? env.REMBERO_WEB_NAMESPACE ?? (demo ? undefined : 'default');
  const seedDemo = options.seedDemo ?? (demo && env.REMBERO_WEB_SEED_DEMO !== 'false');
  return { demo, root, namespace, seedDemo };
}

function jsonBody(request: IncomingMessage): Promise<unknown> {
  return new Promise((resolveBody, reject) => {
    const chunks: Buffer[] = [];
    let bytes = 0;
    request.on('data', (chunk: Buffer) => {
      bytes += chunk.length;
      if (bytes > MAX_INPUT_BYTES) {
        reject(new WebServiceError('request_too_large', 'Request body is too large.', 413));
        request.destroy();
        return;
      }
      chunks.push(chunk);
    });
    request.on('end', () => {
      if (chunks.length === 0) {
        resolveBody({});
        return;
      }
      try {
        resolveBody(JSON.parse(Buffer.concat(chunks).toString('utf8')));
      } catch {
        reject(new WebServiceError('invalid_json', 'Request body must be valid JSON.'));
      }
    });
    request.on('error', reject);
  });
}

function objectBody(value: unknown): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new WebServiceError('invalid_request', 'Request body must be a JSON object.');
  }
  return value as Record<string, unknown>;
}

function stringField(
  body: Record<string, unknown>,
  key: string,
  optional = false
): string | undefined {
  const value = body[key];
  if (value === undefined && optional) return undefined;
  if (typeof value !== 'string') {
    throw new WebServiceError('invalid_request', `${key} must be a string.`);
  }
  return value;
}

function searchKinds(value: unknown): Array<'fact' | 'rule' | 'constraint'> | undefined {
  if (value === undefined) return undefined;
  if (!Array.isArray(value) || value.length === 0 || value.length > 3) {
    throw new WebServiceError('invalid_request', 'kinds must be a non-empty array.');
  }
  const result = [...new Set(value)];
  if (
    result.some(
      (kind) => kind !== 'fact' && kind !== 'rule' && kind !== 'constraint'
    )
  ) {
    throw new WebServiceError(
      'invalid_request',
      'kinds may contain fact, rule, or constraint.'
    );
  }
  return result as Array<'fact' | 'rule' | 'constraint'>;
}

function securityHeaders(response: ServerResponse, dev: boolean): void {
  response.setHeader('X-Content-Type-Options', 'nosniff');
  response.setHeader('Referrer-Policy', 'no-referrer');
  response.setHeader('X-Frame-Options', 'DENY');
  response.setHeader('Permissions-Policy', 'camera=(), microphone=(), geolocation=()');
  if (!dev) {
    response.setHeader(
      'Content-Security-Policy',
      "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
    );
  }
}

function sendJson(response: ServerResponse, value: unknown, status = 200): void {
  const text = stringifyBoundedResult(value, 'web API response');
  response.statusCode = status;
  response.setHeader('Content-Type', 'application/json; charset=utf-8');
  response.setHeader('Cache-Control', 'no-store');
  response.end(text);
}

function assertSameOrigin(request: IncomingMessage): void {
  const origin = request.headers.origin;
  if (origin === undefined) return;
  const host = request.headers.host;
  let originHost: string | undefined;
  try {
    originHost = new URL(origin).host;
  } catch {
    originHost = undefined;
  }
  if (host === undefined || originHost !== host) {
    throw new WebServiceError('origin_rejected', 'Cross-origin requests are not allowed.', 403);
  }
}

async function apiResponse(
  service: RemberoWebService,
  request: IncomingMessage,
  url: URL
): Promise<unknown> {
  const method = request.method ?? 'GET';
  if (method !== 'GET') assertSameOrigin(request);
  if (method === 'GET' && url.pathname === '/api/bootstrap') {
    return service.bootstrap();
  }
  if (method === 'GET' && url.pathname === '/api/document') {
    return service.documentShowcase({
      ...(url.searchParams.get('documentId') === null
        ? {}
        : { documentId: url.searchParams.get('documentId') ?? undefined }),
    });
  }
  if (method === 'GET' && url.pathname === '/api/document/memorg') {
    return service.documentMemorg();
  }
  if (method === 'GET' && url.pathname === '/api/graph') {
    return service.graph({ focus: url.searchParams.get('focus') ?? 'atlas' });
  }
  if (method === 'GET' && url.pathname === '/api/versions') {
    return service.versionWorkspace();
  }
  if (method === 'GET' && url.pathname === '/api/versions/history') {
    return { ref: url.searchParams.get('ref') ?? 'main', history: service.semanticRefHistory(url.searchParams.get('ref') ?? 'main') };
  }
  if (method !== 'POST') {
    throw new WebServiceError('method_not_allowed', 'Method not allowed.', 405);
  }
  const body = objectBody(await jsonBody(request));
  if (url.pathname === '/api/seed') return service.seedDemo();
  if (url.pathname === '/api/document/parse') {
    return service.parseDocument({
      ...(stringField(body, 'documentId', true) === undefined
        ? {}
        : { documentId: stringField(body, 'documentId', true) }),
    });
  }
  if (url.pathname === '/api/document/ask') {
    return service.askDocument({
      ...(stringField(body, 'documentId', true) === undefined
        ? {}
        : { documentId: stringField(body, 'documentId', true) }),
      questionId: stringField(body, 'questionId')!,
    });
  }
  if (url.pathname === '/api/ask') {
    return service.ask({
      question: stringField(body, 'question')!,
      ...(stringField(body, 'presetId', true) === undefined
        ? {}
        : { presetId: stringField(body, 'presetId', true) }),
    });
  }
  if (url.pathname === '/api/search') {
    return service.search({
      text: stringField(body, 'text')!,
      ...(body.kinds === undefined ? {} : { kinds: searchKinds(body.kinds) }),
    });
  }
  if (url.pathname === '/api/memory') {
    return service.addMemory({
      subject: stringField(body, 'subject')!,
      predicate: stringField(body, 'predicate')!,
      object: stringField(body, 'object')!,
      sourceText: stringField(body, 'sourceText')!,
    });
  }
  if (url.pathname === '/api/versions/capture') {
    return service.captureSemanticVersion({
      ...(stringField(body, 'label', true) === undefined
        ? {}
        : { label: stringField(body, 'label', true) }),
      ...(stringField(body, 'ref', true) === undefined
        ? {}
        : { ref: stringField(body, 'ref', true) }),
    });
  }
  if (url.pathname === '/api/versions/review') {
    return service.reviewSemanticVersion({
      candidateVersionDigest: stringField(body, 'candidateVersionDigest')!,
      ...(typeof body.includeDocumentEvaluation === 'boolean'
        ? { includeDocumentEvaluation: body.includeDocumentEvaluation }
        : {}),
    });
  }
  if (url.pathname === '/api/versions/promote') {
    const accepted = body.acceptedReviewDimensions;
    if (
      accepted !== undefined &&
      (!Array.isArray(accepted) || accepted.some((value) => typeof value !== 'string'))
    ) {
      throw new WebServiceError(
        'invalid_request',
        'acceptedReviewDimensions must be an array of strings.'
      );
    }
    return service.promoteSemanticVersion({
      ref: stringField(body, 'ref')!,
      candidateVersionDigest: stringField(body, 'candidateVersionDigest')!,
      assessmentDigest: stringField(body, 'assessmentDigest')!,
      operationId: stringField(body, 'operationId')!,
      ...(accepted === undefined ? {} : { acceptedReviewDimensions: accepted as string[] }),
      ...(stringField(body, 'reason', true) === undefined
        ? {}
        : { reason: stringField(body, 'reason', true) }),
    });
  }
  throw new WebServiceError('not_found', 'API route not found.', 404);
}

function serveFile(
  response: ServerResponse,
  root: string,
  requestedPath: string
): void {
  const relative = requestedPath === '/' ? 'index.html' : requestedPath.replace(/^\/+/, '');
  let file = resolve(root, relative);
  if (!file.startsWith(`${resolve(root)}${sep}`)) {
    throw new WebServiceError('not_found', 'File not found.', 404);
  }
  if (!existsSync(file) || !statSync(file).isFile()) {
    file = resolve(root, 'index.html');
  }
  if (!existsSync(file)) {
    throw new WebServiceError(
      'web_not_built',
      'Web client is not built. Run npm run web:build.',
      503
    );
  }
  response.statusCode = 200;
  response.setHeader(
    'Content-Type',
    MIME_TYPES[extname(file)] ?? 'application/octet-stream'
  );
  response.setHeader(
    'Cache-Control',
    file.endsWith('index.html') ? 'no-cache' : 'public, max-age=31536000, immutable'
  );
  createReadStream(file).pipe(response);
}

function portFromEnv(value: string | undefined): number {
  if (value === undefined) return 4318;
  if (!/^\d+$/.test(value)) throw new Error('REMBERO_WEB_PORT must be an integer');
  const port = Number(value);
  if (!Number.isSafeInteger(port) || port < 1 || port > 65_535) {
    throw new Error('REMBERO_WEB_PORT must be from 1 to 65535');
  }
  return port;
}

export async function startWebServer(options: StartWebServerOptions = {}) {
  loadEnv();
  const dev = options.dev ?? false;
  const host = options.host ?? process.env.REMBERO_WEB_HOST ?? '127.0.0.1';
  if (!LOOPBACK_HOSTS.has(host)) {
    throw new Error('Remembero web console supports loopback hosts only');
  }
  const port = options.port ?? portFromEnv(process.env.REMBERO_WEB_PORT);
  const { root, namespace, seedDemo } = resolveWebConfig(options);
  const store = new MemoryStore(root);
  const semantic = await openSemanticLedgerIfSupported(join(root, 'semantic.sqlite'));
  const llmConfigured = Boolean(process.env.LLM_API_KEY);
  const service = new RemberoWebService({
    store,
    ...(semantic === undefined ? {} : { ledger: semantic.ledger }),
    llm: lazyClientFromEnv(),
    llmConfigured,
    ...(namespace === undefined ? {} : { namespace }),
  });
  if (seedDemo) {
    if (service.bootstrap().empty) service.seedDemo();
    service.parseAllDocuments();
  }
  service.ensureSemanticBaseline();

  const clientRoot = resolve(
    dirname(fileURLToPath(import.meta.url)),
    dev ? '../../web' : '../web-client'
  );
  const vite = dev
    ? await import('vite').then(({ createServer: createViteServer }) =>
        createViteServer({
          root: clientRoot,
          appType: 'spa',
          server: { middlewareMode: true },
        })
      )
    : undefined;

  const server = createServer(async (request, response) => {
    securityHeaders(response, dev);
    try {
      const url = new URL(request.url ?? '/', `http://${request.headers.host ?? host}`);
      if (url.pathname.startsWith('/api/')) {
        sendJson(response, await apiResponse(service, request, url));
        return;
      }
      if (vite !== undefined) {
        vite.middlewares(request, response, (error?: unknown) => {
          if (error !== undefined) {
            sendJson(
              response,
              { error: 'vite_error', message: error instanceof Error ? error.message : String(error) },
              500
            );
          }
        });
        return;
      }
      serveFile(response, clientRoot, url.pathname);
    } catch (error) {
      const status = error instanceof WebServiceError ? error.status : 500;
      sendJson(
        response,
        {
          error: error instanceof WebServiceError ? error.code : 'internal_error',
          message: error instanceof Error ? error.message : String(error),
        },
        status
      );
    }
  });

  await new Promise<void>((resolveListen, reject) => {
    server.once('error', reject);
    server.listen(port, host, () => resolveListen());
  });
  const address = server.address();
  const boundPort = typeof address === 'object' && address !== null
    ? address.port
    : port;
  const url = `http://${host}:${boundPort}`;
  process.stdout.write(`Remembero web console: ${url}\n`);
  process.stdout.write(`Memory root: ${root}\n`);
  return {
    server,
    service,
    url,
    close: async () => {
      await vite?.close();
      semantic?.database.close();
      await new Promise<void>((resolveClose, reject) =>
        server.close((error) => (error === undefined ? resolveClose() : reject(error)))
      );
    },
  };
}

const entry = process.argv[1];
if (entry !== undefined && import.meta.url === pathToFileURL(entry).href) {
  startWebServer({
    dev: process.argv.includes('--dev'),
    ...(process.argv.includes('--demo') ? { demo: true } : {}),
  }).catch((error) => {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  });
}
