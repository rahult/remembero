/**
 * One adapter process for a whole run. Spawned without a shell, fed one JSON line per
 * question, read back one JSON line per question. Requests queue, so an adapter never sees
 * two questions at once and its per-question store stays isolated. stderr is counted and
 * discarded; a response over the byte limit or past the timeout kills the child.
 * A child that dies is respawned on the next request: the protocol requires a fresh store
 * per question anyway, so nothing carries over.
 */
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import {
  parseMemorySystemResponse,
  type MemorySystemClient,
  type MemorySystemRequest,
  type MemorySystemResponse,
} from './memory-systems-protocol.js';

const DEFAULT_TIMEOUT_MS = 900_000;
const DEFAULT_MAX_RESPONSE_BYTES = 4 * 1024 * 1024;
const CLOSE_GRACE_MS = 5_000;

export interface MemorySystemProcessOptions {
  id: string;
  executable: string;
  args?: string[];
  workingDirectory?: string;
  env?: Record<string, string>;
  /** Per request, not per process. */
  timeoutMs?: number;
  maxResponseBytes?: number;
}

export function createMemorySystemProcess(options: MemorySystemProcessOptions): MemorySystemClient {
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const maxResponseBytes = options.maxResponseBytes ?? DEFAULT_MAX_RESPONSE_BYTES;
  let child: ChildProcessWithoutNullStreams | undefined;
  let buffer = '';
  let diagnosticBytes = 0;
  let pending:
    | {
        request: MemorySystemRequest;
        resolve: (value: MemorySystemResponse) => void;
        reject: (error: Error) => void;
        timer: NodeJS.Timeout;
      }
    | undefined;
  let queue: Promise<unknown> = Promise.resolve();

  const settle = (error?: Error, value?: MemorySystemResponse): void => {
    const current = pending;
    if (current === undefined) return;
    pending = undefined;
    clearTimeout(current.timer);
    if (error !== undefined) current.reject(error);
    else current.resolve(value!);
  };

  const consume = (): void => {
    let newline = buffer.indexOf('\n');
    while (newline >= 0) {
      const line = buffer.slice(0, newline);
      buffer = buffer.slice(newline + 1);
      newline = buffer.indexOf('\n');
      if (line.trim() === '') continue;
      let parsed: unknown;
      try {
        parsed = JSON.parse(line) as unknown;
      } catch {
        // A banner or a progress line on stdout is chatter, not an answer: count it with the
        // rest of the suppressed diagnostics and keep waiting for a real response.
        diagnosticBytes += Buffer.byteLength(line, 'utf8');
        continue;
      }
      const current = pending;
      if (current === undefined) continue;
      try {
        settle(undefined, parseMemorySystemResponse(parsed, current.request));
      } catch (error) {
        settle(error instanceof Error ? error : new Error(String(error)));
      }
    }
  };

  const start = (): ChildProcessWithoutNullStreams => {
    if (child !== undefined) return child;
    buffer = '';
    diagnosticBytes = 0;
    const spawned = spawn(options.executable, options.args ?? [], {
      shell: false,
      stdio: ['pipe', 'pipe', 'pipe'],
      cwd: options.workingDirectory,
      env: {
        PATH: process.env.PATH ?? '/usr/bin:/bin',
        HOME: process.env.HOME ?? '',
        LANG: process.env.LANG ?? 'C.UTF-8',
        ...options.env,
      },
    });
    spawned.stdout.setEncoding('utf8');
    spawned.stdout.on('data', (chunk: string) => {
      buffer += chunk;
      if (Buffer.byteLength(buffer, 'utf8') > maxResponseBytes) {
        buffer = '';
        // Disown before killing: the next question must reach a fresh child, and this one's
        // late close event must not settle it.
        if (child === spawned) child = undefined;
        spawned.kill('SIGKILL');
        settle(new Error(`${options.id} response exceeded ${maxResponseBytes} bytes`));
        return;
      }
      consume();
    });
    spawned.stderr.on('data', (chunk: Buffer) => {
      diagnosticBytes += chunk.length;
    });
    // A broken stdin pipe is not the real failure; the close handler reports that.
    spawned.stdin.on('error', () => {});
    // A child we have already disowned (killed at a timeout, at the byte cap, or closed) must
    // never settle the request that came after it.
    spawned.on('error', (error) => {
      if (child !== spawned) return;
      child = undefined;
      settle(error);
    });
    spawned.on('close', (code, signal) => {
      if (child !== spawned) return;
      child = undefined;
      settle(
        new Error(
          `${options.id} exited with ${signal ?? code}; stderr suppressed (${diagnosticBytes} bytes)`,
        ),
      );
    });
    child = spawned;
    return spawned;
  };

  return {
    id: options.id,
    async request(request) {
      const run = queue.then(
        async () =>
          await new Promise<MemorySystemResponse>((resolve, reject) => {
            const spawned = start();
            pending = {
              request,
              resolve,
              reject,
              timer: setTimeout(() => {
                // Disown before killing, so the next question spawns a fresh child instead of
                // inheriting this one's SIGKILL.
                if (child === spawned) child = undefined;
                spawned.kill('SIGKILL');
                settle(
                  new Error(
                    `${options.id} timed out after ${timeoutMs}ms on ${request.questionId}`,
                  ),
                );
              }, timeoutMs),
            };
            spawned.stdin.write(`${JSON.stringify(request)}\n`);
          }),
      );
      queue = run.then(
        () => undefined,
        () => undefined,
      );
      return await run;
    },
    async close() {
      // A request still in flight is answered by nobody once the child is gone; say so rather
      // than leaving its promise pending forever.
      settle(new Error(`${options.id} closed with a request in flight`));
      const spawned = child;
      if (spawned === undefined) return;
      child = undefined;
      spawned.stdin.end();
      await new Promise<void>((resolve) => {
        const timer = setTimeout(() => {
          spawned.kill('SIGKILL');
          resolve();
        }, CLOSE_GRACE_MS);
        spawned.on('close', () => {
          clearTimeout(timer);
          resolve();
        });
      });
    },
  };
}
