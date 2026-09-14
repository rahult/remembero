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
      const current = pending;
      if (line.trim() !== '' && current !== undefined) {
        try {
          settle(
            undefined,
            parseMemorySystemResponse(JSON.parse(line) as unknown, current.request),
          );
        } catch (error) {
          settle(error instanceof Error ? error : new Error(String(error)));
        }
      }
      newline = buffer.indexOf('\n');
    }
  };

  const start = (): ChildProcessWithoutNullStreams => {
    if (child !== undefined) return child;
    buffer = '';
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
    spawned.on('error', (error) => {
      child = undefined;
      settle(error);
    });
    spawned.on('close', (code, signal) => {
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
