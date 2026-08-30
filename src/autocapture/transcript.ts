import {
  closeSync,
  fstatSync,
  lstatSync,
  openSync,
  readSync,
  realpathSync,
  statSync,
} from 'node:fs';
import { homedir } from 'node:os';
import { isAbsolute, join, relative, resolve, sep } from 'node:path';
import { assertBoundedInput } from '../safety.js';

export const DEFAULT_TRANSCRIPT_TAIL_BYTES = 24 * 1024;
export const MAX_TRANSCRIPT_TAIL_BYTES = 48 * 1024;
const MAX_TRANSCRIPT_FILE_BYTES = 512 * 1024 * 1024;
// Agentic transcripts bury user turns under megabytes of tool_result lines, so the
// backward scan may widen well past the base window before it sees user text.
const MAX_READ_WINDOW_BYTES = 16 * 1024 * 1024;

export interface ClaudeStopHookInput {
  sessionId: string;
  transcriptPath: string;
  cwd: string;
  stopHookActive: boolean;
  lastAssistantMessage: string;
}

export interface TranscriptReadOptions {
  claudeConfigDir?: string;
  tailBytes?: number;
  /**
   * Select only user-authored messages and ignore `lastAssistantMessage`.
   * Assistant prose is context, never memory authority; extraction uses this
   * so a model cannot mint "facts" from the assistant's own words.
   */
  userOnly?: boolean;
}

export interface ClaudeTranscriptTail {
  text: string;
  bytes: number;
  messageCount: number;
  userMessageCount: number;
}

function requireString(
  value: unknown,
  field: string,
  options: { allowEmpty?: boolean; maxLength?: number } = {}
): string {
  if (typeof value !== 'string' || (!options.allowEmpty && value.trim() === '')) {
    throw new Error(`Claude Stop hook field '${field}' must be a non-empty string`);
  }
  if (value.length > (options.maxLength ?? 4096)) {
    throw new Error(`Claude Stop hook field '${field}' is too long`);
  }
  return value;
}

export function parseClaudeStopHookInput(raw: string): ClaudeStopHookInput {
  assertBoundedInput(raw, 'Claude Stop hook input');
  let parsed: Record<string, unknown>;
  try {
    parsed = JSON.parse(raw) as Record<string, unknown>;
  } catch {
    throw new Error('Claude Stop hook input must be valid JSON');
  }
  if (parsed === null || Array.isArray(parsed)) {
    throw new Error('Claude Stop hook input must be a JSON object');
  }
  if (parsed.hook_event_name !== 'Stop') {
    throw new Error("auto-capture accepts only Claude's Stop hook event");
  }
  if (typeof parsed.stop_hook_active !== 'boolean') {
    throw new Error("Claude Stop hook field 'stop_hook_active' must be a boolean");
  }
  const sessionId = requireString(parsed.session_id, 'session_id', { maxLength: 256 });
  if (!/^[a-zA-Z0-9._-]+$/.test(sessionId)) {
    throw new Error("Claude Stop hook field 'session_id' has invalid characters");
  }
  return {
    sessionId,
    transcriptPath: requireString(parsed.transcript_path, 'transcript_path'),
    cwd: requireString(parsed.cwd, 'cwd'),
    stopHookActive: parsed.stop_hook_active,
    lastAssistantMessage: requireString(
      parsed.last_assistant_message ?? '',
      'last_assistant_message',
      { allowEmpty: true, maxLength: MAX_TRANSCRIPT_TAIL_BYTES }
    ),
  };
}

export function defaultClaudeConfigDir(env: NodeJS.ProcessEnv = process.env): string {
  return resolve(env.CLAUDE_CONFIG_DIR ?? join(homedir(), '.claude'));
}

function expandHome(path: string): string {
  if (path === '~') return homedir();
  if (path.startsWith(`~${sep}`) || path.startsWith('~/')) {
    return join(homedir(), path.slice(2));
  }
  return path;
}

function validateTailBytes(value: number): number {
  if (!Number.isSafeInteger(value) || value < 1024 || value > MAX_TRANSCRIPT_TAIL_BYTES) {
    throw new Error(
      `transcript tail byte limit must be an integer between 1024 and ${MAX_TRANSCRIPT_TAIL_BYTES}`
    );
  }
  return value;
}

interface TrustedTranscript {
  path: string;
  device: number;
  inode: number;
}

function trustedTranscriptPath(path: string, claudeConfigDir: string): TrustedTranscript {
  const expanded = expandHome(path);
  if (!isAbsolute(expanded)) {
    throw new Error('Claude transcript path must be absolute');
  }
  const candidate = resolve(expanded);
  const linkStat = lstatSync(candidate);
  if (linkStat.isSymbolicLink()) {
    throw new Error('refusing to read a symbolic link as a Claude transcript');
  }
  if (!linkStat.isFile()) throw new Error('Claude transcript path must name a regular file');
  if (linkStat.nlink !== 1) {
    throw new Error('refusing to read a hard-linked file as a Claude transcript');
  }
  if (linkStat.size > MAX_TRANSCRIPT_FILE_BYTES) {
    throw new Error(`Claude transcript exceeds ${MAX_TRANSCRIPT_FILE_BYTES} bytes`);
  }
  if (!candidate.endsWith('.jsonl')) {
    throw new Error('Claude transcript path must end in .jsonl');
  }

  const trustedRoot = realpathSync(join(claudeConfigDir, 'projects'));
  const canonical = realpathSync(candidate);
  const fromRoot = relative(trustedRoot, canonical);
  if (fromRoot === '..' || fromRoot.startsWith(`..${sep}`) || isAbsolute(fromRoot)) {
    throw new Error('Claude transcript path is outside the trusted Claude transcript root');
  }
  return { path: canonical, device: linkStat.dev, inode: linkStat.ino };
}

function textBlocks(content: unknown): string[] {
  if (typeof content === 'string') return [content];
  if (!Array.isArray(content)) return [];
  const text: string[] = [];
  for (const block of content) {
    if (
      typeof block === 'object' &&
      block !== null &&
      (block as Record<string, unknown>).type === 'text' &&
      typeof (block as Record<string, unknown>).text === 'string'
    ) {
      text.push((block as Record<string, unknown>).text as string);
    }
  }
  return text;
}

function removeTranscriptNoise(text: string): string {
  return text
    .replace(/```[\s\S]*?```/g, '')
    .replace(/<system-reminder>[\s\S]*?<\/system-reminder>/gi, '')
    .replace(/<task-notification>[\s\S]*?<\/task-notification>/gi, '')
    .replace(/<tool-use>[\s\S]*?<\/tool-use>/gi, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

function parseTranscriptMessages(text: string): { role: 'user' | 'assistant'; text: string }[] {
  const messages: { role: 'user' | 'assistant'; text: string }[] = [];
  for (const line of text.split('\n')) {
    if (line.trim() === '') continue;
    let entry: Record<string, unknown>;
    try {
      entry = JSON.parse(line) as Record<string, unknown>;
    } catch {
      continue;
    }
    if (entry === null || Array.isArray(entry) || entry.isSidechain === true) continue;
    if (entry.type !== 'user' && entry.type !== 'assistant') continue;
    const message = entry.message;
    if (typeof message !== 'object' || message === null || Array.isArray(message)) continue;
    const role = (message as Record<string, unknown>).role;
    if (role !== 'user' && role !== 'assistant') continue;
    const content = removeTranscriptNoise(
      textBlocks((message as Record<string, unknown>).content).join('\n')
    );
    if (content === '') continue;
    const previous = messages.at(-1);
    if (previous?.role === role && previous.text === content) continue;
    messages.push({ role, text: content });
  }
  return messages;
}

function boundedMessages(
  messages: { role: 'user' | 'assistant'; text: string }[],
  maxBytes: number
): { role: 'user' | 'assistant'; text: string }[] {
  const suffix = (value: string, bytes: number): string => {
    const buffer = Buffer.from(value, 'utf8');
    if (buffer.length <= bytes) return value;
    let start = buffer.length - bytes;
    while (start < buffer.length && (buffer[start] & 0xc0) === 0x80) start += 1;
    return buffer.subarray(start).toString('utf8');
  };
  const perMessageBytes = Math.max(128, Math.floor(maxBytes / 3));
  const compact = messages.map((message) => ({
    ...message,
    text: suffix(message.text, perMessageBytes),
  }));
  let latestUserIndex = -1;
  for (let index = compact.length - 1; index >= 0; index -= 1) {
    if (compact[index].role === 'user') {
      latestUserIndex = index;
      break;
    }
  }
  const selected = new Map<number, { role: 'user' | 'assistant'; text: string }>();
  let used = 0;
  if (latestUserIndex >= 0) {
    const message = compact[latestUserIndex];
    const rendered = `${message.role.toUpperCase()}: ${message.text}`;
    selected.set(latestUserIndex, message);
    used = Buffer.byteLength(rendered, 'utf8');
  }
  for (let index = compact.length - 1; index >= 0; index -= 1) {
    if (selected.has(index)) continue;
    const message = compact[index];
    const rendered = `${message.role.toUpperCase()}: ${message.text}`;
    const bytes = Buffer.byteLength(rendered, 'utf8') + (selected.size > 0 ? 2 : 0);
    if (bytes > maxBytes - used) continue;
    selected.set(index, message);
    used += bytes;
  }
  return [...selected.entries()]
    .sort(([left], [right]) => left - right)
    .map(([, message]) => message);
}

function readTranscriptWindow(trusted: TrustedTranscript, size: number, readBytes: number): string {
  const start = Math.max(0, size - readBytes);
  const buffer = Buffer.alloc(readBytes);
  const descriptor = openSync(trusted.path, 'r');
  try {
    const opened = fstatSync(descriptor);
    if (
      !opened.isFile() ||
      opened.nlink !== 1 ||
      opened.dev !== trusted.device ||
      opened.ino !== trusted.inode
    ) {
      throw new Error('Claude transcript changed during validation');
    }
    if (readBytes > 0) readSync(descriptor, buffer, 0, readBytes, start);
  } finally {
    closeSync(descriptor);
  }
  let raw = buffer.toString('utf8');
  if (start > 0) {
    const firstNewline = raw.indexOf('\n');
    raw = firstNewline === -1 ? '' : raw.slice(firstNewline + 1);
  }
  return raw;
}

export function readClaudeTranscriptTail(
  input: ClaudeStopHookInput,
  options: TranscriptReadOptions = {}
): ClaudeTranscriptTail {
  const tailBytes = validateTailBytes(options.tailBytes ?? DEFAULT_TRANSCRIPT_TAIL_BYTES);
  const configDir = resolve(options.claudeConfigDir ?? defaultClaudeConfigDir());
  const trusted = trustedTranscriptPath(input.transcriptPath, configDir);
  const size = statSync(trusted.path).size;
  let readBytes = Math.min(size, MAX_READ_WINDOW_BYTES, Math.max(64 * 1024, tailBytes * 8));
  let messages = parseTranscriptMessages(readTranscriptWindow(trusted, size, readBytes));
  while (
    !messages.some((message) => message.role === 'user') &&
    readBytes < Math.min(size, MAX_READ_WINDOW_BYTES)
  ) {
    readBytes = Math.min(size, MAX_READ_WINDOW_BYTES, readBytes * 4);
    messages = parseTranscriptMessages(readTranscriptWindow(trusted, size, readBytes));
  }
  if (options.userOnly === true) {
    const users = messages.filter((message) => message.role === 'user');
    const selected = boundedMessages(users, tailBytes);
    const text = selected
      .map((message) => `${message.role.toUpperCase()}: ${message.text}`)
      .join('\n\n');
    return {
      text,
      bytes: Buffer.byteLength(text, 'utf8'),
      messageCount: selected.length,
      userMessageCount: selected.length,
    };
  }
  const lastAssistant = removeTranscriptNoise(input.lastAssistantMessage);
  if (
    lastAssistant !== '' &&
    !(messages.at(-1)?.role === 'assistant' && messages.at(-1)?.text === lastAssistant)
  ) {
    messages.push({ role: 'assistant', text: lastAssistant });
  }
  const selected = boundedMessages(messages, tailBytes);
  const text = selected
    .map((message) => `${message.role.toUpperCase()}: ${message.text}`)
    .join('\n\n');
  return {
    text,
    bytes: Buffer.byteLength(text, 'utf8'),
    messageCount: selected.length,
    userMessageCount: selected.filter((message) => message.role === 'user').length,
  };
}
