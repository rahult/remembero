import { spawnSync } from 'node:child_process';
import {
  installClaudeHook,
  type HookChangeResult,
} from './autocapture/hooks.js';

export type InitExec = (
  command: string,
  args: string[]
) => { status: number | null; stdout: string; stderr: string };

export interface InitOptions {
  settingsPath: string;
  nodePath: string;
  cliPath: string;
  namespace: string;
  dailyCap: number;
  tailBytes: number;
  exec?: InitExec;
  env?: NodeJS.ProcessEnv;
}

export interface InitRegistration {
  command: string[];
  ok: boolean;
  detail: string;
}

export interface InitResult {
  hooks: HookChangeResult;
  registration: InitRegistration;
  claudeMdSnippet: string;
}

function defaultExec(command: string, args: string[]): ReturnType<InitExec> {
  const run = spawnSync(command, args, { encoding: 'utf8', timeout: 30_000 });
  if (run.error) throw run.error;
  return { status: run.status, stdout: run.stdout ?? '', stderr: run.stderr ?? '' };
}

export function claudeMdSnippet(namespace: string): string {
  return `## Memory (Remembero)
- A session-start brief of remembered facts is injected automatically; use the
  \`recall\` or \`query\` tools when you need more than the brief shows.
- When I state something durable — a preference, decision, relationship, or fact
  about me or a project — store it with \`remember\` (namespace '${namespace}').
  Updates ("X is now Y") supersede old facts.
- Never store secrets or transient details. When unsure whether to remember, ask.`;
}

/**
 * One-command onboarding: install both managed Claude hooks, register the
 * core-profile MCP server through the claude CLI, and hand back the CLAUDE.md
 * snippet. Registration failure is reported, never thrown — the hooks and
 * guidance still stand on their own.
 */
export function runInit(options: InitOptions): InitResult {
  const hooks = installClaudeHook({
    settingsPath: options.settingsPath,
    nodePath: options.nodePath,
    cliPath: options.cliPath,
    namespace: options.namespace,
    dailyCap: options.dailyCap,
    tailBytes: options.tailBytes,
  });

  const env = options.env ?? process.env;
  const envFlags = ['-e', 'REMBERO_VALID_TIME_MODE=archive_until'];
  if (env.LLM_API_KEY !== undefined && env.LLM_API_KEY !== '') {
    envFlags.push('-e', `LLM_API_KEY=${env.LLM_API_KEY}`);
  }
  // The claude CLI requires the server name before flag arguments.
  const command = [
    'claude',
    'mcp',
    'add',
    'remembero',
    '--scope',
    'user',
    ...envFlags,
    '--',
    options.nodePath,
    options.cliPath,
    'serve',
    '--profile',
    'core',
    '-n',
    options.namespace,
  ];

  let registration: InitRegistration;
  const exec = options.exec ?? defaultExec;
  try {
    const run = exec(command[0], command.slice(1));
    if (run.status === 0) {
      registration = { command, ok: true, detail: 'registered with Claude Code' };
    } else if (/already exists/i.test(`${run.stdout}\n${run.stderr}`)) {
      registration = { command, ok: true, detail: 'already registered with Claude Code' };
    } else {
      registration = {
        command,
        ok: false,
        detail: `claude mcp add exited with status ${run.status}; run it manually`,
      };
    }
  } catch (error) {
    registration = {
      command,
      ok: false,
      detail: `could not run the claude CLI (${
        error instanceof Error ? error.message : String(error)
      }); run the command manually`,
    };
  }

  return {
    hooks,
    registration,
    claudeMdSnippet: claudeMdSnippet(options.namespace),
  };
}
