import { mkdirSync, mkdtempSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { beforeEach, describe, expect, it } from 'vitest';
import { runInit, type InitExec } from '../src/init.js';

let root: string;
let settingsPath: string;

function options(exec: InitExec, env: NodeJS.ProcessEnv = {}) {
  return {
    settingsPath,
    nodePath: '/usr/local/bin/node',
    cliPath: '/opt/rembero/dist/cli.js',
    namespace: 'personal',
    dailyCap: 10,
    tailBytes: 24_576,
    exec,
    env,
  };
}

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), 'rembero-init-'));
  mkdirSync(join(root, '.claude'), { recursive: true });
  settingsPath = join(root, '.claude', 'settings.json');
});

describe('remembero init', () => {
  it('installs both managed hooks and registers the core-profile MCP server', () => {
    const calls: string[][] = [];
    const result = runInit(
      options((command, args) => {
        calls.push([command, ...args]);
        return { status: 0, stdout: 'added', stderr: '' };
      })
    );

    expect(result.hooks.changed).toBe(true);
    const settings = JSON.parse(readFileSync(settingsPath, 'utf8'));
    expect(settings.hooks.Stop).toBeDefined();
    expect(settings.hooks.SessionStart).toBeDefined();

    // The claude CLI requires the server name before flag arguments.
    expect(calls).toEqual([
      [
        'claude',
        'mcp',
        'add',
        'remembero',
        '--scope',
        'user',
        '-e',
        'REMBERO_VALID_TIME_MODE=archive_until',
        '--',
        '/usr/local/bin/node',
        '/opt/rembero/dist/cli.js',
        'serve',
        '--profile',
        'core',
        '-n',
        'personal',
      ],
    ]);
    expect(result.registration.ok).toBe(true);
    expect(result.claudeMdSnippet).toContain('recall');
    expect(result.claudeMdSnippet).toContain('remember');
  });

  it('forwards a configured LLM_API_KEY into the registration environment', () => {
    const calls: string[][] = [];
    runInit(
      options(
        (command, args) => {
          calls.push([command, ...args]);
          return { status: 0, stdout: '', stderr: '' };
        },
        { LLM_API_KEY: 'sk-or-test' }
      )
    );
    expect(calls[0]).toContain('LLM_API_KEY=sk-or-test');
  });

  it('reports guidance instead of failing when the claude CLI is unavailable', () => {
    const result = runInit(
      options(() => {
        throw new Error('spawn claude ENOENT');
      })
    );
    expect(result.registration.ok).toBe(false);
    expect(result.registration.command.join(' ')).toContain('claude mcp add');
    const settings = JSON.parse(readFileSync(settingsPath, 'utf8'));
    expect(settings.hooks.Stop).toBeDefined();
  });

  it('treats an already-registered server as success', () => {
    const result = runInit(
      options(() => ({
        status: 1,
        stdout: '',
        stderr: 'MCP server remembero already exists in user config',
      }))
    );
    expect(result.registration.ok).toBe(true);
  });
});
