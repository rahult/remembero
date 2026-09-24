/**
 * The decision path as a CLI — the pilot service in miniature:
 *   node dist/evals/run-support-decide.js --case benchmarks/support/demo-case.json \
 *     --question "is a credit due for TCK-1042" [--extract] [--store results/support-verdicts.jsonl]
 *
 * The case file holds a ticket (and optionally chat turns). The question is planned into a
 * closed shape (or rejected), claims are gathered (system fields, plus LLM extraction behind
 * the gate with --extract), the engine decides, and the whole decision basis is appended to
 * the verdict store. The proof is the output — not a confidence, an artifact.
 */

import { readFileSync } from 'node:fs';
import { OpenRouterClient } from '../llm/client.js';
import { admitAll } from '../support/claims.js';
import { decide } from '../support/engine.js';
import { claimsFromTicket, extractClaims } from '../support/extract.js';
import { loadPack } from '../support/pack.js';
import { plan } from '../support/planner.js';
import { chatSource, ticketSource, type ChatInput, type Source, type TicketInput } from '../support/sources.js';
import { VerdictStore } from '../support/store.js';

interface CaseFile {
  ticket: TicketInput;
  chat?: ChatInput;
}

async function main() {
  const argv = process.argv.slice(2);
  const at = (name: string): string | undefined => {
    const i = argv.indexOf(name);
    return i >= 0 && argv[i + 1] !== undefined ? argv[i + 1] : undefined;
  };
  const casePath = at('--case');
  const question = at('--question');
  const extract = argv.includes('--extract');
  const storePath = at('--store') ?? 'results/support-verdicts.jsonl';
  if (!casePath || !question) {
    console.error('usage: run-support-decide --case <case.json> --question "<question>" [--extract] [--store <jsonl>]');
    process.exitCode = 1;
    return;
  }

  const pack = loadPack('benchmarks/support/pack.json');
  const file = JSON.parse(readFileSync(casePath, 'utf8')) as CaseFile;
  const sources: Source[] = [ticketSource(file.ticket)];
  if (file.chat) sources.push(chatSource(file.chat));

  const p = plan(question);
  if (!p.ok) {
    console.log(JSON.stringify({ rejected: p.reason }, null, 1));
    process.exitCode = 1;
    return;
  }

  const inputs = claimsFromTicket(file.ticket);
  if (extract) {
    const useOllama = argv.includes('--ollama');
    const modelName = at('--model') ?? (useOllama ? 'llama3.1:8b' : 'deepseek/deepseek-chat');
    const client = useOllama
      ? new OpenRouterClient({ model: modelName, apiKey: 'ollama', baseUrl: at('--base-url') ?? 'http://127.0.0.1:11434/v1', temperature: 0 })
      : new OpenRouterClient({ model: modelName, apiKey: process.env.OPENROUTER_API_KEY ?? process.env.LLM_API_KEY ?? '', baseUrl: process.env.LLM_BASE_URL ?? 'https://openrouter.ai/api/v1', temperature: 0 });
    const extraction = await extractClaims(client, sources);
    console.error(`gate: ${extraction.admittedCount} admitted, ${extraction.rejectedCount} rejected${extraction.failedSpans ? `, ${extraction.failedSpans} SPANS FAILED (${extraction.lastError ?? 'unknown'})` : ''}`);
    for (const r of extraction.rejections) console.error(`  rejected ${r.predicate}: ${r.reason}`);
  }
  const report = admitAll(inputs, sources);

  const proof = decide(p.shape, p.params, [...pack.claims, ...report.admitted], {
    packId: pack.id,
    packVersion: pack.version,
    spansSearched: sources.reduce((n, s) => n + s.spans.length, 0),
  });

  const store = new VerdictStore(storePath);
  const record = store.append(proof, { id: pack.id, version: pack.version, contentHash: pack.contentHash });
  console.log(JSON.stringify({ seq: record.seq, decisionHash: record.decisionHash, proof }, null, 1));
}

if (process.argv[1]?.endsWith('run-support-decide.js')) {
  main().catch((error: unknown) => {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
  });
}
