// Usage: node scripts/compare-retrieved-sessions.mjs <run-a.json> <run-b.json>
// Exits 1 and lists the question ids whose retrieved session ids differ. The paired proof that
// --memory-system builtin:remembero-raw reproduces the harness's own retrieval exactly.
import { readFileSync } from 'node:fs';

const [a, b] = process.argv.slice(2);
if (a === undefined || b === undefined) {
  console.error('usage: compare-retrieved-sessions.mjs <run-a.json> <run-b.json>');
  process.exit(2);
}
const load = (path) =>
  new Map(
    JSON.parse(readFileSync(path, 'utf8')).observations.map((observation) => [
      observation.questionId,
      (observation.retrievedSessionIds ?? []).join(','),
    ]),
  );
const left = load(a);
const right = load(b);
const differing = [...left.entries()].filter(([id, ids]) => right.get(id) !== ids);
console.log(`${left.size} questions; ${differing.length} differ`);
for (const [id, ids] of differing.slice(0, 20)) {
  console.log(`  ${id}\n    a: ${ids}\n    b: ${right.get(id) ?? '(absent)'}`);
}
process.exit(differing.length === 0 ? 0 : 1);
