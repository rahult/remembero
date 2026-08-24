import { execFileSync } from 'node:child_process';

const raw = execFileSync('npm', ['pack', '--dry-run', '--json'], {
  cwd: process.cwd(),
  encoding: 'utf8',
  stdio: ['ignore', 'pipe', 'pipe'],
});
const report = JSON.parse(raw)[0];
if (report === undefined || !Array.isArray(report.files)) {
  throw new Error('npm pack did not return a file manifest');
}
const paths = report.files.map((file) => file.path);
const forbidden = paths.filter(
  (path) => path.endsWith('.pdf') || path.endsWith('.pyc') || path.includes('/__pycache__/')
);
if (forbidden.length > 0) {
  throw new Error(`package contains forbidden evaluation/runtime files: ${forbidden.join(', ')}`);
}
for (const required of [
  'THIRD_PARTY_NOTICES.md',
  'dist/web-client/documents/document-intelligence.memorg.json',
  'scripts/import-document-memorg.py',
]) {
  if (!paths.includes(required)) throw new Error(`package is missing ${required}`);
}
const maximumTarballBytes = 8 * 1024 * 1024;
if (report.size > maximumTarballBytes) {
  throw new Error(`package tarball is ${report.size} bytes; maximum is ${maximumTarballBytes}`);
}
process.stdout.write(
  `package policy: PASS · ${report.entryCount} files · ${report.size} byte tarball · no PDFs, pyc, or __pycache__\n`
);
