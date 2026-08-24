#!/usr/bin/env node
import { loadEnv } from '../env.js';
import {
  evaluateUnlimitedOcrLive,
  formatUnlimitedOcrLiveHelp,
  maybeWriteUnlimitedOcrLiveOutput,
  parseUnlimitedOcrLiveCliArgs,
  renderUnlimitedOcrLiveOutput,
} from './unlimited-ocr-live.js';

loadEnv();
const options = parseUnlimitedOcrLiveCliArgs(process.argv.slice(2));
if (options.help) {
  process.stdout.write(`${formatUnlimitedOcrLiveHelp()}\n`);
  process.exit(0);
}

const hfToken = process.env.HF_TOKEN?.trim() || undefined;
const report = await evaluateUnlimitedOcrLive({ ...options, hfToken });
const output = renderUnlimitedOcrLiveOutput(report, options);

process.stdout.write(output);
if (report.aggregate.status === 'pass' || options.allowFailedOutput === true) {
  maybeWriteUnlimitedOcrLiveOutput(options.output, output);
} else if (options.output !== undefined) {
  process.stderr.write(
    `Refusing to overwrite ${options.output} with a failed live OCR run. ` +
      'Use --allow-failed-output to preserve an operational failure.\n'
  );
}

if (options.check && report.aggregate.status !== 'pass') {
  process.exitCode = 1;
}
