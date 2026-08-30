import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['tests/**/*.test.ts'],
    // Many tests spawn `node dist/cli.js` subprocesses or run scale-bounded native
    // queries; the vitest default of 5s sits permanently on the edge under load.
    testTimeout: 60_000,
  },
});
