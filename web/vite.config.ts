import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath } from 'node:url';

export default defineConfig({
  root: fileURLToPath(new URL('./', import.meta.url)),
  plugins: [react()],
  build: {
    outDir: '../dist/web-client',
    emptyOutDir: true,
  },
  server: {
    port: 4173,
    host: '127.0.0.1',
  },
});
