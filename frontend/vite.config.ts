import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The e2e suite boots its own backend on a dedicated port and points the dev
// server's proxy at it (see playwright.config.ts).
const backendTarget = process.env.VITE_BACKEND_URL ?? 'http://localhost:8000';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: '0.0.0.0',
    // Same-origin proxy so the UI works out of the box (no VITE_API_URL needed)
    // and artifact downloads stay same-origin: browsers ignore the download
    // attribute on cross-origin links and would navigate away instead.
    proxy: {
      '/api': { target: backendTarget, changeOrigin: true },
      '/ws': { target: backendTarget, ws: true },
    },
  },
});
