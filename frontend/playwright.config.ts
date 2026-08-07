import { defineConfig } from '@playwright/test';

// End-to-end tests against the real stack: a FastAPI backend driven by the
// deterministic mock LLM provider (no API keys needed), real Docker session
// sandboxes, and a Vite dev server proxying to it. Ports differ from the
// development defaults so the suite can run alongside a dev session.
// Requires Docker and the data-copilot-sandbox image (`make build-sandbox`).
export default defineConfig({
  testDir: './e2e',
  // Sandbox container startup plus a full agent turn can be slow on a cold
  // Docker cache, so give each test generous headroom.
  timeout: 180_000,
  expect: { timeout: 30_000 },
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: 'list',
  use: {
    baseURL: 'http://localhost:5174',
    viewport: { width: 1600, height: 900 },
    trace: 'retain-on-failure',
  },
  webServer: [
    {
      command: 'uv run uvicorn app.main:app --app-dir backend --port 8100',
      cwd: '..',
      url: 'http://localhost:8100/health',
      env: { LLM_PROVIDER: 'mock' },
      // Never reuse: reusing a server that was started without the mock
      // provider (or with a stale proxy target) silently tests the wrong
      // stack. The dedicated ports keep this from clashing with dev servers.
      reuseExistingServer: false,
      timeout: 120_000,
    },
    {
      command: 'npm run dev -- --port 5174 --strictPort',
      url: 'http://localhost:5174',
      env: { VITE_BACKEND_URL: 'http://localhost:8100' },
      reuseExistingServer: false,
      timeout: 120_000,
    },
  ],
});
