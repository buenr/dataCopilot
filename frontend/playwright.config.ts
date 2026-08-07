import { defineConfig } from '@playwright/test';

// End-to-end tests against the real stack: FastAPI backends, real Docker
// session sandboxes, and Vite dev servers proxying to them. Ports differ from
// the development defaults so the suite can run alongside a dev session.
// Requires Docker and the data-copilot-sandbox image (`make build-sandbox`).
//
// Two projects:
// - mock (default): deterministic offline mock LLM provider, no API keys.
// - live (E2E_LIVE=1 only): the real OpenAI provider; consumes API credits.
const live = Boolean(process.env.E2E_LIVE);

export default defineConfig({
  testDir: './e2e',
  // Sandbox container startup plus a full agent turn can be slow on a cold
  // Docker cache, so give each test generous headroom.
  timeout: 180_000,
  expect: { timeout: 30_000 },
  workers: 1,
  retries: 0,
  reporter: 'list',
  use: {
    viewport: { width: 1600, height: 900 },
    trace: 'retain-on-failure',
  },
  projects: [
    {
      name: 'mock',
      testMatch: 'workbench.spec.ts',
      use: { baseURL: 'http://localhost:5174' },
    },
    ...(live
      ? [
          {
            name: 'live',
            testMatch: 'live-provider.spec.ts',
            use: { baseURL: 'http://localhost:5175' },
          },
        ]
      : []),
  ],
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
    ...(live
      ? [
          {
            command: 'uv run uvicorn app.main:app --app-dir backend --port 8101',
            cwd: '..',
            url: 'http://localhost:8101/health',
            // The API key comes from the repo-root .env (uvicorn's cwd).
            env: { LLM_PROVIDER: 'openai' },
            reuseExistingServer: false,
            timeout: 120_000,
          },
          {
            command: 'npm run dev -- --port 5175 --strictPort',
            url: 'http://localhost:5175',
            env: { VITE_BACKEND_URL: 'http://localhost:8101' },
            reuseExistingServer: false,
            timeout: 120_000,
          },
        ]
      : []),
  ],
});
