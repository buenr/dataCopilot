import { expect, test, type Page } from '@playwright/test';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// These tests drive the workbench exactly as a user would: open the app,
// upload the sample dataset, ask for a dashboard and a PDF report, and check
// what lands in the canvas. The backend runs the deterministic mock LLM
// provider (see playwright.config.ts), so the flow is offline and repeatable,
// while sessions still execute in real Docker sandboxes.

const SAMPLE_CSV = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  '../../sample_data/WA_Fn-UseC_-HR-Employee-Attrition.csv',
);
const DONE_TEXT = 'Done. The requested output is ready in the canvas.';
const DASHBOARD_HEADING = 'Data Copilot Dashboard';

async function openWorkbench(page: Page) {
  await page.goto('/');
  // The header flips to "Session active" once the sandbox container is up
  // and the websocket handshake has completed.
  await expect(page.getByText('Session active')).toBeVisible();
}

async function uploadSampleDataset(page: Page) {
  await page.locator('#dataset-upload').setInputFiles(SAMPLE_CSV);
  await expect(page.getByText('WA_Fn-UseC_-HR-Employee-Attrition.csv').first()).toBeVisible();
}

async function ask(page: Page, prompt: string) {
  await page.getByLabel('Message Data Copilot').fill(prompt);
  await page.getByLabel('Send message').click();
  await expect(page.getByText(DONE_TEXT).last()).toBeVisible();
}

async function expectDashboardInCanvas(page: Page) {
  const sessionId = await page.evaluate(() => sessionStorage.getItem('datacopilot-session-id'));
  // The canvas iframe mounts as soon as the artifact is published, but the
  // freshly started in-sandbox web server may need a moment before it accepts
  // connections. Wait for the preview endpoint directly, then reload the
  // frame once if it caught the server mid-startup.
  await expect(async () => {
    const response = await page.request.get(`/api/sessions/${sessionId}/preview/8501/dashboard.html`);
    expect(response.ok()).toBeTruthy();
  }).toPass({ timeout: 60_000 });

  const frame = page.frameLocator('iframe[title="Generated web application"]');
  const heading = frame.getByRole('heading', { name: DASHBOARD_HEADING });
  try {
    await expect(heading).toBeVisible({ timeout: 5_000 });
  } catch {
    await page.getByLabel('Refresh artifact').click();
    await expect(heading).toBeVisible();
  }
  // The dashboard must carry real dataset values, not placeholder chrome —
  // the same bar the backend's canvas quality gate enforces.
  await expect(frame.getByText('1,470 rows analyzed')).toBeVisible();
}

test.afterEach(async ({ page, request }) => {
  // Deleting the session tears down its sandbox container right away instead
  // of leaving it for the idle reaper.
  const sessionId = await page
    .evaluate(() => sessionStorage.getItem('datacopilot-session-id'))
    .catch(() => null);
  if (sessionId) await request.delete(`/api/sessions/${sessionId}`).catch(() => undefined);
});

test('a user can upload data and receive a live dashboard and a PDF report', async ({ page }) => {
  await openWorkbench(page);
  await uploadSampleDataset(page);

  await ask(page, 'Build a dashboard of the attrition data.');
  await expectDashboardInCanvas(page);

  await ask(page, 'Create a PDF report of the findings.');
  // Publishing the PDF selects it automatically, so the canvas switches to
  // the document tab without any further interaction.
  const pdfFrame = page.locator('iframe[title="Generated PDF document"]');
  await expect(pdfFrame).toBeVisible();
  await expect(pdfFrame).toHaveAttribute('src', /report\.pdf$/);

  // Both artifacts stay available from the canvas selector.
  await page.getByLabel('Select artifact').selectOption({ label: 'dashboard.html' });
  await expectDashboardInCanvas(page);
});

test('a browser refresh restores the chat and canvas artifacts', async ({ page }) => {
  await openWorkbench(page);
  await uploadSampleDataset(page);
  await ask(page, 'Build a dashboard of the attrition data.');
  await expectDashboardInCanvas(page);

  await page.reload();

  // The conversation and the canvas come back from the session trajectory
  // without re-running the turn; the restored webapp keeps its port so the
  // preview streams from the still-running sandbox.
  await expect(page.getByText('Session active')).toBeVisible();
  await expect(page.getByText('Build a dashboard of the attrition data.')).toBeVisible();
  await expect(page.getByText(DONE_TEXT)).toBeVisible();
  await expectDashboardInCanvas(page);
});
