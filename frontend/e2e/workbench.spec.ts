import { expect, test, type Page } from '@playwright/test';
import {
  deleteSessionAfterTest,
  expectPdfInCanvas,
  expectWebappInCanvas,
  openWorkbench,
  sendPrompt,
  uploadSampleDataset,
} from './support';

// These tests drive the workbench exactly as a user would: open the app,
// upload the sample dataset, ask for a dashboard and a PDF report, and check
// what lands in the canvas. The backend runs the deterministic mock LLM
// provider (see playwright.config.ts), so the flow is offline and repeatable,
// while sessions still execute in real Docker sandboxes.

const DONE_TEXT = 'Done. The requested output is ready in the canvas.';

async function ask(page: Page, prompt: string) {
  await sendPrompt(page, prompt);
  await expect(page.getByText(DONE_TEXT).last()).toBeVisible();
}

async function expectMockDashboardInCanvas(page: Page) {
  await expectWebappInCanvas(page, /1,470 rows analyzed/);
  const frame = page.frameLocator('iframe[title="Generated web application"]');
  await expect(frame.getByRole('heading', { name: 'Data Copilot Dashboard' })).toBeVisible();
}

test.afterEach(async ({ page, request }) => {
  await deleteSessionAfterTest(page, request);
});

test('a user can upload data and receive a live dashboard and a PDF report', async ({ page }) => {
  await openWorkbench(page);
  await uploadSampleDataset(page);

  await ask(page, 'Build a dashboard of the attrition data.');
  await expectMockDashboardInCanvas(page);

  await ask(page, 'Create a PDF report of the findings.');
  // Publishing the PDF selects it automatically, so the canvas switches to
  // the document tab without any further interaction.
  await expectPdfInCanvas(page);

  // Both artifacts stay available from the canvas selector.
  await page.getByLabel('Select artifact').selectOption({ label: 'dashboard.html' });
  await expectMockDashboardInCanvas(page);
});

test('a browser refresh restores the chat and canvas artifacts', async ({ page }) => {
  await openWorkbench(page);
  await uploadSampleDataset(page);
  await ask(page, 'Build a dashboard of the attrition data.');
  await expectMockDashboardInCanvas(page);

  await page.reload();

  // The conversation and the canvas come back from the session trajectory
  // without re-running the turn; the restored webapp keeps its port so the
  // preview streams from the still-running sandbox.
  await expect(page.getByText('Session active')).toBeVisible();
  await expect(page.getByText('Build a dashboard of the attrition data.')).toBeVisible();
  await expect(page.getByText(DONE_TEXT)).toBeVisible();
  await expectMockDashboardInCanvas(page);
});
