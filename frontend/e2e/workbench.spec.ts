import { readFile } from 'node:fs/promises';

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

test('canvas and chat panels survive minimize and expand cycles', async ({ page }) => {
  // Regression: conditionally rendered panels used to crash react-resizable-panels
  // ("Previous layout not found for panel index -1") and blank the whole app.
  const pageErrors: string[] = [];
  page.on('pageerror', (error) => pageErrors.push(String(error)));
  await openWorkbench(page);

  await page.getByRole('button', { name: 'Minimize canvas panel' }).click();
  await page.getByRole('button', { name: 'Expand canvas panel' }).click();
  await expect(page.getByRole('heading', { name: 'Your canvas is ready' })).toBeVisible();

  await page.getByRole('button', { name: 'Minimize chat panel' }).click();
  await page.getByRole('button', { name: 'Expand chat panel' }).click();

  await page.getByRole('button', { name: 'Enter full screen' }).click();
  await page.getByRole('button', { name: 'Exit full screen' }).click();

  // Both panels are back and the app never unmounted.
  await expect(page.getByRole('button', { name: 'Minimize canvas panel' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Minimize chat panel' })).toBeVisible();
  expect(pageErrors).toEqual([]);
});

test('a user can export a spreadsheet of the findings and download it', async ({ page }) => {
  await openWorkbench(page);
  await uploadSampleDataset(page);
  await ask(page, 'Export the attrition summary to Excel.');

  // The data artifact auto-selects onto the Data tab with a download card.
  const downloadButton = page.getByRole('button', { name: 'Download summary.xlsx' });
  await expect(downloadButton).toBeVisible();
  const [download] = await Promise.all([page.waitForEvent('download'), downloadButton.click()]);
  expect(download.suggestedFilename()).toBe('summary.xlsx');
});

test('a user can export the chat transcript as Markdown', async ({ page }) => {
  await openWorkbench(page);
  await uploadSampleDataset(page);
  await ask(page, 'Export the attrition summary to Excel.');

  const [download] = await Promise.all([
    page.waitForEvent('download'),
    page.getByRole('button', { name: 'Export chat transcript' }).click(),
  ]);
  expect(download.suggestedFilename()).toMatch(/^data-copilot-chat-\d{4}-\d{2}-\d{2}\.md$/);
  const transcript = await readFile(await download.path(), 'utf-8');
  expect(transcript).toContain('## You');
  expect(transcript).toContain('Export the attrition summary to Excel.');
  expect(transcript).toContain('## Copilot');
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
