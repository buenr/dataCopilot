import { expect, type Page } from '@playwright/test';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// Shared helpers for driving the workbench exactly as a user would.

export const SAMPLE_CSV = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  '../../sample_data/WA_Fn-UseC_-HR-Employee-Attrition.csv',
);

export async function openWorkbench(page: Page) {
  await page.goto('/');
  // The header flips to "Session active" once the sandbox container is up
  // and the websocket handshake has completed.
  await expect(page.getByText('Session active')).toBeVisible();
}

export async function uploadSampleDataset(page: Page) {
  await page.locator('#dataset-upload').setInputFiles(SAMPLE_CSV);
  await expect(page.getByText('WA_Fn-UseC_-HR-Employee-Attrition.csv').first()).toBeVisible();
}

export async function sendPrompt(page: Page, prompt: string) {
  await page.getByLabel('Message Data Copilot').fill(prompt);
  await page.getByLabel('Send message').click();
  // The composer swaps the send button for a stop button while a turn runs.
  await expect(page.getByLabel('Stop run')).toBeVisible({ timeout: 30_000 });
}

export async function waitForIdle(page: Page, timeout = 60_000) {
  await expect(page.getByLabel('Send message')).toBeVisible({ timeout });
}

// The canvas iframe mounts as soon as the artifact is published, but the
// freshly started in-sandbox web server may need a moment before it accepts
// connections. Wait for the preview endpoint directly, then reload the frame
// once if it caught the server mid-startup.
export async function expectWebappInCanvas(page: Page, textPattern: RegExp, timeout = 120_000) {
  const iframe = page.locator('iframe[title="Generated web application"]');
  await expect(iframe).toBeVisible({ timeout });
  const src = await iframe.getAttribute('src');
  expect(src).toMatch(/\/preview\/\d+\//);
  await expect(async () => {
    const response = await page.request.get(src!);
    expect(response.ok()).toBeTruthy();
  }).toPass({ timeout: 60_000 });

  const frame = page.frameLocator('iframe[title="Generated web application"]');
  try {
    await expect(frame.locator('body')).toContainText(textPattern, { timeout: 10_000 });
  } catch {
    await page.getByLabel('Refresh artifact').click();
    await expect(frame.locator('body')).toContainText(textPattern, { timeout: 30_000 });
  }
}

export async function expectPdfInCanvas(page: Page, timeout = 120_000) {
  const iframe = page.locator('iframe[title="Generated PDF document"]');
  await expect(iframe).toBeVisible({ timeout });
  const src = await iframe.getAttribute('src');
  expect(src).toMatch(/\.pdf(\?.*)?$/);
  const response = await page.request.get(src!);
  expect(response.ok()).toBeTruthy();
  expect(response.headers()['content-type']).toContain('application/pdf');
}

export async function deleteSessionAfterTest(page: Page, request: { delete: (url: string) => Promise<unknown> }) {
  // Deleting the session tears down its sandbox container right away instead
  // of leaving it for the idle reaper.
  const sessionId = await page
    .evaluate(() => sessionStorage.getItem('datacopilot-session-id'))
    .catch(() => null);
  if (sessionId) await request.delete(`/api/sessions/${sessionId}`).catch(() => undefined);
}
