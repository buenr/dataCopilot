import { expect, test } from '@playwright/test';
import {
  deleteSessionAfterTest,
  expectPdfInCanvas,
  expectWebappInCanvas,
  openWorkbench,
  sendPrompt,
  uploadSampleDataset,
  waitForIdle,
} from './support';

// Live-provider coverage: the same user flow, but driven by the real OpenAI
// model instead of the deterministic mock. Only runs when E2E_LIVE=1 is set
// (the matching backend and dev server are only started then), because it
// consumes API credits and takes minutes per turn.
//
// The order matters: the mock suite covers dashboard-then-report, so this
// spec asks for the PDF report first and the dashboard second, with a full
// turn boundary in between — the direction the recovery gates and canvas
// tab-switching are least exercised in.

// A real model turn (multi-round tool calls plus artifact generation) takes
// minutes; give the whole test generous headroom.
const TURN_TIMEOUT = 480_000;

test.setTimeout(15 * 60_000);

test.afterEach(async ({ page, request }) => {
  await deleteSessionAfterTest(page, request);
});

test('openai: a PDF report followed by a dashboard both reach the canvas', async ({ page }) => {
  await openWorkbench(page);
  await uploadSampleDataset(page);

  // Turn 1: the report first.
  await sendPrompt(page, 'Create a polished PDF report of the key attrition drivers.');
  await expectPdfInCanvas(page, TURN_TIMEOUT);
  await waitForIdle(page, TURN_TIMEOUT);

  // Turn 2: the dashboard afterwards, while the PDF sits in the canvas.
  await sendPrompt(page, 'Now build an interactive dashboard of the attrition drivers.');
  await expectWebappInCanvas(page, /attrition|1,470/i, TURN_TIMEOUT);
  await waitForIdle(page, TURN_TIMEOUT);

  // The turn-1 PDF is still listed and renders when selected again.
  const selector = page.getByLabel('Select artifact');
  const pdfLabel = await selector.locator('option', { hasText: '.pdf' }).first().textContent();
  expect(pdfLabel).toBeTruthy();
  await selector.selectOption({ label: pdfLabel!.trim() });
  await expectPdfInCanvas(page);
});
