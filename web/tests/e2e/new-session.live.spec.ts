import { test, expect } from '@playwright/test';

const BASE = process.env.E2E_BASE_URL ?? 'http://localhost:5173';

test('shell renders + New Session creates a INC- id and opens canvas', async ({ page }) => {
  test.setTimeout(60_000);

  page.on('console', (msg) => {
    if (msg.type() === 'error') console.log('[browser-err]', msg.text());
  });
  page.on('pageerror', (err) => console.log('[page-err]', err.message));

  await page.goto(BASE);

  // Shell renders
  await expect(page.getByText(/Sessions/i).first()).toBeVisible();
  await expect(page.getByText(/All Systems Normal|Degraded|Critical/i)).toBeVisible();
  await expect(page.getByText(/Select a session/i)).toBeVisible();

  // Open the modal
  await page.getByRole('button', { name: /New Session/i }).click();
  await expect(page.getByText(/Start a new session/i)).toBeVisible();

  // Fill + submit
  const query = `e2e smoke ${Date.now()}: high p99 on payments-svc`;
  await page.locator('#ns-query').fill(query);
  await page.getByRole('button', { name: /Create session/i }).click();

  // Modal closes
  await expect(page.getByText(/Start a new session/i)).toBeHidden({ timeout: 30_000 });

  // Canvas shows the new session id (backend uses INC-YYYYMMDD-NNN form)
  const sidLocator = page.getByText(/INC-\d{4,}/).first();
  await expect(sidLocator).toBeVisible({ timeout: 30_000 });
  const sid = (await sidLocator.textContent()) ?? '';
  expect(sid).toMatch(/INC-\d{4,}/);

  // Canvas head renders meta row (env + turns counters)
  await expect(page.getByText(/ENV /)).toBeVisible();
  await expect(page.getByText(/TURNS /)).toBeVisible();
});
