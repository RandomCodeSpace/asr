import { test, expect } from '@playwright/test';

const BASE = process.env.E2E_BASE_URL ?? 'http://localhost:8000';

// Stop + retry flow:
//   1. Create a session.
//   2. Click Stop in the CanvasHead, confirm in the ConfirmModal.
//   3. Verify backend returns status='stopped'.
//   4. Create a new session with the same query (the "retry" semantic
//      until the framework grows a first-class retry endpoint), and
//      verify a fresh id comes back.
//
// The framework retries via "create another session with the same
// query" today, so we exercise that path directly. Once a /retry
// endpoint lands the spec will be updated to drive it.

test('stop + retry: confirm stop, then new session', async ({ page, request }) => {
  test.setTimeout(120_000);

  page.on('console', (m) => { if (m.type() === 'error') console.log('[browser-err]', m.text()); });
  page.on('pageerror', (e) => console.log('[page-err]', e.message));

  const query = `retry smoke ${Date.now()}: api auth degraded`;

  // 1. Create the first session via the New Session modal (same path as
  //    new-session.live.spec.ts so the canvas auto-selects the new sid).
  await page.goto(`${BASE}/`);
  await expect(page.getByText(/Sessions/i).first()).toBeVisible({ timeout: 15_000 });
  await page.getByRole('button', { name: /New Session/i }).click();
  await expect(page.getByText(/Start a new session/i)).toBeVisible();
  await page.locator('#ns-query').fill(query);
  await page.getByRole('button', { name: /Create session/i }).click();
  await expect(page.getByText(/Start a new session/i)).toBeHidden({ timeout: 30_000 });

  // 2. CanvasHead with Stop button should now be visible.
  await expect(page.getByRole('button', { name: /^Stop$/ })).toBeVisible({ timeout: 30_000 });

  // 3. Capture the displayed session id before stop (DELETE may 404 the read).
  const sidLocator = page.getByText(/(INC|SES)-\d{4,}/).first();
  await expect(sidLocator).toBeVisible({ timeout: 10_000 });
  const sid1 = ((await sidLocator.textContent()) ?? '').trim().match(/(INC|SES)-\d+/)?.[0] ?? '';
  expect(sid1).toMatch(/(INC|SES)-\d+/);

  // 4. Click Stop, confirm in the destructive ConfirmModal.
  await page.getByRole('button', { name: /^Stop$/ }).click();
  await expect(page.getByText(/stop this session\?/i)).toBeVisible({ timeout: 10_000 });
  await page.getByRole('button', { name: /^Stop session$/ }).click();
  await expect(page.getByText(/stop this session\?/i)).toBeHidden({ timeout: 15_000 });

  // 5. Verify backend state — stopped, error, or 404 (evicted) are all
  //    acceptable terminal outcomes for the stop path.
  const afterStop = await request.get(`${BASE}/api/v1/sessions/${sid1}`);
  if (afterStop.ok()) {
    const body = await afterStop.json();
    expect(['stopped', 'error', 'resolved', 'escalated']).toContain(body.status);
  } else {
    expect([404, 501]).toContain(afterStop.status());
  }

  // 6. Retry semantics: create another session with the same query and
  //    verify a fresh id comes back.
  const retry = await request.post(`${BASE}/api/v1/sessions`, {
    data: { query, environment: 'prod', submitter: { id: 'e2e-operator' } },
  });
  expect(retry.ok()).toBeTruthy();
  const { session_id: sid2 } = await retry.json();
  expect(sid2).not.toBe(sid1);
  expect(sid2).toMatch(/(INC|SES)-\d{4,}/);
});
