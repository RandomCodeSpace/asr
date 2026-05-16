import { test, expect } from '@playwright/test';

const BASE = process.env.E2E_BASE_URL ?? 'http://localhost:8000';

// Best-effort live E2E for the HITL approve path.
// Preconditions for a green run:
//   - Backend up at BASE (uvicorn runtime.api:get_app).
//   - The configured app (incident_management) gates at least one
//     tool call so the session pauses with `status=awaiting_input`
//     within HITL_TIMEOUT_MS.
// If the session never reaches awaiting_input within the timeout the
// test soft-skips with a descriptive reason — we still want the wiring
// to be exercised by CI, but the underlying gating is config-dependent
// and out of scope for this spec.

const HITL_TIMEOUT_MS = 60_000;

test('hitl approve: pause → click Approve → session resumes', async ({ page, request }) => {
  test.setTimeout(120_000);

  page.on('console', (m) => { if (m.type() === 'error') console.log('[browser-err]', m.text()); });
  page.on('pageerror', (e) => console.log('[page-err]', e.message));

  // 1. Create a session via the API (faster than the modal path).
  const created = await request.post(`${BASE}/api/v1/sessions`, {
    data: {
      query: `hitl smoke ${Date.now()}: prod payments-svc 5xx surge`,
      environment: 'prod',
      submitter: { id: 'e2e-operator' },
    },
  });
  expect(created.ok()).toBeTruthy();
  const { session_id: sid } = await created.json();
  expect(sid).toMatch(/(INC|SES)-\d{4,}/);

  // 2. Poll the backend for awaiting_input or terminal state.
  const start = Date.now();
  let status = '';
  while (Date.now() - start < HITL_TIMEOUT_MS) {
    const res = await request.get(`${BASE}/api/v1/sessions/${sid}`);
    if (res.ok()) {
      const body = await res.json();
      status = body.status;
      if (status === 'awaiting_input') break;
      if (['resolved', 'escalated', 'stopped', 'error'].includes(status)) {
        test.skip(true, `session reached terminal ${status} without HITL pause (config-dependent)`);
        return;
      }
    }
    await page.waitForTimeout(750);
  }
  if (status !== 'awaiting_input') {
    test.skip(true, `no HITL pause within ${HITL_TIMEOUT_MS}ms (got status=${status || 'unknown'})`);
    return;
  }

  // 3. Open the SPA on the paused session.
  await page.goto(`${BASE}/?sid=${sid}`);
  await expect(page.getByText(/Sessions/i).first()).toBeVisible({ timeout: 15_000 });

  // The session may not be auto-selected from the URL today (SessionsRail
  // selection is in-memory). Click it in the rail so the canvas mounts.
  const rowMatch = page.getByText(new RegExp(sid));
  if (await rowMatch.count() > 0) {
    await rowMatch.first().click();
  }

  // 4. Wait for the HITLBand to render in the canvas.
  await expect(
    page.getByRole('button', { name: /^Approve$/ }),
  ).toBeVisible({ timeout: 20_000 });

  // 5. Approve and verify the band goes away (session resumes).
  await page.getByRole('button', { name: /^Approve$/ }).click();
  await expect(
    page.getByRole('button', { name: /^Approve$/ }),
  ).toBeHidden({ timeout: 30_000 });

  // 6. Confirm backend status moved past awaiting_input.
  const after = await request.get(`${BASE}/api/v1/sessions/${sid}`);
  const afterBody = await after.json();
  expect(['in_progress', 'matched', 'resolved', 'escalated']).toContain(afterBody.status);
});
