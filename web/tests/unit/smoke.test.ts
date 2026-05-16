// web/tests/unit/smoke.test.ts
import { describe, it, expect } from 'vitest';

describe('vitest smoke', () => {
  it('arithmetic still works', () => {
    expect(1 + 1).toBe(2);
  });
});
