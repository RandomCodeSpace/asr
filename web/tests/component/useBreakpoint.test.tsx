import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useBreakpoint } from '@/state/useBreakpoint';

type Listener = (e: { matches: boolean }) => void;

interface MockMediaQueryList {
  matches: boolean;
  media: string;
  addEventListener: (event: 'change', listener: Listener) => void;
  removeEventListener: (event: 'change', listener: Listener) => void;
  fire: (matches: boolean) => void;
}

const originalMatchMedia = window.matchMedia;

function installMatchMedia(width: number) {
  const lists: MockMediaQueryList[] = [];
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    configurable: true,
    value: (query: string) => {
      const min = /\(min-width:\s*(\d+)px\)/.exec(query);
      const threshold = min ? Number(min[1]) : 0;
      const matches = width >= threshold;
      const listeners = new Set<Listener>();
      const list: MockMediaQueryList = {
        matches,
        media: query,
        addEventListener: (_e, l) => { listeners.add(l); },
        removeEventListener: (_e, l) => { listeners.delete(l); },
        fire: (m) => { list.matches = m; listeners.forEach((l) => l({ matches: m })); },
      };
      lists.push(list);
      return list as unknown as MediaQueryList;
    },
  });
  return {
    setWidth(newWidth: number) {
      width = newWidth;
      for (const list of lists) {
        const min = /\(min-width:\s*(\d+)px\)/.exec(list.media);
        const threshold = min ? Number(min[1]) : 0;
        list.fire(width >= threshold);
      }
    },
  };
}

describe('useBreakpoint', () => {
  beforeEach(() => { /* installed per-test */ });
  afterEach(() => {
    Object.defineProperty(window, 'matchMedia', { writable: true, configurable: true, value: originalMatchMedia });
  });

  it('returns "mobile" when width < 768', () => {
    installMatchMedia(500);
    const { result } = renderHook(() => useBreakpoint());
    expect(result.current).toBe('mobile');
  });

  it('returns "tablet" when 768 <= width < 1200', () => {
    installMatchMedia(900);
    const { result } = renderHook(() => useBreakpoint());
    expect(result.current).toBe('tablet');
  });

  it('returns "desktop" when width >= 1200', () => {
    installMatchMedia(1440);
    const { result } = renderHook(() => useBreakpoint());
    expect(result.current).toBe('desktop');
  });

  it('updates when matchMedia change events fire', () => {
    const mm = installMatchMedia(500);
    const { result } = renderHook(() => useBreakpoint());
    expect(result.current).toBe('mobile');
    act(() => { mm.setWidth(1000); });
    expect(result.current).toBe('tablet');
    act(() => { mm.setWidth(1500); });
    expect(result.current).toBe('desktop');
  });

  it('falls back to "desktop" when matchMedia is unavailable (SSR-safe)', () => {
    Object.defineProperty(window, 'matchMedia', { writable: true, configurable: true, value: undefined });
    const { result } = renderHook(() => useBreakpoint());
    expect(result.current).toBe('desktop');
  });
});
