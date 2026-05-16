import { describe, it, expect } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { SelectedRefProvider, useSelected, useSetSelected } from '@/state/selectedRef';
import type { ReactNode } from 'react';

function wrapper({ children }: { children: ReactNode }) {
  return <SelectedRefProvider>{children}</SelectedRefProvider>;
}

describe('selectedRef', () => {
  it('defaults to {kind: null}', () => {
    const { result } = renderHook(() => useSelected(), { wrapper });
    expect(result.current).toEqual({ kind: null });
  });

  it('setSelected updates the value', () => {
    const { result } = renderHook(
      () => ({ selected: useSelected(), set: useSetSelected() }),
      { wrapper },
    );
    act(() => {
      result.current.set({ kind: 'agent', id: 'intake' });
    });
    expect(result.current.selected).toEqual({ kind: 'agent', id: 'intake' });
  });

  it('setSelected({kind: null}) clears the selection', () => {
    const { result } = renderHook(
      () => ({ selected: useSelected(), set: useSetSelected() }),
      { wrapper },
    );
    act(() => {
      result.current.set({ kind: 'tool_call', id: 'tool-1' });
    });
    expect(result.current.selected.kind).toBe('tool_call');
    act(() => {
      result.current.set({ kind: null });
    });
    expect(result.current.selected).toEqual({ kind: null });
  });

  it('throws when useSelected called outside Provider', () => {
    expect(() => renderHook(() => useSelected())).toThrow(/SelectedRefProvider/);
  });
});
