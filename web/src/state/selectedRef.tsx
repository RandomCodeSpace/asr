import { createContext, useContext, useState, useCallback } from 'react';
import type { ReactNode } from 'react';

export type SelectedKind = 'agent' | 'tool_call' | 'message' | null;

export interface SelectedRef {
  kind: SelectedKind;
  id?: string;
}

const emptyRef: SelectedRef = { kind: null };

const SelectedRefContext = createContext<SelectedRef | undefined>(undefined);
const SetSelectedRefContext = createContext<((ref: SelectedRef) => void) | undefined>(undefined);

export function SelectedRefProvider({ children }: { children: ReactNode }) {
  const [ref, setRef] = useState<SelectedRef>(emptyRef);
  const set = useCallback((next: SelectedRef) => setRef(next), []);
  return (
    <SelectedRefContext.Provider value={ref}>
      <SetSelectedRefContext.Provider value={set}>
        {children}
      </SetSelectedRefContext.Provider>
    </SelectedRefContext.Provider>
  );
}

export function useSelected(): SelectedRef {
  const ctx = useContext(SelectedRefContext);
  if (ctx === undefined) {
    throw new Error('useSelected must be used within a SelectedRefProvider');
  }
  return ctx;
}

export function useSetSelected(): (ref: SelectedRef) => void {
  const ctx = useContext(SetSelectedRefContext);
  if (ctx === undefined) {
    throw new Error('useSetSelected must be used within a SelectedRefProvider');
  }
  return ctx;
}
