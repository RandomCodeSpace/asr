import type { CSSProperties, ReactNode } from 'react';
import * as Dialog from '@radix-ui/react-dialog';

interface MobileSheetProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  children: ReactNode;
  testId?: string;
}

const overlay: CSSProperties = {
  position: 'fixed',
  inset: 0,
  background: 'rgba(21,17,10,0.30)',
  backdropFilter: 'blur(2px)',
  zIndex: 1000,
};

const content: CSSProperties = {
  position: 'fixed',
  left: 0,
  right: 0,
  bottom: 0,
  height: 'min(85vh, 720px)',
  background: 'var(--bg-page)',
  borderTop: '1px solid var(--hair-strong)',
  borderRadius: 0,
  zIndex: 1001,
  display: 'flex',
  flexDirection: 'column',
  boxShadow: 'var(--e-3)',
  animation: 'asr-sheet-slide-up 220ms cubic-bezier(0.16, 1, 0.3, 1)',
};

const handle: CSSProperties = {
  width: 44, height: 4,
  background: 'var(--ink-4)',
  margin: '8px auto 4px',
  opacity: 0.35,
};

const titleRow: CSSProperties = {
  height: 40, padding: '0 16px',
  display: 'flex', alignItems: 'center',
  borderBottom: '1px solid var(--hair)',
};

export function MobileSheet({ open, onOpenChange, title, children, testId }: MobileSheetProps) {
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay style={overlay} />
        <Dialog.Content style={content} data-mobile-sheet={testId ?? ''}>
          <div style={handle} aria-hidden />
          <div style={titleRow}>
            <Dialog.Title
              style={{
                margin: 0,
                fontSize: 10,
                fontFamily: 'var(--ff-mono)',
                letterSpacing: '0.14em',
                textTransform: 'uppercase',
                color: 'var(--ink-3)',
                flex: 1,
              }}
            >
              {title}
            </Dialog.Title>
            <Dialog.Close
              aria-label="Close"
              style={{
                width: 28, height: 28,
                display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                background: 'transparent', border: 'none', color: 'var(--ink-3)',
                cursor: 'pointer', fontSize: 18, lineHeight: 1,
              }}
            >×</Dialog.Close>
          </div>
          <div style={{ flex: 1, overflow: 'auto', minHeight: 0 }}>{children}</div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
