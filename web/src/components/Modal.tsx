import * as Dialog from '@radix-ui/react-dialog';
import type { ReactNode } from 'react';

interface PrimaryAction {
  label: string;
  onClick: () => void;
  disabled?: boolean;
  destructive?: boolean;
}

interface ModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  eyebrow?: string;
  children: ReactNode;
  preview?: ReactNode;
  primaryAction?: PrimaryAction;
  cancelLabel?: string;
}

export function Modal({
  open,
  onOpenChange,
  title,
  eyebrow,
  children,
  preview,
  primaryAction,
  cancelLabel = 'Cancel',
}: ModalProps) {
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(21,17,10,0.18)',
            backdropFilter: 'blur(2px)',
            animation: 'asr-modal-fade-in 180ms cubic-bezier(0.16, 1, 0.3, 1)',
            zIndex: 999,
          }}
        />
        <Dialog.Content
          style={{
            position: 'fixed',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            minWidth: 480,
            maxWidth: 720,
            maxHeight: '85vh',
            display: 'flex',
            flexDirection: 'column',
            background: 'var(--bg-elev)',
            border: '1px solid var(--hair-strong)',
            borderRadius: 0,
            boxShadow: 'var(--e-3)',
            animation: 'asr-modal-scale-in 180ms cubic-bezier(0.16, 1, 0.3, 1)',
            zIndex: 1000,
          }}
        >
          <header
            style={{
              height: 52,
              padding: '0 20px',
              display: 'flex',
              alignItems: 'center',
              borderBottom: '1px solid var(--hair)',
            }}
          >
            <div style={{ flex: 1, minWidth: 0 }}>
              {eyebrow && (
                <div
                  style={{
                    fontSize: 10,
                    color: 'var(--ink-3)',
                    letterSpacing: '0.14em',
                    textTransform: 'uppercase',
                    marginBottom: 2,
                  }}
                >
                  {eyebrow}
                </div>
              )}
              <Dialog.Title
                style={{
                  fontSize: 16,
                  fontWeight: 600,
                  color: 'var(--ink-1)',
                  margin: 0,
                  fontFamily: 'var(--ff-sans)',
                }}
              >
                {title}
              </Dialog.Title>
            </div>
            <Dialog.Close
              aria-label="Close"
              style={{
                width: 28,
                height: 28,
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: 'transparent',
                border: 'none',
                color: 'var(--ink-3)',
                cursor: 'pointer',
                fontSize: 18,
                lineHeight: 1,
              }}
            >
              ×
            </Dialog.Close>
          </header>
          <div style={{ flex: 1, overflow: 'auto', padding: '20px' }}>{children}</div>
          {preview && (
            <div
              style={{
                padding: '12px 20px',
                background: 'var(--bg-subtle)',
                borderTop: '1px solid var(--hair)',
              }}
            >
              {preview}
            </div>
          )}
          <footer
            style={{
              height: 52,
              padding: '0 20px',
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              borderTop: '1px solid var(--hair)',
            }}
          >
            <span
              style={{
                fontFamily: 'var(--ff-mono)',
                fontSize: 10,
                color: 'var(--ink-4)',
              }}
            >
              Esc to close
            </span>
            <span style={{ flex: 1 }} />
            <Dialog.Close asChild>
              <button
                type="button"
                style={{
                  height: 28,
                  padding: '0 14px',
                  fontFamily: 'var(--ff-sans)',
                  fontSize: 'var(--t-body)',
                  color: 'var(--ink-1)',
                  background: 'var(--bg-elev)',
                  border: '1px solid var(--hair-strong)',
                  borderRadius: 0,
                  cursor: 'pointer',
                }}
              >
                {cancelLabel}
              </button>
            </Dialog.Close>
            {primaryAction && (
              <button
                type="button"
                onClick={primaryAction.onClick}
                disabled={primaryAction.disabled}
                data-destructive={primaryAction.destructive ? 'true' : undefined}
                style={{
                  height: 28,
                  padding: '0 14px',
                  fontFamily: 'var(--ff-sans)',
                  fontSize: 'var(--t-body)',
                  fontWeight: 500,
                  color: primaryAction.destructive ? 'var(--bg-elev)' : 'var(--bg-elev)',
                  background: primaryAction.destructive ? 'var(--danger)' : 'var(--ink-1)',
                  border: `1px solid ${primaryAction.destructive ? 'var(--danger)' : 'var(--ink-1)'}`,
                  borderRadius: 0,
                  cursor: primaryAction.disabled ? 'not-allowed' : 'pointer',
                  opacity: primaryAction.disabled ? 0.5 : 1,
                }}
              >
                {primaryAction.label}
              </button>
            )}
          </footer>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
