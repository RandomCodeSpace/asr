import { useState, useEffect } from 'react';
import type { CSSProperties, ReactNode } from 'react';
import { Modal } from '@/components/Modal';

interface ConfirmModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  eyebrow?: string;
  body: ReactNode;
  confirmLabel?: string;
  destructive?: boolean;
  onConfirm: () => void | Promise<void>;
}

const bodyStyle: CSSProperties = {
  fontFamily: 'var(--ff-sans)',
  fontSize: 13,
  color: 'var(--ink-1)',
  lineHeight: 1.55,
};

const errorBox: CSSProperties = {
  marginTop: 12,
  padding: '8px 10px',
  fontFamily: 'var(--ff-mono)',
  fontSize: 11,
  color: 'var(--danger)',
  background: 'var(--danger-bg)',
  border: '1px solid var(--danger)',
};

export function ConfirmModal({
  open,
  onOpenChange,
  title,
  eyebrow,
  body,
  confirmLabel = 'Confirm',
  destructive = false,
  onConfirm,
}: ConfirmModalProps) {
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) {
      setSubmitting(false);
      setError(null);
    }
  }, [open]);

  async function handleConfirm() {
    if (submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      await onConfirm();
      onOpenChange(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setSubmitting(false);
    }
  }

  return (
    <Modal
      open={open}
      onOpenChange={onOpenChange}
      eyebrow={eyebrow ?? (destructive ? 'CONFIRM (DESTRUCTIVE)' : 'CONFIRM')}
      title={title}
      primaryAction={{
        label: submitting ? `${confirmLabel}…` : confirmLabel,
        onClick: () => { void handleConfirm(); },
        disabled: submitting,
        destructive,
      }}
    >
      <div style={bodyStyle}>{body}</div>
      {error && <div role="alert" style={errorBox}>{error}</div>}
    </Modal>
  );
}
