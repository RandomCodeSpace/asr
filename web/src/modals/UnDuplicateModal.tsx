import { useState, useEffect } from 'react';
import type { CSSProperties } from 'react';
import { Modal } from '@/components/Modal';
import { apiFetch, ApiClientError } from '@/api/client';

interface UnDuplicateModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  sessionId: string;
  parentSessionId: string | null;
  retractedBy?: string;
  onSuccess: () => void;
}

const labelStyle: CSSProperties = {
  display: 'block',
  fontFamily: 'var(--ff-mono)',
  fontSize: 10,
  color: 'var(--ink-3)',
  letterSpacing: '0.14em',
  textTransform: 'uppercase',
  marginBottom: 4,
};

const fieldWrap: CSSProperties = { marginBottom: 16 };

const inputStyle: CSSProperties = {
  width: '100%',
  height: 'auto',
  minHeight: 72,
  padding: '8px 10px',
  fontFamily: 'var(--ff-sans)',
  fontSize: 13,
  color: 'var(--ink-1)',
  background: 'var(--bg-elev)',
  border: '1px solid var(--hair)',
  borderRadius: 0,
  outline: 'none',
  boxSizing: 'border-box',
  resize: 'vertical',
  lineHeight: 1.5,
};

export function UnDuplicateModal({
  open,
  onOpenChange,
  sessionId,
  parentSessionId,
  retractedBy = 'operator',
  onSuccess,
}: UnDuplicateModalProps) {
  const [note, setNote] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) {
      setNote('');
      setError(null);
      setSubmitting(false);
    }
  }, [open]);

  async function handleSubmit() {
    if (submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      await apiFetch(`/sessions/${sessionId}/un-duplicate`, {
        method: 'POST',
        json: {
          retracted_by: retractedBy,
          note: note.trim() || null,
        },
      });
      onSuccess();
      onOpenChange(false);
    } catch (e) {
      setError(e instanceof ApiClientError ? `${e.code}: ${e.message}` : String(e));
      setSubmitting(false);
    }
  }

  return (
    <Modal
      open={open}
      onOpenChange={onOpenChange}
      eyebrow="UN-DUPLICATE SESSION"
      title="Retract the duplicate flag?"
      primaryAction={{
        label: submitting ? 'Retracting…' : 'Un-duplicate',
        onClick: () => { void handleSubmit(); },
        disabled: submitting,
      }}
    >
      <p style={{ marginTop: 0, fontSize: 13, lineHeight: 1.55, color: 'var(--ink-1)' }}>
        This will flip <code>{sessionId}</code>'s status back to a runnable
        state and clear the dedup link to{' '}
        <code>{parentSessionId ?? '—'}</code>. An audit row is written in
        the same transaction.
      </p>
      <div style={fieldWrap}>
        <label style={labelStyle} htmlFor="ud-note">Note (optional)</label>
        <textarea
          id="ud-note"
          placeholder="Why was this incorrectly flagged as a duplicate?"
          value={note}
          onChange={(e) => setNote(e.target.value)}
          style={inputStyle}
          disabled={submitting}
        />
      </div>
      {error && (
        <div
          role="alert"
          style={{
            marginTop: 12,
            padding: '8px 10px',
            fontFamily: 'var(--ff-mono)',
            fontSize: 11,
            color: 'var(--danger)',
            background: 'var(--danger-bg)',
            border: '1px solid var(--danger)',
          }}
        >
          {error}
        </div>
      )}
    </Modal>
  );
}
