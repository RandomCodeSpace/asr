import { useState, useEffect } from 'react';
import type { CSSProperties } from 'react';
import { Modal } from '@/components/Modal';
import { apiFetch, ApiClientError } from '@/api/client';

interface ApproveRationaleModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  sessionId: string;
  toolCallId: string;
  approver?: string;
  templates?: string[];
  onApproved: () => void;
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
  height: 32,
  padding: '0 10px',
  fontFamily: 'var(--ff-sans)',
  fontSize: 13,
  color: 'var(--ink-1)',
  background: 'var(--bg-elev)',
  border: '1px solid var(--hair)',
  borderRadius: 0,
  outline: 'none',
  boxSizing: 'border-box',
};

const textareaStyle: CSSProperties = {
  ...inputStyle,
  height: 'auto',
  minHeight: 96,
  padding: '8px 10px',
  resize: 'vertical',
  lineHeight: 1.5,
};

const templateChip: CSSProperties = {
  padding: '4px 8px',
  fontFamily: 'var(--ff-mono)',
  fontSize: 11,
  color: 'var(--ink-2)',
  background: 'var(--bg-subtle)',
  border: '1px solid var(--hair)',
  borderRadius: 0,
  cursor: 'pointer',
};

export function ApproveRationaleModal({
  open,
  onOpenChange,
  sessionId,
  toolCallId,
  approver = 'operator',
  templates = [],
  onApproved,
}: ApproveRationaleModalProps) {
  const [rationale, setRationale] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) {
      setRationale('');
      setError(null);
      setSubmitting(false);
    }
  }, [open]);

  const canSubmit = rationale.trim().length > 0 && !submitting;

  async function handleSubmit() {
    if (!canSubmit) return;
    setSubmitting(true);
    setError(null);
    try {
      await apiFetch(`/sessions/${sessionId}/approvals/${toolCallId}`, {
        method: 'POST',
        json: { decision: 'approve', approver, rationale: rationale.trim() },
      });
      onApproved();
      onOpenChange(false);
    } catch (e) {
      if (e instanceof ApiClientError) {
        setError(`${e.code}: ${e.message}`);
      } else {
        setError(String(e));
      }
      setSubmitting(false);
    }
  }

  return (
    <Modal
      open={open}
      onOpenChange={onOpenChange}
      eyebrow="APPROVE WITH RATIONALE"
      title="Approve this tool call"
      primaryAction={{
        label: submitting ? 'Approving…' : 'Approve',
        onClick: () => { void handleSubmit(); },
        disabled: !canSubmit,
      }}
    >
      <div style={fieldWrap}>
        <label style={labelStyle} htmlFor="ar-rationale">Rationale</label>
        <textarea
          id="ar-rationale"
          autoFocus
          placeholder="Why are you approving this call?"
          value={rationale}
          onChange={(e) => setRationale(e.target.value)}
          style={textareaStyle}
          disabled={submitting}
        />
      </div>
      {templates.length > 0 && (
        <div style={{ ...fieldWrap, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          {templates.map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => setRationale(t)}
              style={templateChip}
              disabled={submitting}
            >
              {t}
            </button>
          ))}
        </div>
      )}
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
