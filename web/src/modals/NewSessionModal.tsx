import { useState, useEffect } from 'react';
import type { CSSProperties } from 'react';
import { Modal } from '@/components/Modal';
import { apiFetch, ApiClientError } from '@/api/client';
import type { SessionStartBody } from '@/api/types';

interface NewSessionModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  environments: string[];
  onCreated: (sid: string) => void;
}

interface StartResponse {
  session_id: string;
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
  fontFamily: 'var(--ff-sans)',
  lineHeight: 1.5,
};

const selectStyle: CSSProperties = {
  ...inputStyle,
  height: 32,
  appearance: 'auto',
};

export function NewSessionModal({ open, onOpenChange, environments, onCreated }: NewSessionModalProps) {
  const [query, setQuery] = useState('');
  const [environment, setEnvironment] = useState(environments[0] ?? 'dev');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) {
      setQuery('');
      setError(null);
      setSubmitting(false);
      setEnvironment(environments[0] ?? 'dev');
    }
  }, [open, environments]);

  const canSubmit = query.trim().length > 0 && !submitting;

  async function handleSubmit() {
    if (!canSubmit) return;
    setSubmitting(true);
    setError(null);
    try {
      const res = await apiFetch<StartResponse>('/sessions', {
        method: 'POST',
        json: {
          query: query.trim(),
          environment,
          submitter: { id: 'operator' },
        } satisfies SessionStartBody,
      });
      onCreated(res.session_id);
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
      eyebrow="NEW SESSION"
      title="Start a new session"
      primaryAction={{
        label: submitting ? 'Starting…' : 'Create session',
        onClick: () => { void handleSubmit(); },
        disabled: !canSubmit,
      }}
    >
      <div style={fieldWrap}>
        <label style={labelStyle} htmlFor="ns-query">Query</label>
        <textarea
          id="ns-query"
          autoFocus
          placeholder="Describe the incident or task in plain language…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          style={textareaStyle}
          disabled={submitting}
        />
      </div>
      <div style={fieldWrap}>
        <label style={labelStyle} htmlFor="ns-env">Environment</label>
        <select
          id="ns-env"
          value={environment}
          onChange={(e) => setEnvironment(e.target.value)}
          style={selectStyle}
          disabled={submitting}
        >
          {environments.map((env) => (
            <option key={env} value={env}>{env}</option>
          ))}
        </select>
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
