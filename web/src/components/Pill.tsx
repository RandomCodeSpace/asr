import type { CSSProperties, ReactNode } from 'react';

type Kind = 'running' | 'paused' | 'error' | 'resolved' | 'neutral';

interface PillProps {
  kind: Kind;
  children: ReactNode;
}

const baseStyle: CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: 4,
  padding: '2px 8px',
  fontFamily: 'var(--ff-sans)',
  fontSize: 'var(--t-micro)',
  fontWeight: 500,
  letterSpacing: '0.08em',
  textTransform: 'uppercase',
  border: '1px solid var(--hair)',
  borderRadius: 0,
  whiteSpace: 'nowrap',
};

const kindStyles: Record<Kind, CSSProperties> = {
  running: { color: 'var(--acc)', borderColor: 'var(--acc-mid)', background: 'var(--acc-soft)' },
  paused: { color: 'var(--warn)', borderColor: 'var(--warn)', background: 'var(--warn-bg)' },
  error: { color: 'var(--danger)', borderColor: 'var(--danger)', background: 'var(--danger-bg)' },
  resolved: { color: 'var(--good)', borderColor: 'var(--good)', background: 'var(--good-bg)' },
  neutral: { color: 'var(--ink-3)', borderColor: 'var(--hair-strong)', background: 'var(--bg-subtle)' },
};

export function Pill({ kind, children }: PillProps) {
  return (
    <span data-kind={kind} style={{ ...baseStyle, ...kindStyles[kind] }}>
      {children}
    </span>
  );
}
