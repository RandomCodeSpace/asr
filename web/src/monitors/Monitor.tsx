import { useState } from 'react';
import type { CSSProperties, ReactNode } from 'react';

interface MonitorProps {
  title: string;
  count?: number;
  pinned: boolean;
  defaultOpen?: boolean;
  children: ReactNode;
}

const wrap: CSSProperties = {
  borderBottom: '1px solid var(--hair)',
};

const header: CSSProperties = {
  height: 32,
  display: 'flex',
  alignItems: 'center',
  gap: 8,
  padding: '0 12px',
  fontFamily: 'var(--ff-sans)',
  fontSize: 11,
  color: 'var(--ink-2)',
  cursor: 'pointer',
  userSelect: 'none',
  background: 'var(--bg-page)',
};

export function Monitor({ title, count, pinned, defaultOpen = false, children }: MonitorProps) {
  const [open, setOpen] = useState<boolean>(pinned || defaultOpen);
  return (
    <section data-monitor={title} data-pinned={pinned} data-collapsed={!open} style={wrap}>
      <header
        onClick={() => setOpen((v) => !v)}
        style={header}
      >
        <span aria-hidden style={{ fontSize: 9, color: 'var(--ink-3)', width: 8 }}>
          {open ? '▾' : '▸'}
        </span>
        {pinned && (
          <span
            data-pinned-dot
            aria-hidden
            style={{
              display: 'inline-block',
              width: 5,
              height: 5,
              borderRadius: '50%',
              background: 'var(--acc)',
            }}
          />
        )}
        <span style={{ fontWeight: 500, letterSpacing: '0.06em', textTransform: 'uppercase' }}>
          {title}
        </span>
        {count !== undefined && (
          <span style={{ marginLeft: 'auto', fontSize: 10, color: 'var(--ink-3)', fontFamily: 'var(--ff-mono)' }}>
            {count}
          </span>
        )}
      </header>
      {open && (
        <div style={{ padding: '8px 12px 12px', background: 'var(--bg-elev)' }}>
          {children}
        </div>
      )}
    </section>
  );
}
