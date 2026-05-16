import { useMemo, useState } from 'react';
import type { CSSProperties, ReactNode } from 'react';
import type { SessionSummary } from '@/state/useSessionList';

const ACTIVE_STATUSES = new Set(['new', 'in_progress', 'awaiting_input', 'matched']);

interface SessionsRailProps {
  sessions: SessionSummary[];
  activeSid: string | null;
  onSelect: (sid: string) => void;
}

const containerStyle: CSSProperties = {
  width: 220,
  display: 'flex',
  flexDirection: 'column',
  background: 'var(--bg-page)',
  borderRight: '1px solid var(--hair)',
  overflow: 'hidden',
  fontFamily: 'var(--ff-sans)',
};

export function SessionsRail({ sessions, activeSid, onSelect }: SessionsRailProps) {
  const [filter, setFilter] = useState('');
  const filtered = useMemo(() => {
    const q = filter.trim().toLowerCase();
    if (!q) return sessions;
    return sessions.filter(
      (s) =>
        s.id.toLowerCase().includes(q) ||
        (s.label !== undefined && s.label.toLowerCase().includes(q)),
    );
  }, [sessions, filter]);

  const active = filtered.filter((s) => ACTIVE_STATUSES.has(s.status));
  const recent = filtered.filter((s) => !ACTIVE_STATUSES.has(s.status));

  return (
    <aside style={containerStyle}>
      <header style={{ padding: '10px 12px', borderBottom: '1px solid var(--hair)' }}>
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'baseline',
            marginBottom: 8,
          }}
        >
          <span style={{ fontSize: 13, fontWeight: 500, color: 'var(--ink-1)' }}>Sessions</span>
          <span style={{ fontSize: 10, color: 'var(--ink-3)' }}>{sessions.length} total</span>
        </div>
        <input
          type="text"
          placeholder="Filter…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          style={{
            width: '100%',
            height: 22,
            padding: '0 8px',
            fontFamily: 'var(--ff-sans)',
            fontSize: 11,
            color: 'var(--ink-1)',
            background: 'var(--bg-elev)',
            border: '1px solid var(--hair)',
            borderRadius: 0,
            outline: 'none',
          }}
        />
      </header>
      <div style={{ flex: 1, overflowY: 'auto' }}>
        {sessions.length === 0 ? (
          <div style={{ padding: 20, fontSize: 11, color: 'var(--ink-3)', textAlign: 'center' }}>
            No sessions yet
          </div>
        ) : (
          <>
            <Group label="Active" count={active.length}>
              {active.map((s) => (
                <Row key={s.id} session={s} active={s.id === activeSid} onSelect={onSelect} />
              ))}
            </Group>
            <Group label="Recent" count={recent.length}>
              {recent.map((s) => (
                <Row key={s.id} session={s} active={s.id === activeSid} onSelect={onSelect} />
              ))}
            </Group>
          </>
        )}
      </div>
    </aside>
  );
}

function Group({ label, count, children }: { label: string; count: number; children: ReactNode }) {
  return (
    <div>
      <div
        style={{
          padding: '8px 12px 4px',
          fontSize: 10,
          color: 'var(--ink-3)',
          letterSpacing: '0.14em',
          textTransform: 'uppercase',
          borderBottom: '1px solid var(--hair)',
        }}
      >
        {label} · {count}
      </div>
      {children}
    </div>
  );
}

function Row({
  session,
  active,
  onSelect,
}: {
  session: SessionSummary;
  active: boolean;
  onSelect: (sid: string) => void;
}) {
  const isPending = session.status === 'awaiting_input';
  return (
    <div
      data-row
      data-active={active}
      onClick={() => onSelect(session.id)}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 6,
        padding: '6px 10px 6px 12px',
        cursor: 'pointer',
        background: active ? 'var(--bg-elev)' : 'transparent',
        borderLeft: active ? '2px solid var(--acc)' : '2px solid transparent',
        position: 'relative',
      }}
    >
      <span
        style={{
          fontFamily: 'var(--ff-mono)',
          fontSize: 11,
          color: active ? 'var(--acc)' : 'var(--ink-2)',
        }}
      >
        {session.id}
      </span>
      <span
        style={{
          flex: 1,
          fontSize: 11,
          color: 'var(--ink-2)',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}
      >
        {session.label ?? ''}
      </span>
      <span
        style={{
          fontFamily: 'var(--ff-mono)',
          fontSize: 9,
          color: 'var(--ink-3)',
          letterSpacing: '0.14em',
          textTransform: 'uppercase',
        }}
      >
        {shortStatus(session.status)}
      </span>
      {isPending && (
        <span
          data-pending-dot
          aria-label="pending approval"
          style={{
            display: 'inline-block',
            width: 5,
            height: 5,
            borderRadius: '50%',
            background: 'var(--warn)',
          }}
        />
      )}
    </div>
  );
}

function shortStatus(s: string): string {
  switch (s) {
    case 'in_progress':
      return 'RUN';
    case 'awaiting_input':
      return 'WAIT';
    case 'resolved':
      return 'DONE';
    case 'escalated':
      return 'ESC';
    case 'error':
      return 'ERR';
    case 'stopped':
      return 'STOP';
    case 'matched':
      return 'MATCH';
    case 'new':
      return 'NEW';
    default:
      return s.slice(0, 4).toUpperCase();
  }
}
