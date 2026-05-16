import type { CSSProperties } from 'react';
import type { SessionSummary } from '@/state/useSessionList';
import { Monitor } from './Monitor';

interface Props {
  queue: SessionSummary[];
  onSelect: (sid: string) => void;
}

const row: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 8,
  padding: '6px 8px',
  borderBottom: '1px solid var(--hair)',
  cursor: 'pointer',
  fontFamily: 'var(--ff-sans)',
};

function ageStr(iso: string): string {
  const t = new Date(iso).getTime();
  if (isNaN(t)) return '—';
  const sec = Math.max(0, Math.floor((Date.now() - t) / 1000));
  if (sec < 60) return `${sec}s`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m`;
  return `${Math.floor(sec / 3600)}h`;
}

export function ApprovalsQueuePanel({ queue, onSelect }: Props) {
  return (
    <Monitor title="Approvals Queue" count={queue.length} pinned>
      {queue.length === 0 ? (
        <div style={{ fontSize: 11, color: 'var(--good)' }}>No approvals waiting.</div>
      ) : (
        queue.map((s) => (
          <div key={s.id} data-row onClick={() => onSelect(s.id)} style={row}>
            <span style={{ fontFamily: 'var(--ff-mono)', fontSize: 11, color: 'var(--ink-1)' }}>{s.id}</span>
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
              {s.label ?? '—'}
            </span>
            <span style={{ fontFamily: 'var(--ff-mono)', fontSize: 10, color: 'var(--warn)' }}>
              {ageStr(s.updated_at)}
            </span>
          </div>
        ))
      )}
    </Monitor>
  );
}
