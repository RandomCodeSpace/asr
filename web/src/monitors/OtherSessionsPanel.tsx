import type { CSSProperties } from 'react';
import type { SessionSummary } from '@/state/useSessionList';
import { Monitor } from './Monitor';

interface Props {
  sessions: SessionSummary[];
  activeSid: string | null;
  onSelect: (sid: string) => void;
}

const tile: CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: 2,
  padding: '6px 8px',
  border: '1px solid var(--hair)',
  background: 'var(--bg-elev)',
  cursor: 'pointer',
  fontFamily: 'var(--ff-sans)',
  marginBottom: 4,
};

const stateColor: Record<string, string> = {
  in_progress: 'var(--acc)',
  awaiting_input: 'var(--warn)',
  resolved: 'var(--good)',
  error: 'var(--danger)',
  stopped: 'var(--ink-3)',
  matched: 'var(--good)',
  escalated: 'var(--danger)',
  new: 'var(--acc)',
};

function shortStatus(s: string): string {
  return s.replace('_', ' ').slice(0, 6).toUpperCase();
}

export function OtherSessionsPanel({ sessions, activeSid, onSelect }: Props) {
  const others = sessions.filter((s) => s.id !== activeSid);
  return (
    <Monitor title="Other Sessions" count={others.length} pinned>
      {others.length === 0 ? (
        <div style={{ fontSize: 11, color: 'var(--ink-3)' }}>
          Only this session is active right now.
        </div>
      ) : (
        others.map((s) => (
          <div key={s.id} data-tile onClick={() => onSelect(s.id)} style={tile}>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 6 }}>
              <span
                style={{
                  fontFamily: 'var(--ff-mono)',
                  fontSize: 11,
                  color: 'var(--ink-1)',
                }}
              >
                {s.id}
              </span>
              <span
                style={{
                  fontFamily: 'var(--ff-mono)',
                  fontSize: 9,
                  color: stateColor[s.status] ?? 'var(--ink-3)',
                  letterSpacing: '0.14em',
                }}
              >
                {shortStatus(s.status)}
              </span>
            </div>
            <div
              style={{
                fontSize: 11,
                color: 'var(--ink-2)',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
            >
              {s.label ?? '—'}
            </div>
            {s.active_agent && (
              <div
                style={{
                  fontFamily: 'var(--ff-mono)',
                  fontSize: 10,
                  color: 'var(--ink-3)',
                }}
              >
                {s.active_agent}
              </div>
            )}
          </div>
        ))
      )}
    </Monitor>
  );
}
