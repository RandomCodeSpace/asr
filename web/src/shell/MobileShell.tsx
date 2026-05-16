import { useState } from 'react';
import type { CSSProperties } from 'react';
import { SessionsRail } from '@/shell/SessionsRail';
import { SessionCanvas } from '@/canvas/SessionCanvas';
import { MonitorRail } from '@/monitors/MonitorRail';
import { MobileSheet } from '@/shell/MobileSheet';
import type { SessionSummary } from '@/state/useSessionList';
import type { AgentDefinition, ToolCall } from '@/api/types';

interface MobileShellProps {
  sessions: SessionSummary[];
  activeSid: string | null;
  onSelectSession: (sid: string) => void;
  queue: SessionSummary[];
  agentsByName: Record<string, AgentDefinition>;
  toolCalls: ToolCall[];
}

const shellWrap: CSSProperties = {
  display: 'grid',
  gridTemplateRows: '1fr auto',
  minHeight: 0,
};

const tabBar: CSSProperties = {
  display: 'grid',
  gridTemplateColumns: 'repeat(3, 1fr)',
  borderTop: '1px solid var(--hair-strong)',
  background: 'var(--bg-elev)',
  height: 56,
};

const tabBtn = (active: boolean): CSSProperties => ({
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  fontFamily: 'var(--ff-mono)',
  fontSize: 11,
  letterSpacing: '0.12em',
  textTransform: 'uppercase',
  color: active ? 'var(--ink-1)' : 'var(--ink-3)',
  background: 'transparent',
  border: 'none',
  borderTop: active ? '2px solid var(--acc)' : '2px solid transparent',
  cursor: 'pointer',
  padding: 0,
});

type ActiveSheet = 'sessions' | 'monitors' | null;

export function MobileShell({
  sessions, activeSid, onSelectSession, queue, agentsByName, toolCalls,
}: MobileShellProps) {
  const [sheet, setSheet] = useState<ActiveSheet>(null);

  function selectAndClose(sid: string) {
    onSelectSession(sid);
    setSheet(null);
  }

  return (
    <div style={shellWrap} data-shell="mobile">
      <div style={{ overflow: 'auto', minHeight: 0 }}>
        <SessionCanvas activeSid={activeSid} />
      </div>
      <nav style={tabBar} aria-label="Mobile navigation">
        <button
          type="button"
          style={tabBtn(sheet === 'sessions')}
          onClick={() => setSheet(sheet === 'sessions' ? null : 'sessions')}
          aria-label="Open sessions"
          aria-pressed={sheet === 'sessions'}
        >
          Sessions
        </button>
        <button
          type="button"
          style={tabBtn(sheet === null)}
          onClick={() => setSheet(null)}
          aria-label="Show canvas"
          aria-pressed={sheet === null}
        >
          Canvas
        </button>
        <button
          type="button"
          style={tabBtn(sheet === 'monitors')}
          onClick={() => setSheet(sheet === 'monitors' ? null : 'monitors')}
          aria-label="Open monitors"
          aria-pressed={sheet === 'monitors'}
        >
          Monitors
        </button>
      </nav>
      <MobileSheet
        open={sheet === 'sessions'}
        onOpenChange={(o) => setSheet(o ? 'sessions' : null)}
        title="Sessions"
        testId="sessions"
      >
        <SessionsRail
          sessions={sessions}
          activeSid={activeSid}
          onSelect={selectAndClose}
        />
      </MobileSheet>
      <MobileSheet
        open={sheet === 'monitors'}
        onOpenChange={(o) => setSheet(o ? 'monitors' : null)}
        title="Monitors"
        testId="monitors"
      >
        <MonitorRail
          sessions={sessions}
          activeSid={activeSid}
          queue={queue}
          agentsByName={agentsByName}
          toolCalls={toolCalls}
          sessionId={activeSid}
          onSelectSession={selectAndClose}
        />
      </MobileSheet>
    </div>
  );
}
