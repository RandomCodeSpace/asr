import { useState } from 'react';
import type { CSSProperties } from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import { SessionsRail } from '@/shell/SessionsRail';
import { SessionCanvas } from '@/canvas/SessionCanvas';
import { MonitorRail } from '@/monitors/MonitorRail';
import type { SessionSummary } from '@/state/useSessionList';
import type { AgentDefinition, ToolCall } from '@/api/types';

interface TabletShellProps {
  sessions: SessionSummary[];
  activeSid: string | null;
  onSelectSession: (sid: string) => void;
  queue: SessionSummary[];
  agentsByName: Record<string, AgentDefinition>;
  toolCalls: ToolCall[];
}

const grid: CSSProperties = {
  display: 'grid',
  gridTemplateColumns: '180px 1fr',
  minHeight: 0,
  position: 'relative',
};

const monitorsBtn: CSSProperties = {
  position: 'absolute',
  top: 12,
  right: 12,
  height: 28,
  padding: '0 12px',
  fontFamily: 'var(--ff-mono)',
  fontSize: 11,
  letterSpacing: '0.06em',
  textTransform: 'uppercase',
  color: 'var(--ink-1)',
  background: 'var(--bg-elev)',
  border: '1px solid var(--hair-strong)',
  borderRadius: 0,
  cursor: 'pointer',
  zIndex: 5,
};

const sheetContent: CSSProperties = {
  position: 'fixed',
  top: 0,
  right: 0,
  bottom: 0,
  width: 'min(360px, 90vw)',
  background: 'var(--bg-page)',
  borderLeft: '1px solid var(--hair-strong)',
  boxShadow: 'var(--e-3)',
  overflow: 'auto',
  zIndex: 1000,
  display: 'flex',
  flexDirection: 'column',
};

export function TabletShell({
  sessions, activeSid, onSelectSession, queue, agentsByName, toolCalls,
}: TabletShellProps) {
  const [monitorsOpen, setMonitorsOpen] = useState(false);
  return (
    <div style={grid} data-shell="tablet">
      <SessionsRail
        sessions={sessions}
        activeSid={activeSid}
        onSelect={onSelectSession}
      />
      <div style={{ position: 'relative', display: 'flex', flexDirection: 'column', minHeight: 0 }}>
        <button
          type="button"
          aria-label="Open monitors"
          style={monitorsBtn}
          onClick={() => setMonitorsOpen(true)}
        >
          Monitors
        </button>
        <SessionCanvas activeSid={activeSid} />
      </div>
      <Dialog.Root open={monitorsOpen} onOpenChange={setMonitorsOpen}>
        <Dialog.Portal>
          <Dialog.Overlay style={{
            position: 'fixed', inset: 0,
            background: 'rgba(21,17,10,0.18)',
            backdropFilter: 'blur(2px)',
            zIndex: 999,
          }} />
          <Dialog.Content style={sheetContent} data-monitors-sheet="">
            <Dialog.Title style={{
              fontSize: 10, color: 'var(--ink-3)',
              letterSpacing: '0.14em', textTransform: 'uppercase',
              padding: '12px 16px', borderBottom: '1px solid var(--hair)',
              margin: 0, fontFamily: 'var(--ff-mono)',
            }}>
              Monitors
            </Dialog.Title>
            <MonitorRail
              sessions={sessions}
              activeSid={activeSid}
              queue={queue}
              agentsByName={agentsByName}
              toolCalls={toolCalls}
              sessionId={activeSid}
              onSelectSession={(sid) => { onSelectSession(sid); setMonitorsOpen(false); }}
            />
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </div>
  );
}
