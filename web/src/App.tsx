import { useState } from 'react';
import type { CSSProperties } from 'react';
import { Topbar, type Health } from '@/shell/Topbar';
import { Statusbar, type ConnectionState, type VmSeqState } from '@/shell/Statusbar';
import { SessionsRail } from '@/shell/SessionsRail';
import { FlowStrip } from '@/shell/FlowStrip';
import { SessionCanvas } from '@/canvas/SessionCanvas';
import { useUiHints } from '@/state/useUiHints';
import { useSessionList } from '@/state/useSessionList';
import { useApprovalsQueue } from '@/state/useApprovalsQueue';
import { useAgentDefinitions } from '@/state/useAgentDefinitions';
import { useSessionFull } from '@/state/useSessionFull';
import { MonitorRail } from '@/monitors/MonitorRail';
import { NewSessionModal } from '@/modals/NewSessionModal';
import { TabletShell } from '@/shell/TabletShell';
import { useBreakpoint } from '@/state/useBreakpoint';

const UI_VERSION = 'v2.0.0-rc1';
const RUNTIME_VERSION_FALLBACK = 'unknown';

const shellStyle: CSSProperties = {
  display: 'grid',
  gridTemplateRows: 'auto auto 1fr auto',
  height: '100vh',
  background: 'var(--bg-page)',
  color: 'var(--ink-1)',
  fontFamily: 'var(--ff-sans)',
};

const paneStyle: CSSProperties = {
  display: 'grid',
  gridTemplateColumns: '220px 1fr 340px',
  minHeight: 0,
};

export function App() {
  const [activeSid, setActiveSid] = useState<string | null>(null);
  const [newSessionOpen, setNewSessionOpen] = useState(false);

  const uiHints = useUiHints();
  const sessionList = useSessionList();
  const approvals = useApprovalsQueue();
  const agents = useAgentDefinitions();
  const sessionFull = useSessionFull(activeSid);
  const breakpoint = useBreakpoint();

  const brandName = uiHints.data?.brand_name ?? 'ASR';
  const envName = uiHints.data?.environments?.[0] ?? 'dev';
  const appName = 'runtime';

  const health: Health =
    sessionList.error || approvals.error || agents.isError
      ? 'down'
      : 'ok';

  const connection: ConnectionState =
    sessionList.error || (sessionFull.error && activeSid !== null)
      ? 'disconnected'
      : 'connected';

  const vmSeqState: VmSeqState = 'in-sync';
  const vmSeq = sessionFull.state.vmSeq;

  return (
    <div style={shellStyle}>
      <Topbar
        brandName={brandName}
        appName={appName}
        envName={envName}
        health={health}
        approvalsCount={approvals.count}
        onSearch={() => {/* Phase 6: open search overlay */}}
        onNew={() => setNewSessionOpen(true)}
        onApprovalsClick={() => {/* Phase 6: open approvals view */}}
      />
      <FlowStrip
        agents={agents.data?.list ?? []}
        activeAgent={null}
        graphVersion={`v${agents.data?.list.length ?? 0}`}
      />
      {breakpoint === 'tablet' ? (
        <TabletShell
          sessions={sessionList.sessions}
          activeSid={activeSid}
          onSelectSession={setActiveSid}
          queue={approvals.queue}
          agentsByName={agents.data?.byName ?? {}}
          toolCalls={sessionFull.state.toolCalls}
        />
      ) : (
        <div style={paneStyle}>
          <SessionsRail
            sessions={sessionList.sessions}
            activeSid={activeSid}
            onSelect={setActiveSid}
          />
          <SessionCanvas activeSid={activeSid} />
          <MonitorRail
            sessions={sessionList.sessions}
            activeSid={activeSid}
            queue={approvals.queue}
            agentsByName={agents.data?.byName ?? {}}
            toolCalls={sessionFull.state.toolCalls}
            sessionId={activeSid}
            onSelectSession={setActiveSid}
          />
        </div>
      )}
      <Statusbar
        connection={connection}
        sseEventCount={sessionFull.state.events.length}
        vmSeq={vmSeq}
        vmSeqState={vmSeqState}
        runtimeVersion={RUNTIME_VERSION_FALLBACK}
        uiVersion={UI_VERSION}
      />
      <NewSessionModal
        open={newSessionOpen}
        onOpenChange={setNewSessionOpen}
        environments={uiHints.data?.environments ?? ['dev']}
        onCreated={(sid) => setActiveSid(sid)}
      />
    </div>
  );
}
