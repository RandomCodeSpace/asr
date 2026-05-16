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

const monitorRailPlaceholder: CSSProperties = {
  background: 'var(--bg-page)',
  borderLeft: '1px solid var(--hair)',
  padding: 16,
  fontSize: 11,
  color: 'var(--ink-3)',
};

export function App() {
  const [activeSid, setActiveSid] = useState<string | null>(null);

  const uiHints = useUiHints();
  const sessionList = useSessionList();
  const approvals = useApprovalsQueue();
  const agents = useAgentDefinitions();
  const sessionFull = useSessionFull(activeSid);

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
        onNew={() => {/* Phase 6: open NewSessionModal */}}
        onApprovalsClick={() => {/* Phase 6: open approvals view */}}
      />
      <FlowStrip
        agents={agents.data?.list ?? []}
        activeAgent={null}
        graphVersion={`v${agents.data?.list.length ?? 0}`}
      />
      <div style={paneStyle}>
        <SessionsRail
          sessions={sessionList.sessions}
          activeSid={activeSid}
          onSelect={setActiveSid}
        />
        <SessionCanvas activeSid={activeSid} />
        <div style={monitorRailPlaceholder}>
          Ambient monitors (Phase 5)
        </div>
      </div>
      <Statusbar
        connection={connection}
        sseEventCount={sessionFull.state.events.length}
        vmSeq={vmSeq}
        vmSeqState={vmSeqState}
        runtimeVersion={RUNTIME_VERSION_FALLBACK}
        uiVersion={UI_VERSION}
      />
    </div>
  );
}
