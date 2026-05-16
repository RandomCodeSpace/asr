import type { CSSProperties } from 'react';
import type { SessionSummary } from '@/state/useSessionList';
import type { AgentDefinition, ToolCall } from '@/api/types';
import { SelectedPanel } from './SelectedPanel';
import { OtherSessionsPanel } from './OtherSessionsPanel';
import { ApprovalsQueuePanel } from './ApprovalsQueuePanel';
import { LessonsPanel } from './LessonsPanel';
import { ToolsPanel } from './ToolsPanel';
import { HealthPanel } from './HealthPanel';

interface MonitorRailProps {
  sessions: SessionSummary[];
  activeSid: string | null;
  queue: SessionSummary[];
  agentsByName: Record<string, AgentDefinition>;
  toolCalls: ToolCall[];
  sessionId: string | null;
  onSelectSession: (sid: string) => void;
}

const wrap: CSSProperties = {
  width: 340,
  display: 'flex',
  flexDirection: 'column',
  background: 'var(--bg-page)',
  borderLeft: '1px solid var(--hair)',
  overflowY: 'auto',
  minHeight: 0,
};

export function MonitorRail({
  sessions, activeSid, queue, agentsByName, toolCalls, sessionId, onSelectSession,
}: MonitorRailProps) {
  return (
    <aside style={wrap}>
      <SelectedPanel agentsByName={agentsByName} toolCalls={toolCalls} />
      <OtherSessionsPanel sessions={sessions} activeSid={activeSid} onSelect={onSelectSession} />
      <ApprovalsQueuePanel queue={queue} onSelect={onSelectSession} />
      <LessonsPanel sessionId={sessionId} />
      <ToolsPanel />
      <HealthPanel />
    </aside>
  );
}
