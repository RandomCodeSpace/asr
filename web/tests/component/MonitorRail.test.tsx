import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen } from '../_helpers/render';
import { MonitorRail } from '@/monitors/MonitorRail';
import type { SessionSummary } from '@/state/useSessionList';
import type { AgentDefinition, ToolCall } from '@/api/types';

const sessions: SessionSummary[] = [
  { id: 'SES-1', status: 'in_progress', label: 'A', created_at: 't0', updated_at: 't1' },
];
const queue: SessionSummary[] = [];
const agents: Record<string, AgentDefinition> = {};
const tools: ToolCall[] = [];

describe('<MonitorRail>', () => {
  beforeEach(() => {
    global.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify([]), { status: 200, headers: { 'content-type': 'application/json' } }),
    );
  });

  it('renders all 6 panel titles', () => {
    render(
      <MonitorRail
        sessions={sessions} activeSid={null} queue={queue}
        agentsByName={agents} toolCalls={tools} sessionId={null}
        onSelectSession={() => {}}
      />,
    );
    expect(screen.getByText(/Selected/)).toBeInTheDocument();
    expect(screen.getByText(/Other Sessions/)).toBeInTheDocument();
    expect(screen.getByText(/Approvals Queue/)).toBeInTheDocument();
    expect(screen.getByText(/Lessons/)).toBeInTheDocument();
    expect(screen.getByText(/Tool Catalog/)).toBeInTheDocument();
    expect(screen.getByText(/System Health/)).toBeInTheDocument();
  });

  it('renders pinned panels open by default (Selected, Others, Approvals)', () => {
    render(
      <MonitorRail
        sessions={sessions} activeSid={null} queue={queue}
        agentsByName={agents} toolCalls={tools} sessionId={null}
        onSelectSession={() => {}}
      />,
    );
    // Selected pinned with empty body → "Click an agent..."
    expect(screen.getByText(/Click an agent, tool, or message/i)).toBeInTheDocument();
  });
});
