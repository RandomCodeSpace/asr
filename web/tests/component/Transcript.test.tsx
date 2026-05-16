import { describe, it, expect } from 'vitest';
import { render, screen } from '../_helpers/render';
import { Transcript } from '@/canvas/Transcript';
import type { AgentRun, ToolCall } from '@/api/types';

const intakeRun: AgentRun = {
  agent: 'intake', started_at: '2026-05-15T14:16:30Z', ended_at: '2026-05-15T14:16:32Z',
  summary: 'Triage observed elevated p99 latency.', confidence: 0.92,
  confidence_rationale: null, signal: null,
};
const triageRun: AgentRun = {
  agent: 'triage', started_at: '2026-05-15T14:16:50Z', ended_at: '2026-05-15T14:16:54Z',
  summary: 'Correlated with deploy.', confidence: 0.88,
  confidence_rationale: null, signal: null,
};
const tcLogs: ToolCall = {
  agent: 'triage', tool: 'obs:get_logs', args: { service: 'api' }, result: { lines: 12 },
  ts: '2026-05-15T14:16:52Z', risk: 'low', status: 'executed',
  approver: null, approved_at: null, approval_rationale: null,
};
const tcPending: ToolCall = {
  agent: 'investigate', tool: 'rem:restart_service',
  args: { service: 'payments-svc' }, result: null, ts: '2026-05-15T14:18:00Z',
  risk: 'high', status: 'pending_approval',
  approver: null, approved_at: null, approval_rationale: null,
};

describe('<Transcript>', () => {
  it('renders one Turn per completed agent run', () => {
    render(
      <Transcript
        agentsRun={[intakeRun, triageRun]}
        toolCalls={[]}
        activeAgent={null}
        hitlContext={null}
        onSelectTool={() => {}}
        onApprove={() => {}}
        onReject={() => {}}
        onApproveWithRationale={() => {}}
      />,
    );
    expect(screen.getByText('intake')).toBeInTheDocument();
    expect(screen.getByText('triage')).toBeInTheDocument();
  });

  it('renders active Turn for currently-running agent (not yet in agentsRun)', () => {
    render(
      <Transcript
        agentsRun={[intakeRun]}
        toolCalls={[]}
        activeAgent={{ name: 'investigate', startedAt: '2026-05-15T14:17:00Z', currentBody: 'Reading deploy diff...' }}
        hitlContext={null}
        onSelectTool={() => {}}
        onApprove={() => {}}
        onReject={() => {}}
        onApproveWithRationale={() => {}}
      />,
    );
    expect(screen.getByText('investigate')).toBeInTheDocument();
    expect(screen.getByText(/Reading deploy diff/)).toBeInTheDocument();
  });

  it('groups tool calls by agent and renders them in the matching Turn sidenote', () => {
    render(
      <Transcript
        agentsRun={[intakeRun, triageRun]}
        toolCalls={[tcLogs]}
        activeAgent={null}
        hitlContext={null}
        onSelectTool={() => {}}
        onApprove={() => {}}
        onReject={() => {}}
        onApproveWithRationale={() => {}}
      />,
    );
    // tcLogs.agent === 'triage', so it appears in triage's sidenote
    expect(screen.getByText('obs:get_logs')).toBeInTheDocument();
  });

  it('renders HITLBand when hitlContext is provided', () => {
    render(
      <Transcript
        agentsRun={[intakeRun]}
        toolCalls={[tcPending]}
        activeAgent={null}
        hitlContext={{
          toolCall: tcPending,
          waitedSeconds: 18,
          question: 'Restart payments-svc?',
          confidence: 0.78,
          turn: 3,
          requestedBy: 'u1',
          policy: 'risk:high requires approval',
        }}
        onSelectTool={() => {}}
        onApprove={() => {}}
        onReject={() => {}}
        onApproveWithRationale={() => {}}
      />,
    );
    expect(screen.getByText(/APPROVAL/)).toBeInTheDocument();
    expect(screen.getByText('Restart payments-svc?')).toBeInTheDocument();
  });

  it('renders empty state when there are no turns', () => {
    render(
      <Transcript
        agentsRun={[]}
        toolCalls={[]}
        activeAgent={null}
        hitlContext={null}
        onSelectTool={() => {}}
        onApprove={() => {}}
        onReject={() => {}}
        onApproveWithRationale={() => {}}
      />,
    );
    expect(screen.getByText(/No turns yet/i)).toBeInTheDocument();
  });
});
