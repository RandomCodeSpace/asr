import { describe, it, expect } from 'vitest';
import { useEffect } from 'react';
import { render, screen } from '../_helpers/render';
import { SelectedPanel } from '@/monitors/SelectedPanel';
import { SelectedRefProvider, useSetSelected } from '@/state/selectedRef';
import type { AgentDefinition, ToolCall } from '@/api/types';

const agents: Record<string, AgentDefinition> = {
  intake: {
    name: 'intake',
    kind: 'responsive',
    model: 'gpt',
    tools: ['obs:get_logs'],
    routes: { success: 'triage' },
    system_prompt_excerpt: 'You triage incidents…',
  },
};

const tc: ToolCall = {
  agent: 'triage',
  tool: 'obs:get_logs',
  args: { service: 'api' },
  result: 'ok',
  ts: '2026-05-15T14:16:50Z',
  risk: 'low',
  status: 'executed',
  approver: null,
  approved_at: null,
  approval_rationale: null,
};

interface SetterRef {
  kind: 'agent' | 'tool_call' | 'message' | null;
  id?: string;
}

function Setter({ ref }: { ref: SetterRef }) {
  const set = useSetSelected();
  useEffect(() => {
    set(ref);
  }, [set, ref]);
  return null;
}

describe('<SelectedPanel>', () => {
  it('renders empty state when nothing selected', () => {
    render(
      <SelectedRefProvider>
        <SelectedPanel agentsByName={agents} toolCalls={[tc]} />
      </SelectedRefProvider>,
    );
    expect(screen.getByText(/Click an agent, tool, or message/i)).toBeInTheDocument();
  });

  it('renders agent details when kind=agent', () => {
    render(
      <SelectedRefProvider>
        <Setter ref={{ kind: 'agent', id: 'intake' }} />
        <SelectedPanel agentsByName={agents} toolCalls={[tc]} />
      </SelectedRefProvider>,
    );
    expect(screen.getByText(/intake/)).toBeInTheDocument();
    expect(screen.getByText(/responsive/)).toBeInTheDocument();
    expect(screen.getByText(/gpt/)).toBeInTheDocument();
  });

  it('renders tool call details when kind=tool_call', () => {
    render(
      <SelectedRefProvider>
        <Setter ref={{ kind: 'tool_call', id: 'obs:get_logs@2026-05-15T14:16:50Z' }} />
        <SelectedPanel agentsByName={agents} toolCalls={[tc]} />
      </SelectedRefProvider>,
    );
    expect(screen.getByText(/obs:get_logs/)).toBeInTheDocument();
    expect(screen.getByText(/low/i)).toBeInTheDocument();
  });

  it('falls back to "not found" when selected id has no match', () => {
    render(
      <SelectedRefProvider>
        <Setter ref={{ kind: 'agent', id: 'mystery' }} />
        <SelectedPanel agentsByName={agents} toolCalls={[tc]} />
      </SelectedRefProvider>,
    );
    expect(screen.getByText(/not found/i)).toBeInTheDocument();
  });
});
