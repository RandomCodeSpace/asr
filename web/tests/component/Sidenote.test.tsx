import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '../_helpers/render';
import { Sidenote } from '@/canvas/Sidenote';
import type { ToolCall } from '@/api/types';

const tool1: ToolCall = {
  agent: 'triage', tool: 'obs:get_logs', args: { service: 'api' },
  result: { lines: 12 }, ts: '2026-05-15T14:16:50Z', risk: 'low',
  status: 'executed', approver: null, approved_at: null, approval_rationale: null,
};

describe('<Sidenote>', () => {
  it('renders confidence / model / duration k/v rows', () => {
    render(
      <Sidenote
        confidence={0.92}
        model="claude-sonnet-4-6"
        durationMs={1230}
        turn={4}
        toolCalls={[]}
        onSelectTool={() => {}}
      />,
    );
    expect(screen.getByText(/conf/i)).toBeInTheDocument();
    expect(screen.getByText(/0\.92/)).toBeInTheDocument();
    expect(screen.getByText(/claude-sonnet-4-6/)).toBeInTheDocument();
    expect(screen.getByText(/1230ms/)).toBeInTheDocument();
  });

  it('renders one tool-call mini-card per ToolCall', () => {
    render(
      <Sidenote
        confidence={null} model="x" durationMs={0} turn={1}
        toolCalls={[tool1, { ...tool1, tool: 'rem:propose_fix', status: 'pending_approval' }]}
        onSelectTool={() => {}}
      />,
    );
    expect(screen.getByText('obs:get_logs')).toBeInTheDocument();
    expect(screen.getByText('rem:propose_fix')).toBeInTheDocument();
  });

  it('calls onSelectTool when a tool card is clicked', () => {
    const onSelect = vi.fn();
    render(
      <Sidenote
        confidence={null} model="x" durationMs={0} turn={1}
        toolCalls={[tool1]}
        onSelectTool={onSelect}
      />,
    );
    fireEvent.click(screen.getByText('obs:get_logs').closest('[data-tool-card]')!);
    expect(onSelect).toHaveBeenCalledWith(tool1);
  });

  it('shows status pill on tool card', () => {
    render(
      <Sidenote
        confidence={null} model="x" durationMs={0} turn={1}
        toolCalls={[{ ...tool1, status: 'pending_approval' }]}
        onSelectTool={() => {}}
      />,
    );
    expect(screen.getByText(/PENDING/i)).toBeInTheDocument();
  });
});
