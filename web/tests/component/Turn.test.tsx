import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '../_helpers/render';
import { Turn } from '@/canvas/Turn';
import type { ToolCall } from '@/api/types';

const tools: ToolCall[] = [];

describe('<Turn>', () => {
  it('renders byline with agent name + timestamp', () => {
    render(
      <Turn
        agent="intake" timestamp="2026-05-15T14:16:32Z" elapsedMs={2200}
        body="Triage observed elevated p99 latency on payments-svc."
        confidence={0.92} model="gpt" durationMs={1230} turn={1}
        toolCalls={tools}
        active={false}
        onSelectTool={() => {}}
      />,
    );
    expect(screen.getByText('intake')).toBeInTheDocument();
    expect(screen.getByText(/14:16:32/)).toBeInTheDocument();
    expect(screen.getByText(/Triage observed/)).toBeInTheDocument();
  });

  it('renders elapsed delta in mono when provided', () => {
    render(
      <Turn
        agent="x" timestamp="2026-05-15T14:16:32Z" elapsedMs={2200}
        body="x" confidence={null} model="x" durationMs={0} turn={1}
        toolCalls={tools} active={false}
        onSelectTool={() => {}}
      />,
    );
    expect(screen.getByText(/\+2\.2s/)).toBeInTheDocument();
  });

  it('marks active turn with data-active="true" and shows typing cursor in body', () => {
    const { container } = render(
      <Turn
        agent="investigate" timestamp="2026-05-15T14:17:30Z" elapsedMs={null}
        body="Reading deploy diff..." confidence={null} model="x" durationMs={0} turn={2}
        toolCalls={tools} active={true}
        onSelectTool={() => {}}
      />,
    );
    expect(container.firstChild).toHaveAttribute('data-active', 'true');
    expect(container.querySelector('[data-typing-cursor]')).not.toBeNull();
  });

  it('non-active turn omits typing cursor', () => {
    const { container } = render(
      <Turn
        agent="x" timestamp="2026-05-15T14:16:00Z" elapsedMs={0}
        body="done" confidence={null} model="x" durationMs={0} turn={1}
        toolCalls={tools} active={false}
        onSelectTool={() => {}}
      />,
    );
    expect(container.querySelector('[data-typing-cursor]')).toBeNull();
    expect(container.firstChild).toHaveAttribute('data-active', 'false');
  });

  it('passes toolCalls through to Sidenote and renders them', () => {
    const tc: ToolCall = {
      agent: 'x', tool: 'obs:get_logs', args: {}, result: null,
      ts: '2026-05-15T14:16:50Z', risk: 'low',
      status: 'executed', approver: null, approved_at: null, approval_rationale: null,
    };
    render(
      <Turn
        agent="x" timestamp="2026-05-15T14:16:32Z" elapsedMs={0}
        body="x" confidence={null} model="x" durationMs={0} turn={1}
        toolCalls={[tc]} active={false}
        onSelectTool={() => {}}
      />,
    );
    expect(screen.getByText('obs:get_logs')).toBeInTheDocument();
  });

  it('forwards onSelectTool to Sidenote', () => {
    const onSelectTool = vi.fn();
    const tc: ToolCall = {
      agent: 'x', tool: 'obs:get_logs', args: {}, result: null,
      ts: 'x', risk: null, status: 'executed',
      approver: null, approved_at: null, approval_rationale: null,
    };
    render(
      <Turn
        agent="x" timestamp="2026-05-15T14:16:32Z" elapsedMs={0}
        body="x" confidence={null} model="x" durationMs={0} turn={1}
        toolCalls={[tc]} active={false}
        onSelectTool={onSelectTool}
      />,
    );
    screen.getByText('obs:get_logs').closest('[data-tool-card]')!.dispatchEvent(
      new MouseEvent('click', { bubbles: true }),
    );
    expect(onSelectTool).toHaveBeenCalledWith(tc);
  });
});
