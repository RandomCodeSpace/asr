import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '../_helpers/render';
import { HITLBand } from '@/canvas/HITLBand';
import type { ToolCall } from '@/api/types';

const tc: ToolCall = {
  agent: 'investigate', tool: 'rem:restart_service',
  args: { service: 'payments-svc' },
  result: null, ts: '2026-05-15T14:18:00Z', risk: 'high',
  status: 'pending_approval', approver: null, approved_at: null, approval_rationale: null,
};

describe('<HITLBand>', () => {
  it('renders the APPROVAL byline + waited duration', () => {
    render(
      <HITLBand
        toolCall={tc}
        waitedSeconds={18}
        question="Restart payments-svc?"
        confidence={0.78}
        turn={3}
        requestedBy="u1@platform"
        policy="risk:high requires approval"
        onApprove={() => {}}
        onReject={() => {}}
        onApproveWithRationale={() => {}}
      />,
    );
    expect(screen.getByText(/APPROVAL/)).toBeInTheDocument();
    expect(screen.getByText(/18 s/)).toBeInTheDocument();
  });

  it('renders the question prominently', () => {
    render(
      <HITLBand
        toolCall={tc} waitedSeconds={1} question="Restart payments-svc?"
        confidence={null} turn={1} requestedBy="u" policy="x"
        onApprove={() => {}} onReject={() => {}} onApproveWithRationale={() => {}}
      />,
    );
    expect(screen.getByText('Restart payments-svc?')).toBeInTheDocument();
  });

  it('renders the tool line with args', () => {
    render(
      <HITLBand
        toolCall={tc} waitedSeconds={1} question="x"
        confidence={null} turn={1} requestedBy="u" policy="x"
        onApprove={() => {}} onReject={() => {}} onApproveWithRationale={() => {}}
      />,
    );
    expect(screen.getByText(/rem:restart_service/)).toBeInTheDocument();
  });

  it('fires onApprove on Approve click', () => {
    const onApprove = vi.fn();
    render(
      <HITLBand
        toolCall={tc} waitedSeconds={1} question="x"
        confidence={null} turn={1} requestedBy="u" policy="x"
        onApprove={onApprove} onReject={() => {}} onApproveWithRationale={() => {}}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: /^Approve$/i }));
    expect(onApprove).toHaveBeenCalledTimes(1);
  });

  it('fires onReject on Reject click', () => {
    const onReject = vi.fn();
    render(
      <HITLBand
        toolCall={tc} waitedSeconds={1} question="x"
        confidence={null} turn={1} requestedBy="u" policy="x"
        onApprove={() => {}} onReject={onReject} onApproveWithRationale={() => {}}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: /Reject/i }));
    expect(onReject).toHaveBeenCalledTimes(1);
  });

  it('fires onApproveWithRationale on rationale click', () => {
    const onRat = vi.fn();
    render(
      <HITLBand
        toolCall={tc} waitedSeconds={1} question="x"
        confidence={null} turn={1} requestedBy="u" policy="x"
        onApprove={() => {}} onReject={() => {}} onApproveWithRationale={onRat}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: /Approve with Rationale/i }));
    expect(onRat).toHaveBeenCalledTimes(1);
  });
});
