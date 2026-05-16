import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '../_helpers/render';
import { CanvasHead } from '@/canvas/CanvasHead';

describe('<CanvasHead>', () => {
  it('renders eyebrow with session id, status, and opened timestamp', () => {
    render(
      <CanvasHead
        sessionId="SES-20260515-042"
        status="in_progress"
        openedAt="2026-05-15T14:16:32Z"
        title="Payments Service Latency Spike"
        env="prod" sev={2} reporter="u1@platform"
        turnCount={4} toolCount={12} agentsActive={3} agentsTotal={4}
        onStop={() => {}} onRetry={() => {}}
      />,
    );
    expect(screen.getByText(/SES-20260515-042/)).toBeInTheDocument();
    expect(screen.getByText(/ACTIVE/i)).toBeInTheDocument();
    expect(screen.getByText(/Payments Service Latency Spike/)).toBeInTheDocument();
  });

  it('renders meta row with env, sev, reporter, counts', () => {
    render(
      <CanvasHead
        sessionId="X" status="in_progress" openedAt="2026-05-15T14:00:00Z"
        title="x" env="prod" sev={2} reporter="u@x"
        turnCount={4} toolCount={12} agentsActive={3} agentsTotal={4}
        onStop={() => {}} onRetry={() => {}}
      />,
    );
    expect(screen.getByText(/ENV prod/)).toBeInTheDocument();
    expect(screen.getByText(/SEV 2/)).toBeInTheDocument();
    expect(screen.getByText(/TURNS 4/)).toBeInTheDocument();
    expect(screen.getByText(/TOOLS 12/)).toBeInTheDocument();
    expect(screen.getByText(/AGENTS 3 of 4/)).toBeInTheDocument();
  });

  it('Stop button calls onStop', () => {
    const onStop = vi.fn();
    render(
      <CanvasHead
        sessionId="X" status="in_progress" openedAt="2026-05-15T14:00:00Z"
        title="x" env="prod" sev={2} reporter="u"
        turnCount={0} toolCount={0} agentsActive={0} agentsTotal={1}
        onStop={onStop} onRetry={() => {}}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: /Stop/i }));
    expect(onStop).toHaveBeenCalledTimes(1);
  });

  it('Retry button only shown when status="error" and calls onRetry', () => {
    const onRetry = vi.fn();
    const { rerender } = render(
      <CanvasHead
        sessionId="X" status="in_progress" openedAt="2026-05-15T14:00:00Z"
        title="x" env="prod" sev={2} reporter="u"
        turnCount={0} toolCount={0} agentsActive={0} agentsTotal={1}
        onStop={() => {}} onRetry={onRetry}
      />,
    );
    expect(screen.queryByRole('button', { name: /Retry/i })).not.toBeInTheDocument();
    rerender(
      <CanvasHead
        sessionId="X" status="error" openedAt="2026-05-15T14:00:00Z"
        title="x" env="prod" sev={2} reporter="u"
        turnCount={0} toolCount={0} agentsActive={0} agentsTotal={1}
        onStop={() => {}} onRetry={onRetry}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: /Retry/i }));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it('truncates titles longer than 80 chars with ellipsis', () => {
    const long = 'a'.repeat(120);
    render(
      <CanvasHead
        sessionId="X" status="in_progress" openedAt="2026-05-15T14:00:00Z"
        title={long} env="x" sev={1} reporter="x"
        turnCount={0} toolCount={0} agentsActive={0} agentsTotal={1}
        onStop={() => {}} onRetry={() => {}}
      />,
    );
    const title = screen.getByText(/a+/);
    expect(title.textContent?.length).toBeLessThanOrEqual(81);  // 80 + ellipsis
    expect(title.textContent).toMatch(/…$/);
  });
});
