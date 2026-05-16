import { describe, it, expect } from 'vitest';
import { render, screen } from '../_helpers/render';
import { Statusbar } from '@/shell/Statusbar';

describe('<Statusbar>', () => {
  it('renders connection state and labels', () => {
    render(
      <Statusbar
        connection="connected"
        sseEventCount={247}
        vmSeq={247}
        vmSeqState="in-sync"
        runtimeVersion="v1.5.2"
        uiVersion="v2.0.0-rc1"
      />,
    );
    expect(screen.getByText(/Connected/)).toBeInTheDocument();
    expect(screen.getByText(/SSE 247 events/)).toBeInTheDocument();
    expect(screen.getByText(/vm_seq 247/)).toBeInTheDocument();
    expect(screen.getByText(/in-sync/)).toBeInTheDocument();
    expect(screen.getByText(/runtime v1\.5\.2/)).toBeInTheDocument();
    expect(screen.getByText(/ui v2\.0\.0-rc1/)).toBeInTheDocument();
  });

  it('uses data-connection attribute for state-driven styling', () => {
    const { container, rerender } = render(
      <Statusbar connection="connected" sseEventCount={0} vmSeq={0} vmSeqState="in-sync" runtimeVersion="x" uiVersion="x" />,
    );
    expect(container.firstChild).toHaveAttribute('data-connection', 'connected');
    rerender(
      <Statusbar connection="degraded" sseEventCount={0} vmSeq={0} vmSeqState="replaying" runtimeVersion="x" uiVersion="x" />,
    );
    expect(container.firstChild).toHaveAttribute('data-connection', 'degraded');
    rerender(
      <Statusbar connection="disconnected" sseEventCount={0} vmSeq={0} vmSeqState="divergent" runtimeVersion="x" uiVersion="x" />,
    );
    expect(container.firstChild).toHaveAttribute('data-connection', 'disconnected');
  });

  it('renders the optional p95 latency when provided', () => {
    render(
      <Statusbar
        connection="connected" sseEventCount={1} vmSeq={1} vmSeqState="in-sync"
        runtimeVersion="x" uiVersion="x" p95Ms={87}
      />,
    );
    expect(screen.getByText(/p95 87 ms/)).toBeInTheDocument();
  });
});
