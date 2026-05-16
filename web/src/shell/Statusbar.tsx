import type { CSSProperties } from 'react';

export type ConnectionState = 'connected' | 'degraded' | 'disconnected';
export type VmSeqState = 'in-sync' | 'replaying' | 'divergent';

interface StatusbarProps {
  connection: ConnectionState;
  sseEventCount: number;
  vmSeq: number;
  vmSeqState: VmSeqState;
  runtimeVersion: string;
  uiVersion: string;
  p95Ms?: number;
}

const dotColor: Record<ConnectionState, string> = {
  connected: 'var(--good)',
  degraded: 'var(--warn)',
  disconnected: 'var(--danger)',
};

const vmSeqColor: Record<VmSeqState, string> = {
  'in-sync': 'var(--good)',
  replaying: 'var(--warn)',
  divergent: 'var(--danger)',
};

const baseStyle: CSSProperties = {
  height: 24,
  display: 'flex',
  alignItems: 'center',
  gap: 12,
  padding: '0 16px',
  fontFamily: 'var(--ff-mono)',
  fontSize: 11,
  color: 'var(--ink-3)',
  background: 'var(--bg-page)',
  borderTop: '1px solid var(--hair)',
  whiteSpace: 'nowrap',
  overflow: 'hidden',
};

const sep = <span style={{ color: 'var(--ink-4)' }}>·</span>;

export function Statusbar({
  connection,
  sseEventCount,
  vmSeq,
  vmSeqState,
  runtimeVersion,
  uiVersion,
  p95Ms,
}: StatusbarProps) {
  const connectionLabel: Record<ConnectionState, string> = {
    connected: 'Connected',
    degraded: 'Reconnecting…',
    disconnected: 'Disconnected',
  };
  return (
    <div data-connection={connection} style={baseStyle}>
      <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
        <span
          aria-hidden
          style={{
            display: 'inline-block',
            width: 6,
            height: 6,
            borderRadius: '50%',
            background: dotColor[connection],
          }}
        />
        {connectionLabel[connection]}
      </span>
      {sep}
      <span>SSE {sseEventCount} events</span>
      {sep}
      <span>
        vm_seq {vmSeq}{' '}
        <span style={{ color: vmSeqColor[vmSeqState] }}>({vmSeqState})</span>
      </span>
      {p95Ms !== undefined && (
        <>
          {sep}
          <span>p95 {p95Ms} ms</span>
        </>
      )}
      <span style={{ flex: 1 }} />
      <span>runtime {runtimeVersion}</span>
      {sep}
      <span>ui {uiVersion}</span>
    </div>
  );
}
