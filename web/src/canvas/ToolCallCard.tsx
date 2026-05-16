import type { CSSProperties } from 'react';
import type { ToolCall } from '@/api/types';

interface ToolCallCardProps {
  tc: ToolCall;
  onClick: () => void;
}

const cardStyle: CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: 2,
  padding: '6px 8px',
  border: '1px solid var(--hair)',
  background: 'var(--bg-elev)',
  cursor: 'pointer',
  fontFamily: 'var(--ff-mono)',
};

const statusColor: Record<string, string> = {
  executed: 'var(--good)',
  executed_with_notify: 'var(--good)',
  pending_approval: 'var(--warn)',
  approved: 'var(--good)',
  rejected: 'var(--danger)',
  auto_rejected: 'var(--danger)',
  timeout: 'var(--danger)',
};

function shortStatus(s: string): string {
  switch (s) {
    case 'executed': return 'OK';
    case 'executed_with_notify': return 'OK*';
    case 'pending_approval': return 'PENDING';
    case 'approved': return 'APPROVED';
    case 'rejected': return 'REJECTED';
    case 'auto_rejected': return 'AUTO-REJ';
    case 'timeout': return 'TIMEOUT';
    default: return s.toUpperCase();
  }
}

export function ToolCallCard({ tc, onClick }: ToolCallCardProps) {
  return (
    <div data-tool-card onClick={onClick} style={cardStyle}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 6 }}>
        <span style={{ fontSize: 11, color: 'var(--ink-1)' }}>{tc.tool}</span>
        <span
          style={{
            fontSize: 9,
            letterSpacing: '0.14em',
            color: statusColor[tc.status] ?? 'var(--ink-3)',
          }}
        >
          {shortStatus(tc.status)}
        </span>
      </div>
      {tc.risk && (
        <span style={{ fontSize: 9, color: 'var(--ink-3)' }}>
          risk: {tc.risk}
        </span>
      )}
    </div>
  );
}
