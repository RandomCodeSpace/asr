import type { CSSProperties } from 'react';
import type { ToolCall } from '@/api/types';
import { Button } from '@/components/Button';

interface HITLBandProps {
  toolCall: ToolCall;
  waitedSeconds: number;
  question: string;
  confidence: number | null;
  turn: number;
  requestedBy: string;
  policy: string;
  onApprove: () => void;
  onReject: () => void;
  onApproveWithRationale: () => void;
}

const grid: CSSProperties = {
  display: 'grid',
  gridTemplateColumns: '96px 1fr 240px',
  gap: 24,
  padding: '20px 24px',
  background: 'linear-gradient(90deg, var(--warn-bg) 0%, rgba(180, 129, 74, 0.02) 100%)',
  borderTop: '1px solid var(--warn)',
  borderBottom: '1px solid var(--warn)',
};

const riskColor: Record<string, string> = {
  low: 'var(--good)',
  medium: 'var(--warn)',
  high: 'var(--danger)',
};

function Kv({ k, v, color }: { k: string; v: string | number; color?: string }) {
  return (
    <div style={{ display: 'flex', gap: 6, fontFamily: 'var(--ff-mono)', fontSize: 11 }}>
      <span style={{ width: 80, color: 'var(--ink-3)' }}>{k}</span>
      <span style={{ flex: 1, color: color ?? 'var(--ink-1)' }}>{String(v)}</span>
    </div>
  );
}

export function HITLBand({
  toolCall, waitedSeconds, question, confidence, turn,
  requestedBy, policy, onApprove, onReject, onApproveWithRationale,
}: HITLBandProps) {
  const argsStr = JSON.stringify(toolCall.args)
    .replace(/^\{|\}$/g, '')
    .replace(/"/g, '');
  return (
    <div style={grid}>
      <div style={{ textAlign: 'right' }}>
        <div
          style={{
            fontFamily: 'var(--ff-mono)', fontSize: 10,
            color: 'var(--warn)', letterSpacing: '0.14em',
            textTransform: 'uppercase', fontWeight: 600,
          }}
        >
          APPROVAL
        </div>
        <div style={{ fontFamily: 'var(--ff-mono)', fontSize: 10, color: 'var(--ink-3)', marginTop: 4 }}>
          waited {waitedSeconds} s
        </div>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        <h2
          style={{
            margin: 0, fontFamily: 'var(--ff-sans)', fontSize: 18,
            fontWeight: 500, color: 'var(--ink-1)', lineHeight: 1.35,
          }}
        >
          {question}
        </h2>
        <div
          style={{
            fontFamily: 'var(--ff-mono)', fontSize: 12, color: 'var(--ink-2)',
            padding: '6px 8px', background: 'var(--bg-elev)',
            border: '1px solid var(--hair)',
          }}
        >
          <span style={{ color: 'var(--acc)' }}>{toolCall.tool}</span>({argsStr})
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Button variant="primary" size="sm" onClick={onApprove}>Approve</Button>
          <Button variant="secondary" size="sm" onClick={onReject}>Reject</Button>
          <span style={{ flex: 1 }} />
          <Button variant="ghost" size="sm" onClick={onApproveWithRationale}>
            Approve with Rationale
          </Button>
        </div>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        <Kv k="Requested by" v={requestedBy} />
        <Kv k="Turn" v={turn} />
        <Kv k="Confidence" v={confidence === null ? '—' : confidence.toFixed(2)} />
        <Kv
          k="Risk"
          v={toolCall.risk ?? 'unknown'}
          color={riskColor[toolCall.risk ?? ''] ?? 'var(--ink-2)'}
        />
        <Kv k="Policy" v={policy} />
      </div>
    </div>
  );
}
