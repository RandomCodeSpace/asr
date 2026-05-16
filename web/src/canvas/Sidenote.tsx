import type { CSSProperties } from 'react';
import type { ToolCall } from '@/api/types';
import { ToolCallCard } from './ToolCallCard';

interface SidenoteProps {
  confidence: number | null;
  model: string;
  durationMs: number;
  turn: number;
  toolCalls: ToolCall[];
  onSelectTool: (tc: ToolCall) => void;
}

const wrap: CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: 6,
  fontFamily: 'var(--ff-mono)',
  fontSize: 11,
  color: 'var(--ink-3)',
};

function Kv({ k, v }: { k: string; v: string | number | null }) {
  return (
    <div style={{ display: 'flex', gap: 6 }}>
      <span style={{ width: 64, color: 'var(--ink-3)' }}>{k}</span>
      <span style={{ flex: 1, color: 'var(--ink-1)' }}>
        {v === null ? '—' : String(v)}
      </span>
    </div>
  );
}

export function Sidenote({
  confidence, model, durationMs, turn, toolCalls, onSelectTool,
}: SidenoteProps) {
  return (
    <div style={wrap}>
      <Kv k="conf" v={confidence === null ? null : confidence.toFixed(2)} />
      <Kv k="model" v={model} />
      <Kv k="dur" v={`${durationMs}ms`} />
      <Kv k="turn" v={turn} />
      {toolCalls.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginTop: 4 }}>
          {toolCalls.map((tc, i) => (
            <ToolCallCard
              key={`${tc.ts}-${i}`}
              tc={tc}
              onClick={() => onSelectTool(tc)}
            />
          ))}
        </div>
      )}
    </div>
  );
}
