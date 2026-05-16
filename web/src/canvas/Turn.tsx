import type { CSSProperties } from 'react';
import type { ToolCall } from '@/api/types';
import { Sidenote } from './Sidenote';

interface TurnProps {
  agent: string;
  timestamp: string;  // ISO UTC
  elapsedMs: number | null;
  body: string;
  confidence: number | null;
  model: string;
  durationMs: number;
  turn: number;
  toolCalls: ToolCall[];
  active: boolean;
  onSelectTool: (tc: ToolCall) => void;
}

const grid: CSSProperties = {
  display: 'grid',
  gridTemplateColumns: '96px 1fr 240px',
  gap: 24,
  padding: '16px 24px',
  borderBottom: '1px solid var(--hair)',
  transition: 'background-color 0.12s',
};

const bylineStyle: CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'flex-end',
  gap: 2,
  fontFamily: 'var(--ff-mono)',
};

function fmtTime(iso: string): string {
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  return d.toISOString().slice(11, 19);
}

function fmtElapsed(ms: number | null): string | null {
  if (ms === null || ms === undefined) return null;
  if (ms === 0) return '+0s';
  const sec = ms / 1000;
  if (sec < 60) return `+${sec.toFixed(1)}s`;
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return `+${m}m${s}s`;
}

export function Turn(props: TurnProps) {
  const elapsedStr = fmtElapsed(props.elapsedMs);
  return (
    <div
      data-turn={props.turn}
      data-active={props.active}
      style={{
        ...grid,
        background: 'transparent',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = 'var(--bg-tint)';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = 'transparent';
      }}
    >
      <div style={bylineStyle}>
        <span
          style={{
            fontSize: 12,
            fontWeight: 500,
            color: props.active ? 'var(--acc)' : 'var(--ink-1)',
          }}
        >
          {props.agent}
        </span>
        <span style={{ fontSize: 10, color: 'var(--ink-3)' }}>
          {fmtTime(props.timestamp)}
        </span>
        {elapsedStr && (
          <span style={{ fontSize: 10, color: 'var(--ink-4)' }}>
            {elapsedStr}
          </span>
        )}
      </div>
      <div
        style={{
          fontFamily: 'var(--ff-sans)',
          fontSize: 14,
          lineHeight: 1.6,
          color: props.active ? 'var(--ink-2)' : 'var(--ink-1)',
          fontStyle: props.active ? 'italic' : 'normal',
        }}
      >
        {props.body}
        {props.active && (
          <span
            data-typing-cursor
            aria-hidden
            style={{
              display: 'inline-block',
              width: 7,
              height: 14,
              marginLeft: 4,
              verticalAlign: -2,
              background: 'var(--ink-2)',
              animation: 'asr-typing 1.2s step-end infinite',
            }}
          />
        )}
      </div>
      <Sidenote
        confidence={props.confidence}
        model={props.model}
        durationMs={props.durationMs}
        turn={props.turn}
        toolCalls={props.toolCalls}
        onSelectTool={props.onSelectTool}
      />
    </div>
  );
}
