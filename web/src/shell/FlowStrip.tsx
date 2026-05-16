import type { CSSProperties } from 'react';
import type { AgentDefinition } from '@/api/types';

export type NodeStatus = 'idle' | 'active' | 'done' | 'gated' | 'error';

interface FlowStripProps {
  agents: AgentDefinition[];
  activeAgent: string | null;
  graphVersion: string;
  /** Agent-name → status overrides (e.g. session.agents_run derives done) */
  statusByAgent?: Record<string, NodeStatus>;
}

const containerStyle: CSSProperties = {
  height: 92,
  display: 'flex',
  flexDirection: 'column',
  background: 'var(--bg-page)',
  borderBottom: '1px solid var(--hair)',
  fontFamily: 'var(--ff-sans)',
  position: 'relative',
};

const NODE_W = 90;
const NODE_W_ACTIVE = 100;
const NODE_H = 40;
const NODE_H_ACTIVE = 44;

export function FlowStrip({ agents, activeAgent, graphVersion, statusByAgent }: FlowStripProps) {
  if (agents.length === 0) {
    return (
      <div style={{ ...containerStyle, alignItems: 'center', justifyContent: 'center' }}>
        <span style={{ fontSize: 11, color: 'var(--ink-3)' }}>No agents loaded</span>
      </div>
    );
  }

  return (
    <div style={containerStyle}>
      <div
        style={{
          padding: '4px 16px 0',
          fontSize: 10,
          color: 'var(--ink-3)',
          letterSpacing: '0.14em',
          textTransform: 'uppercase',
          fontFamily: 'var(--ff-mono)',
        }}
      >
        Runtime · {agents.length} agents · graph {graphVersion}
      </div>
      <div
        style={{
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 32,
          padding: '0 16px',
          position: 'relative',
        }}
      >
        {agents.map((a, i) => {
          const isActive = a.name === activeAgent;
          const status: NodeStatus = isActive
            ? 'active'
            : statusByAgent?.[a.name] ?? 'idle';
          return (
            <div key={a.name} style={{ display: 'flex', alignItems: 'center' }}>
              <FlowNode agent={a} status={status} active={isActive} />
              {i < agents.length - 1 && <FlowEdge />}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function FlowNode({
  agent,
  status,
  active,
}: {
  agent: AgentDefinition;
  status: NodeStatus;
  active: boolean;
}) {
  const w = active ? NODE_W_ACTIVE : NODE_W;
  const h = active ? NODE_H_ACTIVE : NODE_H;
  const fills: Record<NodeStatus, string> = {
    idle: 'var(--bg-elev)',
    active: 'var(--acc-soft)',
    done: 'var(--good-bg)',
    gated: 'var(--warn-bg)',
    error: 'var(--danger-bg)',
  };
  const strokes: Record<NodeStatus, string> = {
    idle: 'var(--hair-strong)',
    active: 'var(--acc)',
    done: 'var(--good)',
    gated: 'var(--warn)',
    error: 'var(--danger)',
  };
  const labelStatusFor: Record<NodeStatus, string> = {
    idle: 'IDLE',
    active: 'ACTIVE',
    done: 'DONE',
    gated: 'GATED',
    error: 'ERROR',
  };
  return (
    <div
      data-flow-node={agent.name}
      data-active={active}
      data-status={status}
      style={{
        width: w,
        height: h,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        background: fills[status],
        border: `${active ? 1.5 : 1}px solid ${strokes[status]}`,
        borderRadius: 0,
        position: 'relative',
        animation: active ? 'asr-flow-halo 4s ease-in-out infinite' : 'none',
      }}
    >
      <span
        style={{
          fontFamily: 'var(--ff-mono)',
          fontSize: 12,
          color: 'var(--ink-1)',
          fontWeight: active ? 600 : 500,
        }}
      >
        {agent.name}
      </span>
      <span
        style={{
          fontFamily: 'var(--ff-mono)',
          fontSize: 9,
          color: 'var(--ink-3)',
          letterSpacing: '0.14em',
          textTransform: 'uppercase',
        }}
      >
        {labelStatusFor[status]}
      </span>
    </div>
  );
}

function FlowEdge() {
  return (
    <svg width={32} height={NODE_H_ACTIVE} aria-hidden style={{ display: 'block' }}>
      <path
        d={`M 0 ${NODE_H_ACTIVE / 2} C 16 ${NODE_H_ACTIVE / 2}, 16 ${NODE_H_ACTIVE / 2}, 32 ${NODE_H_ACTIVE / 2}`}
        stroke="var(--hair-strong)"
        strokeWidth={1}
        fill="none"
      />
    </svg>
  );
}
