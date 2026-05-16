import type { CSSProperties, ReactNode } from 'react';
import type { AgentDefinition, ToolCall } from '@/api/types';
import { useSelected } from '@/state/selectedRef';
import { Monitor } from './Monitor';

interface SelectedPanelProps {
  agentsByName: Record<string, AgentDefinition>;
  toolCalls: ToolCall[];
}

const eyebrow: CSSProperties = {
  display: 'inline-block',
  padding: '2px 6px',
  background: 'var(--acc-soft)',
  color: 'var(--acc)',
  fontFamily: 'var(--ff-mono)',
  fontSize: 9,
  letterSpacing: '0.14em',
  textTransform: 'uppercase',
  marginBottom: 8,
};

const kv: CSSProperties = {
  display: 'flex',
  gap: 6,
  fontFamily: 'var(--ff-mono)',
  fontSize: 11,
  color: 'var(--ink-3)',
  marginBottom: 4,
};

const kvKey: CSSProperties = { width: 80, color: 'var(--ink-3)' };
const kvVal: CSSProperties = { flex: 1, color: 'var(--ink-1)' };

function Empty(): ReactNode {
  return (
    <div style={{ fontSize: 11, color: 'var(--ink-3)' }}>
      Click an agent, tool, or message to see detail.
    </div>
  );
}

function NotFound(): ReactNode {
  return (
    <div style={{ fontSize: 11, color: 'var(--ink-3)' }}>
      Selected item not found in current session.
    </div>
  );
}

function AgentBody({ agent }: { agent: AgentDefinition }): ReactNode {
  return (
    <>
      <div style={eyebrow}>agent · {agent.name}</div>
      <div style={kv}>
        <span style={kvKey}>Kind</span>
        <span style={kvVal}>{agent.kind}</span>
      </div>
      <div style={kv}>
        <span style={kvKey}>Model</span>
        <span style={kvVal}>{agent.model}</span>
      </div>
      <div style={kv}>
        <span style={kvKey}>Tools</span>
        <span style={kvVal}>{agent.tools.join(', ') || '—'}</span>
      </div>
      {Object.keys(agent.routes).length > 0 && (
        <div style={kv}>
          <span style={kvKey}>Routes</span>
          <span style={kvVal}>
            {Object.entries(agent.routes)
              .map(([k, v]) => `${k} → ${v}`)
              .join(' · ')}
          </span>
        </div>
      )}
      {agent.system_prompt_excerpt && (
        <div
          style={{
            marginTop: 8,
            fontSize: 11,
            color: 'var(--ink-2)',
            fontStyle: 'italic',
          }}
        >
          {agent.system_prompt_excerpt}
        </div>
      )}
    </>
  );
}

function ToolCallBody({ tc }: { tc: ToolCall }): ReactNode {
  return (
    <>
      <div style={eyebrow}>tool · {tc.tool}</div>
      <div style={kv}>
        <span style={kvKey}>Agent</span>
        <span style={kvVal}>{tc.agent}</span>
      </div>
      <div style={kv}>
        <span style={kvKey}>Status</span>
        <span style={kvVal}>{tc.status}</span>
      </div>
      <div style={kv}>
        <span style={kvKey}>Risk</span>
        <span style={kvVal}>{tc.risk ?? '—'}</span>
      </div>
      <div style={kv}>
        <span style={kvKey}>Args</span>
      </div>
      <pre
        style={{
          fontFamily: 'var(--ff-mono)',
          fontSize: 11,
          color: 'var(--ink-1)',
          background: 'var(--bg-page)',
          padding: 8,
          border: '1px solid var(--hair)',
          margin: 0,
          overflow: 'auto',
        }}
      >
        {JSON.stringify(tc.args, null, 2)}
      </pre>
    </>
  );
}

function MessageBody({ id }: { id: string | undefined }): ReactNode {
  return (
    <>
      <div style={eyebrow}>message · {id ?? '—'}</div>
      <div style={{ fontSize: 11, color: 'var(--ink-2)' }}>
        Message · {id ?? '—'}
      </div>
    </>
  );
}

export function SelectedPanel({ agentsByName, toolCalls }: SelectedPanelProps) {
  const selected = useSelected();
  let body: ReactNode;
  if (selected.kind === null) {
    body = <Empty />;
  } else if (selected.kind === 'agent') {
    const a = agentsByName[selected.id ?? ''];
    body = a ? <AgentBody agent={a} /> : <NotFound />;
  } else if (selected.kind === 'tool_call') {
    const found = toolCalls.find((t) => `${t.tool}@${t.ts}` === selected.id);
    body = found ? <ToolCallBody tc={found} /> : <NotFound />;
  } else {
    body = <MessageBody id={selected.id} />;
  }
  return (
    <Monitor title="Selected" pinned>
      {body}
    </Monitor>
  );
}
