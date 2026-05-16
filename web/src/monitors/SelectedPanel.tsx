import type { CSSProperties, ReactNode } from 'react';
import type { AgentDefinition, AppView, ToolCall } from '@/api/types';
import { useSelected } from '@/state/selectedRef';
import { useAppViews, filterAppViews } from '@/state/useAppViews';
import { Monitor } from './Monitor';

interface SelectedPanelProps {
  agentsByName: Record<string, AgentDefinition>;
  toolCalls: ToolCall[];
  /** App identifier used to fetch overlay views (Approach C).
   *  v2 has one app per deploy; the backend ignores the value today. */
  appName?: string;
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

function AppViewsBlock({ views }: { views: AppView[] }): ReactNode {
  if (views.length === 0) return null;
  return (
    <div style={{ marginTop: 12, paddingTop: 12, borderTop: '1px solid var(--hair)' }}>
      <div style={{ ...eyebrow, marginBottom: 6 }}>App-specific views →</div>
      <ul style={{ margin: 0, padding: 0, listStyle: 'none' }}>
        {views.map((v) => (
          <li key={v.id} style={{ marginBottom: 4 }}>
            <a
              href={v.url}
              target="_blank"
              rel="noopener noreferrer"
              data-app-view={v.id}
              style={{
                fontFamily: 'var(--ff-sans)',
                fontSize: 12,
                color: 'var(--acc)',
                textDecoration: 'underline',
                textUnderlineOffset: 2,
              }}
            >
              {v.title}
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function SelectedPanel({ agentsByName, toolCalls, appName = 'runtime' }: SelectedPanelProps) {
  const selected = useSelected();
  const appViews = useAppViews(appName);
  const matchingViews = filterAppViews(appViews.data ?? [], {
    kind: selected.kind,
    id: selected.id ?? null,
  });
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
      {selected.kind !== null && <AppViewsBlock views={matchingViews} />}
    </Monitor>
  );
}
