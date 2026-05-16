import type { CSSProperties } from 'react';
import { useSessionFull } from '@/state/useSessionFull';
import { useSetSelected } from '@/state/selectedRef';
import { CanvasHead } from './CanvasHead';
import { Transcript } from './Transcript';
import type { ToolCall } from '@/api/types';

interface SessionCanvasProps {
  activeSid: string | null;
}

const wrap: CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  background: 'var(--bg-elev)',
  borderRight: '1px solid var(--hair)',
  minHeight: 0,
  overflowY: 'auto',
};

const center: CSSProperties = {
  flex: 1,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  padding: 32,
  fontFamily: 'var(--ff-sans)',
  fontSize: 13,
  color: 'var(--ink-3)',
};

function extraStr(extras: Record<string, unknown>, key: string, fallback: string): string {
  const v = extras[key];
  return v === undefined || v === null ? fallback : String(v);
}

function extraNum(extras: Record<string, unknown>, key: string, fallback: number): number {
  const v = extras[key];
  if (typeof v === 'number') return v;
  return fallback;
}

export function SessionCanvas({ activeSid }: SessionCanvasProps) {
  const { state, isLoading, error, refresh } = useSessionFull(activeSid);
  const setSelected = useSetSelected();

  if (activeSid === null) {
    return (
      <div style={wrap}>
        <div style={center}>Select a session from the rail or create a new one.</div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div style={wrap}>
        <div style={center}>Loading…</div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={wrap}>
        <div style={center}>
          <div>
            <div style={{ color: 'var(--danger)', marginBottom: 8 }}>
              {error.message || 'Error loading session'}
            </div>
            <button
              type="button"
              onClick={refresh}
              style={{
                padding: '6px 12px',
                fontFamily: 'var(--ff-sans)',
                fontSize: 12,
                color: 'var(--ink-1)',
                background: 'var(--bg-elev)',
                border: '1px solid var(--hair-strong)',
                borderRadius: 0,
                cursor: 'pointer',
              }}
            >
              Retry
            </button>
          </div>
        </div>
      </div>
    );
  }

  const sess = state.session;
  if (!sess) {
    return <div style={wrap}><div style={center}>Session not found.</div></div>;
  }

  const findings = sess.findings as Record<string, unknown>;
  const title = (findings.title as string | undefined) ?? sess.id;
  const env = extraStr(sess.extra_fields, 'env', 'dev');
  const sev = extraNum(sess.extra_fields, 'sev', 0);
  const reporter = extraStr(sess.extra_fields, 'reporter', '—');

  const onSelectTool = (tc: ToolCall) => {
    setSelected({ kind: 'tool_call', id: `${tc.tool}@${tc.ts}` });
  };

  return (
    <div style={wrap}>
      <CanvasHead
        sessionId={sess.id}
        status={sess.status}
        openedAt={sess.created_at}
        title={title}
        env={env}
        sev={sev}
        reporter={reporter}
        turnCount={state.agentsRun.length}
        toolCount={state.toolCalls.length}
        agentsActive={state.agentsRun.length}
        agentsTotal={Object.keys(state.agentDefinitions).length || state.agentsRun.length}
        onStop={() => { /* Phase 6: confirm modal */ }}
        onRetry={refresh}
      />
      <Transcript
        agentsRun={state.agentsRun}
        toolCalls={state.toolCalls}
        activeAgent={null}
        hitlContext={null}
        onSelectTool={onSelectTool}
        onApprove={() => { /* Phase 6 */ }}
        onReject={() => { /* Phase 6 */ }}
        onApproveWithRationale={() => { /* Phase 6 */ }}
      />
    </div>
  );
}
