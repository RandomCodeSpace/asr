import { useState, useMemo } from 'react';
import type { CSSProperties } from 'react';
import { useSessionFull } from '@/state/useSessionFull';
import { useSetSelected } from '@/state/selectedRef';
import { useUiHints } from '@/state/useUiHints';
import { CanvasHead } from './CanvasHead';
import { Transcript, type HITLContext } from './Transcript';
import { ApproveRationaleModal } from '@/modals/ApproveRationaleModal';
import { apiFetch } from '@/api/client';
import { questionFromToolCall } from '@/lib/hitl/questionFromToolCall';
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

function findPendingApproval(toolCalls: ToolCall[]): { toolCall: ToolCall; idx: number } | null {
  for (let i = 0; i < toolCalls.length; i++) {
    const tc = toolCalls[i];
    if (tc && tc.status === 'pending_approval') return { toolCall: tc, idx: i };
  }
  return null;
}

export function SessionCanvas({ activeSid }: SessionCanvasProps) {
  const { state, isLoading, error, refresh } = useSessionFull(activeSid);
  const setSelected = useSetSelected();
  const uiHints = useUiHints();
  const [rationaleOpen, setRationaleOpen] = useState(false);

  const pending = useMemo(() => findPendingApproval(state.toolCalls), [state.toolCalls]);

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

  const hitlContext: HITLContext | null = pending
    ? {
        toolCall: pending.toolCall,
        waitedSeconds: Math.max(
          0,
          Math.round((Date.now() - new Date(pending.toolCall.ts).getTime()) / 1000),
        ),
        question: questionFromToolCall(
          pending.toolCall,
          uiHints.data?.hitl_question_templates ?? {},
        ),
        confidence: state.agentsRun.at(-1)?.confidence ?? null,
        turn: state.agentsRun.length || 1,
        requestedBy: pending.toolCall.agent,
        policy: pending.toolCall.risk
          ? `risk=${pending.toolCall.risk}`
          : 'risk=unknown',
      }
    : null;

  const sessId = sess.id;
  async function handleApprove() {
    if (!pending) return;
    await apiFetch(`/sessions/${sessId}/approvals/${pending.idx}`, {
      method: 'POST',
      json: { decision: 'approve', approver: 'operator', rationale: null },
    });
    refresh();
  }

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
        onStop={() => { /* Phase 6 / Task 53 confirm modal */ }}
        onRetry={refresh}
      />
      <Transcript
        agentsRun={state.agentsRun}
        toolCalls={state.toolCalls}
        activeAgent={null}
        hitlContext={hitlContext}
        onSelectTool={onSelectTool}
        onApprove={() => { void handleApprove(); }}
        onReject={() => { /* Task 53: confirm modal */ }}
        onApproveWithRationale={() => { if (pending) setRationaleOpen(true); }}
      />
      {pending && (
        <ApproveRationaleModal
          open={rationaleOpen}
          onOpenChange={setRationaleOpen}
          sessionId={sess.id}
          toolCallId={String(pending.idx)}
          templates={uiHints.data?.approval_rationale_templates ?? []}
          onApproved={refresh}
        />
      )}
    </div>
  );
}
