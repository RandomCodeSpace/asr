import type { CSSProperties } from 'react';
import type { AgentRun, ToolCall } from '@/api/types';
import { Turn } from './Turn';
import { HITLBand } from './HITLBand';

export interface ActiveAgentSnapshot {
  name: string;
  startedAt: string;
  currentBody: string;
}

export interface HITLContext {
  toolCall: ToolCall;
  waitedSeconds: number;
  question: string;
  confidence: number | null;
  turn: number;
  requestedBy: string;
  policy: string;
}

interface TranscriptProps {
  agentsRun: AgentRun[];
  toolCalls: ToolCall[];
  activeAgent: ActiveAgentSnapshot | null;
  hitlContext: HITLContext | null;
  onSelectTool: (tc: ToolCall) => void;
  onApprove: () => void;
  onReject: () => void;
  onApproveWithRationale: () => void;
}

const emptyStyle: CSSProperties = {
  padding: 40,
  textAlign: 'center',
  fontFamily: 'var(--ff-sans)',
  fontSize: 13,
  color: 'var(--ink-3)',
};

function durationMs(startIso: string, endIso: string): number {
  const s = new Date(startIso).getTime();
  const e = new Date(endIso).getTime();
  if (isNaN(s) || isNaN(e)) return 0;
  return Math.max(0, e - s);
}

function elapsedFromOpening(openingIso: string, atIso: string): number {
  const o = new Date(openingIso).getTime();
  const a = new Date(atIso).getTime();
  if (isNaN(o) || isNaN(a)) return 0;
  return Math.max(0, a - o);
}

function groupToolsByAgent(toolCalls: ToolCall[]): Map<string, ToolCall[]> {
  const m = new Map<string, ToolCall[]>();
  for (const tc of toolCalls) {
    const arr = m.get(tc.agent) ?? [];
    arr.push(tc);
    m.set(tc.agent, arr);
  }
  return m;
}

export function Transcript({
  agentsRun, toolCalls, activeAgent, hitlContext,
  onSelectTool, onApprove, onReject, onApproveWithRationale,
}: TranscriptProps) {
  const empty = agentsRun.length === 0 && activeAgent === null && hitlContext === null;
  if (empty) {
    return <div style={emptyStyle}>No turns yet.</div>;
  }

  const toolsByAgent = groupToolsByAgent(toolCalls);
  const opening = agentsRun[0]?.started_at ?? activeAgent?.startedAt ?? '';

  return (
    <div style={{ display: 'flex', flexDirection: 'column' }}>
      {agentsRun.map((run, i) => (
        <Turn
          key={`${run.agent}-${run.started_at}-${i}`}
          agent={run.agent}
          timestamp={run.started_at}
          elapsedMs={elapsedFromOpening(opening, run.started_at)}
          body={run.summary}
          confidence={run.confidence}
          model={run.token_usage ? '—' : '—'}
          durationMs={durationMs(run.started_at, run.ended_at)}
          turn={i + 1}
          toolCalls={toolsByAgent.get(run.agent) ?? []}
          active={false}
          onSelectTool={onSelectTool}
        />
      ))}
      {activeAgent && (
        <Turn
          agent={activeAgent.name}
          timestamp={activeAgent.startedAt}
          elapsedMs={elapsedFromOpening(opening, activeAgent.startedAt)}
          body={activeAgent.currentBody}
          confidence={null}
          model="—"
          durationMs={0}
          turn={agentsRun.length + 1}
          toolCalls={toolsByAgent.get(activeAgent.name) ?? []}
          active={true}
          onSelectTool={onSelectTool}
        />
      )}
      {hitlContext && (
        <HITLBand
          toolCall={hitlContext.toolCall}
          waitedSeconds={hitlContext.waitedSeconds}
          question={hitlContext.question}
          confidence={hitlContext.confidence}
          turn={hitlContext.turn}
          requestedBy={hitlContext.requestedBy}
          policy={hitlContext.policy}
          onApprove={onApprove}
          onReject={onReject}
          onApproveWithRationale={onApproveWithRationale}
        />
      )}
    </div>
  );
}
