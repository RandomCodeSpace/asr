import type {
  Session, AgentRun, ToolCall, AgentDefinition,
  SessionEvent, SessionFullBundle,
} from '@/api/types';

export interface SessionState {
  session: Session | null;
  agentsRun: AgentRun[];
  toolCalls: ToolCall[];
  events: SessionEvent[];
  agentDefinitions: Record<string, AgentDefinition>;
  vmSeq: number;
}

export const initialSessionState: SessionState = {
  session: null,
  agentsRun: [],
  toolCalls: [],
  events: [],
  agentDefinitions: {},
  vmSeq: 0,
};

type Action =
  | { type: 'bootstrap'; bundle: SessionFullBundle }
  | { type: 'event'; event: SessionEvent };

export function sessionReducer(state: SessionState, action: Action): SessionState {
  switch (action.type) {
    case 'bootstrap':
      return {
        session: action.bundle.session,
        agentsRun: action.bundle.agents_run,
        toolCalls: action.bundle.tool_calls,
        events: action.bundle.events,
        agentDefinitions: action.bundle.agent_definitions,
        vmSeq: action.bundle.vm_seq,
      };

    case 'event': {
      const ev = action.event;
      if (ev.seq <= state.vmSeq) return state;  // drop dupes / out-of-order
      const events = [...state.events, ev];
      let session = state.session;
      let agentsRun = state.agentsRun;
      let toolCalls = state.toolCalls;

      switch (ev.kind) {
        case 'agent_finished': {
          const p = ev.payload as Partial<AgentRun>;
          agentsRun = [...agentsRun, {
            agent: p.agent ?? '',
            started_at: p.started_at ?? '',
            ended_at: p.ended_at ?? ev.ts,
            summary: p.summary ?? '',
            confidence: p.confidence ?? null,
            confidence_rationale: p.confidence_rationale ?? null,
            signal: p.signal ?? null,
          }];
          break;
        }
        case 'tool_invoked': {
          const p = ev.payload as Partial<ToolCall>;
          toolCalls = [...toolCalls, {
            agent: p.agent ?? '',
            tool: p.tool ?? '',
            args: p.args ?? {},
            result: p.result ?? null,
            ts: ev.ts,
            risk: p.risk ?? null,
            status: 'executed',
            approver: null,
            approved_at: null,
            approval_rationale: null,
          }];
          break;
        }
        case 'approval_pending': {
          const p = ev.payload as Partial<ToolCall>;
          toolCalls = [...toolCalls, {
            agent: p.agent ?? '',
            tool: p.tool ?? '',
            args: p.args ?? {},
            result: null,
            ts: ev.ts,
            risk: p.risk ?? null,
            status: 'pending_approval',
            approver: null,
            approved_at: null,
            approval_rationale: null,
          }];
          break;
        }
        case 'status_changed': {
          const p = ev.payload as { status?: Session['status'] };
          if (session && p.status) session = { ...session, status: p.status };
          break;
        }
      }

      return { ...state, events, session, agentsRun, toolCalls, vmSeq: ev.seq };
    }
  }
}
