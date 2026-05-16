// web/src/api/types.ts
// Hand-authored to match /api/v1/* schemas. v2.1 should generate from OpenAPI.

export type SessionId = string;  // SES-YYYYMMDD-NNN

export interface Session {
  id: SessionId;
  status: 'in_progress' | 'awaiting_input' | 'matched' | 'resolved' | 'escalated' | 'stopped' | 'error' | 'new';
  created_at: string;  // ISO UTC
  updated_at: string;
  deleted_at: string | null;
  agents_run: AgentRun[];
  tool_calls: ToolCall[];
  findings: Record<string, unknown>;
  pending_intervention: Record<string, unknown> | null;
  user_inputs: string[];
  parent_session_id: SessionId | null;
  dedup_rationale: string | null;
  extra_fields: Record<string, unknown>;
  version: number;
}

export interface AgentRun {
  agent: string;
  started_at: string;
  ended_at: string;
  summary: string;
  token_usage?: { input_tokens: number; output_tokens: number; total_tokens: number };
  confidence: number | null;
  confidence_rationale: string | null;
  signal: string | null;
}

export interface ToolCall {
  agent: string;
  tool: string;
  args: Record<string, unknown>;
  result: unknown;
  ts: string;
  risk: 'low' | 'medium' | 'high' | null;
  status: 'executed' | 'executed_with_notify' | 'pending_approval' | 'approved' | 'rejected' | 'timeout' | 'auto_rejected';
  approver: string | null;
  approved_at: string | null;
  approval_rationale: string | null;
}

export interface AgentDefinition {
  name: string;
  kind: string;  // 'responsive' | 'gated' | etc.
  model: string;
  tools: string[];
  routes: Record<string, string>;
  system_prompt_excerpt: string;
}

export interface SessionEvent {
  seq: number;
  kind: string;
  payload: Record<string, unknown>;
  ts: string;
  session_id?: string;  // present on /sessions/recent/events stream
}

export interface SessionFullBundle {
  session: Session;
  agents_run: AgentRun[];
  tool_calls: ToolCall[];
  events: SessionEvent[];
  agent_definitions: Record<string, AgentDefinition>;
  vm_seq: number;
}

export interface UiHints {
  brand_name: string;
  brand_logo_url: string | null;
  approval_rationale_templates: string[];
  hitl_question_templates: Record<string, string>;
  environments: string[];
}

export interface AppView {
  id: string;
  title: string;
  applies_to: string;  // 'always' | 'agent:NAME' | 'tool:NAME'
  url: string;
}

export interface ApiError {
  error: { code: string; message: string; details: Record<string, unknown> };
}
