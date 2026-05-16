import type { ToolCall } from '@/api/types';

const GENERIC = 'Allow {agent} to call {tool} (risk: {risk})?';

export function questionFromToolCall(
  tc: ToolCall,
  templates: Record<string, string>,
): string {
  const template = templates[tc.tool] ?? GENERIC;
  return interpolate(template, {
    agent: tc.agent,
    tool: tc.tool,
    risk: tc.risk ?? 'unknown',
    ...(tc.args as Record<string, unknown>),
  });
}

function interpolate(template: string, vars: Record<string, unknown>): string {
  return template.replace(/\{(\w+)\}/g, (match, key: string) => {
    const v = vars[key];
    if (v === undefined || v === null) return match;
    return String(v);
  });
}
