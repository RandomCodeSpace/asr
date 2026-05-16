import { describe, it, expect } from 'vitest';
import { filterAppViews } from '@/state/useAppViews';
import type { AppView } from '@/api/types';

const views: AppView[] = [
  { id: 'always-1', title: 'Runbook', applies_to: 'always', url: 'https://x' },
  { id: 'agent-intake', title: 'Intake skill prompt', applies_to: 'agent:intake', url: 'https://y' },
  { id: 'tool-search-logs', title: 'Loki query', applies_to: 'tool:search_logs', url: 'https://z' },
];

describe('filterAppViews', () => {
  it('returns empty when no views', () => {
    expect(filterAppViews([], { kind: 'agent', id: 'intake' })).toEqual([]);
  });

  it("always-views match any selection", () => {
    expect(filterAppViews(views, { kind: 'agent', id: 'intake' }).map((v) => v.id)).toContain('always-1');
    expect(filterAppViews(views, { kind: 'tool_call', id: 'search_logs@now' }).map((v) => v.id)).toContain('always-1');
  });

  it('agent:NAME matches only when selected.kind=agent and id=NAME', () => {
    const m = filterAppViews(views, { kind: 'agent', id: 'intake' }).map((v) => v.id);
    expect(m).toContain('agent-intake');
    const n = filterAppViews(views, { kind: 'agent', id: 'triage' }).map((v) => v.id);
    expect(n).not.toContain('agent-intake');
  });

  it('tool:NAME matches when selected.id parsed as `tool@ts` has tool=NAME', () => {
    const m = filterAppViews(views, { kind: 'tool_call', id: 'search_logs@2026-05-16T11:00:00Z' }).map((v) => v.id);
    expect(m).toContain('tool-search-logs');
    const n = filterAppViews(views, { kind: 'tool_call', id: 'create_ticket@2026-05-16T11:00:00Z' }).map((v) => v.id);
    expect(n).not.toContain('tool-search-logs');
  });

  it('falls through unknown applies_to scopes without throwing', () => {
    const odd: AppView[] = [{ id: 'oddly', title: 'X', applies_to: 'plugin:foo', url: '' }];
    expect(filterAppViews(odd, { kind: 'agent', id: 'intake' })).toEqual([]);
  });
});
