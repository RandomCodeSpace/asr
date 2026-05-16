import { describe, it, expect } from 'vitest';
import { render, screen } from '../_helpers/render';
import { FlowStrip } from '@/shell/FlowStrip';
import type { AgentDefinition } from '@/api/types';

const agents: AgentDefinition[] = [
  { name: 'intake', kind: 'responsive', model: 'gpt', tools: [], routes: { success: 'triage' }, system_prompt_excerpt: '' },
  { name: 'triage', kind: 'gated', model: 'claude', tools: ['obs:get_logs'], routes: { success: 'investigate' }, system_prompt_excerpt: '' },
  { name: 'investigate', kind: 'gated', model: 'claude', tools: ['rem:propose_fix'], routes: {}, system_prompt_excerpt: '' },
];

describe('<FlowStrip>', () => {
  it('renders the runtime label with agent count + graph version', () => {
    render(<FlowStrip agents={agents} activeAgent={null} graphVersion="v1.5.2" />);
    expect(screen.getByText(/3 agents/)).toBeInTheDocument();
    expect(screen.getByText(/v1\.5\.2/)).toBeInTheDocument();
  });

  it('renders one node per agent', () => {
    const { container } = render(<FlowStrip agents={agents} activeAgent={null} graphVersion="x" />);
    const nodes = container.querySelectorAll('[data-flow-node]');
    expect(nodes).toHaveLength(3);
  });

  it('marks the active agent with data-active="true"', () => {
    render(<FlowStrip agents={agents} activeAgent="triage" graphVersion="x" />);
    const activeNode = document.querySelector('[data-flow-node="triage"]');
    expect(activeNode).toHaveAttribute('data-active', 'true');
    const idleNode = document.querySelector('[data-flow-node="intake"]');
    expect(idleNode).toHaveAttribute('data-active', 'false');
  });

  it('renders agent names', () => {
    render(<FlowStrip agents={agents} activeAgent={null} graphVersion="x" />);
    expect(screen.getByText('intake')).toBeInTheDocument();
    expect(screen.getByText('triage')).toBeInTheDocument();
    expect(screen.getByText('investigate')).toBeInTheDocument();
  });

  it('renders empty state when agents=[]', () => {
    render(<FlowStrip agents={[]} activeAgent={null} graphVersion="x" />);
    expect(screen.getByText(/No agents loaded/i)).toBeInTheDocument();
  });
});
