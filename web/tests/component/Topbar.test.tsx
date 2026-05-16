import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '../_helpers/render';
import { Topbar } from '@/shell/Topbar';

describe('<Topbar>', () => {
  it('renders brand mark + brand name', () => {
    render(
      <Topbar
        brandName="Acme Agents"
        appName="incident_management"
        envName="production"
        health="ok"
        approvalsCount={0}
        onSearch={() => {}}
        onNew={() => {}}
        onApprovalsClick={() => {}}
      />,
    );
    expect(screen.getByText('Acme Agents')).toBeInTheDocument();
    expect(screen.getByText('A')).toBeInTheDocument();  // brand mark letter
  });

  it('renders breadcrumb with app + env', () => {
    render(
      <Topbar
        brandName="X" appName="my_app" envName="staging"
        health="ok" approvalsCount={0}
        onSearch={() => {}} onNew={() => {}} onApprovalsClick={() => {}}
      />,
    );
    expect(screen.getByText('my_app')).toBeInTheDocument();
    expect(screen.getByText('staging')).toBeInTheDocument();
  });

  it('renders health pill text per state', () => {
    const { rerender } = render(
      <Topbar brandName="X" appName="a" envName="e" health="ok" approvalsCount={0}
              onSearch={() => {}} onNew={() => {}} onApprovalsClick={() => {}} />,
    );
    expect(screen.getByText(/All Systems Normal/)).toBeInTheDocument();
    rerender(
      <Topbar brandName="X" appName="a" envName="e" health="degraded" approvalsCount={0}
              onSearch={() => {}} onNew={() => {}} onApprovalsClick={() => {}} />,
    );
    expect(screen.getByText(/Degraded/)).toBeInTheDocument();
    rerender(
      <Topbar brandName="X" appName="a" envName="e" health="down" approvalsCount={0}
              onSearch={() => {}} onNew={() => {}} onApprovalsClick={() => {}} />,
    );
    expect(screen.getByText(/Critical/)).toBeInTheDocument();
  });

  it('shows approvals badge when count > 0', () => {
    render(
      <Topbar brandName="X" appName="a" envName="e" health="ok" approvalsCount={2}
              onSearch={() => {}} onNew={() => {}} onApprovalsClick={() => {}} />,
    );
    const badge = screen.getByText(/2 Pending Approvals/);
    expect(badge).toBeInTheDocument();
  });

  it('hides approvals badge when count === 0', () => {
    render(
      <Topbar brandName="X" appName="a" envName="e" health="ok" approvalsCount={0}
              onSearch={() => {}} onNew={() => {}} onApprovalsClick={() => {}} />,
    );
    expect(screen.queryByText(/Pending Approvals/)).not.toBeInTheDocument();
  });

  it('fires onNew when New Session button clicked', () => {
    const onNew = vi.fn();
    render(
      <Topbar brandName="X" appName="a" envName="e" health="ok" approvalsCount={0}
              onSearch={() => {}} onNew={onNew} onApprovalsClick={() => {}} />,
    );
    fireEvent.click(screen.getByRole('button', { name: /New Session/i }));
    expect(onNew).toHaveBeenCalledTimes(1);
  });

  it('fires onSearch when search box clicked', () => {
    const onSearch = vi.fn();
    render(
      <Topbar brandName="X" appName="a" envName="e" health="ok" approvalsCount={0}
              onSearch={onSearch} onNew={() => {}} onApprovalsClick={() => {}} />,
    );
    fireEvent.click(screen.getByPlaceholderText(/Search sessions/i));
    expect(onSearch).toHaveBeenCalledTimes(1);
  });
});
