import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '../_helpers/render';
import { Monitor } from '@/monitors/Monitor';

describe('<Monitor>', () => {
  it('renders title + optional count', () => {
    render(
      <Monitor title="Selected" count={3} pinned>
        <div>body content</div>
      </Monitor>,
    );
    expect(screen.getByText('Selected')).toBeInTheDocument();
    expect(screen.getByText('3')).toBeInTheDocument();
    expect(screen.getByText('body content')).toBeInTheDocument();
  });

  it('omits count display when count is undefined', () => {
    render(
      <Monitor title="Health" pinned={false}>
        <div>body</div>
      </Monitor>,
    );
    expect(screen.getByText('Health')).toBeInTheDocument();
    expect(screen.queryByText(/^\d+$/)).not.toBeInTheDocument();
  });

  it('renders pinned dot when pinned=true', () => {
    const { container } = render(
      <Monitor title="X" pinned>
        <div>body</div>
      </Monitor>,
    );
    expect(container.firstChild).toHaveAttribute('data-pinned', 'true');
    expect(container.querySelector('[data-pinned-dot]')).not.toBeNull();
  });

  it('starts collapsed by default unless defaultOpen', () => {
    const { container, rerender } = render(
      <Monitor title="X" pinned={false}>
        <div>body</div>
      </Monitor>,
    );
    expect(container.firstChild).toHaveAttribute('data-collapsed', 'true');
    expect(screen.queryByText('body')).not.toBeInTheDocument();

    rerender(
      <Monitor title="X" pinned={false} defaultOpen>
        <div>body</div>
      </Monitor>,
    );
    expect(screen.getByText('body')).toBeInTheDocument();
  });

  it('toggles open/closed on header click', () => {
    const { container } = render(
      <Monitor title="X" pinned={false}>
        <div>body</div>
      </Monitor>,
    );
    expect(screen.queryByText('body')).not.toBeInTheDocument();
    fireEvent.click(screen.getByText('X'));
    expect(screen.getByText('body')).toBeInTheDocument();
    expect(container.firstChild).toHaveAttribute('data-collapsed', 'false');
  });
});
