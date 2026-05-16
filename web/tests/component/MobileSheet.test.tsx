import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '../_helpers/render';
import { MobileSheet } from '@/shell/MobileSheet';

describe('<MobileSheet>', () => {
  it('does not render content when closed', () => {
    render(
      <MobileSheet open={false} onOpenChange={() => {}} title="Sessions">
        body content
      </MobileSheet>,
    );
    expect(screen.queryByText('body content')).not.toBeInTheDocument();
  });

  it('renders title + body + handle when open', () => {
    render(
      <MobileSheet open onOpenChange={() => {}} title="Sessions" testId="sessions">
        body content
      </MobileSheet>,
    );
    expect(screen.getByText('Sessions')).toBeInTheDocument();
    expect(screen.getByText('body content')).toBeInTheDocument();
    expect(document.querySelector('[data-mobile-sheet="sessions"]')).toBeInTheDocument();
  });

  it('calls onOpenChange(false) when Close is tapped', () => {
    const onOpenChange = vi.fn();
    render(
      <MobileSheet open onOpenChange={onOpenChange} title="X">
        body
      </MobileSheet>,
    );
    fireEvent.click(screen.getByRole('button', { name: /close/i }));
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });
});
