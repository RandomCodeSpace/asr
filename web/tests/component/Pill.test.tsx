import { describe, it, expect } from 'vitest';
import { render, screen } from '../_helpers/render';
import { Pill } from '@/components/Pill';

describe('<Pill>', () => {
  it('renders SHORT_CAPS label text', () => {
    render(<Pill kind="running">Running</Pill>);
    // Implementation transforms via CSS text-transform: uppercase;
    // the underlying text node should still contain the literal children.
    expect(screen.getByText('Running')).toBeInTheDocument();
  });

  it('exposes the kind via data-kind', () => {
    render(<Pill kind="error">Error</Pill>);
    expect(screen.getByText('Error').closest('[data-kind]')).toHaveAttribute('data-kind', 'error');
  });

  it.each(['running', 'paused', 'error', 'resolved', 'neutral'] as const)(
    'accepts kind="%s"',
    (kind) => {
      render(<Pill kind={kind}>label</Pill>);
      expect(screen.getByText('label').closest('[data-kind]')).toHaveAttribute('data-kind', kind);
    },
  );
});
