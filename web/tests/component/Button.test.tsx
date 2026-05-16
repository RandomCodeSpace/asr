import { describe, it, expect } from 'vitest';
import { render, screen } from '../_helpers/render';
import { Button } from '@/components/Button';

describe('<Button>', () => {
  it('renders the label text', () => {
    render(<Button>Click me</Button>);
    expect(screen.getByRole('button')).toHaveTextContent('Click me');
  });

  it('default variant uses ink-on-bg style (data-variant="primary")', () => {
    render(<Button>x</Button>);
    expect(screen.getByRole('button')).toHaveAttribute('data-variant', 'primary');
  });

  it('respects variant="secondary"', () => {
    render(<Button variant="secondary">x</Button>);
    expect(screen.getByRole('button')).toHaveAttribute('data-variant', 'secondary');
  });

  it('respects variant="ghost"', () => {
    render(<Button variant="ghost">x</Button>);
    expect(screen.getByRole('button')).toHaveAttribute('data-variant', 'ghost');
  });

  it('disabled state is non-interactive', () => {
    render(<Button disabled>x</Button>);
    expect(screen.getByRole('button')).toBeDisabled();
  });
});
