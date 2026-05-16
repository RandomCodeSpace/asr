import { describe, it, expect } from 'vitest';
import { render, screen } from '../_helpers/render';
import { Tag } from '@/components/Tag';

describe('<Tag>', () => {
  it('renders children text', () => {
    render(<Tag>hello</Tag>);
    expect(screen.getByText('hello')).toBeInTheDocument();
  });

  it('defaults to variant="default"', () => {
    render(<Tag>x</Tag>);
    expect(screen.getByText('x').closest('[data-variant]')).toHaveAttribute('data-variant', 'default');
  });

  it('accepts variant="mono"', () => {
    render(<Tag variant="mono">slack.post_message</Tag>);
    expect(screen.getByText('slack.post_message').closest('[data-variant]')).toHaveAttribute('data-variant', 'mono');
  });
});
