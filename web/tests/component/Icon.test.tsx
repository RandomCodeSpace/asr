import { describe, it, expect } from 'vitest';
import { render, screen } from '../_helpers/render';
import { Icon } from '@/icons/Icon';
import { IconSprite } from '@/icons/sprite';

describe('<Icon>', () => {
  it('renders the named SVG via <use>', () => {
    render(<><IconSprite /><Icon name="check" /></>);
    const svg = screen.getByRole('img', { hidden: true });
    expect(svg.querySelector('use')?.getAttribute('href')).toBe('#i-check');
  });

  it('inherits currentColor', () => {
    render(<><IconSprite /><Icon name="x" /></>);
    const svg = screen.getByRole('img', { hidden: true });
    expect(svg.getAttribute('width')).toBe('14');
  });
});
