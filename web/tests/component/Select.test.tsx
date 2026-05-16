import { describe, it, expect } from 'vitest';
import { render, screen } from '../_helpers/render';
import { Select } from '@/components/Select';

describe('<Select>', () => {
  it('renders the trigger with the placeholder when no value', () => {
    render(
      <Select placeholder="Choose…" options={[{ value: 'a', label: 'Alpha' }]} />,
    );
    expect(screen.getByText('Choose…')).toBeInTheDocument();
  });

  it('exposes the value via the trigger when value is set', () => {
    render(
      <Select value="a" placeholder="Choose…" options={[{ value: 'a', label: 'Alpha' }]} />,
    );
    expect(screen.getByText('Alpha')).toBeInTheDocument();
  });
});
