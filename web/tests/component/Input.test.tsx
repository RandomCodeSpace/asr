import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '../_helpers/render';
import { Input } from '@/components/Input';

describe('<Input>', () => {
  it('renders an <input> element', () => {
    render(<Input placeholder="ask" />);
    expect(screen.getByPlaceholderText('ask')).toBeInTheDocument();
  });

  it('forwards value + onChange', () => {
    const onChange = vi.fn();
    render(<Input value="hi" onChange={onChange} readOnly />);
    expect(screen.getByDisplayValue('hi')).toBeInTheDocument();
  });

  it('passes data-size for size="sm"', () => {
    render(<Input size="sm" placeholder="x" />);
    expect(screen.getByPlaceholderText('x')).toHaveAttribute('data-size', 'sm');
  });

  it('fires onChange on input', () => {
    const onChange = vi.fn();
    render(<Input onChange={onChange} placeholder="x" />);
    fireEvent.change(screen.getByPlaceholderText('x'), { target: { value: 'a' } });
    expect(onChange).toHaveBeenCalledTimes(1);
  });
});
