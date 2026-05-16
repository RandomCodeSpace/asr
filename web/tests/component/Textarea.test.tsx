import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '../_helpers/render';
import { Textarea } from '@/components/Textarea';

describe('<Textarea>', () => {
  it('renders a <textarea>', () => {
    render(<Textarea placeholder="notes" />);
    expect(screen.getByPlaceholderText('notes').tagName).toBe('TEXTAREA');
  });

  it('fires onChange', () => {
    const onChange = vi.fn();
    render(<Textarea onChange={onChange} placeholder="x" />);
    fireEvent.change(screen.getByPlaceholderText('x'), { target: { value: 'a' } });
    expect(onChange).toHaveBeenCalled();
  });
});
