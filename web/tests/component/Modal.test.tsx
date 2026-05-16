import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '../_helpers/render';
import { Modal } from '@/components/Modal';

describe('<Modal>', () => {
  it('renders nothing when closed', () => {
    render(
      <Modal open={false} onOpenChange={() => {}} title="Hi">
        body content
      </Modal>,
    );
    expect(screen.queryByText('body content')).not.toBeInTheDocument();
  });

  it('renders title + body when open', () => {
    render(
      <Modal open onOpenChange={() => {}} eyebrow="NEW THING" title="Create something">
        body content
      </Modal>,
    );
    expect(screen.getByText('NEW THING')).toBeInTheDocument();
    expect(screen.getByText('Create something')).toBeInTheDocument();
    expect(screen.getByText('body content')).toBeInTheDocument();
  });

  it('calls onOpenChange(false) when close button clicked', () => {
    const onOpenChange = vi.fn();
    render(
      <Modal open onOpenChange={onOpenChange} title="Hi">
        body
      </Modal>,
    );
    screen.getByRole('button', { name: /close/i }).click();
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });

  it('renders footer Cancel + primary action when provided', () => {
    const onPrimary = vi.fn();
    render(
      <Modal
        open
        onOpenChange={() => {}}
        title="Hi"
        primaryAction={{ label: 'Submit', onClick: onPrimary }}
      >
        body
      </Modal>,
    );
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeInTheDocument();
    const submit = screen.getByRole('button', { name: 'Submit' });
    submit.click();
    expect(onPrimary).toHaveBeenCalledTimes(1);
  });
});
