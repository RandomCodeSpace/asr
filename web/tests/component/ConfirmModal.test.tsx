import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '../_helpers/render';
import { ConfirmModal } from '@/modals/ConfirmModal';

describe('<ConfirmModal>', () => {
  it('does not render when closed', () => {
    render(
      <ConfirmModal
        open={false} onOpenChange={() => {}}
        title="T" body="b" onConfirm={() => {}}
      />,
    );
    expect(screen.queryByText('T')).not.toBeInTheDocument();
  });

  it('renders title + body + default eyebrow', () => {
    render(
      <ConfirmModal
        open onOpenChange={() => {}}
        title="Stop session?" body="The session will be cancelled."
        onConfirm={() => {}}
      />,
    );
    expect(screen.getByText('Stop session?')).toBeInTheDocument();
    expect(screen.getByText(/the session will be cancelled/i)).toBeInTheDocument();
    expect(screen.getByText('CONFIRM')).toBeInTheDocument();
  });

  it('uses destructive eyebrow + destructive button styling when destructive', () => {
    render(
      <ConfirmModal
        open onOpenChange={() => {}}
        title="Drop table?" body="bye" destructive
        confirmLabel="Drop"
        onConfirm={() => {}}
      />,
    );
    expect(screen.getByText(/CONFIRM \(DESTRUCTIVE\)/)).toBeInTheDocument();
    const btn = screen.getByRole('button', { name: 'Drop' });
    expect(btn).toHaveAttribute('data-destructive', 'true');
  });

  it('calls onConfirm + closes on confirm click', async () => {
    const onConfirm = vi.fn();
    const onOpenChange = vi.fn();
    render(
      <ConfirmModal
        open onOpenChange={onOpenChange}
        title="T" body="b" confirmLabel="Yes" onConfirm={onConfirm}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: 'Yes' }));
    await waitFor(() => expect(onConfirm).toHaveBeenCalled());
    await waitFor(() => expect(onOpenChange).toHaveBeenCalledWith(false));
  });

  it('shows error from rejected onConfirm and keeps modal open', async () => {
    const onConfirm = vi.fn(() => Promise.reject(new Error('boom')));
    const onOpenChange = vi.fn();
    render(
      <ConfirmModal
        open onOpenChange={onOpenChange}
        title="T" body="b" confirmLabel="Go" onConfirm={onConfirm}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: 'Go' }));
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('boom'));
    expect(onOpenChange).not.toHaveBeenCalledWith(false);
  });

  it('honours custom eyebrow when supplied', () => {
    render(
      <ConfirmModal
        open onOpenChange={() => {}}
        title="T" eyebrow="REJECT TOOL CALL" body="b" onConfirm={() => {}}
      />,
    );
    expect(screen.getByText('REJECT TOOL CALL')).toBeInTheDocument();
  });
});
