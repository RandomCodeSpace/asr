import type { CSSProperties } from 'react';
import { Button } from '@/components/Button';
import { Icon } from '@/icons/Icon';

interface CanvasHeadProps {
  sessionId: string;
  status: string;
  openedAt: string;  // ISO UTC
  title: string;
  env: string;
  sev: number;
  reporter: string;
  turnCount: number;
  toolCount: number;
  agentsActive: number;
  agentsTotal: number;
  onStop: () => void;
  onRetry: () => void;
  /** When supplied, Retry is enabled iff retry.enabled; the reason is
   *  shown as the button's title tooltip. Without it, legacy heuristic
   *  applies (button visible only when status === 'error', always enabled). */
  retry?: { enabled: boolean; reason: string };
  /** Shown only when the session was flipped by the dedup pipeline
   *  (status === 'duplicate'). Operator-triggered correction. */
  onUnDuplicate?: () => void;
}

const wrap: CSSProperties = {
  padding: '20px 24px 12px',
  borderBottom: '1px solid var(--hair)',
  background: 'var(--bg-elev)',
};

const labelStyle: CSSProperties = {
  fontFamily: 'var(--ff-mono)',
  fontSize: 10,
  color: 'var(--ink-3)',
  letterSpacing: '0.14em',
  textTransform: 'uppercase',
};

function statusLabel(s: string): string {
  return s.replace(/_/g, ' ').toUpperCase();
}

function fmtOpenedAt(iso: string): string {
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  return d.toISOString().slice(11, 19);
}

function truncate(s: string, n: number): string {
  if (s.length <= n) return s;
  return s.slice(0, n) + '…';
}

export function CanvasHead(props: CanvasHeadProps) {
  const truncTitle = truncate(props.title, 80);
  const showRetry = props.retry !== undefined
    ? props.retry.enabled
    : props.status === 'error';
  const retryEnabled = props.retry ? props.retry.enabled : true;
  const retryReason = props.retry?.reason ?? '';
  const isDuplicate = props.status === 'duplicate';
  const isActive = props.status === 'in_progress' || props.status === 'new';
  const eyebrowParts = [
    props.sessionId,
    isActive ? 'ACTIVE' : statusLabel(props.status),
    `OPENED ${fmtOpenedAt(props.openedAt)} UTC`,
  ];
  return (
    <header style={wrap}>
      <div style={{ ...labelStyle, display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
        {isActive && (
          <span
            aria-hidden
            style={{
              display: 'inline-block', width: 6, height: 6,
              borderRadius: '50%', background: 'var(--acc)',
              animation: 'asr-pulse 2.4s ease-in-out infinite',
            }}
          />
        )}
        <span>{eyebrowParts.join(' · ')}</span>
      </div>
      <h1
        style={{
          margin: '0 0 8px',
          fontFamily: 'var(--ff-sans)',
          fontSize: 30,
          fontWeight: 500,
          letterSpacing: '-0.018em',
          color: 'var(--ink-1)',
          lineHeight: 1.2,
        }}
      >
        {truncTitle}
      </h1>
      <div
        style={{
          display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap',
          fontFamily: 'var(--ff-mono)', fontSize: 11, color: 'var(--ink-2)',
          letterSpacing: '0.06em', textTransform: 'uppercase',
        }}
      >
        <span>ENV {props.env}</span>
        <span style={{ color: 'var(--ink-4)' }}>·</span>
        <span>SEV {props.sev}</span>
        <span style={{ color: 'var(--ink-4)' }}>·</span>
        <span>REPORTER {props.reporter}</span>
        <span style={{ color: 'var(--ink-4)' }}>·</span>
        <span>TURNS {props.turnCount}</span>
        <span style={{ color: 'var(--ink-4)' }}>·</span>
        <span>TOOLS {props.toolCount}</span>
        <span style={{ color: 'var(--ink-4)' }}>·</span>
        <span>AGENTS {props.agentsActive} of {props.agentsTotal}</span>
        <span style={{ flex: 1 }} />
        <Button variant="ghost" size="sm" onClick={props.onStop}>
          <Icon name="stop" size={12} /> Stop
        </Button>
        {showRetry && (
          <Button
            variant="secondary"
            size="sm"
            onClick={props.onRetry}
            disabled={!retryEnabled}
            title={retryReason || undefined}
          >
            <Icon name="retry" size={12} /> Retry
          </Button>
        )}
        {isDuplicate && props.onUnDuplicate && (
          <Button variant="secondary" size="sm" onClick={props.onUnDuplicate}>
            Un-duplicate
          </Button>
        )}
      </div>
    </header>
  );
}
