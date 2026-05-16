import type { CSSProperties } from 'react';
import { Icon } from '@/icons/Icon';
import { Button } from '@/components/Button';

export type Health = 'ok' | 'degraded' | 'down';

interface TopbarProps {
  brandName: string;
  appName: string;
  envName: string;
  health: Health;
  approvalsCount: number;
  user?: string;
  onSearch: () => void;
  onNew: () => void;
  onApprovalsClick: () => void;
}

const containerStyle: CSSProperties = {
  height: 48,
  display: 'flex',
  alignItems: 'center',
  gap: 12,
  padding: '0 16px',
  background: 'var(--bg-elev)',
  borderBottom: '1px solid var(--hair)',
  fontFamily: 'var(--ff-sans)',
};

const sep = (
  <span style={{ width: 1, height: 18, background: 'var(--hair)' }} aria-hidden />
);

const healthLabel: Record<Health, string> = {
  ok: 'All Systems Normal',
  degraded: 'Degraded',
  down: 'Critical',
};

const healthColor: Record<Health, string> = {
  ok: 'var(--good)',
  degraded: 'var(--warn)',
  down: 'var(--danger)',
};

export function Topbar({
  brandName, appName, envName, health, approvalsCount,
  user = 'Operator', onSearch, onNew, onApprovalsClick,
}: TopbarProps) {
  const initial = (brandName[0] ?? 'A').toUpperCase();
  return (
    <header style={containerStyle}>
      <div style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>
        <span
          aria-hidden
          style={{
            width: 22,
            height: 22,
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'var(--ink-1)',
            color: 'var(--bg-elev)',
            fontSize: 12,
            fontWeight: 600,
            fontFamily: 'var(--ff-sans)',
          }}
        >
          {initial}
        </span>
        <span style={{ fontSize: 13, fontWeight: 500, color: 'var(--ink-1)' }}>{brandName}</span>
      </div>
      {sep}
      <div style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--ink-2)' }}>
        <span>{appName}</span>
        <span style={{ color: 'var(--ink-4)' }}>/</span>
        <span>{envName}</span>
      </div>
      {sep}
      <div style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 11, color: 'var(--ink-2)' }}>
        <span
          aria-hidden
          style={{
            display: 'inline-block',
            width: 6,
            height: 6,
            borderRadius: '50%',
            background: healthColor[health],
          }}
        />
        {healthLabel[health]}
      </div>
      <span style={{ flex: 1 }} />
      <div
        onClick={onSearch}
        style={{
          width: 220,
          height: 28,
          display: 'inline-flex',
          alignItems: 'center',
          gap: 8,
          padding: '0 10px',
          background: 'var(--bg-page)',
          border: '1px solid var(--hair)',
          cursor: 'pointer',
        }}
      >
        <Icon name="search" size={12} />
        <input
          type="text"
          readOnly
          placeholder="Search sessions, agents, tools…"
          style={{
            flex: 1,
            border: 'none',
            background: 'transparent',
            fontFamily: 'var(--ff-sans)',
            fontSize: 12,
            color: 'var(--ink-2)',
            outline: 'none',
            cursor: 'pointer',
          }}
        />
      </div>
      {approvalsCount > 0 && (
        <button
          type="button"
          onClick={onApprovalsClick}
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: 6,
            height: 26,
            padding: '0 10px',
            fontFamily: 'var(--ff-sans)',
            fontSize: 11,
            fontWeight: 500,
            color: 'var(--warn)',
            background: 'var(--warn-bg)',
            border: '1px solid var(--warn)',
            borderRadius: 0,
            cursor: 'pointer',
          }}
        >
          <span
            aria-hidden
            style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--warn)' }}
          />
          {approvalsCount} Pending Approvals
        </button>
      )}
      <Button onClick={onNew} variant="primary" size="sm">
        <Icon name="plus" size={12} />
        New Session
      </Button>
      {sep}
      <span style={{ fontSize: 11, color: 'var(--ink-2)' }}>{user}</span>
    </header>
  );
}
