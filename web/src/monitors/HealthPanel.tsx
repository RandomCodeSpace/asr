import { useQuery } from '@tanstack/react-query';
import { Monitor } from './Monitor';

interface Health {
  status: 'ok' | 'degraded' | 'down' | string;
  uptime_seconds?: number;
}

const dotColor: Record<string, string> = {
  ok: 'var(--good)',
  degraded: 'var(--warn)',
  down: 'var(--danger)',
};

function uptimeStr(sec: number | undefined): string | null {
  if (sec === undefined || sec === null) return null;
  if (sec < 60) return `${sec}s`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m`;
  if (sec < 86400) return `${Math.floor(sec / 3600)}h`;
  return `${Math.floor(sec / 86400)}d`;
}

export function HealthPanel() {
  const { data, dataUpdatedAt } = useQuery<Health>({
    queryKey: ['health'],
    // /health is at the server root (NOT /api/v1) — bypass apiFetch's prefix.
    queryFn: async () => {
      const r = await fetch('/health', { method: 'GET' });
      if (!r.ok) throw new Error('health check failed');
      return (await r.json()) as Health;
    },
    refetchInterval: 30_000,
  });

  const status = data?.status ?? 'unknown';
  const uptime = uptimeStr(data?.uptime_seconds);
  const lastPoll = dataUpdatedAt
    ? new Date(dataUpdatedAt).toLocaleTimeString(undefined, {
        hour12: false,
      })
    : null;

  return (
    <Monitor title="System Health" pinned={false}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
        <span
          data-health-dot
          data-status={status}
          aria-hidden
          style={{
            display: 'inline-block',
            width: 6,
            height: 6,
            borderRadius: '50%',
            background: dotColor[status] ?? 'var(--ink-3)',
          }}
        />
        <span style={{ fontFamily: 'var(--ff-sans)', fontSize: 12, color: 'var(--ink-1)' }}>
          {status}
        </span>
      </div>
      {uptime !== null && (
        <div style={{ fontFamily: 'var(--ff-mono)', fontSize: 11, color: 'var(--ink-3)' }}>
          uptime {uptime}
        </div>
      )}
      {lastPoll !== null && (
        <div style={{ fontFamily: 'var(--ff-mono)', fontSize: 11, color: 'var(--ink-3)' }}>
          last poll {lastPoll}
        </div>
      )}
    </Monitor>
  );
}
