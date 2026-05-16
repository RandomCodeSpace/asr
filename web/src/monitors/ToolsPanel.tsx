import { useQuery } from '@tanstack/react-query';
import { apiFetch } from '@/api/client';
import { Monitor } from './Monitor';

interface Tool {
  name: string;
  description?: string;
  risk?: 'low' | 'medium' | 'high';
}

const riskColor: Record<string, string> = {
  low: 'var(--good)',
  medium: 'var(--warn)',
  high: 'var(--danger)',
};

export function ToolsPanel() {
  const { data } = useQuery<Tool[]>({
    queryKey: ['tools'],
    queryFn: () => apiFetch<Tool[]>('/tools'),
    staleTime: Infinity,
    gcTime: Infinity,
  });

  return (
    <Monitor title="Tool Catalog" count={data?.length ?? 0} pinned={false}>
      {!data || data.length === 0 ? (
        <div style={{ fontSize: 11, color: 'var(--ink-3)' }}>No tools registered.</div>
      ) : (
        data.map((t) => (
          <div key={t.name} style={{ display: 'flex', flexDirection: 'column', gap: 2, marginBottom: 6 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 6 }}>
              <span style={{ fontFamily: 'var(--ff-mono)', fontSize: 11, color: 'var(--ink-1)' }}>{t.name}</span>
              {t.risk && (
                <span style={{ fontFamily: 'var(--ff-mono)', fontSize: 9, color: riskColor[t.risk] ?? 'var(--ink-3)', letterSpacing: '0.14em' }}>
                  {t.risk.toUpperCase()}
                </span>
              )}
            </div>
            {t.description && (
              <span style={{ fontFamily: 'var(--ff-sans)', fontSize: 11, color: 'var(--ink-2)' }}>
                {t.description}
              </span>
            )}
          </div>
        ))
      )}
    </Monitor>
  );
}
