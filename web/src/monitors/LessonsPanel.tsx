import { useQuery } from '@tanstack/react-query';
import { apiFetch } from '@/api/client';
import { Monitor } from './Monitor';

interface Lesson {
  id?: string;
  title?: string;
  summary?: string;
  agent?: string;
  [k: string]: unknown;
}

interface Props {
  sessionId: string | null;
}

export function LessonsPanel({ sessionId }: Props) {
  const { data } = useQuery<Lesson[]>({
    queryKey: ['lessons', sessionId],
    queryFn: () => apiFetch<Lesson[]>(`/sessions/${sessionId}/lessons`),
    enabled: sessionId !== null,
    staleTime: 60_000,
  });

  return (
    <Monitor title="Lessons" count={data?.length ?? 0} pinned={false}>
      {!data || data.length === 0 ? (
        <div style={{ fontSize: 11, color: 'var(--ink-3)' }}>
          No lessons relevant to this session yet.
        </div>
      ) : (
        data.map((l, i) => (
          <div key={l.id ?? i} style={{ marginBottom: 8 }}>
            <div
              style={{
                fontFamily: 'var(--ff-sans)',
                fontSize: 12,
                color: 'var(--ink-1)',
                fontWeight: 500,
              }}
            >
              {l.title ?? '—'}
            </div>
            {l.summary && (
              <div
                style={{
                  fontFamily: 'var(--ff-sans)',
                  fontSize: 11,
                  color: 'var(--ink-2)',
                  marginTop: 2,
                }}
              >
                {l.summary}
              </div>
            )}
            {l.agent && (
              <div
                style={{
                  fontFamily: 'var(--ff-mono)',
                  fontSize: 10,
                  color: 'var(--ink-3)',
                  marginTop: 2,
                }}
              >
                {l.agent}
              </div>
            )}
          </div>
        ))
      )}
    </Monitor>
  );
}
