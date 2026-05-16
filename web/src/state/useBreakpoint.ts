import { useEffect, useState } from 'react';

export type Breakpoint = 'mobile' | 'tablet' | 'desktop';

const TABLET_MIN = '(min-width: 768px)';
const DESKTOP_MIN = '(min-width: 1200px)';

function readNow(): Breakpoint {
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') {
    return 'desktop';
  }
  if (window.matchMedia(DESKTOP_MIN).matches) return 'desktop';
  if (window.matchMedia(TABLET_MIN).matches) return 'tablet';
  return 'mobile';
}

export function useBreakpoint(): Breakpoint {
  const [bp, setBp] = useState<Breakpoint>(readNow);

  useEffect(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return;
    const tabletMq = window.matchMedia(TABLET_MIN);
    const desktopMq = window.matchMedia(DESKTOP_MIN);
    const update = () => setBp(readNow());
    tabletMq.addEventListener('change', update);
    desktopMq.addEventListener('change', update);
    update();
    return () => {
      tabletMq.removeEventListener('change', update);
      desktopMq.removeEventListener('change', update);
    };
  }, []);

  return bp;
}
