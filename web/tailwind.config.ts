import type { Config } from 'tailwindcss';

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        'bg-page': 'var(--bg-page)',
        'bg-elev': 'var(--bg-elev)',
        'bg-subtle': 'var(--bg-subtle)',
        'bg-deep': 'var(--bg-deep)',
        'bg-tint': 'var(--bg-tint)',
        'ink-1': 'var(--ink-1)',
        'ink-2': 'var(--ink-2)',
        'ink-3': 'var(--ink-3)',
        'ink-4': 'var(--ink-4)',
        hair: 'var(--hair)',
        'hair-strong': 'var(--hair-strong)',
        acc: 'var(--acc)',
        'acc-dim': 'var(--acc-dim)',
        warn: 'var(--warn)',
        danger: 'var(--danger)',
        good: 'var(--good)',
        info: 'var(--info)',
      },
      fontFamily: {
        sans: ['Geist', 'Inter', 'system-ui', 'sans-serif'],
        mono: ['"Geist Mono"', '"JetBrains Mono"', 'ui-monospace', 'monospace'],
      },
      fontSize: {
        micro: '10px',
        meta: '11px',
        body: '13px',
        lead: '14px',
        h3: '15px',
        h2: '18px',
        h1: '24px',
        display: '30px',
      },
      spacing: {
        1: '4px', 2: '8px', 3: '12px', 4: '16px', 5: '24px', 6: '32px',
      },
      borderRadius: { none: '0', DEFAULT: '0' },
    },
  },
  plugins: [],
} satisfies Config;
