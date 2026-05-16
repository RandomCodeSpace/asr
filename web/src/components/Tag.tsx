import type { CSSProperties, ReactNode } from 'react';

type Variant = 'default' | 'mono';

interface TagProps {
  variant?: Variant;
  children: ReactNode;
}

const baseStyle: CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  padding: '1px 6px',
  fontSize: 'var(--t-meta)',
  color: 'var(--ink-2)',
  border: '1px solid var(--hair)',
  background: 'var(--bg-subtle)',
  borderRadius: 0,
  whiteSpace: 'nowrap',
};

const variantStyles: Record<Variant, CSSProperties> = {
  default: { fontFamily: 'var(--ff-sans)' },
  mono: { fontFamily: 'var(--ff-mono)' },
};

export function Tag({ variant = 'default', children }: TagProps) {
  return (
    <span data-variant={variant} style={{ ...baseStyle, ...variantStyles[variant] }}>
      {children}
    </span>
  );
}
