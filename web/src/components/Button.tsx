import type { ButtonHTMLAttributes, CSSProperties, ReactNode } from 'react';

type Variant = 'primary' | 'secondary' | 'ghost';
type Size = 'default' | 'sm';

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  children: ReactNode;
}

const baseStyle: CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: 'var(--s-2)',
  fontFamily: 'var(--ff-sans)',
  fontSize: 'var(--t-body)',
  fontWeight: 500,
  padding: '8px var(--s-4)',
  border: '1px solid transparent',
  cursor: 'pointer',
  letterSpacing: '0.005em',
  transition: 'background-color 0.12s, border-color 0.12s, color 0.12s',
  borderRadius: 0,
};

const variantStyles: Record<Variant, CSSProperties> = {
  primary: {
    background: 'var(--ink-1)',
    color: 'var(--bg-elev)',
    borderColor: 'var(--ink-1)',
  },
  secondary: {
    background: 'var(--bg-elev)',
    color: 'var(--ink-1)',
    border: '1px solid var(--hair-strong)',
  },
  ghost: {
    background: 'transparent',
    color: 'var(--ink-3)',
    textDecoration: 'underline',
    textDecorationColor: 'var(--ink-4)',
    textUnderlineOffset: 2,
    padding: '8px var(--s-3)',
    border: 'none',
  },
};

export function Button({
  variant = 'primary',
  size = 'default',
  children,
  style,
  ...rest
}: ButtonProps) {
  const sizeOverride: CSSProperties =
    size === 'sm' ? { padding: '4px var(--s-3)', fontSize: 'var(--t-meta)' } : {};
  return (
    <button
      data-variant={variant}
      data-size={size}
      style={{ ...baseStyle, ...variantStyles[variant], ...sizeOverride, ...style }}
      {...rest}
    >
      {children}
    </button>
  );
}
