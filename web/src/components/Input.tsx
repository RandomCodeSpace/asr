import { forwardRef } from 'react';
import type { InputHTMLAttributes } from 'react';

type Size = 'default' | 'sm';

interface InputProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'size'> {
  size?: Size;
}

const baseClass = 'asr-input';

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ size = 'default', style, className, ...rest }, ref) => {
    const height = size === 'sm' ? 22 : 28;
    return (
      <input
        ref={ref}
        data-size={size}
        className={[baseClass, className].filter(Boolean).join(' ')}
        style={{
          height,
          padding: '0 10px',
          fontFamily: 'var(--ff-sans)',
          fontSize: size === 'sm' ? 'var(--t-meta)' : 'var(--t-body)',
          color: 'var(--ink-1)',
          background: 'var(--bg-elev)',
          border: '1px solid var(--hair)',
          borderRadius: 0,
          outline: 'none',
          transition: 'border-color 0.12s, box-shadow 0.12s',
          ...style,
        }}
        onFocus={(e) => {
          e.currentTarget.style.borderColor = 'var(--hair-strong)';
          e.currentTarget.style.boxShadow = '0 0 0 1px var(--acc-soft)';
          rest.onFocus?.(e);
        }}
        onBlur={(e) => {
          e.currentTarget.style.borderColor = 'var(--hair)';
          e.currentTarget.style.boxShadow = 'none';
          rest.onBlur?.(e);
        }}
        {...rest}
      />
    );
  },
);
Input.displayName = 'Input';
