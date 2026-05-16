import { forwardRef } from 'react';
import type { TextareaHTMLAttributes } from 'react';

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaHTMLAttributes<HTMLTextAreaElement>>(
  ({ style, ...rest }, ref) => (
    <textarea
      ref={ref}
      style={{
        padding: '8px 10px',
        fontFamily: 'var(--ff-sans)',
        fontSize: 'var(--t-body)',
        color: 'var(--ink-1)',
        background: 'var(--bg-elev)',
        border: '1px solid var(--hair)',
        borderRadius: 0,
        outline: 'none',
        resize: 'vertical',
        minHeight: 64,
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
  ),
);
Textarea.displayName = 'Textarea';
