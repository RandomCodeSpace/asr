import { forwardRef } from 'react';
import type { SelectHTMLAttributes } from 'react';

export interface SelectOption {
  value: string;
  label: string;
}

interface SelectProps extends Omit<SelectHTMLAttributes<HTMLSelectElement>, 'children'> {
  options: SelectOption[];
  placeholder?: string;
}

export const Select = forwardRef<HTMLSelectElement, SelectProps>(
  ({ options, placeholder, style, value, ...rest }, ref) => (
    <select
      ref={ref}
      value={value ?? ''}
      style={{
        height: 28,
        padding: '0 10px',
        fontFamily: 'var(--ff-sans)',
        fontSize: 'var(--t-body)',
        color: 'var(--ink-1)',
        background: 'var(--bg-elev)',
        border: '1px solid var(--hair)',
        borderRadius: 0,
        ...style,
      }}
      {...rest}
    >
      {placeholder !== undefined && <option value="" disabled>{placeholder}</option>}
      {options.map((o) => (
        <option key={o.value} value={o.value}>{o.label}</option>
      ))}
    </select>
  ),
);
Select.displayName = 'Select';
