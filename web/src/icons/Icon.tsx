type IconName =
  | 'check' | 'x' | 'plus' | 'stop' | 'retry' | 'search' | 'alert' | 'info'
  | 'arrow-right' | 'back' | 'list' | 'network' | 'message';

interface IconProps {
  name: IconName;
  size?: 12 | 14 | 16;
}

export function Icon({ name, size = 14 }: IconProps) {
  return (
    <svg
      role="img"
      width={size}
      height={size}
      style={{ flexShrink: 0, verticalAlign: -2 }}
      aria-hidden
    >
      <use href={`#i-${name}`} />
    </svg>
  );
}
