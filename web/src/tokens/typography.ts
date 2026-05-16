// web/src/tokens/typography.ts
export const typography = {
  ffSans: '"Geist", "Inter", system-ui, sans-serif',
  ffMono: '"Geist Mono", "JetBrains Mono", ui-monospace, monospace',
  micro: 10,
  meta: 11,
  body: 13,
  lead: 14,
  h3: 15,
  h2: 18,
  h1: 24,
  display: 30,
} as const;

export type TypographyToken = keyof typeof typography;
