// web/src/tokens/elevation.ts
export const elevation = {
  e1: '0 1px 2px rgba(21,17,10,0.04), 0 0 0 1px #E6E1D4',
  e2: '0 2px 4px rgba(21,17,10,0.05), 0 8px 16px rgba(21,17,10,0.04), 0 0 0 1px #E6E1D4',
  e3: '0 4px 12px rgba(21,17,10,0.07), 0 16px 32px rgba(21,17,10,0.06), 0 0 0 1px #D4CDB8',
} as const;

export type ElevationToken = keyof typeof elevation;
