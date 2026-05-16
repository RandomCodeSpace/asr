// web/src/tokens/colors.ts
export const colors = {
  bgPage: '#FBFAF6',
  bgElev: '#FFFFFF',
  bgSubtle: '#F4F2EC',
  bgDeep: '#ECE7DB',
  bgTint: '#FAF6EA',
  ink1: '#15110A',
  ink2: '#4A4540',
  ink3: '#918A80',
  ink4: '#C8C2B6',
  hair: '#E6E1D4',
  hairStrong: '#D4CDB8',
  acc: '#2A4365',
  accDim: '#1F3147',
  accSoft: 'rgba(42, 67, 101, 0.08)',
  accMid: 'rgba(42, 67, 101, 0.18)',
  warn: '#B4814A',
  warnBg: 'rgba(180, 129, 74, 0.08)',
  danger: '#B85A4F',
  dangerBg: 'rgba(184, 90, 79, 0.08)',
  good: '#5C8862',
  goodBg: 'rgba(92, 136, 98, 0.08)',
  info: '#4F6F8E',
} as const;

export type ColorToken = keyof typeof colors;
