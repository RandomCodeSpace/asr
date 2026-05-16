#!/usr/bin/env node
// web/scripts/check-size.mjs
// Hard gate for the production JS bundle. Run after `vite build`.
//
// Budgets are intentionally conservative — they sit ~25% above the
// current production size so we get warned before drift turns into a
// regression. Move them down deliberately when the bundle shrinks,
// up only with a written rationale on the commit.

import { readdirSync, readFileSync, statSync } from 'node:fs';
import { gzipSync } from 'node:zlib';
import { join } from 'node:path';

const DIST_ASSETS = 'dist/assets';
const BUDGETS = {
  rawBytes: 400 * 1024,    // 400 kB raw
  gzipBytes: 130 * 1024,   // 130 kB gzip
};

function findBundle() {
  const entries = readdirSync(DIST_ASSETS);
  const candidates = entries.filter((f) => /^index-.*\.js$/.test(f));
  if (candidates.length === 0) {
    console.error(`✗ no index-*.js bundle found in ${DIST_ASSETS}/`);
    process.exit(2);
  }
  // Largest wins (defensive: vite shouldn't emit multiple index-* but
  // chunk splits could introduce variants).
  return candidates
    .map((f) => ({ name: f, path: join(DIST_ASSETS, f), size: statSync(join(DIST_ASSETS, f)).size }))
    .sort((a, b) => b.size - a.size)[0];
}

function format(n) {
  return `${(n / 1024).toFixed(2)} kB`;
}

const bundle = findBundle();
const raw = bundle.size;
const gzip = gzipSync(readFileSync(bundle.path)).length;

console.log(`bundle: ${bundle.name}`);
console.log(`  raw:  ${format(raw)} / budget ${format(BUDGETS.rawBytes)}`);
console.log(`  gzip: ${format(gzip)} / budget ${format(BUDGETS.gzipBytes)}`);

let fail = false;
if (raw > BUDGETS.rawBytes) {
  console.error(`✗ raw size exceeds budget by ${format(raw - BUDGETS.rawBytes)}`);
  fail = true;
}
if (gzip > BUDGETS.gzipBytes) {
  console.error(`✗ gzip size exceeds budget by ${format(gzip - BUDGETS.gzipBytes)}`);
  fail = true;
}

if (fail) process.exit(1);
console.log('✓ bundle within budget');
