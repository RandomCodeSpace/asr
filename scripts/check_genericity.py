#!/usr/bin/env python3
"""Genericity ratchet for src/runtime/.

Counts occurrences of domain-specific tokens that should NOT exist in the
generic framework layer. Apps under examples/ are free to use these tokens
as much as they want — only src/runtime/ is policed.

Used as both:
  1. A CLI: `python scripts/check_genericity.py [--baseline N]` returns the count
     and exits non-zero if it exceeds the baseline.
  2. A pytest fixture (tests/test_genericity_ratchet.py) that asserts the count
     is at or below the recorded baseline.

The baseline is hand-recorded after this script first lands. To update:
  1. Bring the count down via a refactor commit.
  2. Update BASELINE in this file in the same commit.
  3. The ratchet enforces "monotonically non-increasing" thereafter.
"""

from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path


# Tokens to count. Case-insensitive word matches inside src/runtime/*.py only.
# Excluded: comments and docstrings (judged via simple heuristic — see below)
TOKENS = ("incident", "severity", "reporter")


def _strip_comments_and_strings(text: str) -> str:
    """Crude pass: drop trailing comments after '#' on a line.

    String literals stay because they may carry domain leaks (e.g. hardcoded
    prompts like 'incident reports for an SRE platform' in dedup.py — those
    are exactly what we want the ratchet to catch).
    """
    out = []
    for line in text.splitlines():
        if "#" in line:
            # Naive comment strip — does not handle '#' inside strings, fine for our purposes.
            line = line.split("#", 1)[0]
        out.append(line)
    return "\n".join(out)


def count_runtime_leaks(root: Path | None = None) -> dict[str, int]:
    """Walk src/runtime/, return a dict of token → match count."""
    root = root or Path("src/runtime")
    pattern = re.compile(r"\b(?:" + "|".join(TOKENS) + r")\b", re.IGNORECASE)
    counts = {t: 0 for t in TOKENS}
    for path in sorted(root.rglob("*.py")):
        text = _strip_comments_and_strings(path.read_text(encoding="utf-8"))
        for match in pattern.finditer(text):
            tok = match.group(0).lower()
            counts[tok] = counts.get(tok, 0) + 1
    return counts


def total(counts: dict[str, int]) -> int:
    return sum(counts.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=int, default=None,
                        help="Fail if the leak count exceeds this number.")
    parser.add_argument("--root", type=str, default="src/runtime")
    args = parser.parse_args(argv)
    counts = count_runtime_leaks(Path(args.root))
    t = total(counts)
    print("Runtime leak counts (excluding comments):")
    for tok in TOKENS:
        print(f"  {tok}: {counts[tok]}")
    print(f"  TOTAL: {t}")
    if args.baseline is not None and t > args.baseline:
        print(f"\nFAIL: total leak count {t} exceeds baseline {args.baseline}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
