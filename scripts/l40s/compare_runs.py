#!/usr/bin/env python3
"""Compare two counting output dirs (e.g. main branch vs l40optim) for exact equality.

    python scripts/l40s/compare_runs.py results/baseline results/optimized
"""

import sys
from pathlib import Path

FILES = ["tracks.jsonl", "records.csv", "interval_counts.csv", "interval_counts_3class.csv"]


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    a_dir, b_dir = Path(sys.argv[1]), Path(sys.argv[2])
    ok = True
    for name in FILES:
        a, b = a_dir / name, b_dir / name
        if not a.exists() and not b.exists():
            print(f"[skip] {name}: in neither")
            continue
        if not a.exists() or not b.exists():
            print(f"[DIFF] {name}: only in {a_dir if a.exists() else b_dir}")
            ok = False
            continue
        la, lb = a.read_text().splitlines(), b.read_text().splitlines()
        if la == lb:
            print(f"[same] {name} ({len(la)} lines)")
            continue
        ok = False
        print(f"[DIFF] {name}: {len(la)} vs {len(lb)} lines")
        shown = 0
        for i, (x, y) in enumerate(zip(la, lb)):
            if x != y:
                print(f"   line {i + 1}:\n     A: {x}\n     B: {y}")
                shown += 1
                if shown == 5:
                    break
    print("PASS: identical results" if ok else "FAIL: results differ")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
