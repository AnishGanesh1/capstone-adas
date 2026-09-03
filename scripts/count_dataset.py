#!/usr/bin/env python3
"""What have I actually got, and what --frames do I need to hit a target?

  python3 count_dataset.py --root ~/dataset/train                  # current state
  python3 count_dataset.py --root ~/dataset/train --target 10000   # solve for --frames

Capture is resumable per (town, condition), so the final total is
    sum over (town, condition) of max(existing, frames)
which is monotonic in `frames`. This binary-searches it rather than making you
guess, and it accounts for conditions you have already over-filled.
"""

import argparse
import collections
import glob
import json
import os

import config as C


def load_counts(root):
    counts = collections.defaultdict(int)
    for m in glob.glob(os.path.join(root, 'meta', '*.json')):
        try:
            with open(m) as fh:
                j = json.load(fh)
        except Exception:
            continue
        counts[(j.get('town', '?'), j.get('condition', '?'))] += 1
    return counts


def projected(counts, towns, conds, frames):
    return sum(max(counts.get((t, c), 0), frames) for t in towns for c in conds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.expanduser('~/dataset/train'))
    ap.add_argument('--tier', default='adverse', choices=list(C.TIERS))
    ap.add_argument('--target', type=int, default=0)
    ap.add_argument('--frames', type=int, default=0)
    args = ap.parse_args()

    counts = load_counts(args.root)
    if not counts:
        print('no meta json under %s - nothing captured yet' % args.root)
        total_now = 0
    else:
        total_now = sum(counts.values())

    towns = C.ALL_TOWNS
    conds = C.TIERS[args.tier]

    print('root : %s' % args.root)
    print('tier : %s  (%d conditions x %d towns = %d cells)'
          % (args.tier, len(conds), len(towns), len(conds) * len(towns)))
    print('have : %d frames total\n' % total_now)

    # ---- per-condition table -------------------------------------------
    width = max(len(c) for c in list(conds) + ['condition']) + 1
    hdr = '  %-*s' % (width, 'condition') + ''.join('%12s' % t[:11] for t in towns) + '%9s' % 'total'
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    grand = 0
    for c in conds:
        row = [counts.get((t, c), 0) for t in towns]
        grand += sum(row)
        flag = '' if all(row) else '   <- gap'
        print('  %-*s' % (width, c) + ''.join('%12d' % v for v in row)
              + '%9d' % sum(row) + flag)
    print('  ' + '-' * (len(hdr) - 2))
    print('  %-*s' % (width, 'TIER TOTAL') + ' ' * (12 * len(towns)) + '%9d' % grand)

    # anything outside the tier (e.g. clear, or old severity tags)
    other = {k: v for k, v in counts.items() if k[1] not in conds}
    if other:
        by_cond = collections.Counter()
        for (t, c), v in other.items():
            by_cond[c] += v
        print('\n  outside this tier: %d frames' % sum(by_cond.values()))
        for c, v in by_cond.most_common():
            note = '  (reference set - keep out of training)' if c == 'clear' else ''
            print('    %-22s %6d%s' % (c, v, note))

    # ---- solve ----------------------------------------------------------
    if args.frames:
        p = projected(counts, towns, conds, args.frames)
        print('\n--frames %d  ->  projected total %d  (+%d new)'
              % (args.frames, p, p - grand))

    # ---- planning table -------------------------------------------------
    print('\n  PLANNING - what each --frames value gives you')
    print('  %8s %10s %10s %14s' % ('--frames', 'total', 'new', 'min per cell'))
    for n in (50, 75, 100, 125, 150, 200, 250, 300, 400):
        pj = projected(counts, towns, conds, n)
        lowest = min(max(counts.get((t, c), 0), n) for t in towns for c in conds)
        print('  %8d %10d %10d %14d' % (n, pj, pj - grand, lowest))
    print('\n  "min per cell" is what your WORST-covered condition ends up with.')
    print('  A tier with 8 frames in it is not a tier, it is noise - pick a')
    print('  --frames that fills the gaps, even if the total overshoots.')

    if args.target:
        lo, hi = 0, args.target
        while lo < hi:
            mid = (lo + hi) // 2
            if projected(counts, towns, conds, mid) >= args.target:
                hi = mid
            else:
                lo = mid + 1
        p = projected(counts, towns, conds, lo)
        print('\n' + '=' * 60)
        print('  TARGET %d' % args.target)
        print('  use --frames %d   ->  %d total  (+%d new to capture)'
              % (lo, p, p - grand))
        print('=' * 60)
        print('\n  python3 capture_drive.py --town <T> --tier %s --frames %d \\'
              % (args.tier, lo))
        print('          --no-lidar --out %s' % args.root)


if __name__ == '__main__':
    main()
