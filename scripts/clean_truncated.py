#!/usr/bin/env python3
"""Find and remove frames left half-written by a killed capture run.

A capture that dies mid-frame (power cut, SIGKILL, destroyed-actor abort) can
leave a 0-byte or truncated meta json, plus its image / label / npy siblings.
Every downstream script then dies on JSONDecodeError.

  python3 clean_truncated.py --root ~/dataset/train            # report only
  python3 clean_truncated.py --root ~/dataset/train --delete   # remove them
"""

import argparse
import glob
import json
import os

SIBS = (('images', '.jpg'), ('labels', '.txt'), ('lidar', '.npy'), ('sem', '.png'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--delete', action='store_true')
    args = ap.parse_args()
    root = os.path.expanduser(args.root)

    metas = sorted(glob.glob(os.path.join(root, 'meta', '*.json')))
    print('root  : %s' % root)
    print('meta  : %d files' % len(metas))

    bad = []
    for p in metas:
        try:
            if os.path.getsize(p) == 0:
                bad.append((p, 'empty')); continue
            with open(p) as fh:
                j = json.load(fh)
            if 'town' not in j or 'condition' not in j:
                bad.append((p, 'missing keys'))
        except json.JSONDecodeError:
            bad.append((p, 'truncated json'))
        except Exception as e:
            bad.append((p, type(e).__name__))

    # zero-byte payload files whose meta parsed fine
    empty_sibs = []
    for sub, ext in SIBS:
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        for p in glob.glob(os.path.join(d, '*' + ext)):
            if os.path.getsize(p) == 0:
                empty_sibs.append(p)

    print('bad meta        : %d' % len(bad))
    print('zero-byte files : %d' % len(empty_sibs))
    for p, why in bad[:10]:
        print('   %-55s %s' % (os.path.basename(p), why))
    if len(bad) > 10:
        print('   ... and %d more' % (len(bad) - 10))

    stems = {os.path.splitext(os.path.basename(p))[0] for p, _ in bad}
    stems |= {os.path.splitext(os.path.basename(p))[0] for p in empty_sibs}

    if not stems:
        print('\nnothing to clean.')
        return

    victims = []
    for stem in sorted(stems):
        victims.append(os.path.join(root, 'meta', stem + '.json'))
        for sub, ext in SIBS:
            victims.append(os.path.join(root, sub, stem + ext))
    victims = [p for p in victims if os.path.exists(p)]

    print('\n%d frames affected, %d files' % (len(stems), len(victims)))

    if not args.delete:
        print('\ndry run. rerun with --delete to remove them.')
        return

    n = 0
    for p in victims:
        try:
            os.remove(p); n += 1
        except Exception as e:
            print('  could not remove %s: %s' % (p, e))
    print('removed %d files (%d frames)' % (n, len(stems)))
    print('\nre-run count_dataset.py - the totals will drop slightly, and the')
    print('next capture run will refill those indices.')


if __name__ == '__main__':
    main()
