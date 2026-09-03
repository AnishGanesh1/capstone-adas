#!/usr/bin/env python3
"""Split the BEV dataset by town and write a YOLOv8-OBB data.yaml.

  python3 make_splits_bev.py --src ~/dataset/lidar_bev --meta ~/dataset/lidar_raw \
                             --dst ~/dataset/bev_split

Splitting by town, not randomly, for the same reason as the camera dataset:
random splitting puts frames from the same junction in train and test, the model
memorises the junction, and the reported number is fiction.
"""

import argparse
import collections
import glob
import json
import os
import shutil

import config as C


def split_of(town):
    if town in C.TOWNS_TRAIN:
        return 'train'
    if town in C.TOWNS_VAL:
        return 'val'
    if town in C.TOWNS_TEST:
        return 'test'
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=os.path.expanduser('~/dataset/lidar_bev'))
    ap.add_argument('--meta', default=os.path.expanduser('~/dataset/lidar_raw'),
                    help='where the meta json lives (for the town field)')
    ap.add_argument('--dst', default=os.path.expanduser('~/dataset/bev_split'))
    ap.add_argument('--copy', action='store_true', help='copy instead of symlink')
    args = ap.parse_args()

    for sp in ('train', 'val', 'test'):
        os.makedirs(os.path.join(args.dst, 'images', sp), exist_ok=True)
        os.makedirs(os.path.join(args.dst, 'labels', sp), exist_ok=True)

    counts = collections.Counter()
    stems = collections.defaultdict(set)
    unsplit = collections.Counter()
    missing_meta = 0

    for img in sorted(glob.glob(os.path.join(args.src, 'images', '*.png'))):
        stem = os.path.splitext(os.path.basename(img))[0]
        mp = os.path.join(args.meta, 'meta', stem + '.json')
        if not os.path.exists(mp):
            missing_meta += 1
            continue
        with open(mp) as fh:
            town = json.load(fh)['town']
        sp = split_of(town)
        if sp is None:
            unsplit[town] += 1
            continue

        lbl = os.path.join(args.src, 'labels', stem + '.txt')
        if not os.path.exists(lbl):
            continue

        for s, d in ((os.path.abspath(img), os.path.join(args.dst, 'images', sp, stem + '.png')),
                     (os.path.abspath(lbl), os.path.join(args.dst, 'labels', sp, stem + '.txt'))):
            if os.path.lexists(d):
                os.remove(d)
            if args.copy:
                shutil.copy2(s, d)
            else:
                os.symlink(s, d)

        counts[sp] += 1
        stems[sp].add(stem)

    for a in ('train', 'val', 'test'):
        for b in ('train', 'val', 'test'):
            if a < b:
                overlap = stems[a] & stems[b]
                assert not overlap, 'LEAKAGE: %d stems in %s and %s' % (len(overlap), a, b)

    yaml_p = os.path.join(args.dst, 'data.yaml')
    with open(yaml_p, 'w') as fh:
        fh.write('path: %s\n' % os.path.abspath(args.dst))
        fh.write('train: images/train\nval: images/val\ntest: images/test\n')
        fh.write('nc: %d\n' % len(C.BEV_CLASSES))
        fh.write('names: %s\n' % C.BEV_CLASSES)

    print('\nBEV split by town:')
    for sp in ('train', 'val', 'test'):
        towns = {'train': C.TOWNS_TRAIN, 'val': C.TOWNS_VAL, 'test': C.TOWNS_TEST}[sp]
        print('  %-6s %5d frames   %s' % (sp, counts[sp], ', '.join(towns)))
    if unsplit:
        print('\n  IGNORED (town not in any split): %s' % dict(unsplit))
    if missing_meta:
        print('  %d BEV images had no meta json - check --meta points at the right dir'
              % missing_meta)
    print('\nleakage check passed')
    print('wrote %s' % yaml_p)


if __name__ == '__main__':
    main()
