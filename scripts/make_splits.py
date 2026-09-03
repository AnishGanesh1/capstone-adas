#!/usr/bin/env python3
"""Assemble the captured frames into a YOLO dataset, split BY TOWN.

  python3 make_splits.py --src ~/dataset/train --dst ~/dataset/camera_tl

Splitting by town, not randomly, is the single most important methodological
choice here. Random splitting puts frames from the same junction in both train
and test, the model memorises the junction, and the reported accuracy is fiction.

Uses symlinks by default so nothing is duplicated on disk. --copy if your
filesystem or tooling dislikes symlinks.
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
    ap.add_argument('--src', default=os.path.expanduser('~/dataset/train'))
    ap.add_argument('--dst', default=os.path.expanduser('~/dataset/camera_tl'))
    ap.add_argument('--copy', action='store_true', help='copy instead of symlink')
    args = ap.parse_args()

    for sp in ('train', 'val', 'test'):
        os.makedirs(os.path.join(args.dst, 'images', sp), exist_ok=True)
        os.makedirs(os.path.join(args.dst, 'labels', sp), exist_ok=True)

    counts = collections.Counter()
    unsplit = collections.Counter()
    stems_by_split = collections.defaultdict(set)

    for meta_p in sorted(glob.glob(os.path.join(args.src, 'meta', '*.json'))):
        stem = os.path.splitext(os.path.basename(meta_p))[0]
        with open(meta_p) as fh:
            town = json.load(fh)['town']
        sp = split_of(town)
        if sp is None:
            unsplit[town] += 1
            continue

        img_src = os.path.abspath(os.path.join(args.src, 'images', stem + '.jpg'))
        lbl_src = os.path.abspath(os.path.join(args.src, 'labels', stem + '.txt'))
        if not (os.path.exists(img_src) and os.path.exists(lbl_src)):
            continue

        img_dst = os.path.join(args.dst, 'images', sp, stem + '.jpg')
        lbl_dst = os.path.join(args.dst, 'labels', sp, stem + '.txt')
        for s, d in ((img_src, img_dst), (lbl_src, lbl_dst)):
            if os.path.lexists(d):
                os.remove(d)
            if args.copy:
                shutil.copy2(s, d)
            else:
                os.symlink(s, d)

        counts[sp] += 1
        stems_by_split[sp].add(stem)

    # leakage assertion
    for a in ('train', 'val', 'test'):
        for b in ('train', 'val', 'test'):
            if a < b:
                overlap = stems_by_split[a] & stems_by_split[b]
                assert not overlap, 'LEAKAGE: %d stems in both %s and %s' % (len(overlap), a, b)

    yaml_p = os.path.join(args.dst, 'data.yaml')
    with open(yaml_p, 'w') as fh:
        fh.write('path: %s\n' % os.path.abspath(args.dst))
        fh.write('train: images/train\nval: images/val\ntest: images/test\n')
        fh.write('nc: %d\n' % len(C.TL_CLASSES))
        fh.write('names: %s\n' % C.TL_CLASSES)

    print('\nsplit by town:')
    for sp in ('train', 'val', 'test'):
        towns = {'train': C.TOWNS_TRAIN, 'val': C.TOWNS_VAL, 'test': C.TOWNS_TEST}[sp]
        print('  %-6s %5d frames   %s' % (sp, counts[sp], ', '.join(towns)))
    if unsplit:
        print('\n  IGNORED (town not in any split): %s' % dict(unsplit))
    print('\nno stem appears in two splits - leakage check passed')
    print('wrote %s' % yaml_p)


if __name__ == '__main__':
    main()
