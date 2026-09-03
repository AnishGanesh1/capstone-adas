#!/usr/bin/env python3
"""Verification gate for the LiDAR dataset. Same idea as verify_dataset.py.

  python3 verify_lidar.py --root ~/dataset/lidar_raw --bev ~/dataset/lidar_bev --draw 20

Checks:
  1. point count per condition   - fog/storm MUST be below clear
  2. objects per frame           - are there things to detect at all
  3. BEV label statistics        - class balance, boxes per image
  4. file integrity              - orphans, malformed OBB rows
  5. draws BEV images with boxes - you must look at these
"""

import argparse
import collections
import glob
import json
import os
import random

import numpy as np
from PIL import Image, ImageDraw

import config as C

COLOURS = [(255, 90, 90), (255, 170, 60), (255, 230, 80),
           (120, 220, 255), (170, 140, 255), (120, 255, 150)]


def check_points(root):
    print('\n[1] LiDAR points per condition')
    per = collections.defaultdict(list)
    for m in glob.glob(os.path.join(root, 'meta', '*.json')):
        with open(m) as fh:
            j = json.load(fh)
        per[j.get('condition', '?')].append(j.get('n_lidar_points', 0))
    if not per:
        print('    FAIL: no meta json')
        return False
    base = float(np.mean(per['clear'])) if 'clear' in per else None
    for c in sorted(per):
        mean = float(np.mean(per[c]))
        rel = '' if base is None else '  (%+.1f%% vs clear)' % (100 * (mean / base - 1))
        print('    %-14s n=%4d  mean %8.0f%s' % (c, len(per[c]), mean, rel))
    ok = True
    if base is not None:
        for harsh in ('fog', 'fog_dense', 'fog_night', 'storm_day', 'storm_night', 'hail_severe'):
            if harsh in per and float(np.mean(per[harsh])) >= base * 0.98:
                print('    FAIL: %s is not below clear - LiDAR weather coupling did not apply'
                      % harsh)
                print('          (attributes were set without RESPAWNING the sensor)')
                ok = False
    return ok


def check_objects(root):
    print('\n[2] objects per frame (from meta boxes_3d)')
    counts = []
    groups = collections.Counter()
    for m in glob.glob(os.path.join(root, 'meta', '*.json')):
        with open(m) as fh:
            j = json.load(fh)
        b = j.get('boxes_3d', [])
        counts.append(len(b))
        for x in b:
            groups[x.get('group', '?')] += 1
    if not counts:
        print('    FAIL: no boxes_3d in meta')
        return False
    counts = np.array(counts)
    print('    frames %d | mean %.1f objects | median %d | max %d | zero-object frames %d'
          % (len(counts), counts.mean(), int(np.median(counts)), counts.max(),
             int((counts == 0).sum())))
    for g, n in groups.most_common():
        print('    %-12s %d' % (g, n))
    if counts.mean() < 1.0:
        print('    WARN: fewer than one object per frame - raise --vehicles')
    return True


def check_bev(bev):
    print('\n[3] BEV label statistics')
    lbls = sorted(glob.glob(os.path.join(bev, 'labels', '*.txt')))
    if not lbls:
        print('    SKIP: no BEV labels yet (run lidar_to_bev.py first)')
        return None
    per_cls = collections.Counter()
    per_img = []
    bad = 0
    for p in lbls:
        n = 0
        for line in open(p):
            v = line.split()
            if len(v) != 9:
                if line.strip():
                    bad += 1
                continue
            try:
                cid = int(v[0])
                vals = [float(x) for x in v[1:]]
            except ValueError:
                bad += 1
                continue
            if not (0 <= cid < len(C.BEV_CLASSES)) or any(x < 0 or x > 1 for x in vals):
                bad += 1
                continue
            per_cls[cid] += 1
            n += 1
        per_img.append(n)
    print('    images %d | mean %.1f boxes | empty %d'
          % (len(per_img), float(np.mean(per_img)), int(sum(1 for x in per_img if x == 0))))
    for i, name in enumerate(C.BEV_CLASSES):
        print('    %-12s %d' % (name, per_cls.get(i, 0)))
    if bad:
        print('    FAIL: %d malformed OBB rows' % bad)
        return False
    print('    all OBB rows well formed (9 fields, normalised)')
    return True


def check_integrity(root, bev):
    print('\n[4] file integrity')
    lid = {os.path.splitext(os.path.basename(p))[0]
           for p in glob.glob(os.path.join(root, 'lidar', '*.npy'))}
    met = {os.path.splitext(os.path.basename(p))[0]
           for p in glob.glob(os.path.join(root, 'meta', '*.json'))}
    print('    lidar %d | meta %d' % (len(lid), len(met)))
    ok = True
    if lid - met:
        print('    FAIL: %d sweeps with no meta' % len(lid - met))
        ok = False
    if met - lid:
        print('    FAIL: %d meta with no sweep' % len(met - lid))
        ok = False
    if os.path.isdir(os.path.join(bev, 'images')):
        bi = {os.path.splitext(os.path.basename(p))[0]
              for p in glob.glob(os.path.join(bev, 'images', '*.png'))}
        bl = {os.path.splitext(os.path.basename(p))[0]
              for p in glob.glob(os.path.join(bev, 'labels', '*.txt'))}
        print('    bev images %d | bev labels %d' % (len(bi), len(bl)))
        if bi - bl:
            print('    FAIL: %d BEV images with no label file' % len(bi - bl))
            ok = False
    return ok


def draw_bev(bev, n, rng):
    print('\n[5] drawing %d BEV frames with boxes' % n)
    out = os.path.join(bev, '_verify')
    lbls = [p for p in sorted(glob.glob(os.path.join(bev, 'labels', '*.txt')))
            if os.path.getsize(p) > 0]
    if not lbls:
        print('    SKIP: no non-empty BEV labels')
        return None
    os.makedirs(out, exist_ok=True)
    for p in rng.sample(lbls, min(n, len(lbls))):
        stem = os.path.splitext(os.path.basename(p))[0]
        ip = os.path.join(bev, 'images', stem + '.png')
        if not os.path.exists(ip):
            continue
        im = Image.open(ip).convert('RGB')
        d = ImageDraw.Draw(im)
        W, H = im.size
        for line in open(p):
            v = line.split()
            if len(v) != 9:
                continue
            cid = int(v[0])
            pts = [(float(v[1 + 2 * i]) * W, float(v[2 + 2 * i]) * H) for i in range(4)]
            d.polygon(pts, outline=COLOURS[cid % len(COLOURS)])
            d.text(pts[0], C.BEV_CLASSES[cid][:3], fill=COLOURS[cid % len(COLOURS)])
        im.save(os.path.join(out, stem + '.png'))
    print('    wrote to %s' % out)
    print('    ACTION: open them. Bright rectangles in the image must line up with polygons.')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.expanduser('~/dataset/lidar_raw'))
    ap.add_argument('--bev', default=os.path.expanduser('~/dataset/lidar_bev'))
    ap.add_argument('--draw', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    print('verifying %s' % args.root)
    results = [
        ('points per condition', check_points(args.root)),
        ('objects per frame', check_objects(args.root)),
        ('bev labels', check_bev(args.bev)),
        ('file integrity', check_integrity(args.root, args.bev)),
        ('draw bev', draw_bev(args.bev, args.draw, rng) if args.draw else None),
    ]
    print('\n' + '=' * 60)
    for name, ok in results:
        print('  %-24s %s' % (name, 'SKIP' if ok is None else ('PASS' if ok else 'FAIL')))
    print('=' * 60)
    hard = [ok for _, ok in results if ok is not None]
    if all(hard):
        print('All applicable checks passed.')
        print('STILL REQUIRED: open %s/_verify/ and look.' % args.bev)
    else:
        print('Fix the failures before training.')


if __name__ == '__main__':
    main()
