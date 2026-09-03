#!/usr/bin/env python3
"""Week-1 verification gate. Do not train until every check passes.

  python3 verify_dataset.py --root ~/dataset/train --draw 30

Writes annotated frames to <root>/_verify/ so you can LOOK at them. Mislabelled
data is invisible in a loss curve and obvious in a picture.
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

COLOURS = {0: (255, 60, 60), 1: (255, 200, 0), 2: (60, 220, 60), 3: (150, 150, 150)}


def load_rows(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if len(p) == 5:
                rows.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return rows


def check_draw(root, n, rng):
    print('\n[1] drawing boxes on %d random frames' % n)
    out = os.path.join(root, '_verify')
    os.makedirs(out, exist_ok=True)
    labels = sorted(glob.glob(os.path.join(root, 'labels', '*.txt')))
    with_boxes = [p for p in labels if os.path.getsize(p) > 0]
    if not with_boxes:
        print('    FAIL: no label file contains a single box')
        return False
    pick = rng.sample(with_boxes, min(n, len(with_boxes)))
    for p in pick:
        stem = os.path.splitext(os.path.basename(p))[0]
        img_p = os.path.join(root, 'images', stem + '.jpg')
        if not os.path.exists(img_p):
            continue
        im = Image.open(img_p).convert('RGB')
        d = ImageDraw.Draw(im)
        W, H = im.size
        for cls, cx, cy, bw, bh in load_rows(p):
            x1, y1 = (cx - bw / 2) * W, (cy - bh / 2) * H
            x2, y2 = (cx + bw / 2) * W, (cy + bh / 2) * H
            d.rectangle([x1, y1, x2, y2], outline=COLOURS.get(cls, (255, 255, 255)), width=2)
            d.text((x1, max(0, y1 - 12)), C.TL_CLASSES[cls],
                   fill=COLOURS.get(cls, (255, 255, 255)))
        im.save(os.path.join(out, stem + '.jpg'), quality=92)
    print('    wrote %d annotated frames to %s' % (len(pick), out))
    print('    ACTION: open them. Boxes must sit on lamp heads, colour must match the lit lamp.')
    return True


def check_lidar_by_condition(root):
    print('\n[2] LiDAR point count per condition (fog MUST be lower than clear)')
    per = collections.defaultdict(list)
    for m in glob.glob(os.path.join(root, 'meta', '*.json')):
        with open(m) as fh:
            j = json.load(fh)
        per[j['condition']].append(j.get('n_lidar_points', 0))
    if not per:
        print('    FAIL: no meta json found')
        return False
    base = np.mean(per['clear']) if 'clear' in per else None
    ok = True
    for c in C.CONDITIONS:
        if c not in per:
            continue
        mean = np.mean(per[c])
        rel = '' if base is None else '  (%+.1f%% vs clear)' % (100 * (mean / base - 1))
        print('    %-10s n=%4d  mean points %8.0f%s' % (c, len(per[c]), mean, rel))
    if base is not None and 'fog' in per:
        if np.mean(per['fog']) >= base * 0.98:
            print('    FAIL: fog point count is not below clear.')
            print('          The LiDAR attributes did not take effect - you almost certainly')
            print('          set them without RESPAWNING the sensor.')
            ok = False
    return ok


def check_class_balance(root):
    print('\n[3] class balance per condition')
    counts = collections.defaultdict(collections.Counter)
    for m in glob.glob(os.path.join(root, 'meta', '*.json')):
        stem = os.path.splitext(os.path.basename(m))[0]
        lp = os.path.join(root, 'labels', stem + '.txt')
        if not os.path.exists(lp):
            continue
        with open(m) as fh:
            cond = json.load(fh)['condition']
        for cls, *_ in load_rows(lp):
            counts[cond][cls] += 1
    if not counts:
        print('    FAIL: no labels matched to meta')
        return False
    hdr = '    %-10s' % 'condition' + ''.join('%10s' % c for c in C.TL_CLASSES) + '%9s' % 'total'
    print(hdr)
    grand = collections.Counter()
    for cond in C.CONDITIONS:
        if cond not in counts:
            continue
        row = counts[cond]
        grand.update(row)
        print('    %-10s' % cond + ''.join('%10d' % row.get(i, 0)
                                           for i in range(len(C.TL_CLASSES)))
              + '%9d' % sum(row.values()))
    print('    %-10s' % 'ALL' + ''.join('%10d' % grand.get(i, 0)
                                        for i in range(len(C.TL_CLASSES)))
          + '%9d' % sum(grand.values()))
    if grand.get(1, 0) < 0.02 * max(sum(grand.values()), 1):
        print('    WARN: yellow is under 2% of boxes. Expected - a natural cycle rarely shows')
        print('          amber. Plan to oversample or weight the loss.')
    return True


def check_box_sizes(root):
    print('\n[4] box height histogram (pixels)')
    heights = []
    for p in glob.glob(os.path.join(root, 'labels', '*.txt')):
        for _, _, _, _, bh in load_rows(p):
            heights.append(bh * C.IMAGE_H)
    if not heights:
        print('    FAIL: no boxes at all')
        return False
    heights = np.array(heights)
    bins = [0, 8, 12, 16, 24, 32, 48, 64, 1e9]
    names = ['<8', '8-12', '12-16', '16-24', '24-32', '32-48', '48-64', '>64']
    hist, _ = np.histogram(heights, bins=bins)
    for n, h in zip(names, hist):
        bar = '#' * int(60 * h / max(hist.max(), 1))
        print('    %-7s %6d  %s' % (n, h, bar))
    print('    median %.1f px, 90th pct %.1f px' % (np.median(heights), np.percentile(heights, 90)))
    if hist[0] > 0:
        print('    FAIL: boxes below %d px got through the size filter' % C.MIN_BOX_PX)
        return False
    return True


def check_integrity(root):
    print('\n[5] file integrity')
    imgs = {os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(root, 'images', '*.jpg'))}
    lbls = {os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(root, 'labels', '*.txt'))}
    lid = {os.path.splitext(os.path.basename(p))[0]
           for p in glob.glob(os.path.join(root, 'lidar', '*.npy'))}
    met = {os.path.splitext(os.path.basename(p))[0]
           for p in glob.glob(os.path.join(root, 'meta', '*.json'))}
    print('    images %d | labels %d | lidar %d | meta %d' % (len(imgs), len(lbls), len(lid), len(met)))
    ok = True
    for name, s in (('labels', lbls), ('lidar', lid), ('meta', met)):
        missing = imgs - s
        if missing:
            print('    FAIL: %d images have no %s (e.g. %s)'
                  % (len(missing), name, sorted(missing)[0]))
            ok = False
    bad = 0
    for p in glob.glob(os.path.join(root, 'labels', '*.txt')):
        for cls, cx, cy, bw, bh in load_rows(p):
            if not (0 <= cls < len(C.TL_CLASSES)) or not all(
                    0.0 <= v <= 1.0 for v in (cx, cy, bw, bh)) or bw <= 0 or bh <= 0:
                bad += 1
    if bad:
        print('    FAIL: %d malformed label rows (class id or out-of-range coords)' % bad)
        ok = False
    else:
        print('    all label rows well formed')
    return ok


def check_condition_counts(root):
    print('\n[6] frames per condition')
    per = collections.Counter()
    for m in glob.glob(os.path.join(root, 'meta', '*.json')):
        with open(m) as fh:
            per[json.load(fh)['condition']] += 1
    if not per:
        return False
    for c in C.CONDITIONS:
        print('    %-10s %5d' % (c, per.get(c, 0)))
    print('    TOTAL      %5d' % sum(per.values()))
    lo, hi = min(per.values()), max(per.values())
    if hi > 0 and lo < 0.5 * hi:
        print('    WARN: conditions are badly unbalanced (%d vs %d)' % (lo, hi))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.expanduser('~/dataset/train'))
    ap.add_argument('--draw', type=int, default=30)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    print('verifying %s' % args.root)
    results = [
        ('draw boxes', check_draw(args.root, args.draw, rng)),
        ('lidar per condition', check_lidar_by_condition(args.root)),
        ('class balance', check_class_balance(args.root)),
        ('box sizes', check_box_sizes(args.root)),
        ('file integrity', check_integrity(args.root)),
        ('condition counts', check_condition_counts(args.root)),
    ]
    print('\n' + '=' * 60)
    for name, ok in results:
        print('  %-24s %s' % (name, 'PASS' if ok else 'FAIL'))
    print('=' * 60)
    if all(ok for _, ok in results):
        print('All automated checks passed.')
        print('STILL REQUIRED: open %s/_verify/ and look at the frames.' % args.root)
    else:
        print('Fix the failures before capturing the full dataset.')


if __name__ == '__main__':
    main()
