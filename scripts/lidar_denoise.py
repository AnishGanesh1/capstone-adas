#!/usr/bin/env python3
"""Statistical Outlier Removal on the LiDAR sweeps.

This is the Phase-1 commitment: "LiDAR may detect raindrops as objects; reduced
using Adaptive Filters or Statistical Outlier Removal."

SOR computes, for every point, the mean distance to its k nearest neighbours.
Points whose mean distance exceeds (global mean + std_ratio x global std) are
dropped. Rain and hail returns are sparse and isolated, so they sit in that tail;
returns off real surfaces are dense and survive.

  python3 lidar_denoise.py --in ~/dataset/lidar_raw --out ~/dataset/lidar_sor

Writes denoised .npy files plus a per-condition removal table you can put
straight into the report - "SOR removed 0.4% in clear, 6.1% in storm_night" is
a result, not just a preprocessing note.
"""

import argparse
import collections
import glob
import json
import os
import shutil
import time

import numpy as np


def _sor_scipy(points, k, std_ratio):
    from scipy.spatial import cKDTree
    xyz = points[:, :3]
    tree = cKDTree(xyz)
    # k+1 because the first neighbour of a point is itself
    d, _ = tree.query(xyz, k=min(k + 1, len(xyz)), workers=-1)
    mean_d = d[:, 1:].mean(axis=1)
    thresh = mean_d.mean() + std_ratio * mean_d.std()
    return points[mean_d <= thresh]


def _sor_open3d(points, k, std_ratio):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))
    _, keep = pcd.remove_statistical_outlier(nb_neighbors=k, std_ratio=std_ratio)
    return points[np.asarray(keep)]


def sor(points, k=20, std_ratio=2.0, backend='auto'):
    if points.shape[0] < k + 2:
        return points
    if backend in ('auto', 'scipy'):
        try:
            return _sor_scipy(points, k, std_ratio)
        except ImportError:
            if backend == 'scipy':
                raise
    return _sor_open3d(points, k, std_ratio)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', default=os.path.expanduser('~/dataset/lidar_raw'))
    ap.add_argument('--out', dest='out', default=os.path.expanduser('~/dataset/lidar_sor'))
    ap.add_argument('--k', type=int, default=20, help='neighbours per point')
    ap.add_argument('--std-ratio', type=float, default=2.0, help='lower = more aggressive')
    ap.add_argument('--backend', default='auto', choices=['auto', 'scipy', 'open3d'])
    ap.add_argument('--limit', type=int, default=0, help='process only N files (for testing)')
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, 'lidar'), exist_ok=True)
    os.makedirs(os.path.join(args.out, 'meta'), exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.inp, 'lidar', '*.npy')))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit('no .npy files in %s/lidar' % args.inp)
    print('%d sweeps, k=%d std_ratio=%.1f' % (len(files), args.k, args.std_ratio))

    stats = collections.defaultdict(lambda: dict(n=0, before=0, after=0))
    t0 = time.time()

    for i, p in enumerate(files):
        stem = os.path.splitext(os.path.basename(p))[0]
        meta_p = os.path.join(args.inp, 'meta', stem + '.json')
        if not os.path.exists(meta_p):
            continue
        with open(meta_p) as fh:
            meta = json.load(fh)
        cond = meta.get('condition', 'unknown')

        pts = np.load(p)
        clean = sor(pts, args.k, args.std_ratio, args.backend)

        np.save(os.path.join(args.out, 'lidar', stem + '.npy'), clean.astype(np.float32))
        meta['n_lidar_points_raw'] = int(pts.shape[0])
        meta['n_lidar_points'] = int(clean.shape[0])
        meta['sor'] = dict(k=args.k, std_ratio=args.std_ratio)
        with open(os.path.join(args.out, 'meta', stem + '.json'), 'w') as fh:
            json.dump(meta, fh)

        s = stats[cond]
        s['n'] += 1
        s['before'] += int(pts.shape[0])
        s['after'] += int(clean.shape[0])

        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print('  %d/%d  %.1f f/s' % (i + 1, len(files), (i + 1) / max(el, 1)), flush=True)

    print('\n================ SOR REMOVAL BY CONDITION ================')
    print('  %-14s %6s %12s %12s %9s' % ('condition', 'frames', 'mean before', 'mean after', 'removed'))
    rows = []
    for cond in sorted(stats):
        s = stats[cond]
        b = s['before'] / max(s['n'], 1)
        a = s['after'] / max(s['n'], 1)
        pct = 100.0 * (1 - a / max(b, 1))
        rows.append((cond, s['n'], b, a, pct))
        print('  %-14s %6d %12.0f %12.0f %8.2f%%' % (cond, s['n'], b, a, pct))

    with open(os.path.join(args.out, 'sor_stats.json'), 'w') as fh:
        json.dump([dict(condition=c, frames=n, mean_before=b, mean_after=a, removed_pct=p)
                   for c, n, b, a, p in rows], fh, indent=1)
    print('\nwrote %s' % os.path.join(args.out, 'sor_stats.json'))

    if rows:
        clear = [r for r in rows if r[0] == 'clear']
        worst = max(rows, key=lambda r: r[4])
        if clear and worst[4] > clear[0][4] + 1.0:
            print('\nGOOD: SOR removes more in %s (%.2f%%) than in clear (%.2f%%),'
                  % (worst[0], worst[4], clear[0][4]))
            print('      which is exactly the weather-artefact behaviour you want to report.')
        elif clear:
            print('\nNOTE: removal is similar across conditions. Either the weather coupling')
            print('      is too mild, or std_ratio is too high. Try --std-ratio 1.5.')


if __name__ == '__main__':
    main()
