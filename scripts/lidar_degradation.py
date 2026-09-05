#!/usr/bin/env python3
"""LiDAR degradation table: points retained per condition against clear.

Reads n_lidar_points and the sensor parameters straight out of the capture
meta json, so the table is measured, not asserted.

  python3 lidar_degradation.py --raw ~/dataset/lidar_raw --ref ~/dataset/lidar_ref
"""

import argparse, collections, glob, json, os, statistics, sys

ap = argparse.ArgumentParser()
ap.add_argument('--raw', default=os.path.expanduser('~/dataset/lidar_raw'))
ap.add_argument('--ref', default=os.path.expanduser('~/dataset/lidar_ref'))
ap.add_argument('--csv', default=os.path.expanduser('~/capstone/docs/lidar_degradation.csv'))
ap.add_argument('--md',  default=os.path.expanduser('~/capstone/docs/lidar_degradation.md'))
a = ap.parse_args()

TIER = {}
try:
    sys.path.insert(0, os.path.expanduser('~/capstone/scripts'))
    import config as C
    for c in C.CONDITIONS_MILD:    TIER[c] = 'mild'
    for c in C.CONDITIONS_SEVERE:  TIER[c] = 'severe'
    for c in C.CONDITIONS_EXTREME: TIER[c] = 'extreme'
    TIER['clear'] = 'reference'
except Exception:
    pass

def collect(root):
    pts, par = collections.defaultdict(list), {}
    for p in glob.glob(os.path.join(os.path.expanduser(root), 'meta', '*.json')):
        try:
            j = json.load(open(p))
        except Exception:
            continue
        c = j.get('condition', '?')
        n = j.get('n_lidar_points')
        if n:
            pts[c].append(n)
        if c not in par and 'lidar_params' in j:
            par[c] = j['lidar_params']
    return pts, par

pts, par = collect(a.raw)
rpts, rpar = collect(a.ref)
pts.update(rpts); par.update(rpar)

if 'clear' not in pts:
    sys.exit('no clear frames found under %s - cannot compute a baseline' % a.ref)

base = statistics.mean(pts['clear'])
print('baseline: clear = %.0f points (n=%d)\n' % (base, len(pts['clear'])))

rows = []
for c, v in pts.items():
    p = par.get(c, {})
    rows.append(dict(
        condition=c, tier=TIER.get(c, ''), n=len(v),
        mean=statistics.mean(v),
        std=statistics.pstdev(v) if len(v) > 1 else 0.0,
        retained=100.0 * statistics.mean(v) / base,
        atten=p.get('atmosphere_attenuation_rate', float('nan')),
        dropoff=p.get('dropoff_general_rate', float('nan')),
        noise=p.get('noise_stddev', float('nan'))))
rows.sort(key=lambda r: -r['mean'])

hdr = '%-20s %-9s %6s %9s %8s %9s %8s %8s %7s' % (
    'condition', 'tier', 'n', 'mean pts', 'sd', 'retained', 'atten', 'dropoff', 'noise')
print(hdr); print('-' * len(hdr))
for r in rows:
    print('%-20s %-9s %6d %9.0f %8.0f %8.1f%% %8.3f %8.2f %7.2f' % (
        r['condition'], r['tier'], r['n'], r['mean'], r['std'],
        r['retained'], r['atten'], r['dropoff'], r['noise']))

os.makedirs(os.path.dirname(a.csv), exist_ok=True)
with open(a.csv, 'w') as fh:
    fh.write('condition,tier,n,mean_points,sd,retained_pct,attenuation,dropoff,noise\n')
    for r in rows:
        fh.write('%s,%s,%d,%.0f,%.0f,%.2f,%.4f,%.2f,%.3f\n' % (
            r['condition'], r['tier'], r['n'], r['mean'], r['std'],
            r['retained'], r['atten'], r['dropoff'], r['noise']))

with open(a.md, 'w') as fh:
    fh.write('# LiDAR degradation by weather condition\n\n')
    fh.write('Baseline: `clear` = %.0f points per sweep (n=%d).\n\n' % (base, len(pts['clear'])))
    fh.write('| Condition | Tier | Frames | Mean points | Retained | Attenuation | Dropoff | Noise |\n')
    fh.write('|---|---|---|---|---|---|---|---|\n')
    for r in rows:
        fh.write('| `%s` | %s | %d | %.0f | %.1f%% | %.3f | %.2f | %.2f |\n' % (
            r['condition'], r['tier'], r['n'], r['mean'],
            r['retained'], r['atten'], r['dropoff'], r['noise']))

print('\nwritten: %s' % a.csv)
print('written: %s' % a.md)

nights = [r for r in rows if r['condition'] in ('night', 'midnight')]
if nights:
    print('\nDarkness check (active sensor - should be near 100%%):')
    for r in nights:
        print('  %-12s %.1f%% of clear' % (r['condition'], r['retained']))
