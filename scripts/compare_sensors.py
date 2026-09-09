import csv, os
def load(p):
    p = os.path.expanduser(p)
    if not os.path.exists(p): return None
    with open(p) as fh:
        return {r['condition']: float(r['map50']) for r in csv.DictReader(fh)}
cam = load('~/capstone/docs/eval_tl_raw960.csv')
lid = load('~/capstone/docs/eval_bev_raw.csv')
if not (cam and lid):
    raise SystemExit('need both eval CSVs first')
rows = [(c, cam[c], lid[c], lid[c] - cam[c]) for c in cam if c in lid]
rows.sort(key=lambda r: (r[0] != 'ALL', -r[3]))
print('%-22s %9s %9s %11s' % ('condition', 'camera', 'lidar', 'lidar edge'))
print('-' * 55)
for c, a, b, d in rows:
    print('%-22s %9.4f %9.4f %+11.4f' % (c, a, b, d))
with open(os.path.expanduser('~/capstone/docs/sensor_comparison.md'), 'w') as fh:
    fh.write('| Condition | Camera mAP@50 | LiDAR mAP@50 | LiDAR advantage |\n|---|---|---|---|\n')
    for c, a, b, d in rows:
        fh.write('| `%s` | %.3f | %.3f | %+.3f |\n' % (c, a, b, d))
print('\nwritten: ~/capstone/docs/sensor_comparison.md')
