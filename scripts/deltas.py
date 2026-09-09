import csv, os, sys
def load(p):
    with open(os.path.expanduser(p)) as fh:
        return {r['condition']: r for r in csv.DictReader(fh)}
pairs = [
    ('~/capstone/docs/eval_tl_raw960.csv', '~/capstone/docs/eval_tl_enh960.csv',
     'CAMERA   raw -> enhanced'),
    ('~/capstone/docs/eval_bev_raw.csv',   '~/capstone/docs/eval_bev_sor.csv',
     'BEV      raw -> SOR denoised'),
]
for a, b, title in pairs:
    try:
        A, B = load(a), load(b)
    except FileNotFoundError:
        print('\n%s : not ready' % title); continue
    print('\n' + title)
    print('%-22s %9s %9s %9s' % ('condition', 'before', 'after', 'delta'))
    print('-' * 52)
    wins = 0; tot = 0
    for c in sorted(A, key=lambda k: (k != 'ALL', k)):
        if c not in B: continue
        x, y = float(A[c]['map50']), float(B[c]['map50'])
        print('%-22s %9.4f %9.4f %+9.4f' % (c, x, y, y - x))
        if c != 'ALL':
            tot += 1; wins += (y > x)
    print('improved in %d of %d conditions' % (wins, tot))
