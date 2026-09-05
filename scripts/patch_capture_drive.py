#!/usr/bin/env python3
"""Add --negatives-only and --tag to capture_drive.py. Idempotent."""
import shutil, sys
p = 'capture_drive.py'
s = open(p).read()
if '--negatives-only' in s:
    print('already patched'); sys.exit(0)

a1 = """    ap.add_argument('--no-lidar', action='store_true',"""
a2 = """    sev_tag = '' if sev == 1.0 else '_s%d' % round(sev * 100)"""
a3 = """                if not rows and rng.random() > C.BACKGROUND_KEEP:
                    n_skip_bg += 1
                    continue
"""
for n, a in (('a1', a1), ('a2', a2), ('a3', a3)):
    if s.count(a) != 1:
        sys.exit('anchor %s found %d times - file differs, aborting' % (n, s.count(a)))

shutil.copy(p, p + '.bak')

s = s.replace(a1, """    ap.add_argument('--negatives-only', action='store_true',
                    help='keep ONLY frames with no traffic light - street layout negatives')
    ap.add_argument('--tag', default='',
                    help='extra stem tag, e.g. "neg", so these get their own counter '
                         'and never collide with positives in the same directory')
""" + a1)

s = s.replace(a2, a2 + """
    if args.tag:
        sev_tag += '_' + args.tag""")

s = s.replace(a3, """                if args.negatives_only:
                    if rows:
                        n_skip_bg += 1
                        continue
                elif not rows and rng.random() > C.BACKGROUND_KEEP:
                    n_skip_bg += 1
                    continue
""")
open(p, 'w').write(s)
print('patched capture_drive.py (backup: capture_drive.py.bak)')
