#!/usr/bin/env python3
"""Contact sheet of one frame per condition, labelled. Figure for the report."""
import argparse, os, sys, cv2, numpy as np
sys.path.insert(0, os.path.expanduser('~/capstone/scripts'))
import config as C

ap = argparse.ArgumentParser()
ap.add_argument('--in', dest='src', required=True)
ap.add_argument('--out', default=os.path.expanduser('~/condition_grid.jpg'))
ap.add_argument('--cols', type=int, default=6)
ap.add_argument('--cell', type=int, default=380)
a = ap.parse_args()

order = ['clear'] + C.CONDITIONS_ADVERSE
tiers = {'clear': 'REF'}
for c in C.CONDITIONS_MILD:    tiers[c] = 'MILD'
for c in C.CONDITIONS_SEVERE:  tiers[c] = 'SEVERE'
for c in C.CONDITIONS_EXTREME: tiers[c] = 'EXTREME'
colour = {'REF': (200,200,200), 'MILD': (120,220,120),
          'SEVERE': (80,180,255), 'EXTREME': (80,80,255)}

cells = []
for c in order:
    for ext in ('.png', '.jpg'):
        p = os.path.join(os.path.expanduser(a.src), c + ext)
        if os.path.exists(p): break
    else:
        continue
    im = cv2.imread(p)
    if im is None: continue
    h, w = im.shape[:2]
    s = a.cell / max(h, w)
    im = cv2.resize(im, (int(w*s), int(h*s)))
    pad = np.zeros((a.cell, a.cell, 3), np.uint8)
    y, x = (a.cell-im.shape[0])//2, (a.cell-im.shape[1])//2
    pad[y:y+im.shape[0], x:x+im.shape[1]] = im
    t = tiers.get(c, '')
    cv2.rectangle(pad, (0, a.cell-30), (a.cell, a.cell), (0,0,0), -1)
    cv2.putText(pad, c, (6, a.cell-10), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (255,255,255), 1, cv2.LINE_AA)
    cv2.rectangle(pad, (0,0), (a.cell-1, a.cell-1), colour.get(t,(255,255,255)), 3)
    cells.append(pad)

if not cells:
    sys.exit('no images found in %s' % a.src)

cols = a.cols
rows = [cells[i:i+cols] for i in range(0, len(cells), cols)]
while len(rows[-1]) < cols:
    rows[-1].append(np.zeros((a.cell, a.cell, 3), np.uint8))
cv2.imwrite(a.out, np.vstack([np.hstack(r) for r in rows]),
            [cv2.IMWRITE_JPEG_QUALITY, 92])
print('wrote %s  (%d cells)' % (a.out, len(cells)))
