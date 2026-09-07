#!/usr/bin/env python3
"""Per-condition mAP. Slices the test split by weather and evaluates each."""
import argparse, collections, glob, json, os, shutil, sys, tempfile
import yaml
from ultralytics import YOLO

def condition_of(stem, meta_dir, known):
    p = os.path.join(meta_dir, stem + '.json')
    if os.path.exists(p):
        try:
            return json.load(open(p)).get('condition', '?')
        except Exception:
            pass
    for c in sorted(known, key=len, reverse=True):
        if '_%s_' % c in stem or stem.endswith('_' + c):
            return c
    return '?'

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--data', required=True)
ap.add_argument('--split', default='test', choices=['train','val','test'])
ap.add_argument('--meta', default='')
ap.add_argument('--imgsz', type=int, default=640)
ap.add_argument('--batch', type=int, default=16)
ap.add_argument('--device', default='0')
ap.add_argument('--conf', type=float, default=0.001)
ap.add_argument('--iou', type=float, default=0.6)
ap.add_argument('--obb', action='store_true')
ap.add_argument('--min-images', type=int, default=25)
ap.add_argument('--out', default=os.path.expanduser('~/capstone/docs/eval_by_condition.csv'))
args = ap.parse_args()

root = os.path.expanduser(args.data)
img_dir = os.path.join(root, 'images', args.split)
lbl_dir = os.path.join(root, 'labels', args.split)
if not os.path.isdir(img_dir):
    sys.exit('no such split: %s' % img_dir)

cfg = yaml.safe_load(open(os.path.join(root, 'data.yaml')))
names = cfg.get('names')
if isinstance(names, dict):
    names = [names[k] for k in sorted(names)]

meta_dir = os.path.expanduser(args.meta) if args.meta else ''
try:
    sys.path.insert(0, os.path.expanduser('~/capstone/scripts'))
    import config as C
    known = list(C.CONDITIONS)
except Exception:
    known = []

imgs = sorted(glob.glob(os.path.join(img_dir, '*.*')))
print('model : %s' % args.model)
print('data  : %s  (%s, %d images)' % (root, args.split, len(imgs)))
print('meta  : %s%s' % (meta_dir, '' if os.path.isdir(meta_dir) else '  [absent - using filenames]'))

groups = collections.defaultdict(list)
for p in imgs:
    stem = os.path.splitext(os.path.basename(p))[0]
    groups[condition_of(stem, meta_dir, known)].append(p)

print('\nconditions found: %d' % len(groups))
for c in sorted(groups):
    print('  %-22s %5d' % (c, len(groups[c])))

model = YOLO(args.model)
task = 'obb' if args.obb else 'detect'
rows = []
work = tempfile.mkdtemp(prefix='evalcond_')
try:
    for cond in ['__ALL__'] + sorted(groups):
        files = imgs if cond == '__ALL__' else groups[cond]
        if cond != '__ALL__' and len(files) < args.min_images:
            print('\nskipping %s (%d images)' % (cond, len(files))); continue
        d = os.path.join(work, cond.strip('_') or 'ALL')
        io_, lo_ = os.path.join(d,'images'), os.path.join(d,'labels')
        os.makedirs(io_, exist_ok=True); os.makedirs(lo_, exist_ok=True)
        for p in files:
            stem = os.path.splitext(os.path.basename(p))[0]
            l = os.path.join(lbl_dir, stem + '.txt')
            di = os.path.join(io_, os.path.basename(p))
            dl = os.path.join(lo_, stem + '.txt')
            if not os.path.lexists(di): os.symlink(os.path.abspath(p), di)
            if os.path.exists(l) and not os.path.lexists(dl):
                os.symlink(os.path.abspath(l), dl)
        y = os.path.join(d, 'data.yaml')
        yaml.safe_dump({'path': d, 'train': 'images', 'val': 'images',
                        'names': {i: n for i, n in enumerate(names)}}, open(y,'w'))
        print('\n=== %s  (%d images) ===' % (cond, len(files)), flush=True)
        try:
            m = model.val(data=y, split='val', imgsz=args.imgsz, batch=args.batch,
                          device=args.device, conf=args.conf, iou=args.iou,
                          task=task, plots=False, verbose=False,
                          project=work, name=(cond.strip('_') or 'ALL')+'_v', exist_ok=True)
            box = getattr(m, 'obb', None) or m.box
            rows.append(dict(condition=cond.strip('_') or 'ALL', n=len(files),
                             map50=float(box.map50), map=float(box.map),
                             precision=float(box.mp), recall=float(box.mr)))
            print('  mAP50 %.4f  mAP50-95 %.4f  P %.3f  R %.3f'
                  % (box.map50, box.map, box.mp, box.mr))
        except Exception as e:
            print('  FAILED: %s' % e)

    rows.sort(key=lambda r: (r['condition'] != 'ALL', -r['map50']))
    print('\n' + '='*74)
    print('%-22s %7s %9s %10s %8s %8s' % ('condition','images','mAP@50','mAP@50-95','P','R'))
    print('-'*74)
    for r in rows:
        print('%-22s %7d %9.4f %10.4f %8.3f %8.3f'
              % (r['condition'], r['n'], r['map50'], r['map'], r['precision'], r['recall']))

    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out,'w') as fh:
        fh.write('condition,images,map50,map50_95,precision,recall\n')
        for r in rows:
            fh.write('%s,%d,%.4f,%.4f,%.4f,%.4f\n' % (r['condition'], r['n'],
                     r['map50'], r['map'], r['precision'], r['recall']))
    with open(out.replace('.csv','.md'),'w') as fh:
        fh.write('| Condition | Images | mAP@50 | mAP@50-95 | P | R |\n|---|---|---|---|---|---|\n')
        for r in rows:
            fh.write('| `%s` | %d | %.3f | %.3f | %.3f | %.3f |\n' % (r['condition'],
                     r['n'], r['map50'], r['map'], r['precision'], r['recall']))
    print('\nwritten: %s (and .md)' % out)
finally:
    shutil.rmtree(work, ignore_errors=True)
