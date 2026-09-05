#!/usr/bin/env python3
"""Per-condition image enhancement for adverse-weather traffic-light detection.

Routes each frame to a method chosen by its weather condition, writes an
enhanced copy, and (with --metrics) reports PSNR/SSIM against the clear
reference set.

  python3 enhance_images.py --in ~/dataset/train --out ~/dataset/train_enh
  python3 enhance_images.py --in ~/dataset/train --out ~/dataset/train_enh \
          --conditions fog_dense,monsoon_night --limit 20 --draw

THE CONSTRAINT THAT SHAPES EVERYTHING HERE:
traffic-light STATE is the class label, and state is carried by HUE. Any
operation that shifts colour destroys the label. So contrast work happens on
the LAB L-channel only, never on RGB, and retinex is applied to luminance
rather than per-channel.
"""

import argparse
import glob
import json
import os
import time

import cv2
import numpy as np

# --------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------

def clahe_l(bgr, clip=2.5, grid=8):
    """CLAHE on the LAB lightness channel. Hue and chroma untouched."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def dark_channel(bgr, patch=15):
    mn = np.min(bgr, axis=2)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    return cv2.erode(mn, k)


def atmospheric_light(bgr, dark, top=0.001):
    h, w = dark.shape
    n = max(int(h * w * top), 1)
    idx = np.argpartition(dark.ravel(), -n)[-n:]
    return bgr.reshape(-1, 3)[idx].max(axis=0).astype(np.float64)


def dcp(bgr, omega=0.92, patch=15, t0=0.1, guided=True):
    """Dark Channel Prior dehazing (He et al.), with guided-filter refinement."""
    f = bgr.astype(np.float64)
    dark = dark_channel(bgr, patch)
    A = np.maximum(atmospheric_light(bgr, dark), 1.0)

    norm = f / A
    t = 1.0 - omega * dark_channel(np.uint8(np.clip(norm * 255, 0, 255)), patch) / 255.0

    if guided:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
        t = guided_filter(gray, t, r=40, eps=1e-3)

    t = np.clip(t, t0, 1.0)[:, :, None]
    out = (f - A) / t + A
    return np.uint8(np.clip(out, 0, 255))


def guided_filter(guide, src, r=40, eps=1e-3):
    mg = cv2.boxFilter(guide, cv2.CV_64F, (r, r))
    ms = cv2.boxFilter(src,   cv2.CV_64F, (r, r))
    cov = cv2.boxFilter(guide * src, cv2.CV_64F, (r, r)) - mg * ms
    var = cv2.boxFilter(guide * guide, cv2.CV_64F, (r, r)) - mg * mg
    a = cov / (var + eps)
    b = ms - a * mg
    return cv2.boxFilter(a, cv2.CV_64F, (r, r)) * guide + \
           cv2.boxFilter(b, cv2.CV_64F, (r, r))


def msr_luminance(bgr, scales=(15, 80, 250)):
    """Multi-scale retinex applied to LUMINANCE only.

    Classic MSRCR runs per-channel and shifts colour balance - unusable here,
    because a hue shift relabels a red lamp as amber. Restricting it to L keeps
    the illumination correction and leaves chroma alone.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    lf = l.astype(np.float64) + 1.0

    acc = np.zeros_like(lf)
    for s in scales:
        blur = cv2.GaussianBlur(lf, (0, 0), s)
        acc += np.log10(lf) - np.log10(blur + 1.0)
    acc /= len(scales)

    lo, hi = np.percentile(acc, 1), np.percentile(acc, 99)
    out = np.clip((acc - lo) / max(hi - lo, 1e-6), 0, 1) * 255.0
    return cv2.cvtColor(cv2.merge([np.uint8(out), a, b]), cv2.COLOR_LAB2BGR)


def adaptive_gamma(bgr, target=0.45):
    """Push mean luminance toward `target`. Gamma is monotonic per channel, so
    it rescales brightness without rotating hue."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    mean = lab[:, :, 0].mean() / 255.0
    mean = min(max(mean, 0.02), 0.98)
    g = np.log(target) / np.log(mean)
    g = float(np.clip(g, 0.4, 2.5))
    lut = np.array([((i / 255.0) ** g) * 255 for i in range(256)], np.uint8)
    return cv2.LUT(bgr, lut)


def bilateral(bgr, d=7, sc=60, ss=60):
    """Edge-preserving smoothing - removes rain streaks without softening lamps."""
    return cv2.bilateralFilter(bgr, d, sc, ss)


def median(bgr, k=5):
    """Kills impulsive hail/snow particles."""
    return cv2.medianBlur(bgr, k)


# --------------------------------------------------------------------------
# per-condition routing
# --------------------------------------------------------------------------

def enh_fog(img):        return clahe_l(adaptive_gamma(dcp(img)), clip=2.0)
def enh_fog_night(img):  return clahe_l(msr_luminance(dcp(img, omega=0.85)), clip=2.0)
def enh_night(img):      return clahe_l(msr_luminance(img), clip=2.0)
def enh_rain(img):       return clahe_l(bilateral(img), clip=2.5)
def enh_rain_night(img): return clahe_l(msr_luminance(bilateral(img)), clip=2.0)
def enh_hail(img):       return clahe_l(median(img), clip=2.5)
def enh_glare(img):      return clahe_l(adaptive_gamma(img, target=0.42), clip=1.8)
def enh_glare_fog(img):  return clahe_l(adaptive_gamma(dcp(img), target=0.42), clip=1.8)

ROUTE = {
    # fog family - scattering dominates, dehaze first
    'fog':               ('DCP + adaptive gamma + CLAHE-L', enh_fog),
    'haze':              ('DCP + adaptive gamma + CLAHE-L', enh_fog),
    'fog_dense':         ('DCP + adaptive gamma + CLAHE-L', enh_fog),
    'fog_whiteout':      ('DCP + adaptive gamma + CLAHE-L', enh_fog),
    # fog at night - dehaze then retinex
    'fog_night':         ('DCP + MSR-L + CLAHE-L', enh_fog_night),
    'fog_night_extreme': ('DCP + MSR-L + CLAHE-L', enh_fog_night),
    'deep_night_fog':    ('DCP + MSR-L + CLAHE-L', enh_fog_night),
    # darkness - illumination correction only
    'night':             ('MSR-L + CLAHE-L', enh_night),
    'midnight':          ('MSR-L + CLAHE-L', enh_night),
    'wet_night':         ('MSR-L + CLAHE-L', enh_night),
    # rain - edge-preserving denoise
    'rain':              ('bilateral + CLAHE-L', enh_rain),
    'storm_day':         ('bilateral + CLAHE-L', enh_rain),
    'monsoon':           ('bilateral + CLAHE-L', enh_rain),
    'storm_night':       ('bilateral + MSR-L + CLAHE-L', enh_rain_night),
    'monsoon_night':     ('bilateral + MSR-L + CLAHE-L', enh_rain_night),
    'storm_dusk':        ('bilateral + MSR-L + CLAHE-L', enh_rain_night),
    # impulsive particles
    'hail':              ('median + CLAHE-L', enh_hail),
    'hail_severe':       ('median + CLAHE-L', enh_hail),
    'blizzard':          ('median + CLAHE-L', enh_hail),
    # backlight
    'glare_dawn':        ('adaptive gamma + CLAHE-L', enh_glare),
    'glare_wet':         ('adaptive gamma + CLAHE-L', enh_glare),
    'glare_fog':         ('DCP + adaptive gamma + CLAHE-L', enh_glare_fog),
    # reference - untouched
    'clear':             ('none (reference)', lambda x: x),
}


def condition_of(stem, meta_dir):
    p = os.path.join(meta_dir, stem + '.json')
    if os.path.exists(p):
        try:
            with open(p) as fh:
                return json.load(fh).get('condition', '?')
        except Exception:
            pass
    for c in sorted(ROUTE, key=len, reverse=True):     # longest match first
        if '_%s_' % c in stem:
            return c
    return '?'


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in',  dest='src', required=True)
    ap.add_argument('--out', dest='dst', required=True)
    ap.add_argument('--conditions', default='', help='comma list; default all')
    ap.add_argument('--limit', type=int, default=0, help='first N per condition')
    ap.add_argument('--draw', action='store_true',
                    help='also write side-by-side before/after into _pairs/')
    ap.add_argument('--metrics', action='store_true',
                    help='PSNR/SSIM of original and enhanced vs a clear reference')
    ap.add_argument('--ref', default=os.path.expanduser('~/dataset/reference'))
    ap.add_argument('--jobs', type=int, default=0, help='0 = cpu_count - 1')
    args = ap.parse_args()

    src = os.path.expanduser(args.src)
    dst = os.path.expanduser(args.dst)
    meta_dir = os.path.join(src, 'meta')
    os.makedirs(os.path.join(dst, 'images'), exist_ok=True)
    os.makedirs(os.path.join(dst, 'labels'), exist_ok=True)
    if args.draw:
        os.makedirs(os.path.join(dst, '_pairs'), exist_ok=True)

    want = {c.strip() for c in args.conditions.split(',') if c.strip()}

    imgs = sorted(glob.glob(os.path.join(src, 'images', '*.jpg')))
    print('source : %s  (%d images)' % (src, len(imgs)))
    print('dest   : %s' % dst)

    by_cond = {}
    for p in imgs:
        stem = os.path.splitext(os.path.basename(p))[0]
        c = condition_of(stem, meta_dir)
        if want and c not in want:
            continue
        by_cond.setdefault(c, []).append((stem, p))

    print('\n  %-20s %8s  %s' % ('condition', 'frames', 'method'))
    print('  ' + '-' * 62)
    for c in sorted(by_cond):
        method = ROUTE.get(c, ('UNROUTED - copied as-is', None))[0]
        print('  %-20s %8d  %s' % (c, len(by_cond[c]), method))

    ref_img = None
    if args.metrics:
        try:
            from skimage.metrics import peak_signal_noise_ratio as psnr
            from skimage.metrics import structural_similarity as ssim
        except ImportError:
            print('\n  scikit-image missing:  pip install scikit-image --break-system-packages')
            args.metrics = False
        else:
            refs = sorted(glob.glob(os.path.join(os.path.expanduser(args.ref),
                                                 'images', '*_clear_*.jpg')))
            if not refs:
                print('\n  no clear reference images under %s - metrics disabled' % args.ref)
                args.metrics = False
            else:
                ref_img = cv2.imread(refs[0])
                print('\n  metrics reference: %s' % os.path.basename(refs[0]))

    t0 = time.time()
    n = 0
    rows = {}
    for c in sorted(by_cond):
        items = by_cond[c]
        if args.limit:
            items = items[:args.limit]
        fn = ROUTE.get(c, (None, lambda x: x))[1]
        acc = []
        for stem, p in items:
            img = cv2.imread(p)
            if img is None:
                continue
            out = fn(img)
            cv2.imwrite(os.path.join(dst, 'images', stem + '.jpg'), out,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

            lbl = os.path.join(src, 'labels', stem + '.txt')
            if os.path.exists(lbl):
                with open(lbl) as a, open(os.path.join(dst, 'labels', stem + '.txt'), 'w') as b:
                    b.write(a.read())          # geometry unchanged, labels carry over

            if args.draw and len(acc) < 3:
                cv2.imwrite(os.path.join(dst, '_pairs', '%s_%s.jpg' % (c, stem)),
                            np.hstack([img, out]))

            if args.metrics and ref_img is not None and ref_img.shape == img.shape:
                g = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
                acc.append((
                    psnr(ref_img, img), psnr(ref_img, out),
                    ssim(g, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)),
                    ssim(g, cv2.cvtColor(out, cv2.COLOR_BGR2GRAY))))
            n += 1
            if n % 200 == 0:
                el = time.time() - t0
                print('  %6d done  %.1f img/s' % (n, n / max(el, 1)), flush=True)
        if acc:
            a = np.array(acc)
            rows[c] = a.mean(axis=0)

    if rows:
        print('\n  PSNR / SSIM vs clear reference')
        print('  %-20s %10s %10s %10s %10s' %
              ('condition', 'PSNR raw', 'PSNR enh', 'SSIM raw', 'SSIM enh'))
        print('  ' + '-' * 66)
        for c in sorted(rows):
            r = rows[c]
            print('  %-20s %10.2f %10.2f %10.4f %10.4f' % (c, r[0], r[1], r[2], r[3]))
        with open(os.path.join(dst, 'metrics.json'), 'w') as fh:
            json.dump({c: dict(psnr_raw=r[0], psnr_enh=r[1],
                               ssim_raw=r[2], ssim_enh=r[3])
                       for c, r in rows.items()}, fh, indent=2)
        print('\n  written: %s/metrics.json' % dst)
        print('  NOTE: PSNR/SSIM are proxies. A method that raises SSIM but')
        print('  lowers mAP is not an improvement for this task.')

    print('\nDONE  %d images  %.0f s' % (n, time.time() - t0))


if __name__ == '__main__':
    main()
