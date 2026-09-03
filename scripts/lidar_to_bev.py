#!/usr/bin/env python3
"""
LiDAR sweep -> BEV pseudo-image + YOLOv8-OBB labels.

Reads the .npy point clouds and .json metadata written by carla_capture.py and produces
a YOLO-OBB dataset that YOLOv8 can train on directly.

  python lidar_to_bev.py --in ./dataset --out ./dataset/lidar_bev

Channels of the output image:
  R = normalised max height in cell
  G = normalised log point density
  B = max intensity

Label format (YOLOv8 OBB), one line per object, all values normalised 0-1:
  <class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
"""

import argparse
import glob
import json
import os

import numpy as np
from PIL import Image

# BEV extent in metres, relative to the LiDAR
X_RANGE = (-50.0, 50.0)
Y_RANGE = (-50.0, 50.0)
Z_RANGE = (-2.5, 3.5)
RES = 0.1                       # metres per pixel -> 1000 x 1000
MIN_POINTS_IN_BOX = 5           # drop objects the LiDAR never actually saw

W = int((X_RANGE[1] - X_RANGE[0]) / RES)
H = int((Y_RANGE[1] - Y_RANGE[0]) / RES)

CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle", "pedestrian"]


def classify(type_id, group):
    if group == "pedestrian":
        return 5
    t = type_id.lower()
    if "motorcycle" in t or "harley" in t or "yamaha" in t or "kawasaki" in t or "vespa" in t:
        return 3
    if "bike" in t or "bicycle" in t or "crossbike" in t or "omafiets" in t or "century" in t:
        return 4
    if "bus" in t:
        return 2
    if "truck" in t or "carlacola" in t or "firetruck" in t or "ambulance" in t or "sprinter" in t:
        return 1
    return 0


def to_pixel(x, y):
    """LiDAR metres -> BEV pixels. x forward -> image up, y right -> image right."""
    col = (y - Y_RANGE[0]) / RES
    row = (X_RANGE[1] - x) / RES
    return col, row


def rasterise(points):
    """points: (N,4) x,y,z,intensity -> (H,W,3) uint8"""
    mask = ((points[:, 0] > X_RANGE[0]) & (points[:, 0] < X_RANGE[1]) &
            (points[:, 1] > Y_RANGE[0]) & (points[:, 1] < Y_RANGE[1]) &
            (points[:, 2] > Z_RANGE[0]) & (points[:, 2] < Z_RANGE[1]))
    p = points[mask]

    height = np.zeros((H, W), dtype=np.float32)
    density = np.zeros((H, W), dtype=np.float32)
    intensity = np.zeros((H, W), dtype=np.float32)
    if p.shape[0] == 0:
        return np.zeros((H, W, 3), dtype=np.uint8)

    cols = np.clip(((p[:, 1] - Y_RANGE[0]) / RES).astype(np.int32), 0, W - 1)
    rows = np.clip(((X_RANGE[1] - p[:, 0]) / RES).astype(np.int32), 0, H - 1)

    np.maximum.at(height, (rows, cols), p[:, 2])
    np.add.at(density, (rows, cols), 1.0)
    np.maximum.at(intensity, (rows, cols), p[:, 3])

    height = (height - Z_RANGE[0]) / (Z_RANGE[1] - Z_RANGE[0])
    height = np.clip(height, 0.0, 1.0)
    density = np.log1p(density) / np.log(64.0)
    density = np.clip(density, 0.0, 1.0)
    intensity = np.clip(intensity, 0.0, 1.0)

    img = np.stack([height, density, intensity], axis=-1)
    return (img * 255).astype(np.uint8)


def footprint_corners(verts):
    """8 box vertices in the LiDAR frame -> 4 footprint corners, ordered.

    The vertices are axis-aligned in the actor's local frame before transform, so the
    footprint is the convex hull of the xy projection. With a rigid transform that hull
    is exactly the 4 distinct xy positions.
    """
    xy = np.array([[v[0], v[1]] for v in verts], dtype=np.float64)
    uniq = []
    for p in xy:
        if not any(np.allclose(p, u, atol=1e-3) for u in uniq):
            uniq.append(p)
    uniq = np.array(uniq)
    if uniq.shape[0] < 4:
        return None
    if uniq.shape[0] > 4:                      # numerical noise - take extreme 4 by angle
        c = uniq.mean(axis=0)
        ang = np.arctan2(uniq[:, 1] - c[1], uniq[:, 0] - c[0])
        order = np.argsort(ang)
        uniq = uniq[order]
        step = max(1, len(uniq) // 4)
        uniq = uniq[::step][:4]
    c = uniq.mean(axis=0)
    ang = np.arctan2(uniq[:, 1] - c[1], uniq[:, 0] - c[0])
    return uniq[np.argsort(ang)]


def points_in_box(points, verts):
    zs = [v[2] for v in verts]
    corners = footprint_corners(verts)
    if corners is None:
        return 0
    lo = np.array([corners[:, 0].min(), corners[:, 1].min(), min(zs)])
    hi = np.array([corners[:, 0].max(), corners[:, 1].max(), max(zs)])
    m = np.all((points[:, :3] >= lo) & (points[:, :3] <= hi), axis=1)
    return int(np.count_nonzero(m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="./dataset")
    ap.add_argument("--out", dest="out", default="./dataset/lidar_bev")
    ap.add_argument("--lidar-dir", default="", help="override <in>/lidar (e.g. SOR output)")
    args = ap.parse_args()

    img_dir = os.path.join(args.out, "images")
    lbl_dir = os.path.join(args.out, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    lidar_dir = args.lidar_dir or os.path.join(args.inp, "lidar")
    files = sorted(glob.glob(os.path.join(lidar_dir, "*.npy")))
    print(f"{len(files)} sweeps")

    kept_total = dropped_total = 0

    for i, npy in enumerate(files):
        stem = os.path.splitext(os.path.basename(npy))[0]
        meta_path = os.path.join(args.inp, "meta", stem + ".json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as fh:
            meta = json.load(fh)

        points = np.load(npy)
        Image.fromarray(rasterise(points)).save(os.path.join(img_dir, stem + ".png"))

        lines = []
        for box in meta.get("boxes_3d", []):
            verts = box["verts"]
            if points_in_box(points, verts) < MIN_POINTS_IN_BOX:
                dropped_total += 1
                continue
            corners = footprint_corners(verts)
            if corners is None:
                continue
            pix = [to_pixel(x, y) for x, y in corners]
            if any(c < -W or c > 2 * W or r < -H or r > 2 * H for c, r in pix):
                continue
            if all(c < 0 or c > W or r < 0 or r > H for c, r in pix):
                continue
            cls = classify(box["type_id"], box["group"])
            vals = []
            for c, r in pix:
                vals += [min(max(c / W, 0.0), 1.0), min(max(r / H, 0.0), 1.0)]
            lines.append(str(cls) + " " + " ".join(f"{v:.6f}" for v in vals))
            kept_total += 1

        with open(os.path.join(lbl_dir, stem + ".txt"), "w") as fh:
            fh.write("\n".join(lines))

        if (i + 1) % 100 == 0:
            print(f"{i+1}/{len(files)}")

    with open(os.path.join(args.out, "data.yaml"), "w") as fh:
        fh.write(f"path: {os.path.abspath(args.out)}\n")
        fh.write("train: images/train\nval: images/val\ntest: images/test\n")
        fh.write(f"nc: {len(CLASSES)}\nnames: {CLASSES}\n")

    print(f"kept {kept_total} boxes, dropped {dropped_total} with too few points")
    print("Now split images/ and labels/ into train/val/test BY TOWN, not randomly.")


if __name__ == "__main__":
    main()
