# Week 1 — Capture Infrastructure
## Every command, in order, to produce a verified 2,000-image dataset

Files in `scripts/`:

| File | What it does |
|---|---|
| `config.py` | Single source of truth — towns, classes, sensors, weather + LiDAR coupling |
| `carla_utils.py` | Shared: camera maths, sensor rig, traffic-light labelling, 3D boxes |
| `detect_tl_tag.py` | Works out the semantic tag id for traffic lights in **your** build |
| `capture_drive.py` | Capture Mode 1 — bulk training data |
| `verify_dataset.py` | The six-check gate. Do not train until it passes. |
| `make_splits.py` | Assembles the YOLO dataset, split **by town** |

Plus the two you already have: `carla_capture.py` (frozen-pose eval set) and `lidar_to_bev.py`.

**Target: 95 frames × 7 conditions × 3 towns = 1,995 images**, about 2.5 GB.

---

## STEP 0 — Get the scripts onto the box

From your **laptop** (PowerShell):

```powershell
ssh s21@100.107.251.96 "mkdir -p ~/capstone/scripts ~/logs ~/dataset"
scp scripts\*.py s21@100.107.251.96:/home/s21/capstone/scripts/
scp carla_capture.py lidar_to_bev.py s21@100.107.251.96:/home/s21/capstone/scripts/
```

Then get on the box and stay there for everything below:

```powershell
ssh s21@100.107.251.96
```

---

## STEP 1 — Environment

```bash
source ~/venv-carla/bin/activate
pip install --upgrade pip
pip install numpy pillow opencv-python scikit-image tqdm ultralytics open3d

cd ~/capstone/scripts
python3 -c "import carla, numpy, PIL, cv2, skimage; print('deps ok')"
echo $CARLA_ROOT && ls $CARLA_ROOT/CarlaUE4.sh
df -h ~ | tail -1
```

You need ~5 GB free for this run. If `df` says otherwise, stop and clear space first.

Record the versions now — you need them in the report and you will forget:

```bash
mkdir -p ~/capstone/docs
{ python3 -c "
import sys, carla, torch, ultralytics
print('python     ', sys.version.split()[0])
print('carla-py   ', carla.__file__)
print('torch      ', torch.__version__, 'cuda', torch.cuda.is_available())
print('ultralytics', ultralytics.__version__)"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  cat $CARLA_ROOT/VERSION 2>/dev/null | sed 's/^/carla-server /'
} | tee ~/capstone/docs/versions.txt
```

---

## STEP 2 — Start the CARLA server

```bash
tmux new -s carla
```

Inside tmux (one line, no backslashes):

```bash
cd $CARLA_ROOT && ./CarlaUE4.sh -RenderOffScreen -quality-level=Epic -carla-rpc-port=2000 -carla-streaming-port=2001
```

Banner appears, then silence — that is the server running. Wait ~60 s, then **`Ctrl+B`, release, `D`**.

```bash
ss -tlnp | grep -E '2000|2001'      # both must be LISTEN
```

`-quality-level=Epic` is not optional. On `Low`, CARLA disables fog, rain and wet-road
rendering, and all seven conditions come out looking identical.

---

## STEP 3 — Find the traffic-light semantic tag (5 minutes, saves a week)

`config.py` guesses tag 18. The palette changes between CARLA releases. Check yours:

```bash
cd ~/capstone/scripts
python3 detect_tl_tag.py --town Town03 --samples 25
```

It projects real traffic-light boxes into the semantic image and reports which tag dominates
inside them. If it prints something other than 18:

```bash
sed -i 's/^SEM_TAG_TRAFFIC_LIGHT = .*/SEM_TAG_TRAFFIC_LIGHT = <THE NUMBER>/' config.py
grep SEM_TAG config.py
```

If this is wrong, the occlusion filter rejects every box and you get an empty dataset.

---

## STEP 4 — Smoke test (10 frames, one condition, ~2 minutes)

Never launch a long run without this.

```bash
cd ~/capstone/scripts
python3 capture_drive.py --town Town03 --frames 10 --conditions clear \
        --vehicles 40 --walkers 15 --out ~/dataset/smoke
```

Expect lines like `clear 10/10  tl=2 pts=64231`. Then:

```bash
ls ~/dataset/smoke/images | head
python3 verify_dataset.py --root ~/dataset/smoke --draw 10
```

**Now actually look at the pictures.** Copy them to your laptop:

```powershell
scp -r s21@100.107.251.96:/home/s21/dataset/smoke/_verify ./smoke_check
```

Open them. Boxes must sit on lamp heads — not on poles, not floating in the sky — and the label
colour must match the lit lamp. If `tl=0` on every line, the semantic tag is wrong (Step 3).

---

## STEP 5 — Two-condition test (checks the LiDAR coupling, ~5 minutes)

```bash
python3 capture_drive.py --town Town03 --frames 15 --conditions clear,fog \
        --out ~/dataset/smoke2
python3 verify_dataset.py --root ~/dataset/smoke2 --draw 6
```

Look at check `[2]`. Fog **must** show fewer LiDAR points than clear — around −20% to −30%. If it
shows 0%, the LiDAR attributes did not take effect, and your entire adverse-weather LiDAR claim is
unsupported. `capture_drive.py` respawns the sensor per condition, which is what makes it work.

```bash
rm -rf ~/dataset/smoke ~/dataset/smoke2      # only once both smoke tests pass
```

---

## STEP 6 — The real run: 1,995 images

Three towns, in tmux, logged. Roughly 20–40 minutes each.

```bash
tmux new -s capture
cd ~/capstone/scripts
source ~/venv-carla/bin/activate

for T in Town01 Town03 Town10HD_Opt; do
  echo "=== $T ==="
  python3 capture_drive.py --town $T --frames 95 --vehicles 60 --walkers 25 \
          --out ~/dataset/train 2>&1 | tee ~/logs/drive_$T.log
done
echo "CAPTURE COMPLETE"
```

`Ctrl+B` then `D` to detach. Watch from your laptop without attaching:

```powershell
ssh s21@100.107.251.96 "tail -5 ~/logs/drive_Town03.log; df -h ~ | tail -1"
```

Notes:

- `--frames` is **per condition**, so 95 gives 665 per town.
- The script is **resumable** — if it dies, rerun the same command and it skips stems already
  written.
- Town02 is deliberately not in this list: it is your validation town, capture it in Week 2 once
  the pipeline is proven. If you want it now, add it to the loop.

---

## STEP 7 — The verification gate

```bash
python3 verify_dataset.py --root ~/dataset/train --draw 30 | tee ~/capstone/docs/verify_week1.txt
```

All six must pass:

| # | Check | Fails when |
|---|---|---|
| 1 | Boxes drawn on 30 frames | — (you must look) |
| 2 | LiDAR points per condition | Fog not below clear → attributes not applied |
| 3 | Class balance per condition | — (expect green ≫ red ≫ yellow) |
| 4 | Box height histogram | Boxes under 8 px got through |
| 5 | File integrity | Orphan images, malformed rows |
| 6 | Frames per condition | Badly unbalanced |

Then pull the annotated frames and **look at all 30**:

```powershell
scp -r s21@100.107.251.96:/home/s21/dataset/train/_verify ./verify_week1
```

This is the gate. Everything downstream assumes these labels are right.

---

## STEP 8 — Build the YOLO dataset

```bash
python3 make_splits.py --src ~/dataset/train --dst ~/dataset/camera_tl
cat ~/dataset/camera_tl/data.yaml
```

Split is **by town** — Town01 + Town03 train, Town02 val, Town10HD_Opt test. The script asserts
no stem appears in two splits. Random splitting would leak the same junction into train and test
and inflate every number you report.

With only Town01/Town03/Town10HD_Opt captured, `val` will be empty until you add Town02. For a
first training run you can temporarily point val at test, but fix it before any reported result.

---

## STEP 9 — LiDAR → BEV

```bash
python3 lidar_to_bev.py --in ~/dataset/train --out ~/dataset/lidar_bev
ls ~/dataset/lidar_bev/images | head
```

Check the output by eye — draw the OBB corners on 10 BEV images:

```bash
python3 - <<'EOF'
import glob, os, random
from PIL import Image, ImageDraw
root = os.path.expanduser('~/dataset/lidar_bev')
out = os.path.join(root, '_verify'); os.makedirs(out, exist_ok=True)
files = [p for p in sorted(glob.glob(root + '/labels/*.txt')) if os.path.getsize(p) > 0]
for p in random.sample(files, min(10, len(files))):
    stem = os.path.splitext(os.path.basename(p))[0]
    im = Image.open(f'{root}/images/{stem}.png').convert('RGB')
    d = ImageDraw.Draw(im); W, H = im.size
    for line in open(p):
        v = line.split()
        if len(v) != 9: continue
        pts = [(float(v[1+2*i])*W, float(v[2+2*i])*H) for i in range(4)]
        d.polygon(pts, outline=(255, 80, 80))
    im.save(f'{out}/{stem}.png')
print('wrote', out)
EOF
```

Rectangles in the BEV image must line up with the drawn polygons. If boxes float in empty space,
the "fewer than 5 points → drop" filter is not firing.

---

## STEP 10 — Record the statistics

```bash
python3 verify_dataset.py --root ~/dataset/train --draw 0 > ~/capstone/docs/dataset_stats.txt
du -sh ~/dataset/*
cat ~/capstone/docs/dataset_stats.txt
```

`dataset_stats.txt` is a report section and a slide, produced for free. Commit it.

```bash
cd ~/capstone && git add -A && git commit -m "week 1: capture pipeline + 2k dataset verified"
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `tl=0` on every line | Wrong semantic tag | Step 3 |
| `RuntimeError: time-out` | Server not up | `ss -tlnp \| grep 2000`; restart Step 2 |
| `... overran frame N - sync broken` | Async mode, or a sensor died | Restart the server and the script |
| Fog point count == clear | LiDAR attrs not applied | Respawn the sensor — `capture_drive.py` already does; check you edited the right config |
| Everything looks the same in all weather | `-quality-level=Low` | Use `Epic` |
| `ModuleNotFoundError: config` | Not in `scripts/` | `cd ~/capstone/scripts` first |
| `module 'carla' has no attribute 'Client'` | A folder named `carla` shadows the module | Never name a directory `carla` |
| Capture dies when SSH drops | Not in tmux | Step 6 |
| Disk full mid-run | Shared box | `df -h ~` first; the run is resumable |
| Very few frames with traffic lights | Town has sparse junctions | Town03 and Town10HD_Opt are dense; Town01 is sparse |

---

## What "done" looks like

- [ ] `versions.txt` written
- [ ] Semantic tag verified, not assumed
- [ ] Smoke test passed and **30 annotated frames looked at by a human**
- [ ] Fog LiDAR measurably sparser than clear
- [ ] ~1,995 images with labels, LiDAR sweeps and meta
- [ ] All six verification checks PASS
- [ ] `data.yaml` written, split by town, leakage assertion passed
- [ ] BEV images and OBB labels generated and eyeballed
- [ ] `dataset_stats.txt` committed
