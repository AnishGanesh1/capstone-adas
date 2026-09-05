#!/usr/bin/env python3
"""Capture Mode 1 - drive and capture. Bulk training data.

Ego on autopilot, weather cycled through every condition, a frame saved every
~0.5 s of driving. Writes RGB, LiDAR sweep, YOLO traffic-light labels and a meta
json holding per-box distance plus the 3D actor boxes the BEV stage needs.

  python3 capture_drive.py --town Town03 --frames 100 --out ~/dataset/train

--frames is PER CONDITION, so 100 with seven conditions gives 700 frames per town.

Resumable: rerun the same command and it skips stems already on disk.
"""

import argparse
import json
import os
import random
import time

import numpy as np
import carla
from PIL import Image

import config as C
from carla_utils import Rig, rgb_array, semantic_tags, lidar_array, \
    traffic_light_labels, actor_boxes_3d


def already_saved(out, town, cond):
    n = 0
    while os.path.exists(os.path.join(out, 'meta', '%s_%s_%05d.json' % (town, cond, n))):
        n += 1
    return n


def make_dirs(root):
    for sub in ('images', 'labels', 'lidar', 'meta'):
        os.makedirs(os.path.join(root, sub), exist_ok=True)


def spawn_traffic(world, tm, n_vehicles, n_walkers, rng):
    """Returns (actor_ids_to_destroy, walker_controllers)."""
    bp_lib = world.get_blueprint_library()
    spawned = []

    spawns = world.get_map().get_spawn_points()
    rng.shuffle(spawns)
    for sp in spawns[:n_vehicles]:
        bp = rng.choice(bp_lib.filter('vehicle.*'))
        if bp.has_attribute('color'):
            bp.set_attribute('color', rng.choice(bp.get_attribute('color').recommended_values))
        v = world.try_spawn_actor(bp, sp)
        if v:
            v.set_autopilot(True, tm.get_port())
            spawned.append(v)

    world.tick()          # vehicles must exist before anything else touches them

    controllers = []
    if n_walkers > 0:
        walker_bps = bp_lib.filter('walker.pedestrian.*')
        ctrl_bp = bp_lib.find('controller.ai.walker')
        walkers = []
        for _ in range(n_walkers):
            loc = world.get_random_location_from_navigation()
            if loc is None:
                continue
            w = world.try_spawn_actor(rng.choice(walker_bps), carla.Transform(loc))
            if w:
                spawned.append(w)
                walkers.append(w)
        world.tick()      # walkers must exist before attaching controllers

        for w in walkers:
            if not w.is_alive:
                continue
            c = world.try_spawn_actor(ctrl_bp, carla.Transform(), attach_to=w)
            if c:
                spawned.append(c)
                controllers.append(c)
        world.tick()      # controllers must exist before start()

        for c in controllers:
            try:
                c.start()
                dest = world.get_random_location_from_navigation()
                if dest is not None:
                    c.go_to_location(dest)
                c.set_max_speed(1.2 + random.random() * 0.6)
            except Exception:
                pass
        world.tick()
    return spawned, controllers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='localhost')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--town', default='Town03')
    ap.add_argument('--frames', type=int, default=100, help='frames PER CONDITION')
    ap.add_argument('--vehicles', type=int, default=60)
    ap.add_argument('--walkers', type=int, default=25)
    ap.add_argument('--out', default=os.path.expanduser('~/dataset/train'))
    ap.add_argument('--seed', type=int, default=C.SEED)
    ap.add_argument('--tm-port', type=int, default=0,
                    help='traffic manager port; 0 = pick a free one (avoids stale TM state)')
    ap.add_argument('--conditions', default='',
                    help='explicit comma list; overrides --tier')
    ap.add_argument('--tier', default='adverse',
                    choices=['mild', 'severe', 'extreme', 'adverse', 'all', 'ref'],
                    help='which preset group to capture (default: adverse - '
                         'every bad-weather preset, clear excluded)')
    ap.add_argument('--severity', type=float, default=1.0,
                    help='scale every preset harder, e.g. 1.3 = 30%% worse')
    ap.add_argument('--save-sem', action='store_true', help='also write semantic PNGs (big)')
    ap.add_argument('--no-lidar', action='store_true',
                    help='camera / traffic-light only: skip LiDAR spawn and .npy writes '
                         '(roughly 2x faster, ~1 MB per frame saved)')
    args = ap.parse_args()

    conds = [c.strip() for c in args.conditions.split(',') if c.strip()] or C.TIERS[args.tier]
    for c in conds:
        if c not in C.CONDITIONS:
            raise SystemExit('unknown condition: %s' % c)
    sev = args.severity
    sev_tag = '' if sev == 1.0 else '_s%d' % round(sev * 100)
    print('tier=%s  %d conditions  severity=%.2f%s'
          % (args.tier, len(conds), sev, '  (ADVERSE ONLY)' if 'clear' not in conds else ''))

    rng = random.Random(args.seed)
    make_dirs(args.out)
    if args.save_sem:
        os.makedirs(os.path.join(args.out, 'sem'), exist_ok=True)

    client = carla.Client(args.host, args.port)
    client.set_timeout(180.0)
    print('loading %s ...' % args.town)
    world = client.load_world(args.town)

    original = world.get_settings()
    s = world.get_settings()
    s.synchronous_mode = True
    s.fixed_delta_seconds = C.FIXED_DELTA
    world.apply_settings(s)

    tm_port = args.tm_port
    if tm_port == 0:
        import socket
        sk = socket.socket(); sk.bind(('', 0)); tm_port = sk.getsockname()[1]; sk.close()
    print('traffic manager port %d' % tm_port)
    tm = client.get_trafficmanager(tm_port)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(args.seed)

    spawned, controllers = [], []
    rig = None
    ego = None
    t_start = time.time()
    n_saved = n_skip_dup = n_skip_bg = 0

    try:
        # ---- ego FIRST, so it gets a guaranteed spawn point ----------------
        bp_lib = world.get_blueprint_library()
        print('spawning ego ...', flush=True)
        for sp in world.get_map().get_spawn_points():
            ego = world.try_spawn_actor(bp_lib.find('vehicle.tesla.model3'), sp)
            if ego:
                break
        if ego is None:
            raise SystemExit('could not spawn ego vehicle')
        world.tick()                       # actor is not real until the next tick
        print('    ego id %d alive=%s' % (ego.id, ego.is_alive), flush=True)
        spawned.append(ego)

        ego.set_autopilot(True, tm.get_port())
        try:
            tm.ignore_lights_percentage(ego, 0)      # obey lights: we want red/green variety
        except Exception as e:
            print('    ignore_lights_percentage unsupported:', e, flush=True)
        world.tick()

        # ---- sensors -------------------------------------------------------
        print('attaching sensors ...', flush=True)
        rig = Rig(world, ego, save_semantic=args.save_sem)

        # ---- background traffic last --------------------------------------
        if args.vehicles or args.walkers:
            print('spawning traffic ...', flush=True)
            extra, controllers = spawn_traffic(world, tm, args.vehicles, args.walkers, rng)
            spawned.extend(extra)
            print('    %d background actors' % len(extra), flush=True)
        world.tick()

        for cond_name in conds:
            cond = C.intensify(C.CONDITIONS[cond_name], sev)
            print('\n=== %s / %s ===' % (args.town, cond_name), flush=True)
            if args.no_lidar:
                print('  lidar disabled (--no-lidar)', flush=True)
            else:
                print('  respawning lidar ...', flush=True)
                rig.set_lidar(cond['lidar'])
            print('  setting weather ...', flush=True)
            world.set_weather(carla.WeatherParameters(**cond['weather']))
            print('  settling %d ticks ...' % C.SETTLE_TICKS, flush=True)
            for _ in range(C.SETTLE_TICKS):
                world.tick()
            saved = already_saved(args.out, args.town, cond_name + sev_tag)
            if saved >= args.frames:
                print('  already have %d/%d - skipping condition' % (saved, args.frames),
                      flush=True)
                continue
            if saved:
                print('  resuming from %d already on disk' % saved, flush=True)
            print('  capturing', flush=True)

            skips = 0
            last_loc = ego.get_transform().location

            while saved < args.frames:
                for _ in range(C.TICKS_BETWEEN_SAVES):
                    world.tick()
                fid = world.tick()
                data = rig.grab(fid)

                loc = ego.get_transform().location
                moved = loc.distance(last_loc)
                if moved < C.MIN_MOVE_M and skips < C.MAX_SKIPS:
                    skips += 1
                    n_skip_dup += 1
                    continue
                skips = 0
                last_loc = loc

                sem = semantic_tags(data['sem'])
                rows, meta_rows = traffic_light_labels(
                    world, rig.sensors['rgb'], rig.k, sem)

                if not rows and rng.random() > C.BACKGROUND_KEEP:
                    n_skip_bg += 1
                    continue

                stem = '%s_%s%s_%05d' % (args.town, cond_name, sev_tag, saved)
                lbl_path = os.path.join(args.out, 'labels', stem + '.txt')
                if os.path.exists(lbl_path):     # resumable
                    saved += 1
                    continue

                Image.fromarray(rgb_array(data['rgb'])).save(
                    os.path.join(args.out, 'images', stem + '.jpg'), quality=95)
                if args.save_sem:
                    Image.fromarray(sem).save(os.path.join(args.out, 'sem', stem + '.png'))

                if args.no_lidar:
                    pts = np.zeros((0, 4), dtype=np.float32)
                else:
                    pts = lidar_array(data['lidar'])
                    np.save(os.path.join(args.out, 'lidar', stem + '.npy'), pts)

                with open(lbl_path, 'w') as fh:
                    for cls, cx, cy, bw, bh in rows:
                        fh.write('%d %.6f %.6f %.6f %.6f\n' % (cls, cx, cy, bw, bh))

                tf = ego.get_transform()
                meta = dict(
                    stem=stem, town=args.town, condition=cond_name, severity=sev,
                    n_lidar_points=int(pts.shape[0]),
                    weather=cond['weather'], lidar_params=cond['lidar'],
                    ego=dict(x=round(tf.location.x, 2), y=round(tf.location.y, 2),
                             z=round(tf.location.z, 2), yaw=round(tf.rotation.yaw, 2)),
                    speed_kmh=round(3.6 * ego.get_velocity().length(), 1),
                    traffic_lights=meta_rows,
                    boxes_3d=[] if args.no_lidar else actor_boxes_3d(world, rig.lidar))
                with open(os.path.join(args.out, 'meta', stem + '.json'), 'w') as fh:
                    json.dump(meta, fh)

                saved += 1
                n_saved += 1
                if saved % 25 == 0:
                    el = time.time() - t_start
                    print('  %s %4d/%d   tl=%d pts=%d   [%d saved, %.0fs, %.2f f/s]'
                          % (cond_name, saved, args.frames, len(rows), pts.shape[0],
                             n_saved, el, n_saved / max(el, 1)))

    finally:
        # order matters: stop callbacks, drain a tick, then destroy children
        # before parents. Getting this wrong aborts the process in C++.
        for c in controllers:
            try:
                c.stop()
            except Exception:
                pass
        if rig is not None:
            try:
                rig.destroy()
            except Exception as e:
                print('cleanup: rig -', e)
        try:
            if ego is not None and ego.is_alive:
                ego.set_autopilot(False)
        except Exception:
            pass
        try:
            client.apply_batch([carla.command.DestroyActor(a) for a in spawned
                                if a is not None and a.is_alive])
            time.sleep(0.5)
        except Exception:
            for a in spawned:
                try:
                    if a.is_alive:
                        a.destroy()
                except Exception:
                    pass
        try:
            world.apply_settings(original)
            tm.set_synchronous_mode(False)
        except Exception as e:
            print('cleanup: settings -', e)
        el = time.time() - t_start
        print('\nDONE  saved=%d  skipped(duplicate)=%d  skipped(no traffic light)=%d  %.0f s'
              % (n_saved, n_skip_dup, n_skip_bg, el))


if __name__ == '__main__':
    main()
