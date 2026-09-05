#!/usr/bin/env python3
"""Traffic-light-focused capture. Almost every frame contains a labelled light.

capture_drive.py drives on autopilot and hopes to pass a junction. In a town
with few signals that means most frames are background, they get discarded, and
throughput collapses (0.36 f/s observed in Town02).

This script instead enumerates every traffic light in the map, builds viewing
poses on the approach lanes at a spread of distances, and teleports the ego to
each one. Consequences:

  * ~100% of saved frames contain a traffic light
  * no driving time, so roughly 5-10x the throughput
  * lights are FROZEN and cycled red/yellow/green, so the classes come out
    balanced instead of following whatever the simulation happened to show

Output format is byte-identical to capture_drive.py, so count_dataset.py,
verify_dataset.py and make_splits.py all work unchanged.

  python3 capture_tl_focused.py --town Town02 --tier adverse --frames 171 \
          --out ~/dataset/train
"""

import argparse
import json
import os
import random
import socket
import time

import numpy as np
import carla
from PIL import Image

import config as C
from carla_utils import Rig, rgb_array, semantic_tags, traffic_light_labels

STATES = [carla.TrafficLightState.Red,
          carla.TrafficLightState.Yellow,
          carla.TrafficLightState.Green]


def already_saved(out, town, cond):
    n = 0
    while os.path.exists(os.path.join(out, 'meta', '%s_%s_%05d.json' % (town, cond, n))):
        n += 1
    return n


def make_dirs(root):
    for sub in ('images', 'labels', 'meta'):
        os.makedirs(os.path.join(root, sub), exist_ok=True)


def light_poses(world, dists):
    """One ego pose per (traffic light, approach lane, distance).

    Uses the light's own stop waypoints, so the pose sits in a lane the light
    actually controls and is oriented along it - the light is then ahead in the
    camera frustum by construction.
    """
    amap = world.get_map()
    poses = []
    lights = list(world.get_actors().filter('traffic.traffic_light*'))
    for tl in lights:
        wps = []
        try:
            wps = tl.get_stop_waypoints()
        except Exception:
            pass
        if not wps:
            wp = amap.get_waypoint(tl.get_transform().location, project_to_road=True)
            if wp:
                wps = [wp]
        for wp in wps:
            for d in dists:
                back = wp.previous(float(d))
                if not back:
                    continue
                t = back[0].transform
                poses.append((tl.id, d, carla.Transform(
                    carla.Location(x=t.location.x, y=t.location.y, z=t.location.z + 0.35),
                    carla.Rotation(pitch=0.0, yaw=t.rotation.yaw, roll=0.0))))
    return poses, len(lights)


def jitter(tf, rng, lat=0.9, yaw=5.0, z=0.15):
    """Small pose perturbation so repeated visits are not duplicates."""
    yaw_rad = np.radians(tf.rotation.yaw)
    dx, dy = -np.sin(yaw_rad), np.cos(yaw_rad)          # lane-lateral unit vector
    off = rng.uniform(-lat, lat)
    return carla.Transform(
        carla.Location(x=tf.location.x + dx * off,
                       y=tf.location.y + dy * off,
                       z=tf.location.z + rng.uniform(-z, z)),
        carla.Rotation(pitch=rng.uniform(-1.5, 1.5),
                       yaw=tf.rotation.yaw + rng.uniform(-yaw, yaw),
                       roll=0.0))


def spawn_traffic(world, tm, n_vehicles, n_walkers, rng):
    """Background actors, purely for realistic occlusion and scene clutter."""
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
    world.tick()

    controllers = []
    if n_walkers > 0:
        wbps = bp_lib.filter('walker.pedestrian.*')
        ctrl = bp_lib.find('controller.ai.walker')
        walkers = []
        for _ in range(n_walkers):
            loc = world.get_random_location_from_navigation()
            if loc is None:
                continue
            w = world.try_spawn_actor(rng.choice(wbps), carla.Transform(loc))
            if w:
                spawned.append(w); walkers.append(w)
        world.tick()
        for w in walkers:
            if not w.is_alive:
                continue
            c = world.try_spawn_actor(ctrl, carla.Transform(), attach_to=w)
            if c:
                spawned.append(c); controllers.append(c)
        world.tick()
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
    ap.add_argument('--town', default='Town02')
    ap.add_argument('--frames', type=int, default=171, help='PER CONDITION')
    ap.add_argument('--out', default=os.path.expanduser('~/dataset/train'))
    ap.add_argument('--tier', default='adverse',
                    choices=['mild', 'severe', 'extreme', 'adverse', 'all', 'ref'])
    ap.add_argument('--conditions', default='')
    ap.add_argument('--severity', type=float, default=1.0)
    ap.add_argument('--vehicles', type=int, default=35)
    ap.add_argument('--walkers', type=int, default=15)
    ap.add_argument('--seed', type=int, default=C.SEED)
    ap.add_argument('--tm-port', type=int, default=0)
    ap.add_argument('--min-lights', type=int, default=1,
                    help='discard a frame with fewer labelled lights than this')
    ap.add_argument('--background', type=float, default=0.03,
                    help='fraction of light-free frames to keep as negatives')
    ap.add_argument('--dists', default='8,13,19,26,34,44,56',
                    help='ego distances from the stop line, metres')
    ap.add_argument('--no-balance', action='store_true',
                    help='do not freeze/cycle light states (use simulation states)')
    ap.add_argument('--min-free-gb', type=float, default=3.0)
    args = ap.parse_args()

    conds = [c.strip() for c in args.conditions.split(',') if c.strip()] or C.TIERS[args.tier]
    for c in conds:
        if c not in C.CONDITIONS:
            raise SystemExit('unknown condition: %s' % c)
    sev = args.severity
    sev_tag = '' if sev == 1.0 else '_s%d' % round(sev * 100)
    dists = [float(x) for x in args.dists.split(',') if x.strip()]

    rng = random.Random(args.seed)
    nprng = np.random.default_rng(args.seed)
    make_dirs(args.out)

    client = carla.Client(args.host, args.port)
    client.set_timeout(180.0)
    print('loading %s ...' % args.town, flush=True)
    world = client.load_world(args.town)

    original = world.get_settings()
    s = world.get_settings()
    s.synchronous_mode = True
    s.fixed_delta_seconds = C.FIXED_DELTA
    world.apply_settings(s)

    tm_port = args.tm_port
    if tm_port == 0:
        sk = socket.socket(); sk.bind(('', 0)); tm_port = sk.getsockname()[1]; sk.close()
    tm = client.get_trafficmanager(tm_port)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(args.seed)
    print('traffic manager port %d' % tm_port, flush=True)

    spawned, controllers, rig, ego = [], [], None, None
    t0 = time.time()
    n_saved = n_nolight = 0

    try:
        # ---- poses -----------------------------------------------------
        poses, n_lights = light_poses(world, dists)
        if not poses:
            raise SystemExit('no traffic lights found in %s' % args.town)
        rng.shuffle(poses)
        print('%d traffic lights -> %d viewing poses' % (n_lights, len(poses)), flush=True)

        # ---- ego -------------------------------------------------------
        bp_lib = world.get_blueprint_library()
        ego = world.try_spawn_actor(bp_lib.find('vehicle.tesla.model3'), poses[0][2])
        if ego is None:
            for sp in world.get_map().get_spawn_points():
                ego = world.try_spawn_actor(bp_lib.find('vehicle.tesla.model3'), sp)
                if ego:
                    break
        if ego is None:
            raise SystemExit('could not spawn ego')
        world.tick()                      # actor is not real until the next tick
        spawned.append(ego)
        ego.set_simulate_physics(False)   # teleporting a physics body makes it bounce
        world.tick()
        print('ego id %d' % ego.id, flush=True)

        rig = Rig(world, ego, save_semantic=False)

        if args.vehicles or args.walkers:
            extra, controllers = spawn_traffic(world, tm, args.vehicles, args.walkers, rng)
            spawned.extend(extra)
            print('%d background actors' % len(extra), flush=True)
        world.tick()

        # ---- freeze lights so states can be balanced -------------------
        lights = list(world.get_actors().filter('traffic.traffic_light*'))
        if not args.no_balance:
            for tl in lights:
                try:
                    tl.freeze(True)
                except Exception:
                    pass
            print('froze %d lights - states will be cycled R/Y/G' % len(lights), flush=True)

        # ---- capture ---------------------------------------------------
        for cond_name in conds:
            cond = C.intensify(C.CONDITIONS[cond_name], sev)
            print('\n=== %s / %s ===' % (args.town, cond_name), flush=True)
            world.set_weather(carla.WeatherParameters(**cond['weather']))
            for _ in range(C.SETTLE_TICKS):
                world.tick()

            saved = already_saved(args.out, args.town, cond_name + sev_tag)
            if saved >= args.frames:
                print('  already have %d/%d - skipping' % (saved, args.frames), flush=True)
                continue
            if saved:
                print('  resuming from %d' % saved, flush=True)

            st = os.statvfs(args.out)
            if st.f_bavail * st.f_frsize / 1e9 < args.min_free_gb:
                print('  STOPPING: free disk below --min-free-gb', flush=True)
                raise SystemExit(3)

            i = 0
            attempts = 0
            max_attempts = args.frames * 6
            while saved < args.frames and attempts < max_attempts:
                attempts += 1
                tl_id, dist, base = poses[i % len(poses)]
                i += 1

                if not args.no_balance:
                    want = STATES[saved % 3]
                    for tl in lights:
                        try:
                            tl.set_state(want)
                        except Exception:
                            pass

                ego.set_transform(jitter(base, nprng))
                world.tick()                 # apply the teleport
                fid = world.tick()           # render at the new pose
                data = rig.grab(fid)

                sem = semantic_tags(data['sem'])
                rows, meta_rows = traffic_light_labels(
                    world, rig.sensors['rgb'], rig.k, sem)

                if len(rows) < args.min_lights:
                    n_nolight += 1
                    if rng.random() > args.background:
                        continue

                stem = '%s_%s%s_%05d' % (args.town, cond_name, sev_tag, saved)
                lbl_path = os.path.join(args.out, 'labels', stem + '.txt')
                if os.path.exists(lbl_path):
                    saved += 1
                    continue

                Image.fromarray(rgb_array(data['rgb'])).save(
                    os.path.join(args.out, 'images', stem + '.jpg'), quality=95)
                with open(lbl_path, 'w') as fh:
                    for cls, cx, cy, bw, bh in rows:
                        fh.write('%d %.6f %.6f %.6f %.6f\n' % (cls, cx, cy, bw, bh))

                tf = ego.get_transform()
                json.dump(dict(
                    stem=stem, town=args.town, condition=cond_name, severity=sev,
                    capture_mode='tl_focused', n_lidar_points=0,
                    weather=cond['weather'], lidar_params=cond['lidar'],
                    ego=dict(x=round(tf.location.x, 2), y=round(tf.location.y, 2),
                             z=round(tf.location.z, 2), yaw=round(tf.rotation.yaw, 2)),
                    speed_kmh=0.0, target_light=tl_id, target_dist=dist,
                    traffic_lights=meta_rows, boxes_3d=[]),
                    open(os.path.join(args.out, 'meta', stem + '.json'), 'w'))

                saved += 1
                n_saved += 1
                if saved % 25 == 0:
                    el = time.time() - t0
                    print('  %s %4d/%d  tl=%d  [%d saved, %.0fs, %.2f f/s]'
                          % (cond_name, saved, args.frames, len(rows),
                             n_saved, el, n_saved / max(el, 1)), flush=True)

            if attempts >= max_attempts:
                print('  WARNING: hit the attempt cap with %d/%d saved - very few '
                      'lights are visible in this town' % (saved, args.frames), flush=True)

    finally:
        for c in controllers:
            try:
                c.stop()
            except Exception:
                pass
        if rig is not None:
            try:
                rig.destroy()
            except Exception as e:
                print('cleanup rig:', e)
        try:
            client.apply_batch([carla.command.DestroyActor(a) for a in spawned
                                if a is not None and a.is_alive])
            time.sleep(0.5)
        except Exception:
            pass
        try:
            world.apply_settings(original)
            tm.set_synchronous_mode(False)
        except Exception as e:
            print('cleanup settings:', e)
        el = time.time() - t0
        print('\nDONE saved=%d  discarded(no light)=%d  %.0f s  %.2f f/s'
              % (n_saved, n_nolight, el, n_saved / max(el, 1)))


if __name__ == '__main__':
    main()
