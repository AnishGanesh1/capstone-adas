#!/usr/bin/env python3
"""LiDAR-only capture. No cameras, so it runs far faster than capture_drive.py.

Writes, per frame:
  lidar/<stem>.npy   float32 (N,4) = x, y, z, intensity in the LiDAR frame
  meta/<stem>.json   condition, point count, ego pose, and the 3D actor boxes
                     that lidar_to_bev.py turns into oriented labels

  python3 capture_lidar.py --town Town03 --frames 125 --out ~/dataset/lidar_raw

--frames is PER CONDITION. Resumable: rerun and it skips stems already on disk.

REMEMBER: CARLA does not couple weather to the LiDAR. The per-condition
attenuation / dropoff / noise values in config.py are what make a "foggy" sweep
actually different from a clear one, and they are blueprint attributes, so the
sensor is destroyed and respawned whenever the condition changes.
"""

import argparse
import json
import os
import random
import socket
import time

import numpy as np
import carla

import config as C
from carla_utils import lidar_array, actor_boxes_3d


def free_gb(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def already_saved(out, town, cond):
    """Count contiguous stems already on disk so a top-up run skips them
    instantly instead of re-simulating them."""
    n = 0
    while os.path.exists(os.path.join(out, 'meta', '%s_%s_%05d.json' % (town, cond, n))):
        n += 1
    return n


def make_dirs(root):
    for sub in ('lidar', 'meta'):
        os.makedirs(os.path.join(root, sub), exist_ok=True)


def purge(client, world):
    stale = [a for a in world.get_actors()
             if a.type_id.startswith(('sensor.', 'vehicle.', 'walker.', 'controller.'))]
    if not stale:
        return 0
    for a in stale:
        if a.type_id.startswith('sensor.'):
            try:
                a.stop()
            except Exception:
                pass
    client.apply_batch_sync([carla.command.DestroyActor(a) for a in stale], True)
    return len(stale)


class LidarOnly:
    """Just the ray-cast LiDAR, respawned per weather condition."""

    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self.bp = world.get_blueprint_library()
        self.sensor = None
        self.q = None
        self.tf = carla.Transform(carla.Location(x=C.LIDAR_X, z=C.LIDAR_Z))

    def _tick(self):
        try:
            self.world.tick()
        except Exception:
            pass
        time.sleep(0.05)

    def set_params(self, params):
        import queue
        if self.sensor is not None:
            try:
                self.sensor.stop()
            except Exception:
                pass
            self._tick()
            try:
                if self.sensor.is_alive:
                    self.sensor.destroy()
            except Exception:
                pass
            self.sensor = None
            self.q = None
            self._tick()

        if not self.vehicle.is_alive:
            raise RuntimeError('ego died before LiDAR spawn')

        bp = self.bp.find('sensor.lidar.ray_cast')
        for k, v in C.LIDAR_BASE.items():
            if bp.has_attribute(k):
                bp.set_attribute(k, v)
        for k, v in params.items():
            if bp.has_attribute(k):
                bp.set_attribute(k, str(v))
            else:
                print('    WARN no lidar attribute %r' % k, flush=True)

        self.sensor = self.world.spawn_actor(bp, self.tf, attach_to=self.vehicle)
        print('    lidar id %d' % self.sensor.id, flush=True)
        self._tick()
        self.q = queue.Queue()
        self.sensor.listen(self.q.put)

    def grab(self, frame_id, timeout=10.0):
        while True:
            d = self.q.get(timeout=timeout)
            if d.frame == frame_id:
                return d
            if d.frame > frame_id:
                raise RuntimeError('lidar overran frame %d' % frame_id)

    def destroy(self):
        if self.sensor is None:
            return
        try:
            self.sensor.stop()
        except Exception:
            pass
        self._tick()
        try:
            if self.sensor.is_alive:
                self.sensor.destroy()
        except Exception:
            pass
        self.sensor = None
        self.q = None


def spawn_traffic(world, tm, n_vehicles, n_walkers, rng):
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
        world.tick()
        for w in walkers:
            if not w.is_alive:
                continue
            c = world.try_spawn_actor(ctrl_bp, carla.Transform(), attach_to=w)
            if c:
                spawned.append(c)
                controllers.append(c)
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
    ap.add_argument('--town', default='Town03')
    ap.add_argument('--frames', type=int, default=125, help='frames PER CONDITION')
    ap.add_argument('--vehicles', type=int, default=70)
    ap.add_argument('--walkers', type=int, default=30)
    ap.add_argument('--out', default=os.path.expanduser('~/dataset/lidar_raw'))
    ap.add_argument('--conditions', default='',
                    help='explicit comma list; overrides --tier')
    ap.add_argument('--tier', default='adverse',
                    choices=['mild', 'severe', 'extreme', 'adverse', 'all', 'ref'],
                    help='which preset group to capture (default: adverse - '
                         'every bad-weather preset, clear excluded)')
    ap.add_argument('--severity', type=float, default=1.0,
                    help='scale every preset harder, e.g. 1.3 = 30%% worse')
    ap.add_argument('--seed', type=int, default=C.SEED)
    ap.add_argument('--tm-port', type=int, default=0)
    ap.add_argument('--min-objects', type=int, default=1,
                    help='require at least this many actors in LiDAR range')
    ap.add_argument('--min-free-gb', type=float, default=5.0,
                    help='stop cleanly if free disk falls below this')
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

    client = carla.Client(args.host, args.port)
    client.set_timeout(180.0)

    world = client.get_world()
    n = purge(client, world)
    if n:
        print('purged %d leftover actors' % n)
    if args.town not in world.get_map().name:
        print('loading %s ...' % args.town)
        world = client.load_world(args.town)
    print('map:', world.get_map().name)

    original = world.get_settings()
    s = world.get_settings()
    s.synchronous_mode = True
    s.fixed_delta_seconds = C.FIXED_DELTA
    world.apply_settings(s)

    tm_port = args.tm_port
    if tm_port == 0:
        sk = socket.socket()
        sk.bind(('', 0))
        tm_port = sk.getsockname()[1]
        sk.close()
    print('traffic manager port %d' % tm_port)
    tm = client.get_trafficmanager(tm_port)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(args.seed)

    spawned, controllers = [], []
    lid = None
    ego = None
    t0 = time.time()
    n_saved = n_dup = n_empty = 0

    try:
        bp_lib = world.get_blueprint_library()
        print('spawning ego ...', flush=True)
        for sp in world.get_map().get_spawn_points():
            ego = world.try_spawn_actor(bp_lib.find('vehicle.tesla.model3'), sp)
            if ego:
                break
        if ego is None:
            raise SystemExit('could not spawn ego')
        world.tick()
        spawned.append(ego)
        ego.set_autopilot(True, tm.get_port())
        world.tick()
        print('    ego id %d' % ego.id, flush=True)

        lid = LidarOnly(world, ego)

        print('spawning traffic ...', flush=True)
        extra, controllers = spawn_traffic(world, tm, args.vehicles, args.walkers, rng)
        spawned.extend(extra)
        print('    %d background actors' % len(extra), flush=True)
        world.tick()

        for cond_name in conds:
            cond = C.intensify(C.CONDITIONS[cond_name], sev)
            print('\n=== %s / %s ===' % (args.town, cond_name), flush=True)
            print('  respawning lidar with weather-coupled attributes ...', flush=True)
            lid.set_params(cond['lidar'])
            world.set_weather(carla.WeatherParameters(**cond['weather']))
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
                meas = lid.grab(fid)

                loc = ego.get_transform().location
                if loc.distance(last_loc) < C.MIN_MOVE_M and skips < C.MAX_SKIPS:
                    skips += 1
                    n_dup += 1
                    continue
                skips = 0
                last_loc = loc

                boxes = actor_boxes_3d(world, lid.sensor)
                if len(boxes) < args.min_objects and rng.random() > C.BACKGROUND_KEEP:
                    n_empty += 1
                    continue

                stem = '%s_%s%s_%05d' % (args.town, cond_name, sev_tag, saved)
                meta_p = os.path.join(args.out, 'meta', stem + '.json')
                if os.path.exists(meta_p):
                    saved += 1
                    continue

                if n_saved % 100 == 0 and free_gb(args.out) < args.min_free_gb:
                    print('\n  STOPPING: free disk %.1f GB below --min-free-gb %.1f'
                          % (free_gb(args.out), args.min_free_gb), flush=True)
                    raise SystemExit(3)

                pts = lidar_array(meas)
                np.save(os.path.join(args.out, 'lidar', stem + '.npy'), pts)

                tf = ego.get_transform()
                json.dump(dict(
                    stem=stem, town=args.town, condition=cond_name, severity=sev,
                    n_lidar_points=int(pts.shape[0]),
                    lidar_params=cond['lidar'], weather=cond['weather'],
                    ego=dict(x=round(tf.location.x, 2), y=round(tf.location.y, 2),
                             z=round(tf.location.z, 2), yaw=round(tf.rotation.yaw, 2)),
                    speed_kmh=round(3.6 * ego.get_velocity().length(), 1),
                    boxes_3d=boxes), open(meta_p, 'w'))

                saved += 1
                n_saved += 1
                if saved % 25 == 0:
                    el = time.time() - t0
                    print('  %s %4d/%d  pts=%6d objs=%2d  [%d saved, %.0fs, %.2f f/s]'
                          % (cond_name, saved, args.frames, pts.shape[0], len(boxes),
                             n_saved, el, n_saved / max(el, 1)), flush=True)

    finally:
        for c in controllers:
            try:
                c.stop()
            except Exception:
                pass
        if lid is not None:
            try:
                lid.destroy()
            except Exception as e:
                print('cleanup lidar:', e)
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
            pass
        try:
            world.apply_settings(original)
            tm.set_synchronous_mode(False)
        except Exception as e:
            print('cleanup settings:', e)
        print('\nDONE saved=%d  dup-skipped=%d  empty-skipped=%d  %.0f s'
              % (n_saved, n_dup, n_empty, time.time() - t0))


if __name__ == '__main__':
    main()
