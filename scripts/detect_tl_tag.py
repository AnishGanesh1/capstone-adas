#!/usr/bin/env python3
"""Work out which semantic tag id means 'traffic light' in YOUR CARLA build.

The palette has changed between releases, so hardcoding 18 from a blog post is a
guess. This projects real traffic-light boxes into the semantic image and reports
which tag actually dominates inside them.

  python3 detect_tl_tag.py

Then set SEM_TAG_TRAFFIC_LIGHT in config.py to whatever it prints.
"""

import argparse
import collections

import numpy as np
import carla

import config as C
from carla_utils import Rig, semantic_tags, world_to_image, _tl_boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='localhost')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--town', default='Town03')
    ap.add_argument('--samples', type=int, default=25)
    ap.add_argument('--force-reload', action='store_true',
                    help='reload the map even if it is already loaded')
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0)

    # Purge orphans BEFORE touching the map. A crashed previous run leaves
    # sensors attached to destroyed vehicles, and load_world() then aborts the
    # client in C++ with "trying to operate on a destroyed actor".
    world = client.get_world()
    stale = [a for a in world.get_actors()
             if a.type_id.startswith(('sensor.', 'vehicle.', 'walker.', 'controller.'))]
    if stale:
        print('purging %d leftover actors from a previous run' % len(stale))
        for a in stale:
            if a.type_id.startswith('sensor.'):
                try:
                    a.stop()
                except Exception:
                    pass
        client.apply_batch_sync([carla.command.DestroyActor(a) for a in stale], True)

    current = world.get_map().name.split('/')[-1]
    if args.force_reload or args.town not in current:
        print('loading %s (current: %s)' % (args.town, current))
        world = client.load_world(args.town)
    else:
        print('already on %s - not reloading' % current)

    original = world.get_settings()
    s = world.get_settings()
    s.synchronous_mode = True
    s.fixed_delta_seconds = C.FIXED_DELTA
    world.apply_settings(s)

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)

    bp_lib = world.get_blueprint_library()
    spawns = world.get_map().get_spawn_points()
    ego = None
    for sp in spawns:
        ego = world.try_spawn_actor(bp_lib.find('vehicle.tesla.model3'), sp)
        if ego:
            break
    if ego is None:
        raise SystemExit('could not spawn ego')
    ego.set_autopilot(True, tm.get_port())

    rig = Rig(world, ego)
    rig.set_lidar(C.CONDITIONS['clear']['lidar'])
    world.set_weather(carla.WeatherParameters(**C.CONDITIONS['clear']['weather']))
    for _ in range(30):
        world.tick()

    votes = collections.Counter()
    all_tags = collections.Counter()
    got = 0
    try:
        for _ in range(args.samples * 40):
            if got >= args.samples:
                break
            fid = world.tick()
            data = rig.grab(fid)
            sem = semantic_tags(data['sem'])
            all_tags.update(np.unique(sem).tolist())

            w2c = np.array(rig.sensors['rgb'].get_transform().get_inverse_matrix())
            cam_loc = rig.sensors['rgb'].get_transform().location
            for tl in world.get_actors().filter('traffic.traffic_light*'):
                if tl.get_transform().location.distance(cam_loc) > 40.0:
                    continue
                for verts in _tl_boxes(tl):
                    us, vs, ok = [], [], True
                    for v in verts:
                        u, vv, d = world_to_image(v, w2c, rig.k)
                        if d <= 0:
                            ok = False
                            break
                        us.append(u)
                        vs.append(vv)
                    if not ok:
                        continue
                    x1, x2 = max(0, int(min(us))), min(C.IMAGE_W, int(max(us)))
                    y1, y2 = max(0, int(min(vs))), min(C.IMAGE_H, int(max(vs)))
                    if x2 - x1 < 4 or y2 - y1 < 10:
                        continue
                    patch = sem[y1:y2, x1:x2]
                    if patch.size:
                        vals, counts = np.unique(patch, return_counts=True)
                        votes[int(vals[np.argmax(counts)])] += 1
                        got += 1
    finally:
        try:
            rig.destroy()
        except Exception as e:
            print('cleanup: rig -', e)
        try:
            ego.set_autopilot(False)
            if ego.is_alive:
                ego.destroy()
        except Exception as e:
            print('cleanup: ego -', e)
        try:
            world.apply_settings(original)
            tm.set_synchronous_mode(False)
        except Exception as e:
            print('cleanup: settings -', e)

    print('\ntags present anywhere in the semantic image:', sorted(all_tags))
    print('\ndominant tag inside projected traffic-light boxes:')
    for tag, n in votes.most_common(6):
        print('   tag %3d  ->  %d boxes' % (tag, n))
    if votes:
        best = votes.most_common(1)[0][0]
        print('\n==> set SEM_TAG_TRAFFIC_LIGHT = %d in config.py' % best)
        if best != C.SEM_TAG_TRAFFIC_LIGHT:
            print('    (config.py currently says %d - CHANGE IT)' % C.SEM_TAG_TRAFFIC_LIGHT)
        else:
            print('    (config.py already matches)')
    else:
        print('\nNo boxes sampled. Traffic lights may be far away - rerun with more --samples.')


if __name__ == '__main__':
    main()
