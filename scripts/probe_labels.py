#!/usr/bin/env python3
"""Why is tl=0? Prints the label funnel one stage at a time."""
import argparse, os, time
import numpy as np
import carla
from PIL import Image
import config as C
import carla_utils as U

ap = argparse.ArgumentParser()
ap.add_argument('--host', default='localhost')
ap.add_argument('--port', type=int, default=2000)
ap.add_argument('--town', default='Town02')
ap.add_argument('--poses', type=int, default=6)
ap.add_argument('--dist', type=float, default=18.0)
ap.add_argument('--save', default='/tmp/probe')
args = ap.parse_args()

os.makedirs(args.save, exist_ok=True)
client = carla.Client(args.host, args.port); client.set_timeout(180.0)
print('loading %s ...' % args.town, flush=True)
world = client.load_world(args.town)

orig = world.get_settings()
s = world.get_settings()
s.synchronous_mode = True
s.fixed_delta_seconds = C.FIXED_DELTA
world.apply_settings(s)

print('\nCONFIG')
for k in ('SEM_TAG_TRAFFIC_LIGHT','MIN_BOX_PX','MIN_VISIBLE_FRAC',
          'MAX_TL_DIST','IMAGE_W','IMAGE_H','FOV'):
    print('  %-24s %s' % (k, getattr(C, k, '<missing>')))

ego = rig = None; spawned = []
try:
    amap = world.get_map()
    lights = list(world.get_actors().filter('traffic.traffic_light*'))
    print('\ntraffic lights in map: %d' % len(lights))
    if not lights:
        raise SystemExit('none found - wrong town?')

    poses = []
    for tl in lights:
        try: wps = tl.get_stop_waypoints()
        except Exception: wps = []
        if not wps:
            wp = amap.get_waypoint(tl.get_transform().location, project_to_road=True)
            wps = [wp] if wp else []
        for wp in wps:
            b = wp.previous(args.dist)
            if b:
                t = b[0].transform
                poses.append((tl, carla.Transform(
                    carla.Location(t.location.x, t.location.y, t.location.z + 0.35),
                    carla.Rotation(0.0, t.rotation.yaw, 0.0))))
        if len(poses) >= args.poses: break
    print('probe poses: %d' % len(poses))
    if not poses: raise SystemExit('no approach poses could be built')

    bp = world.get_blueprint_library().find('vehicle.tesla.model3')
    ego = world.try_spawn_actor(bp, poses[0][1])
    if ego is None:
        for sp in amap.get_spawn_points():
            ego = world.try_spawn_actor(bp, sp)
            if ego: break
    world.tick(); spawned.append(ego)
    ego.set_simulate_physics(False); world.tick()

    rig = U.Rig(world, ego, save_semantic=False)
    world.set_weather(carla.WeatherParameters(**C.CONDITIONS['clear']['weather']))
    for _ in range(C.SETTLE_TICKS): world.tick()

    cam = rig.sensors['rgb']; K = rig.k
    tag = getattr(C, 'SEM_TAG_TRAFFIC_LIGHT', None)

    for pi, (target, tf) in enumerate(poses[:args.poses]):
        ego.set_transform(tf)
        world.tick(); fid = world.tick(); world.tick()
        data = rig.grab(fid)
        sem = U.semantic_tags(data['sem'])
        rgb = U.rgb_array(data['rgb'])

        uniq, cnt = np.unique(sem, return_counts=True)
        top = sorted(zip(cnt, uniq), reverse=True)[:8]
        npix = int((sem == tag).sum()) if tag is not None else -1

        print('\n--- pose %d  (light %d, %.0f m) ---' % (pi, target.id, args.dist))
        print('  tags present (tag:pixels): %s'
              % ', '.join('%d:%d' % (u, c) for c, u in top))
        print('  pixels with tag %s : %d' % (tag, npix))

        cloc = cam.get_transform().location
        n_near = n_front = n_inimg = 0
        for tl in lights:
            d = tl.get_transform().location.distance(cloc)
            if d > getattr(C, 'MAX_TL_DIST', 90.0): continue
            n_near += 1
            try:
                uv = U.world_to_image(tl.get_transform().location, cam, K)
            except Exception as e:
                print('  world_to_image failed:', e); break
            if uv is None: continue
            if len(uv) == 3: u, v, depth = uv
            else: (u, v), depth = uv, 1.0
            if depth <= 0: continue
            n_front += 1
            if 0 <= u < C.IMAGE_W and 0 <= v < C.IMAGE_H: n_inimg += 1

        rows, meta_rows = U.traffic_light_labels(world, cam, K, sem)
        print('  within %.0f m       : %d' % (getattr(C,'MAX_TL_DIST',90.0), n_near))
        print('  in front (depth>0) : %d' % n_front)
        print('  inside image       : %d' % n_inimg)
        print('  LABELS WRITTEN     : %d' % len(rows))
        if meta_rows: print('  meta[0]: %s' % (meta_rows[0],))

        Image.fromarray(rgb).save('%s/pose%02d_rgb.jpg' % (args.save, pi), quality=92)
        if tag is not None:
            Image.fromarray(((sem == tag)*255).astype(np.uint8)).save(
                '%s/pose%02d_mask.png' % (args.save, pi))

    print('\nimages + masks in %s' % args.save)
finally:
    try:
        if rig is not None: rig.destroy()
    except Exception: pass
    try:
        client.apply_batch([carla.command.DestroyActor(a) for a in spawned
                            if a is not None and a.is_alive]); time.sleep(0.5)
    except Exception: pass
    try: world.apply_settings(orig)
    except Exception: pass
