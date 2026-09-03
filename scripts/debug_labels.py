#!/usr/bin/env python3
"""Why am I getting zero traffic-light labels?

Teleports the ego to a pose in front of each traffic light and reports the
filter funnel. Deliberately uses NO Traffic Manager, NO autopilot and NO NPCs,
because a stale Traffic Manager holding references to destroyed vehicles is a
common source of "trying to operate on a destroyed actor" aborts.

  python3 debug_labels.py --town Town03 --lights 20
"""

import argparse
import collections
import os

import numpy as np
import carla
from PIL import Image, ImageDraw

import config as C
from carla_utils import Rig, rgb_array, semantic_tags, world_to_image, _tl_boxes


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


def poses_facing_lights(world, back=25.0):
    """A pose `back` metres behind each stop line, facing the signal."""
    out = []
    cmap = world.get_map()
    for tl in world.get_actors().filter('traffic.traffic_light*'):
        wps = []
        try:
            wps = tl.get_stop_waypoints()
        except Exception:
            pass
        if not wps:
            wp = cmap.get_waypoint(tl.get_transform().location, project_to_road=True)
            if wp:
                wps = [wp]
        if not wps:
            continue
        tf = wps[0].transform
        f = tf.get_forward_vector()
        loc = carla.Location(x=tf.location.x - f.x * back,
                             y=tf.location.y - f.y * back,
                             z=tf.location.z + 0.5)
        out.append((tl, carla.Transform(loc, tf.rotation)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='localhost')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--town', default='Town03')
    ap.add_argument('--lights', type=int, default=20, help='how many signals to visit')
    ap.add_argument('--back', type=float, default=25.0, help='metres behind the stop line')
    ap.add_argument('--out', default=os.path.expanduser('~/dataset/debug'))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0)
    world = client.get_world()

    n = purge(client, world)
    if n:
        print('purged %d leftover actors' % n)

    if args.town not in world.get_map().name:
        print('loading', args.town)
        world = client.load_world(args.town)
    print('map:', world.get_map().name)

    lights = world.get_actors().filter('traffic.traffic_light*')
    print('traffic lights in map:', len(lights))
    if len(lights) == 0:
        raise SystemExit('this map has no traffic lights - use Town03 or Town10HD_Opt')

    original = world.get_settings()
    s = world.get_settings()
    s.synchronous_mode = True
    s.fixed_delta_seconds = C.FIXED_DELTA
    world.apply_settings(s)

    ego = None
    rig = None
    funnel = collections.Counter()
    tag_hist = collections.Counter()
    fracs = []
    examples = []

    try:
        print('computing poses in front of signals ...', flush=True)
        poses = poses_facing_lights(world, args.back)[:args.lights]
        print('    %d poses' % len(poses), flush=True)
        if not poses:
            raise SystemExit('could not derive poses - no stop waypoints found')

        bp_lib = world.get_blueprint_library()
        print('spawning ego (no autopilot, no traffic manager) ...', flush=True)
        ego = world.try_spawn_actor(bp_lib.find('vehicle.tesla.model3'), poses[0][1])
        if ego is None:
            for sp in world.get_map().get_spawn_points():
                ego = world.try_spawn_actor(bp_lib.find('vehicle.tesla.model3'), sp)
                if ego:
                    break
        if ego is None:
            raise SystemExit('could not spawn ego')
        world.tick()
        ego.set_simulate_physics(False)       # stay exactly where we put it
        world.tick()
        print('    ego id %d' % ego.id, flush=True)

        rig = Rig(world, ego)
        print('spawning lidar ...', flush=True)
        rig.set_lidar(C.CONDITIONS['clear']['lidar'])
        print('setting weather ...', flush=True)
        world.set_weather(carla.WeatherParameters(**C.CONDITIONS['clear']['weather']))
        print('settling ...', flush=True)
        for _ in range(10):
            world.tick()
        print('visiting signals ...', flush=True)

        for idx, (tl, pose) in enumerate(poses):
            ego.set_transform(pose)
            for _ in range(4):
                world.tick()
            fid = world.tick()
            data = rig.grab(fid)
            sem = semantic_tags(data['sem'])
            cam = rig.sensors['rgb']
            w2c = np.array(cam.get_transform().get_inverse_matrix())
            cam_loc = cam.get_transform().location

            frame_labels = []
            for other in world.get_actors().filter('traffic.traffic_light*'):
                d = other.get_transform().location.distance(cam_loc)
                if d > C.MAX_TL_DIST:
                    continue
                funnel['1_within_range'] += 1
                for verts in _tl_boxes(other):
                    us, vs, ok = [], [], True
                    for v in verts:
                        u, vv, depth = world_to_image(v, w2c, rig.k)
                        if depth <= 0:
                            ok = False
                            break
                        us.append(u)
                        vs.append(vv)
                    if not ok:
                        funnel['2a_behind_camera'] += 1
                        continue
                    funnel['2_in_front'] += 1

                    x1, x2 = max(0.0, min(us)), min(float(C.IMAGE_W), max(us))
                    y1, y2 = max(0.0, min(vs)), min(float(C.IMAGE_H), max(vs))
                    if x2 <= x1 or y2 <= y1:
                        funnel['3a_off_screen'] += 1
                        continue
                    funnel['3_on_screen'] += 1

                    if (y2 - y1) < C.MIN_BOX_PX:
                        funnel['4a_too_small'] += 1
                        continue
                    funnel['4_big_enough'] += 1

                    patch = sem[int(y1):int(y2), int(x1):int(x2)]
                    if patch.size == 0:
                        continue
                    vals, counts = np.unique(patch, return_counts=True)
                    for v_, c_ in zip(vals, counts):
                        tag_hist[int(v_)] += int(c_)
                    frac = float(np.count_nonzero(patch == C.SEM_TAG_TRAFFIC_LIGHT)) / patch.size
                    fracs.append(frac)
                    if frac < C.MIN_VISIBLE_FRAC:
                        funnel['5a_occlusion_reject'] += 1
                        continue
                    funnel['5_KEPT'] += 1
                    frame_labels.append((x1, y1, x2, y2, str(other.get_state())))

            if frame_labels:
                examples.append((len(frame_labels), idx, rgb_array(data['rgb']), frame_labels))
            if (idx + 1) % 5 == 0:
                print('    %d/%d visited, %d kept so far'
                      % (idx + 1, len(poses), funnel['5_KEPT']), flush=True)

        # ---------------------------------------------------------------- report
        print('\n================ FILTER FUNNEL ================')
        for kname in ['1_within_range', '2a_behind_camera', '2_in_front', '3a_off_screen',
                      '3_on_screen', '4a_too_small', '4_big_enough',
                      '5a_occlusion_reject', '5_KEPT']:
            print('  %-22s %6d' % (kname, funnel[kname]))

        print('\n========= SEMANTIC TAGS INSIDE CANDIDATE BOXES =========')
        print('  config.py SEM_TAG_TRAFFIC_LIGHT = %d' % C.SEM_TAG_TRAFFIC_LIGHT)
        total = sum(tag_hist.values()) or 1
        for tag, cnt in tag_hist.most_common(8):
            mark = '   <-- config' if tag == C.SEM_TAG_TRAFFIC_LIGHT else ''
            print('    tag %3d : %8d px (%5.1f%%)%s' % (tag, cnt, 100.0 * cnt / total, mark))

        if fracs:
            v = np.array(fracs)
            print('\n  visible fraction: min %.3f  median %.3f  max %.3f  (threshold %.2f)'
                  % (v.min(), np.median(v), v.max(), C.MIN_VISIBLE_FRAC))

        print('\n================ DIAGNOSIS ================')
        if funnel['1_within_range'] == 0:
            print('  No lights within %.0f m of any pose - poses are wrong.' % C.MAX_TL_DIST)
        elif funnel['3_on_screen'] == 0:
            print('  Nearby but never on screen -> camera transform / projection is wrong,')
            print('  or the ego is facing away from the signals (try --back 15).')
        elif funnel['4_big_enough'] == 0:
            print('  Always smaller than %d px -> try --back 12' % C.MIN_BOX_PX)
        elif funnel['5_KEPT'] == 0:
            top = tag_hist.most_common(1)[0][0] if tag_hist else None
            print('  Boxes reach the occlusion check and ALL get rejected.')
            print('  -> SEM_TAG_TRAFFIC_LIGHT is wrong.')
            if top is not None:
                print('  -> run this, then rerun:')
                print('     sed -i "s/^SEM_TAG_TRAFFIC_LIGHT = .*/SEM_TAG_TRAFFIC_LIGHT = %d/" config.py'
                      % top)
        else:
            print('  OK: %d boxes survived every filter across %d poses.'
                  % (funnel['5_KEPT'], len(poses)))

        examples.sort(key=lambda t: -t[0])
        for cnt, idx, img, labels in examples[:6]:
            im = Image.fromarray(img)
            d = ImageDraw.Draw(im)
            for x1, y1, x2, y2, st in labels:
                d.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
                d.text((x1, max(0, y1 - 12)), st, fill=(255, 255, 0))
            im.save(os.path.join(args.out, 'debug_%03d_%dboxes.jpg' % (idx, cnt)), quality=92)
        if examples:
            print('\n  wrote %d annotated examples to %s'
                  % (min(6, len(examples)), args.out))
            print('  LOOK AT THEM - boxes must sit on lamp heads, label must match the lit lamp.')

    finally:
        if rig is not None:
            try:
                rig.destroy()
            except Exception as e:
                print('cleanup rig:', e)
        try:
            if ego is not None and ego.is_alive:
                ego.destroy()
        except Exception as e:
            print('cleanup ego:', e)
        try:
            world.apply_settings(original)
        except Exception as e:
            print('cleanup settings:', e)


if __name__ == '__main__':
    main()
