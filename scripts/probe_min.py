#!/usr/bin/env python3
"""Minimal end-to-end probe. Prints a line before every CARLA call, so when it
dies you know exactly which call killed it.

  python3 probe_min.py                 # use whatever map is loaded
  python3 probe_min.py --town Town03   # reload the map first (riskier)

Success looks like: 'PROBE PASSED' and a file at ~/probe.jpg
"""

import argparse
import queue
import time

import numpy as np
from PIL import Image

import carla


def step(msg):
    print('>>> ' + msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='localhost')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--town', default='')
    ap.add_argument('--out', default='/home/s21/probe.jpg')
    args = ap.parse_args()

    step('connecting')
    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0)
    print('    client', client.get_client_version(), '/ server', client.get_server_version())

    if args.town:
        step('load_world(%s)  <-- most likely place to abort if orphans exist' % args.town)
        world = client.load_world(args.town)
    else:
        step('get_world()  (no reload)')
        world = client.get_world()
    print('    map =', world.get_map().name)

    step('checking for leftover actors')
    leftover = [a.type_id for a in world.get_actors()
                if a.type_id.startswith(('sensor.', 'vehicle.', 'walker.'))]
    print('    leftover:', len(leftover), leftover[:5])
    if leftover:
        print('    !! run reset_world.py first')

    original = world.get_settings()
    ego = None
    cam = None
    try:
        step('synchronous mode on')
        s = world.get_settings()
        s.synchronous_mode = True
        s.fixed_delta_seconds = 0.05
        world.apply_settings(s)

        step('spawning ego')
        bp_lib = world.get_blueprint_library()
        for sp in world.get_map().get_spawn_points():
            ego = world.try_spawn_actor(bp_lib.find('vehicle.tesla.model3'), sp)
            if ego:
                break
        if ego is None:
            raise SystemExit('could not spawn ego')
        print('    ego id', ego.id)

        step('attaching rgb camera')
        bp = bp_lib.find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', '1280')
        bp.set_attribute('image_size_y', '720')
        bp.set_attribute('fov', '90')
        cam = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.5, z=2.4)),
                                attach_to=ego)
        q = queue.Queue()
        cam.listen(q.put)
        print('    camera id', cam.id)

        step('ticking 20 frames')
        img = None
        for i in range(20):
            fid = world.tick()
            while True:
                img = q.get(timeout=10.0)
                if img.frame == fid:
                    break
        print('    got frame', img.frame)

        step('saving image')
        a = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
        Image.fromarray(a[:, :, :3][:, :, ::-1]).save(args.out, quality=92)
        print('    wrote', args.out)

        step('counting traffic lights')
        print('    ', len(world.get_actors().filter('traffic.traffic_light*')))

        print('\nPROBE PASSED')

    finally:
        step('cleanup')
        try:
            if cam is not None:
                cam.stop()
        except Exception as e:
            print('    cam.stop:', e)
        try:
            world.tick()
        except Exception:
            pass
        time.sleep(0.3)
        try:
            if cam is not None and cam.is_alive:
                cam.destroy()
                print('    camera destroyed')
        except Exception as e:
            print('    cam.destroy:', e)
        try:
            if ego is not None and ego.is_alive:
                ego.destroy()
                print('    ego destroyed')
        except Exception as e:
            print('    ego.destroy:', e)
        try:
            world.apply_settings(original)
            print('    settings restored')
        except Exception as e:
            print('    settings:', e)


if __name__ == '__main__':
    main()
