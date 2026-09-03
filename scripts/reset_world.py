#!/usr/bin/env python3
"""Put the CARLA world back into a clean, usable state.

Run this after ANY crashed script. A crashed run leaves two kinds of mess:

  1. Orphaned sensors still registered but attached to a destroyed vehicle.
     The next load_world() trips over them and aborts the client in C++.
  2. The server stuck in synchronous mode, so the next script hangs forever
     on world.tick() with no error message.

  python3 reset_world.py            # purge actors, restore async mode
  python3 reset_world.py --sync     # purge, then leave it in sync mode
"""

import argparse

import carla

KILL_PREFIXES = ('sensor.', 'vehicle.', 'walker.', 'controller.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='localhost')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--sync', action='store_true', help='leave synchronous mode ON')
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world = client.get_world()
    print('connected. map =', world.get_map().name)

    # 1. async first, so ticking is not required for anything below
    s = world.get_settings()
    print('was synchronous_mode =', s.synchronous_mode)
    s.synchronous_mode = False
    s.fixed_delta_seconds = None
    world.apply_settings(s)

    # 2. stop sensor callbacks BEFORE destroying anything
    actors = world.get_actors()
    sensors = [a for a in actors if a.type_id.startswith('sensor.')]
    for a in sensors:
        try:
            a.stop()
        except Exception:
            pass
    print('stopped %d sensors' % len(sensors))

    # 3. destroy children (sensors, controllers) before parents (vehicles, walkers)
    def purge(pred, label):
        batch = [carla.command.DestroyActor(a) for a in world.get_actors()
                 if pred(a.type_id)]
        if not batch:
            print('  %-12s none' % label)
            return
        client.apply_batch_sync(batch, True)
        print('  %-12s destroyed %d' % (label, len(batch)))

    purge(lambda t: t.startswith('sensor.'), 'sensors')
    purge(lambda t: t.startswith('controller.'), 'controllers')
    purge(lambda t: t.startswith('walker.'), 'walkers')
    purge(lambda t: t.startswith('vehicle.'), 'vehicles')

    left = [a.type_id for a in world.get_actors() if a.type_id.startswith(KILL_PREFIXES)]
    print('remaining spawnable actors:', len(left))
    if left:
        print('  ', left[:10])

    tls = len(world.get_actors().filter('traffic.traffic_light*'))
    print('traffic lights in map:', tls)

    if args.sync:
        s = world.get_settings()
        s.synchronous_mode = True
        s.fixed_delta_seconds = 0.05
        world.apply_settings(s)
        print('left in SYNCHRONOUS mode at 20 Hz')
    else:
        print('left in ASYNCHRONOUS mode')


if __name__ == '__main__':
    main()
