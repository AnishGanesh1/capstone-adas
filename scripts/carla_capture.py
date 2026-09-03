#!/usr/bin/env python3
"""
CARLA adverse-weather capture for traffic-light (camera) and object (LiDAR) detection.

What it does
------------
For each ego pose, freezes the scene, then sweeps every weather condition capturing a
pixel-aligned set of: RGB, semantic segmentation, depth, LiDAR sweep, and YOLO-format
traffic-light labels.

Run
---
  ./CarlaUE4.sh -RenderOffScreen -quality-level=Epic
  python carla_capture.py --town Town03 --poses 40 --out ./dataset

NOTE: this has been syntax-checked but not executed against a live CARLA server.
Verify the two version-dependent items flagged with `VERIFY:` before a full run.
"""

import argparse
import json
import math
import os
import queue
import random

import numpy as np

try:
    import carla
except ImportError:
    raise SystemExit("carla module not found. pip install carla==<your server version>")


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

IMAGE_W, IMAGE_H, FOV = 1920, 1080, 90.0
FIXED_DELTA = 0.05                      # 20 Hz
SETTLE_TICKS = 15                       # ticks after a weather change before capturing
MIN_BOX_PX = 8                          # discard traffic lights shorter than this
MIN_VISIBLE_FRAC = 0.35                 # discard if less than this fraction visible

# VERIFY: semantic tag id for traffic lights changes between CARLA releases.
# Dump one semantic frame and check before trusting this.
SEM_TAG_TRAFFIC_LIGHT = 18

TL_CLASS = {"Red": 0, "Yellow": 1, "Green": 2, "Off": 3}

# camera weather + matching LiDAR degradation (CARLA does NOT couple these itself)
CONDITIONS = {
    "clear": dict(
        weather=dict(cloudiness=0, precipitation=0, precipitation_deposits=0,
                     wind_intensity=0, sun_altitude_angle=70, sun_azimuth_angle=0,
                     fog_density=0, fog_distance=0, fog_falloff=0.1, wetness=0),
        lidar=dict(atmosphere_attenuation_rate=0.004, dropoff_general_rate=0.45,
                   noise_stddev=0.0)),
    "rain": dict(
        weather=dict(cloudiness=80, precipitation=80, precipitation_deposits=60,
                     wind_intensity=60, sun_altitude_angle=45, sun_azimuth_angle=0,
                     fog_density=10, fog_distance=60, fog_falloff=0.5, wetness=60),
        lidar=dict(atmosphere_attenuation_rate=0.020, dropoff_general_rate=0.55,
                   noise_stddev=0.04)),
    "fog": dict(
        weather=dict(cloudiness=60, precipitation=0, precipitation_deposits=0,
                     wind_intensity=10, sun_altitude_angle=45, sun_azimuth_angle=0,
                     fog_density=70, fog_distance=10, fog_falloff=1.5, wetness=0),
        lidar=dict(atmosphere_attenuation_rate=0.030, dropoff_general_rate=0.55,
                   noise_stddev=0.03)),
    "haze": dict(
        weather=dict(cloudiness=40, precipitation=0, precipitation_deposits=0,
                     wind_intensity=10, sun_altitude_angle=60, sun_azimuth_angle=0,
                     fog_density=30, fog_distance=60, fog_falloff=0.8, wetness=0),
        lidar=dict(atmosphere_attenuation_rate=0.010, dropoff_general_rate=0.45,
                   noise_stddev=0.01)),
    "hail": dict(
        weather=dict(cloudiness=90, precipitation=90, precipitation_deposits=80,
                     wind_intensity=90, sun_altitude_angle=40, sun_azimuth_angle=0,
                     fog_density=20, fog_distance=40, fog_falloff=0.8, wetness=70),
        lidar=dict(atmosphere_attenuation_rate=0.025, dropoff_general_rate=0.60,
                   noise_stddev=0.06)),
    "night": dict(
        weather=dict(cloudiness=20, precipitation=0, precipitation_deposits=0,
                     wind_intensity=10, sun_altitude_angle=-20, sun_azimuth_angle=0,
                     fog_density=5, fog_distance=60, fog_falloff=0.5, wetness=0),
        # LiDAR is active - darkness does not degrade it. This is the point.
        lidar=dict(atmosphere_attenuation_rate=0.004, dropoff_general_rate=0.45,
                   noise_stddev=0.0)),
    "wet_night": dict(
        weather=dict(cloudiness=30, precipitation=0, precipitation_deposits=70,
                     wind_intensity=20, sun_altitude_angle=-20, sun_azimuth_angle=0,
                     fog_density=10, fog_distance=50, fog_falloff=0.5, wetness=80),
        lidar=dict(atmosphere_attenuation_rate=0.006, dropoff_general_rate=0.45,
                   noise_stddev=0.01)),
}

LIDAR_BASE = dict(channels="64", range="100.0", points_per_second="1300000",
                  rotation_frequency=str(1.0 / FIXED_DELTA),
                  upper_fov="10.0", lower_fov="-30.0", horizontal_fov="360.0")


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def build_intrinsics(w, h, fov):
    f = w / (2.0 * math.tan(fov * math.pi / 360.0))
    k = np.identity(3)
    k[0, 0] = k[1, 1] = f
    k[0, 2] = w / 2.0
    k[1, 2] = h / 2.0
    return k


def world_to_image(point, w2c, k):
    """3D world point -> (u, v, depth). depth <= 0 means behind the camera."""
    p = np.array([point.x, point.y, point.z, 1.0])
    p_cam = np.dot(w2c, p)
    # UE axes (x fwd, y right, z up) -> standard camera axes
    p_std = np.array([p_cam[1], -p_cam[2], p_cam[0]])
    depth = p_std[2]
    if depth <= 0.0:
        return None, None, depth
    p_img = np.dot(k, p_std)
    return p_img[0] / p_img[2], p_img[1] / p_img[2], depth


def decode_bgra(image):
    a = np.frombuffer(image.raw_data, dtype=np.uint8)
    return np.reshape(a, (image.height, image.width, 4))


def semantic_tags(image):
    """CARLA packs the semantic tag into the red channel of the raw buffer."""
    return decode_bgra(image)[:, :, 2]


def lidar_to_numpy(measurement):
    a = np.frombuffer(measurement.raw_data, dtype=np.float32)
    return np.reshape(a, (-1, 4)).copy()          # x, y, z, intensity


# --------------------------------------------------------------------------------------
# Sensor rig
# --------------------------------------------------------------------------------------

class Rig:
    """Cameras stay alive for the whole run; the LiDAR is respawned per condition
    because blueprint attributes are immutable once the sensor exists."""

    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self.bp = world.get_blueprint_library()
        self.queues = {}
        self.sensors = {}
        self.lidar = None
        self.lidar_q = None

        cam_tf = carla.Transform(carla.Location(x=1.5, z=2.4))
        for name, bp_id in (("rgb", "sensor.camera.rgb"),
                            ("sem", "sensor.camera.semantic_segmentation"),
                            ("depth", "sensor.camera.depth")):
            bp = self.bp.find(bp_id)
            bp.set_attribute("image_size_x", str(IMAGE_W))
            bp.set_attribute("image_size_y", str(IMAGE_H))
            bp.set_attribute("fov", str(FOV))
            s = world.spawn_actor(bp, cam_tf, attach_to=vehicle)
            q = queue.Queue()
            s.listen(q.put)
            self.sensors[name] = s
            self.queues[name] = q

        self.lidar_tf = carla.Transform(carla.Location(x=0.0, z=2.4))
        self.k = build_intrinsics(IMAGE_W, IMAGE_H, FOV)

    def set_lidar(self, params):
        if self.lidar is not None:
            self.lidar.stop()
            self.lidar.destroy()
        bp = self.bp.find("sensor.lidar.ray_cast")
        for key, val in LIDAR_BASE.items():
            bp.set_attribute(key, val)
        for key, val in params.items():
            bp.set_attribute(key, str(val))
        self.lidar_q = queue.Queue()
        self.lidar = self.world.spawn_actor(bp, self.lidar_tf, attach_to=self.vehicle)
        self.lidar.listen(self.lidar_q.put)

    def grab(self, frame_id, timeout=5.0):
        """Block until every sensor has delivered the given frame."""
        out = {}
        for name, q in list(self.queues.items()) + [("lidar", self.lidar_q)]:
            while True:
                data = q.get(timeout=timeout)
                if data.frame == frame_id:
                    out[name] = data
                    break
                if data.frame > frame_id:
                    raise RuntimeError(f"{name} overran frame {frame_id} - sync broken")
        return out

    def destroy(self):
        for s in self.sensors.values():
            s.stop()
            s.destroy()
        if self.lidar is not None:
            self.lidar.stop()
            self.lidar.destroy()


# --------------------------------------------------------------------------------------
# Traffic light labelling
# --------------------------------------------------------------------------------------

def traffic_light_boxes(world, camera, k, sem_tags, max_dist=90.0):
    """Return YOLO rows (cls, cx, cy, w, h) normalised, for visible traffic lights."""
    rows = []
    w2c = np.array(camera.get_transform().get_inverse_matrix())
    cam_loc = camera.get_transform().location

    for tl in world.get_actors().filter("traffic.traffic_light*"):
        if tl.get_transform().location.distance(cam_loc) > max_dist:
            continue

        state = str(tl.get_state())
        cls = TL_CLASS.get(state)
        if cls is None:
            continue

        # VERIFY: get_light_boxes() exists in recent releases and gives per-lamp-head
        # boxes, which is what we want. actor.bounding_box often encloses the whole pole.
        if hasattr(tl, "get_light_boxes"):
            boxes = tl.get_light_boxes()
            verts_sets = [b.get_world_vertices(carla.Transform()) for b in boxes]
        else:
            bb = tl.bounding_box
            verts_sets = [bb.get_world_vertices(tl.get_transform())]

        for verts in verts_sets:
            us, vs, ok = [], [], True
            for v in verts:
                u, vv, depth = world_to_image(v, w2c, k)
                if depth <= 0.0:            # behind the camera - discard the whole box
                    ok = False
                    break
                us.append(u)
                vs.append(vv)
            if not ok:
                continue

            x1, x2 = max(0.0, min(us)), min(float(IMAGE_W), max(us))
            y1, y2 = max(0.0, min(vs)), min(float(IMAGE_H), max(vs))
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh < MIN_BOX_PX:
                continue

            patch = sem_tags[int(y1):int(y2), int(x1):int(x2)]
            if patch.size == 0:
                continue
            visible = float(np.count_nonzero(patch == SEM_TAG_TRAFFIC_LIGHT)) / patch.size
            if visible < MIN_VISIBLE_FRAC:
                continue

            rows.append((cls,
                         (x1 + bw / 2) / IMAGE_W, (y1 + bh / 2) / IMAGE_H,
                         bw / IMAGE_W, bh / IMAGE_H))
    return rows


def actor_boxes_3d(world, lidar, max_dist=60.0):
    """3D boxes of vehicles and walkers in the LiDAR frame, for the BEV stage."""
    out = []
    l2w = lidar.get_transform()
    w2l = np.array(l2w.get_inverse_matrix())
    for wildcard, cls in (("vehicle.*", "vehicle"), ("walker.pedestrian.*", "pedestrian")):
        for a in world.get_actors().filter(wildcard):
            if a.get_transform().location.distance(l2w.location) > max_dist:
                continue
            bb = a.bounding_box
            tf = a.get_transform()
            verts = []
            for v in bb.get_world_vertices(tf):
                p = np.dot(w2l, np.array([v.x, v.y, v.z, 1.0]))
                verts.append([float(p[0]), float(p[1]), float(p[2])])
            out.append(dict(type_id=a.type_id, group=cls, verts=verts))
    return out


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def pick_poses(world, n, seed=42):
    """Spawn points nearest to traffic lights, as stand-ins for junction approaches."""
    rng = random.Random(seed)
    spawns = world.get_map().get_spawn_points()
    lights = [tl.get_transform().location
              for tl in world.get_actors().filter("traffic.traffic_light*")]
    scored = []
    for sp in spawns:
        if not lights:
            break
        d = min(sp.location.distance(l) for l in lights)
        if 10.0 < d < 80.0:
            scored.append((d, sp))
    rng.shuffle(scored)
    return [sp for _, sp in scored[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--town", default="Town03")
    ap.add_argument("--poses", type=int, default=40)
    ap.add_argument("--traffic", type=int, default=60)
    ap.add_argument("--out", default="./dataset")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    for sub in ("images", "labels", "lidar", "meta", "sem", "depth"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world = client.load_world(args.town)

    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DELTA
    world.apply_settings(settings)

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(args.seed)

    bp_lib = world.get_blueprint_library()
    spawned = []

    try:
        # ---- background traffic -------------------------------------------------
        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        for sp in spawn_points[:args.traffic]:
            bp = random.choice(bp_lib.filter("vehicle.*"))
            v = world.try_spawn_actor(bp, sp)
            if v:
                v.set_autopilot(True, tm.get_port())
                spawned.append(v)

        # ---- ego ---------------------------------------------------------------
        ego_bp = bp_lib.find("vehicle.tesla.model3")
        ego = None
        for sp in spawn_points:
            ego = world.try_spawn_actor(ego_bp, sp)
            if ego:
                break
        if ego is None:
            raise SystemExit("could not spawn ego vehicle")
        spawned.append(ego)

        rig = Rig(world, ego)
        for _ in range(20):
            world.tick()

        poses = pick_poses(world, args.poses, args.seed)
        print(f"{len(poses)} poses selected in {args.town}")

        idx = 0
        for pose_i, pose in enumerate(poses):
            ego.set_transform(pose)
            for _ in range(10):
                world.tick()

            # freeze the scene so every condition sees identical geometry
            world.freeze_all_traffic_lights(True)
            for a in world.get_actors().filter("vehicle.*"):
                a.set_simulate_physics(False)
            for a in world.get_actors().filter("walker.*"):
                a.set_simulate_physics(False)

            for state_name, state in (("red", carla.TrafficLightState.Red),
                                      ("yellow", carla.TrafficLightState.Yellow),
                                      ("green", carla.TrafficLightState.Green)):
                for tl in world.get_actors().filter("traffic.traffic_light*"):
                    tl.set_state(state)
                world.tick()

                for cond_name, cond in CONDITIONS.items():
                    rig.set_lidar(cond["lidar"])
                    world.set_weather(carla.WeatherParameters(**cond["weather"]))
                    for _ in range(SETTLE_TICKS):
                        world.tick()

                    frame_id = world.tick()
                    data = rig.grab(frame_id)

                    stem = f"{args.town}_{cond_name}_{state_name}_p{pose_i:03d}_{idx:06d}"
                    data["rgb"].save_to_disk(
                        os.path.join(args.out, "images", stem + ".png"))
                    data["sem"].save_to_disk(
                        os.path.join(args.out, "sem", stem + ".png"))
                    data["depth"].save_to_disk(
                        os.path.join(args.out, "depth", stem + ".png"))

                    pts = lidar_to_numpy(data["lidar"])
                    np.save(os.path.join(args.out, "lidar", stem + ".npy"), pts)

                    rows = traffic_light_boxes(world, rig.sensors["rgb"], rig.k,
                                               semantic_tags(data["sem"]))
                    with open(os.path.join(args.out, "labels", stem + ".txt"), "w") as fh:
                        for cls, cx, cy, bw, bh in rows:
                            fh.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

                    meta = dict(
                        stem=stem, town=args.town, condition=cond_name,
                        light_state=state_name, pose_index=pose_i,
                        lidar_params=cond["lidar"], weather=cond["weather"],
                        n_lidar_points=int(pts.shape[0]),
                        n_traffic_lights=len(rows),
                        ego=dict(x=ego.get_transform().location.x,
                                 y=ego.get_transform().location.y,
                                 z=ego.get_transform().location.z,
                                 yaw=ego.get_transform().rotation.yaw),
                        boxes_3d=actor_boxes_3d(world, rig.lidar))
                    with open(os.path.join(args.out, "meta", stem + ".json"), "w") as fh:
                        json.dump(meta, fh)

                    idx += 1
                    print(f"[{idx}] {stem}  tl={len(rows)}  pts={pts.shape[0]}")

            # unfreeze and let the world move on
            for a in world.get_actors().filter("vehicle.*"):
                a.set_simulate_physics(True)
            for a in world.get_actors().filter("walker.*"):
                a.set_simulate_physics(True)
            world.freeze_all_traffic_lights(False)
            for _ in range(40):
                world.tick()

        rig.destroy()

    finally:
        world.apply_settings(original)
        tm.set_synchronous_mode(False)
        for a in spawned:
            try:
                a.destroy()
            except Exception:
                pass
        print("done")


if __name__ == "__main__":
    main()
