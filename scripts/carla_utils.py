"""Shared helpers: camera maths, sensor rig, traffic-light labelling, 3D actor boxes."""

import math
import queue
import time

import numpy as np
import carla

import config as C


# ----------------------------------------------------------------- camera maths
def build_intrinsics(w=C.IMAGE_W, h=C.IMAGE_H, fov=C.FOV):
    f = w / (2.0 * math.tan(fov * math.pi / 360.0))
    k = np.identity(3)
    k[0, 0] = k[1, 1] = f
    k[0, 2] = w / 2.0
    k[1, 2] = h / 2.0
    return k


def world_to_image(point, w2c, k):
    """3D world point -> (u, v, depth). depth <= 0 means BEHIND the camera.

    Objects behind the camera project to plausible-looking boxes in front of you
    if you forget the depth check. It is the most common CARLA labelling bug.
    """
    p = np.array([point.x, point.y, point.z, 1.0])
    p_cam = np.dot(w2c, p)
    p_std = np.array([p_cam[1], -p_cam[2], p_cam[0]])   # UE axes -> camera axes
    depth = p_std[2]
    if depth <= 0.0:
        return None, None, depth
    p_img = np.dot(k, p_std)
    return p_img[0] / p_img[2], p_img[1] / p_img[2], depth


# ----------------------------------------------------------------- buffers
def decode_bgra(image):
    a = np.frombuffer(image.raw_data, dtype=np.uint8)
    return np.reshape(a, (image.height, image.width, 4))


def rgb_array(image):
    return decode_bgra(image)[:, :, :3][:, :, ::-1].copy()      # BGRA -> RGB


def semantic_tags(image):
    """CARLA packs the semantic tag into the red channel of the raw buffer."""
    return decode_bgra(image)[:, :, 2].copy()


def lidar_array(measurement):
    a = np.frombuffer(measurement.raw_data, dtype=np.float32)
    return np.reshape(a, (-1, 4)).copy()                        # x, y, z, intensity


# ----------------------------------------------------------------- sensor rig
class Rig:
    """RGB + semantic cameras live for the whole run.

    The LiDAR is destroyed and respawned whenever the weather changes, because
    blueprint attributes are immutable once the sensor exists.
    """

    def __init__(self, world, vehicle, save_semantic=False):
        self.world = world
        self.vehicle = vehicle
        self.bp = world.get_blueprint_library()
        self.save_semantic = save_semantic
        self.queues = {}
        self.sensors = {}
        self.lidar = None
        self.lidar_q = None
        self.k = build_intrinsics()

        if not vehicle.is_alive:
            raise RuntimeError('parent vehicle is not alive - tick after spawning it')

        cam_tf = carla.Transform(carla.Location(x=C.CAM_X, z=C.CAM_Z))
        for name, bp_id in (('rgb', 'sensor.camera.rgb'),
                            ('sem', 'sensor.camera.semantic_segmentation')):
            bp = self.bp.find(bp_id)
            bp.set_attribute('image_size_x', str(C.IMAGE_W))
            bp.set_attribute('image_size_y', str(C.IMAGE_H))
            bp.set_attribute('fov', str(C.FOV))
            s = world.spawn_actor(bp, cam_tf, attach_to=vehicle)
            self.sensors[name] = s
            print('    attached %s (id %d)' % (name, s.id), flush=True)

        # In synchronous mode an actor is not fully created until the next tick.
        # Registering listen() or attaching more children before that tick is
        # what produces "trying to operate on a destroyed actor" + abort.
        self._tick()

        for name, s in self.sensors.items():
            q = queue.Queue()
            s.listen(q.put)
            self.queues[name] = q

        self.lidar_tf = carla.Transform(carla.Location(x=C.LIDAR_X, z=C.LIDAR_Z))

    def _tick(self):
        try:
            if self.world.get_settings().synchronous_mode:
                self.world.tick()
            else:
                self.world.wait_for_tick()
        except Exception:
            pass
        time.sleep(0.05)

    def set_lidar(self, params):
        """Respawn the LiDAR. Blueprint attributes are immutable after spawn,
        so changing weather coupling means a new sensor."""
        if self.lidar is not None:
            try:
                self.lidar.stop()
            except Exception:
                pass
            self._tick()
            try:
                if self.lidar.is_alive:
                    self.lidar.destroy()
            except Exception as e:
                print('    lidar destroy:', e, flush=True)
            self.lidar = None
            self.lidar_q = None
            self._tick()

        if not self.vehicle.is_alive:
            raise RuntimeError('parent vehicle died before LiDAR spawn')

        bp = self.bp.find('sensor.lidar.ray_cast')
        for k, v in C.LIDAR_BASE.items():
            if bp.has_attribute(k):
                bp.set_attribute(k, v)
            else:
                print('    WARN lidar has no attribute %r - skipped' % k, flush=True)
        for k, v in params.items():
            if bp.has_attribute(k):
                bp.set_attribute(k, str(v))
            else:
                print('    WARN lidar has no attribute %r - skipped' % k, flush=True)

        self.lidar = self.world.spawn_actor(bp, self.lidar_tf, attach_to=self.vehicle)
        print('    attached lidar (id %d)' % self.lidar.id, flush=True)
        self._tick()
        self.lidar_q = queue.Queue()
        self.lidar.listen(self.lidar_q.put)

    def grab(self, frame_id, timeout=10.0):
        """Block until every sensor has delivered THIS frame. Guarantees alignment."""
        out = {}
        pairs = list(self.queues.items())
        if self.lidar_q is not None:
            pairs.append(('lidar', self.lidar_q))
        for name, q in pairs:
            while True:
                data = q.get(timeout=timeout)
                if data.frame == frame_id:
                    out[name] = data
                    break
                if data.frame > frame_id:
                    raise RuntimeError('%s overran frame %d - sync broken' % (name, frame_id))
        return out

    def destroy(self):
        """Stop every callback, let one tick drain, THEN destroy.

        Destroying a sensor while its listen callback is still firing makes the
        CARLA client throw a C++ exception that aborts the process rather than
        raising something Python can catch.
        """
        actors = list(self.sensors.values())
        if self.lidar is not None:
            actors.append(self.lidar)

        for s in actors:
            try:
                s.stop()
            except Exception:
                pass
        try:
            self.world.tick()          # drain anything already in flight
        except Exception:
            pass
        time.sleep(0.2)

        for s in actors:
            try:
                if s.is_alive:
                    s.destroy()
            except Exception:
                pass

        self.sensors = {}
        self.queues = {}
        self.lidar = None
        self.lidar_q = None


# ----------------------------------------------------------------- traffic lights
def _tl_boxes(tl):
    """Per-lamp-head boxes if the build supports it, else the whole-actor box.

    actor.bounding_box on a traffic light often encloses the entire pole and mast
    arm, which is not the object you want to detect.
    """
    if hasattr(tl, 'get_light_boxes'):
        try:
            return [b.get_world_vertices(carla.Transform()) for b in tl.get_light_boxes()]
        except Exception:
            pass
    return [tl.bounding_box.get_world_vertices(tl.get_transform())]


def traffic_light_labels(world, camera, k, sem, tl_tag=C.SEM_TAG_TRAFFIC_LIGHT):
    """Return (yolo_rows, meta_rows).

    yolo_rows : (cls, cx, cy, w, h) normalised 0-1
    meta_rows : dicts with pixel box, distance, state, visible fraction
    """
    yolo_rows, meta_rows = [], []
    w2c = np.array(camera.get_transform().get_inverse_matrix())
    cam_loc = camera.get_transform().location

    for tl in world.get_actors().filter('traffic.traffic_light*'):
        dist = tl.get_transform().location.distance(cam_loc)
        if dist > C.MAX_TL_DIST:
            continue
        state = str(tl.get_state())
        cls = C.TL_CLASS_ID.get(state)
        if cls is None:
            continue

        for verts in _tl_boxes(tl):
            us, vs, ok = [], [], True
            for v in verts:
                u, vv, depth = world_to_image(v, w2c, k)
                if depth <= 0.0:                 # behind the camera
                    ok = False
                    break
                us.append(u)
                vs.append(vv)
            if not ok:
                continue

            x1, x2 = max(0.0, min(us)), min(float(C.IMAGE_W), max(us))
            y1, y2 = max(0.0, min(vs)), min(float(C.IMAGE_H), max(vs))
            bw, bh = x2 - x1, y2 - y1
            if bw <= 1 or bh < C.MIN_BOX_PX:
                continue

            patch = sem[int(y1):int(y2), int(x1):int(x2)]
            if patch.size == 0:
                continue
            visible = float(np.count_nonzero(patch == tl_tag)) / patch.size
            if visible < C.MIN_VISIBLE_FRAC:
                continue

            yolo_rows.append((cls,
                              (x1 + bw / 2) / C.IMAGE_W, (y1 + bh / 2) / C.IMAGE_H,
                              bw / C.IMAGE_W, bh / C.IMAGE_H))
            meta_rows.append(dict(cls=cls, state=state,
                                  box=[round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                                  dist_m=round(dist, 1), visible=round(visible, 3)))
    return yolo_rows, meta_rows


# ----------------------------------------------------------------- 3D boxes for BEV
def actor_boxes_3d(world, lidar, max_dist=60.0):
    """Vehicle and walker boxes expressed in the LiDAR frame, for lidar_to_bev.py."""
    out = []
    l2w = lidar.get_transform()
    w2l = np.array(l2w.get_inverse_matrix())
    for wildcard, group in (('vehicle.*', 'vehicle'), ('walker.pedestrian.*', 'pedestrian')):
        for a in world.get_actors().filter(wildcard):
            d = a.get_transform().location.distance(l2w.location)
            if d > max_dist or d < 0.5:          # skip the ego itself
                continue
            verts = []
            for v in a.bounding_box.get_world_vertices(a.get_transform()):
                p = np.dot(w2l, np.array([v.x, v.y, v.z, 1.0]))
                verts.append([float(p[0]), float(p[1]), float(p[2])])
            out.append(dict(type_id=a.type_id, group=group,
                            dist_m=round(d, 1), verts=verts))
    return out
