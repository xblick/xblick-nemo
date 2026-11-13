"""durchblick-nemo: NeMO (nemo.py) on cameras' colour and depth: objects known from a few images of them found,
segmented and posed (metric, with the depth) in every frame; any number of cameras, started and stopped at run time.

    durchblick-nemo --objects-dir DIR [--object NAME ...] [--camera CAMERA[:color=NAME][:depth=NAME] ...]
                    [--node nemo] [--segments] [--autocast bfloat16]

A camera is ``<CAMERA>/<color>`` (rgb8 or bgr8) and ``<CAMERA>/<depth>`` (16UC1 mm in the colour camera's geometry),
paired by stamp, at the depth's resolution, undistorted (blick.rgbd). An object (``<DIR>/NeMO/<name>/``,
objects.py; made by durchblick-nemo-make from images or by this node from a camera's views) is, per camera:

* found while lost: NeMO decodes the whole frame in patches (nemo.NeMO.detect), its largest region over
  --mask-threshold of at least --min-pixels is the object (at most every --detect-every seconds per object and camera:
  hundreds of milliseconds with several objects)
* then tracked: each frame a square crop around its last mask (1.4 times its extent) is decoded, all tracked objects of
  a frame in one batch; its pose from its confident mask pixels' NeMO points and their measured depth (nemo.NeMO.pose);
  fewer than --min-inliers inliers, or under --min-inlier-ratio of the mask's pixels: lost

Published per camera, stamped as the frame, in the colour camera's frame (undistorted: the camera of its P):

* ``<CAMERA>/<node>/detections_3d``  vision_msgs/Detection3DArray: per object its name (class_id; score: the inlier
  ratio) and pose (NeMO's frame scaled to metres in the camera), the box (its surface points' 2-98 % extent: centre
  pose and size), ``id`` the object's instance id (1, 2, ... in --object order, then as added)
* with --segments, as durchblick-sam3 does (so durchblick-foundationpose can take NeMO as its segmenter,
  ``:segmenter=<node>``): ``<CAMERA>/<node>/instances`` (mono16, the colour image's size, the instance ids) and
  ``<CAMERA>/<node>/detections_2d``, in the colour camera's own image (its size; a camera with lens distortion through
  blick.rgbd.Distorter: the masks are found in the undistorted frame)

Queryables: ``<node>/cameras`` ({"start": SPEC} | {"stop": CAMERA}; durchblick-nemo-objects cameras) and
``<node>/objects`` ({"add": NAME} | {"remove": NAME} | {"make": {"camera": CAMERA, "name": NAME, "label": LABEL |
"sam_id": ID, "views": 4, "min_angle": 15, "timeout": 60}}; durchblick-nemo-objects): make collects the camera's
next views of the object at least ``min_angle`` degrees apart seen from it (Durchblick's collect_views: the camera's
pose from <camera>/tf; move the camera or the object), the object's mask in each from the camera's segmenter (its
spec's ``:segmenter=NAME``, e.g. a durchblick-sam3 instance: by its label, the largest such instance, or by its id),
and makes, saves and adds the object.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from dataclasses import dataclass, field

import blick as db
import numpy as np
from blick import node, topics
from blick.paths import expand_path
from blick.rgbd import CameraSpec, Distorter, FramePreparer, RGBDInputs, collect_views
from blick.session import add_arguments, open_from_args

NODE = "nemo"
CAMERAS = "cameras"
OBJECTS = "objects"


@dataclass
class Track:
    box: tuple | None = None  # the square crop (x0, y0, side) of the next frame; None: lost
    last_detect: float = -math.inf


@dataclass
class Camera:
    spec: CameraSpec
    inputs: RGBDInputs
    prepare: FramePreparer
    detections_3d: object
    detections_2d: object = None
    instances: object = None  # with --segments: created with the first frame (the colour camera_info)
    distorter: object = None  # with --segments and lens distortion: the frames' labels back into the colour image
    distorter_key: object = None
    report: object = None  # blick.node.Reporter
    tracks: dict = field(default_factory=dict)
    last_stamp: int | None = None


def quaternion(R: np.ndarray) -> db.Quaternion:
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return db.Quaternion(x, y, z, w)


def detection_3d(header, obj, pose, instance_id: int) -> db.Detection3D:
    """An object's pose (NeMO's frame scaled to metres) and box as a Detection3D."""
    R = pose.matrix[:3, :3] / pose.scale
    q = quaternion(R)
    centre = pose.matrix[:3, :3] @ ((obj.lo + obj.hi) / 2) + pose.matrix[:3, 3]
    size = (obj.hi - obj.lo) * pose.scale
    return db.Detection3D(
        header, [db.ObjectHypothesisWithPose(db.ObjectHypothesis(obj.name, pose.inliers / max(pose.mask_pixels, 1)),
                                             db.PoseWithCovariance(db.Pose(db.Point(*pose.matrix[:3, 3]), q)))],
        db.BoundingBox3D(db.Pose(db.Point(*centre), q), db.Vector3(*size.astype(float))), str(instance_id))


# the parameters that change while the node runs (Parameters.declare_options); every other option applies at the start.
# object: the objects to find (a new list loads the new ones and drops the others)
LIVE = {"mask_threshold": {"range": (0.0, 1.0, 0.01)},
        "conf_threshold": {"range": (0.0, 10.0, 0.05)},
        "inlier_distance": {"range": (0.0, 0.1, 0.001), "unit": "m"},
        "min_inliers": {"range": (3, 10000, 1)},
        "min_inlier_ratio": {"range": (0.0, 1.0, 0.01)},
        "min_pixels": {"range": (0, 100000, 10), "unit": "pixels"},
        "detect_every": {"range": (0.0, 60.0, 0.1), "unit": "s"}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="durchblick-nemo", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--objects-dir", required=True, help="where the objects are (<DIR>/NeMO/<name>/)")
    parser.add_argument("--object", action="append", default=[],
                        help="an object to find (repeated; at run time: the parameter object, or <node>/objects)")
    parser.add_argument("--camera", action="append", default=[], metavar="SPEC",
                        help="CAMERA[:color=NAME][:depth=NAME][:segmenter=NAME] (repeated; also at run time)")
    parser.add_argument("--node", default=NODE)
    parser.add_argument("--segments", action="store_true",
                        help="also publish instances and detections_2d (a segmenter for durchblick-foundationpose)")
    parser.add_argument("--autocast", choices=["bfloat16", "none"], default="bfloat16",
                        help="NeMO's decoder in bfloat16 (the default: as accurate as float32 on the benchmark, the "
                             "whole-frame search 160 -> 61 ms) or float32 (none)")
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--conf-threshold", type=float, default=1.2, help="NeMO's confidence of a point (its examples)")
    parser.add_argument("--inlier-distance", type=float, default=0.01, help="[m] of the pose fit")
    parser.add_argument("--min-inliers", type=int, default=50)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.3)
    parser.add_argument("--min-pixels", type=int, default=200, help="smallest found region [px at the depth's size]")
    parser.add_argument("--detect-every", type=float, default=0.5, help="[s] between searches for a lost object")
    parser.add_argument("--history", type=int, default=10, help="frames of each stream kept for pairing by stamp")
    parser.add_argument("--max-offset-ms", type=float, default=None,
                        help="pair a colour frame with the nearest depth frame this close (default: half the "
                             "depth's frame period, measured: each colour frame with its nearest depth frame, as a "
                             "camera whose depth is stamped apart from its colour needs, one measured Azure Kinect: "
                             "11-17 ms); 0: only equal stamps")
    parser.add_argument("--device", default="cuda")
    add_arguments(parser)
    # --max-hz: 20 colour frames/s (about 10 measured; 0: every frame)
    # --local: auto (the CUDA ring on this GPU, else shared memory), shm or cuda
    node.add_arguments(parser, max_hz=20.0, local=("auto", "shm", "cuda"))
    node.Parameters.add_arguments(parser)  # --param NAME:=VALUE, --params-file, --dump-params, --params-read-only
    args = parser.parse_args(argv)
    # the options as parameters (the terminal interface shows them): read-only, but LIVE and the objects
    session = open_from_args(args)
    params = node.Parameters(session, args.node)
    params.configure(args)

    def check_objects(value) -> str | None:
        """A new object list: refused with the reason if an object is not in the objects directory."""
        from . import objects  # noqa: PLC0415 - with the model's imports

        missing = sorted(set(value) - set(objects.names(args.objects_dir)) - set(loaded))
        return f"{missing}: not in {args.objects_dir}" if missing else None

    params.declare_options(parser, args, live={**LIVE, "object": {"write": check_objects}}, leave_out=("node",))

    def library() -> list[str]:
        from . import objects  # noqa: PLC0415 - with the model's imports

        return objects.names(expand_path(args.objects_dir))

    # what the parameter object takes: the objects in the objects directory (made ones too), read at each request
    params.declare("library", [], description="the objects in --objects-dir (object takes their names)", read_only=True,
                   read=library)
    params.check_overrides()
    try:
        args.objects_dir = expand_path(args.objects_dir)  # ${VAR} and ~ (some launchers pass arguments as they stand)
        specs = [CameraSpec.parse(c) for c in args.camera]
        for spec in specs:
            topics.check_name(spec.camera)
    except ValueError as error:
        parser.error(str(error))
    if len({s.camera for s in specs}) != len(specs):
        parser.error("--camera: a camera given twice")

    # DINOv2 without xFormers: slower, the same
    node.start("durchblick-nemo", ignore=["xFormers is not available"], max_hz=args.max_hz, local=args.local)

    import cv2  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from . import objects  # noqa: PLC0415
    from .nemo import NeMO, crop_images, crop_pixels, square_box  # noqa: PLC0415

    device = torch.device(args.device)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    objects_dir = args.objects_dir
    objects_dir.mkdir(parents=True, exist_ok=True)
    nemo = NeMO(device, autocast=None if args.autocast == "none" else args.autocast)
    generator = torch.Generator(device=device).manual_seed(0)
    loaded: dict[str, objects.NeuralObject] = {}
    instance_ids: dict[str, int] = {}
    lock = threading.Lock()  # loaded, cameras: the queryables' threads and the loop

    def add(name: str):
        obj = objects.load(objects_dir, name, device)
        with lock:
            loaded[name] = obj
            instance_ids.setdefault(name, len(instance_ids) + 1)
            for camera in cameras.values():
                camera.tracks[name] = Track()
        print(f"durchblick-nemo: {name} (instance {instance_ids[name]})", flush=True)

    def remove(name: str):
        with lock:
            if loaded.pop(name, None) is None:
                raise ValueError(f"{name}: not loaded")
            for camera in cameras.values():
                camera.tracks.pop(name, None)

    wake = threading.Event()
    cameras: dict[str, Camera] = {}
    for name in args.object:
        try:
            add(name)
        except (FileNotFoundError, ValueError) as error:
            raise SystemExit(str(error)) from None

    max_offset_ns = None if args.max_offset_ms is None else round(args.max_offset_ms * 1e6)

    def start(spec: CameraSpec):
        with lock:
            if spec.camera in cameras:
                raise ValueError(f"{spec.camera}: already read (as {cameras[spec.camera].spec}); stop it first")
            camera = Camera(spec, RGBDInputs(session, spec, args.history, args.local, wake, max_hz=args.max_hz,
                                             device=device, max_offset_ns=max_offset_ns),
                            FramePreparer(device),
                            db.DetectionPublisher(session, topics.detections(spec.camera, args.node, "detections_3d"),
                                                  db.Detection3DArray))
            camera.report = node.Reporter(session, spec.camera, args.node, "durchblick-nemo",
                                          log_prefix=f"{spec.camera}: ", max_hz=args.max_hz)
            camera.report.watch(camera.inputs)
            if args.segments:
                camera.detections_2d = db.DetectionPublisher(
                    session, topics.detections(spec.camera, args.node, "detections_2d"), db.Detection2DArray)
            camera.tracks = {n: Track() for n in loaded}
            cameras[spec.camera] = camera
        print(f"durchblick-nemo: reading {spec} ({', '.join(spec.keys.values())})", flush=True)

    def stop(name: str):
        with lock:
            camera = cameras.pop(name, None)
        if camera is None:
            raise ValueError(f"{name}: not read; read are {sorted(cameras)}")
        camera.inputs.close()
        camera.detections_3d.close()
        camera.report.close()
        if camera.detections_2d is not None:
            camera.detections_2d.close()
        if camera.instances is not None:
            camera.instances.close()
        print(f"durchblick-nemo: stopped {name}", flush=True)

    for spec in specs:
        start(spec)

    def make(request: dict) -> dict:
        """The object from a camera's next views (its segmenter's masks), saved and added."""
        camera_name, name = str(request["camera"]), str(request["name"])
        objects.folder(objects_dir, name)
        label, sam_id = request.get("label"), request.get("sam_id")
        if (label is None) == (sam_id is None):
            raise ValueError("label or sam_id (one of them)")
        n_views, min_angle = int(request.get("views", 4)), float(request.get("min_angle", 15.0))
        timeout = float(request.get("timeout", 60.0))
        with lock:
            camera = cameras.get(camera_name)
        if camera is None:
            raise ValueError(f"{camera_name}: not read; read are {sorted(cameras)}")
        if camera.spec.segmenter is None:
            raise ValueError(f"{camera.spec}: no segmenter (its spec's :segmenter=NAME) for the views' masks")
        views = collect_views(session, camera.spec, n_views, device, instance=sam_id, label=label,
                              min_angle=min_angle, timeout=timeout, min_pixels=args.min_pixels, local=args.local,
                              log=lambda line: print(f"durchblick-nemo: {name}: {line}", flush=True))
        crops = [crop_images(v.frame.rgb.float() / 255.0, [square_box(mask=v.mask.cpu().numpy())],
                             mode="bicubic")[0].clamp(0, 1) for v in views]
        features, points = nemo.encode(torch.stack(crops))
        out = objects.save(objects_dir, name, features, points, torch.stack(crops),
                           {"source": "camera", "camera": str(camera.spec), "label": label,
                            "instance": views[0].instance, "min_angle": min_angle,
                            "views": [{"stamp_ns": v.stamp, "cam_in_world": None if v.cam_in_world is None
                                       else v.cam_in_world.tolist()} for v in views]})
        add(name)
        return {"made": name, "path": str(out), "views": len(views)}

    def answer(query, handler):
        try:
            payload = query.payload.to_string() if query.payload is not None else ""
            result = handler(json.loads(payload) if payload.strip() else None)
            query.reply(query.key_expr, json.dumps(result))
        except Exception as error:  # noqa: BLE001 - the asker gets the error, the node goes on
            query.reply_err(f"{type(error).__name__}: {error}")

    def answer_objects(query):
        """make takes seconds (the views): in a thread of its own, the frames go on; the rest at once."""
        payload = query.payload.to_string() if query.payload is not None else ""
        if payload.strip() and "make" in json.loads(payload):
            threading.Thread(target=answer, args=(query, on_objects), name="make", daemon=True).start()
        else:
            answer(query, on_objects)

    def on_cameras(request):
        if request is not None:
            if set(request) - {"start", "stop"} or len(request) != 1:
                raise ValueError('{"start": SPEC} or {"stop": CAMERA} expected')
            if "start" in request:
                spec = CameraSpec.parse(str(request["start"]))
                topics.check_name(spec.camera)
                start(spec)
            else:
                stop(str(request["stop"]))
        with lock:
            return {"cameras": [str(c.spec) for c in cameras.values()]}

    def on_objects(request):
        if request is not None:
            if set(request) - {"add", "remove", "make"} or len(request) != 1:
                raise ValueError('{"add": NAME}, {"remove": NAME} or {"make": {...}} expected')
            if "add" in request:
                add(str(request["add"]))
            elif "remove" in request:
                remove(str(request["remove"]))
            else:
                made = make(request["make"])
                follow()
                return made
            follow()
        with lock:
            return {"objects": {n: instance_ids[n] for n in loaded}, "available": objects.names(objects_dir)}

    def follow():
        """The parameter object follows the objects loaded (after the objects key's add, remove, make)."""
        with lock:
            names = sorted(loaded)
        applied[0] = names
        params.set("object", names)

    def apply_parameters():
        """The live parameters' values now, in this thread: the thresholds at once; a new object list loads the new
        objects and drops the others."""
        for name in LIVE:
            setattr(args, name, params[name])
        wanted = list(dict.fromkeys(params["object"]))
        if sorted(wanted) == applied[0]:
            return
        with lock:
            now = set(loaded)
        try:
            for name in wanted:
                if name not in now:
                    add(name)
        except (FileNotFoundError, ValueError) as error:  # gone since the check: what is loaded stays the list
            print(f"durchblick-nemo: parameter object: {error}", flush=True)
        for name in now - set(wanted):
            remove(name)
        follow()
        print(f"durchblick-nemo: objects {applied[0]} (parameter object)", flush=True)

    applied = [sorted(loaded)]
    # answered in the loop, between frames (not in Zenoh's threads: make waits for views)
    cameras_queryable = session.declare_queryable(topics.check_name(f"{args.node}/{CAMERAS}"))
    objects_queryable = session.declare_queryable(topics.check_name(f"{args.node}/{OBJECTS}"))
    print(f"durchblick-nemo: objects {sorted(loaded)} of {objects_dir}; {args.node}/{CAMERAS}, {args.node}/{OBJECTS}",
          flush=True)

    def process(camera: Camera, frame, stamp: int):
        with lock:
            objs = {n: loaded[n] for n in camera.tracks if n in loaded}
        now = time.monotonic()
        due = [n for n, t in camera.tracks.items()
               if n in objs and t.box is None and now - t.last_detect >= args.detect_every]
        if due:
            probs = nemo.detect(frame.rgb, torch.stack([objs[n].features for n in due]))
            for i, n in enumerate(due):
                camera.tracks[n].last_detect = now
                region = (probs[i] > args.mask_threshold).to(torch.uint8).cpu().numpy()
                count, _, stats, _ = cv2.connectedComponentsWithStats(region)
                if count > 1:
                    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                    x, y, w, h, area = stats[k]
                    if area >= args.min_pixels:
                        camera.tracks[n].box = square_box(box=(x, y, x + w, y + h))
        tracked = [n for n, t in camera.tracks.items() if n in objs and t.box is not None]
        header = db.Header(db.Time.from_ns(stamp), frame.frame_id)
        found3d, found2d = [], []
        hh, ww = frame.depth.shape
        labels = torch.zeros((hh, ww), dtype=torch.int32, device=device) if args.segments else None
        if tracked:
            boxes = [camera.tracks[n].box for n in tracked]
            crops = crop_images(frame.rgb.float() / 255.0, boxes)[:, None]  # (B, 1, 3, 224, 224)
            d = nemo.decode(crops, torch.stack([objs[n].features for n in tracked]))
            for i, n in enumerate(tracked):
                track = camera.tracks[n]
                pose = NeMO.pose(d.points[i, 0], d.confidence[i, 0], d.mask[i, 0], frame.depth, frame.K, track.box,
                                 args.conf_threshold, args.mask_threshold, args.inlier_distance, generator)
                if (pose is None or pose.inliers < args.min_inliers
                        or pose.inliers < args.min_inlier_ratio * pose.mask_pixels):
                    track.box = None
                    continue
                u, v = crop_pixels(track.box)
                selected = (d.mask[i, 0] > args.mask_threshold).cpu().numpy()
                us, vs = np.round(u[selected]).astype(int), np.round(v[selected]).astype(int)
                inside = (us >= 0) & (vs >= 0) & (us < ww) & (vs < hh)
                us, vs = us[inside], vs[inside]
                if len(us) < args.min_pixels:
                    track.box = None
                    continue
                x0, y0, x1, y1 = us.min(), vs.min(), us.max() + 1, vs.max() + 1
                track.box = square_box(box=(x0, y0, x1, y1))
                found3d.append(detection_3d(header, objs[n], pose, instance_ids[n]))
                if labels is not None:
                    labels[torch.as_tensor(vs, device=device), torch.as_tensor(us, device=device)] = instance_ids[n]
                    found2d.append(db.Detection2D(
                        header, [db.ObjectHypothesisWithPose(db.ObjectHypothesis(n, pose.inliers / pose.mask_pixels))],
                        db.BoundingBox2D(db.Pose2D(db.Point2D((x0 + x1) / 2, (y0 + y1) / 2), 0.0), x1 - x0, y1 - y0),
                        str(instance_ids[n])))
        camera.detections_3d.publish(db.Detection3DArray(header, found3d))
        if args.segments:
            publish_segments(camera, frame, stamp, header, labels, found2d)
        camera.report.counts["frames"] += 1
        camera.report.counts["found"] += len(found3d)

    def publish_segments(camera: Camera, frame, stamp, header, labels, found2d):
        """The instances in the colour camera's own image (its size; with lens distortion through
        blick.rgbd.Distorter), the 2D boxes from them."""
        info = camera.inputs.readers["color"].subscriber.camera_info
        if camera.distorter_key != camera.prepare.model_key:  # the first frame, or the camera's model changed
            u = camera.prepare.undistorter
            camera.distorter = None if u.identity else Distorter(info, u.K, *labels.shape, device)
            camera.distorter_key = camera.prepare.model_key
        if camera.instances is None:
            camera.instances = db.CameraPublisher(session, f"{camera.spec.camera}/{args.node}/instances",
                                                  encoding="mono16", frame_id=frame.frame_id, camera_info=info,
                                                  source="nemo", cuda_device=device.index, labels=True,
                                                  codec="png")
        camera.instances.camera_info, camera.instances.frame_id = info, frame.frame_id  # a camera back changed
        if camera.distorter is not None:
            image = camera.distorter.apply(labels)
        elif (info.height, info.width) != tuple(labels.shape):
            image = torch.nn.functional.interpolate(labels[None, None].float(), size=(info.height, info.width),
                                                    mode="nearest")[0, 0].int()
        else:
            image = labels
        boxed = []
        for det in found2d:
            vs, us = torch.nonzero(image == int(det.id), as_tuple=True)
            if len(us) == 0:
                continue  # outside the colour image
            x0, x1, y0, y1 = int(us.min()), int(us.max()) + 1, int(vs.min()), int(vs.max()) + 1
            det.bbox = db.BoundingBox2D(db.Pose2D(db.Point2D((x0 + x1) / 2, (y0 + y1) / 2), 0.0), x1 - x0, y1 - y0)
            boxed.append(det)
        camera.instances.publish(image.to(torch.uint16).contiguous(), stamp_ns=stamp,
                                 stream=torch.cuda.current_stream(device))
        camera.detections_2d.publish(db.Detection2DArray(header, boxed))

    def report(camera: Camera):
        done, seconds = camera.report.take()
        frames, per_frame = done["frames"] / seconds, done["found"] / max(done["frames"], 1)
        level, message = node.state(f"{frames:.1f} frames/s, {per_frame:.1f} objects a frame",
                                    waiting=None if done["frames"] else camera.inputs.waiting())
        camera.report.publish(level, message, {"frames/s": frames, "objects/frame": per_frame,
                                               "objects loaded": len(loaded), "streams": str(camera.spec)})

    try:
        while True:
            wake.wait(timeout=0.1)
            wake.clear()
            while (q := cameras_queryable.try_recv()) is not None:
                answer(q, on_cameras)
            while (q := objects_queryable.try_recv()) is not None:
                answer_objects(q)
            apply_parameters()
            with lock:
                active = list(cameras.values())
            for camera in active:
                camera.inputs.check()
                stamp = camera.inputs.newest_complete()
                if stamp is None or stamp == camera.last_stamp:
                    continue
                camera.last_stamp = stamp
                frame = camera.prepare(camera.inputs, stamp)
                if frame is not None:
                    process(camera, frame, stamp)
            for camera in active:
                if camera.report.due():
                    report(camera)
    except KeyboardInterrupt:
        pass
    finally:
        cameras_queryable.undeclare()
        objects_queryable.undeclare()
        for name in list(cameras):
            stop(name)
        params.close()
        session.close()
    return 0


def query(node: str, part: str, payload, timeout: float, args) -> dict:
    """Asks the node's queryable (with ``payload``) and returns its answer; SystemExit on its error or without an answer
    within ``timeout``."""
    key = topics.check_name(f"{node}/{part}")
    session = open_from_args(args)
    end = time.monotonic() + timeout
    try:
        while time.monotonic() < end:  # right after the session opens a query can reach no queryable yet: ask again
            for reply in session.get(key, payload=payload, timeout=max(1.0, end - time.monotonic())):
                if reply.ok is not None:
                    return json.loads(reply.ok.payload.to_string())
                raise SystemExit(f"{key}: {reply.err.payload.to_string()}")
            time.sleep(0.2)
    finally:
        session.close()
    raise SystemExit(f"{key}: no answer within {timeout} s (is durchblick-nemo running?)")


def objects_client(argv=None) -> int:
    """durchblick-nemo-objects [add NAME | remove NAME | make CAMERA NAME (--label L | --sam-id ID) | cameras [start
    SPEC | stop CAMERA]] [--node nemo]: the node's objects (loaded, available) or cameras, or changes them."""
    parser = argparse.ArgumentParser(prog="durchblick-nemo-objects", description=objects_client.__doc__)
    parser.add_argument("action", nargs="?", choices=["add", "remove", "make", "cameras"], default=None)
    parser.add_argument("targets", nargs="*")
    parser.add_argument("--label")
    parser.add_argument("--sam-id")
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--min-angle", type=float, default=15.0,
                        help="[deg] between the views seen from the object (move the camera meanwhile)")
    parser.add_argument("--node", default=NODE)
    parser.add_argument("--timeout", type=float, default=90.0)
    add_arguments(parser)
    args = parser.parse_args(argv)
    part, payload = OBJECTS, None
    if args.action in ("add", "remove"):
        if len(args.targets) != 1:
            parser.error(f"{args.action} NAME")
        payload = json.dumps({args.action: args.targets[0]})
    elif args.action == "make":
        if len(args.targets) != 2 or (args.label is None) == (args.sam_id is None):
            parser.error("make CAMERA NAME with --label or --sam-id")
        request = {"camera": args.targets[0], "name": args.targets[1], "views": args.views, "min_angle": args.min_angle,
                   "timeout": args.timeout - 5}
        request |= {"label": args.label} if args.label is not None else {"sam_id": args.sam_id}
        payload = json.dumps({"make": request})
    elif args.action == "cameras":
        part = CAMERAS
        if args.targets:
            if len(args.targets) != 2 or args.targets[0] not in ("start", "stop"):
                parser.error("cameras [start SPEC | stop CAMERA]")
            payload = json.dumps({args.targets[0]: args.targets[1]})
    print(json.dumps(query(args.node, part, payload, args.timeout, args), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
