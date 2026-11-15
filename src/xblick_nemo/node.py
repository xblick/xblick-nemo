"""xblick-nemo: NeMO (nemo.py) on cameras' colour and depth: objects known from a few images of them found,
segmented and posed (metric, with the depth) in every frame; any number of cameras, started and stopped at run time.

    xblick-nemo [--objects-dir DIR] [--object NAME ...] [--camera CAMERA[:color=NAME][:depth=NAME] ...]
                    [--node nemo] [--segments] [--autocast bfloat16] [--no-download]

A camera is ``<CAMERA>/<color>`` (rgb8 or bgr8) and ``<CAMERA>/<depth>`` (16UC1 mm in the colour camera's geometry),
paired by stamp, at the depth's resolution, undistorted (xblick.rgbd). An object (``<DIR>/NeMO/<name>/``,
objects.py; made by xblick-nemo-make from images or by this node from a camera's views) is, per camera:

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
* with --segments, as xblick-sam3 does (so xblick-foundationpose can take NeMO as its segmenter,
  ``:segmenter=<node>``): ``<CAMERA>/<node>/instances`` (mono16, the colour image's size, the instance ids) and
  ``<CAMERA>/<node>/detections_2d``, in the colour camera's own image (its size; a camera with lens distortion through
  xblick.rgbd.Distorter: the masks are found in the undistorted frame)

Queryables: ``<node>/cameras`` ({"start": SPEC} | {"stop": CAMERA}; xblick-nemo-objects cameras) and
``<node>/objects`` ({"add": NAME} | {"remove": NAME} | {"make": {"camera": CAMERA, "name": NAME, "label": LABEL |
"sam_id": ID, "views": 4, "min_angle": 15, "timeout": 60}}; xblick-nemo-objects). They answer at once, while the
model works: a start, stop, add or remove is checked and answered with the cameras or objects as they are from the
next frame on, and the frame loop applies it then. make collects the camera's next views of the object at least
``min_angle`` degrees apart seen from it (Durchblick's collect_views: the camera's pose from <camera>/tf_static and
<camera>/tf; move the camera or the object), the object's mask in each from the camera's segmenter (its spec's
``:segmenter=NAME``, e.g. an xblick-sam3 instance: by its label, the largest such instance, or by its id), makes and
saves the object, adds it from the next frame on, and answers then: within the request's ``timeout`` and the time to
encode and save, in a thread of its own (the frames go on). The cameras can be given at the start (--camera) or later
on ``<node>/cameras`` alone.

Weights (weights.py; in the node's cache directory ~/.cache/xblick/nemo, or --weights-dir): a missing checkpoint is
joined from its shards in the submodule at the start, before the model loads; --no-download ends the node instead.
``xblick-nemo --download-only`` only makes and checks it (no camera, no Zenoh, no GPU), for preparation scripts. The
submodule and DINOv2's code come from tools/setup.sh.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xblick as db
from xblick import mps, node, topics
from xblick import weights as xblick_weights
from xblick.camera import check_local
from xblick.history import PAIR_MAX_OFFSET_NS
from xblick.paths import data_dir, expand_path
from xblick.rgbd import CameraSpec, Distorter, FramePreparer, RGBDInputs, collect_views
from xblick.session import add_arguments, open_from_args, replies

from . import weights

NODE = "nemo"
CAMERAS = "cameras"
OBJECTS = "objects"


def parse_device(text: str) -> str:
    """``--device``: cuda (GPU 0) or cuda:N, as ``cuda:N``."""
    if text == "cuda":
        return "cuda:0"
    if text.startswith("cuda:") and text[5:].isdigit():
        return text
    raise argparse.ArgumentTypeError(f"{text!r}: cuda (GPU 0) or cuda:N")


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
    report: object = None  # xblick.node.Reporter
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


def objects_dir_of(args) -> Path:
    """--objects-dir (``${VAR}`` and ``~`` expanded), else the node's data directory: the objects a user made are data,
    never in the cache or the checkout."""
    return expand_path(args.objects_dir) if args.objects_dir else data_dir(weights.NODE)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="xblick-nemo", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--objects-dir", default=None,
                        help="where the objects are (<DIR>/NeMO/<name>/; default: the node's data directory, "
                             "~/.local/share/xblick/nemo or $XBLICK_HOME/data/nemo)")
    parser.add_argument("--object", action="append", default=[],
                        help="an object to find (repeated; at run time: the parameter object, or <node>/objects)")
    parser.add_argument("--camera", action="append", default=[], metavar="SPEC",
                        help="CAMERA[:color=NAME][:depth=NAME][:segmenter=NAME] (repeated; also at run time)")
    parser.add_argument("--node", default=NODE)
    parser.add_argument("--segments", action="store_true",
                        help="also publish instances and detections_2d (a segmenter for xblick-foundationpose)")
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
    parser.add_argument("--max-offset-ms", type=float, default=PAIR_MAX_OFFSET_NS / 1e6,
                        help="pair a colour frame with the nearest depth frame at most this far apart, else skip it "
                             "(counted); 20 covers the Azure Kinect's colour and depth of one capture (up to 17 ms "
                             "apart) and the rc_visard's (equal stamps), and pairs no neighbouring frame of a 30 or "
                             "15 Hz depth stream (default: %(default)s); 0: only equal stamps")
    parser.add_argument("--device", type=parse_device, default="cuda:0",
                        help="the GPU: cuda (GPU 0) or cuda:N; a GPU that cannot be used is an error (default: cuda:0)")
    xblick_weights.add_arguments(parser)  # --download-only (a mode of the command line), --no-download
    add_arguments(parser)
    # --max-hz: 20 colour frames/s (about 10 measured; 0: every frame)
    # --local: auto (the CUDA ring on this GPU, else shared memory), shm or cuda
    node.add_arguments(parser, max_hz=20.0, local=("auto", "shm", "cuda"))
    node.Parameters.add_arguments(parser)  # --param NAME:=VALUE, --params-file, --dump-params, --params-read-only
    args = xblick_weights.parse_args(parser, argv)  # --download-only: --objects-dir not required
    if args.download_only:
        return weights.store_for(parser, args).download_only(
            weights.needed(), options=weights.options(args))
    if args.max_offset_ms < 0:
        parser.error("--max-offset-ms: not negative")
    mps.check()  # with CUDA_MPS_PIPE_DIRECTORY (MPS meant) its daemon has to run: else stop here
    # the options as parameters (the terminal interface shows them): read-only, but LIVE and the objects
    session = open_from_args(args)
    params = node.Parameters(session, args.node)
    params.configure(args)

    def check_objects(value) -> str | None:
        """A new object list: refused with the reason if an object is not in the objects directory."""
        from . import objects  # noqa: PLC0415 - with the model's imports

        missing = sorted(set(value) - set(objects.names(objects_dir_of(args))) - set(loaded))
        return f"{missing}: not in {objects_dir_of(args)}" if missing else None

    params.declare_options(parser, args, live={**LIVE, "object": {"write": check_objects}},
                           leave_out=("node", "download_only"))

    def library() -> list[str]:
        from . import objects  # noqa: PLC0415 - with the model's imports

        return objects.names(objects_dir_of(args))

    # what the parameter object takes: the objects in the objects directory (made ones too), read at each request
    params.declare("library", [], description="the objects in --objects-dir (object takes their names)", read_only=True,
                   read=library)
    params.check_overrides()
    try:
        args.objects_dir = objects_dir_of(args)  # ${VAR} and ~ (some launchers pass arguments as they stand)
        specs = [CameraSpec.parse(c) for c in args.camera]
        for spec in specs:
            topics.check_name(spec.camera)
    except ValueError as error:
        parser.error(str(error))
    if len({s.camera for s in specs}) != len(specs):
        parser.error("--camera: a camera given twice")

    # DINOv2 without xFormers: slower, the same
    node.start("xblick-nemo", ignore=["xFormers is not available"], max_hz=args.max_hz, local=args.local)
    store = weights.store_for(parser, args)  # --weights-dir, also given as a parameter
    try:
        store.ensure(weights.needed(), download=not args.no_download,
                     options=weights.options(args))
    except xblick_weights.WeightsError as error:
        params.close()
        session.close()
        return store.fail(error)

    import cv2  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from . import objects  # noqa: PLC0415
    from .nemo import NeMO, crop_images, crop_pixels, square_box  # noqa: PLC0415

    device = torch.device(args.device)
    if device.index >= torch.cuda.device_count():  # no CUDA, or not that many GPUs: one line, never another device
        print(f"xblick-nemo: --device {args.device}: no such GPU ({torch.cuda.device_count()} visible)",
              file=sys.stderr, flush=True)
        params.close()
        session.close()
        return 1
    torch.cuda.set_device(device)
    objects_dir = args.objects_dir
    objects_dir.mkdir(parents=True, exist_ok=True)
    nemo = NeMO(device, store.root, autocast=None if args.autocast == "none" else args.autocast)
    generator = torch.Generator(device=device).manual_seed(0)
    loaded: dict[str, objects.NeuralObject] = {}
    instance_ids: dict[str, int] = {}
    lock = threading.Lock()  # loaded, cameras, accepted, camera_changes, instance_ids: the queryables and the loop
    # the objects key and the make thread change the parameter object (read, changed, set: one at a time)
    object_lock = threading.Lock()

    def add(name: str):
        obj = objects.load(objects_dir, name, device)
        with lock:
            loaded[name] = obj
            instance_ids.setdefault(name, len(instance_ids) + 1)
            for camera in cameras.values():
                camera.tracks[name] = Track()
        print(f"xblick-nemo: {name} (instance {instance_ids[name]})", flush=True)

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

    max_offset_ns = round(args.max_offset_ms * 1e6)
    accepted: dict[str, CameraSpec] = {s.camera: s for s in specs}  # the cameras from the next frame on
    camera_changes: list[tuple[str, object]] = []  # ("start", spec) or ("stop", name), accepted, for the loop

    def start(spec: CameraSpec):
        with lock:
            if spec.camera in cameras:
                raise ValueError(f"{spec.camera}: already read (as {cameras[spec.camera].spec}); stop it first")
            camera = Camera(spec, RGBDInputs(session, spec, args.history, args.local, wake, max_hz=args.max_hz,
                                             device=device, max_offset_ns=max_offset_ns),
                            FramePreparer(device),
                            db.DetectionPublisher(session, topics.detections(spec.camera, args.node, "detections_3d"),
                                                  db.Detection3DArray))
            camera.report = node.Reporter(session, spec.camera, args.node, "xblick-nemo",
                                          log_prefix=f"{spec.camera}: ", max_hz=args.max_hz)
            camera.report.watch(camera.inputs)
            if args.segments:
                camera.detections_2d = db.DetectionPublisher(
                    session, topics.detections(spec.camera, args.node, "detections_2d"), db.Detection2DArray)
            camera.tracks = {n: Track() for n in loaded}
            cameras[spec.camera] = camera
        print(f"xblick-nemo: reading {spec} ({', '.join(spec.keys.values())})", flush=True)

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
        print(f"xblick-nemo: stopped {name}", flush=True)

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
                              log=lambda line: print(f"xblick-nemo: {name}: {line}", flush=True))
        crops = [crop_images(v.frame.rgb.float() / 255.0, [square_box(mask=v.mask.cpu().numpy())],
                             mode="bicubic")[0].clamp(0, 1) for v in views]
        features, points = nemo.encode(torch.stack(crops))
        out = objects.save(objects_dir, name, features, points, torch.stack(crops),
                           {"source": "camera", "camera": str(camera.spec), "label": label,
                            "instance": views[0].instance, "min_angle": min_angle,
                            "views": [{"stamp_ns": v.stamp, "cam_in_world": None if v.cam_in_world is None
                                       else v.cam_in_world.tolist()} for v in views]})
        with object_lock:  # found from the next frame on (the loop loads it), as an add on the objects key
            set_objects(wanted_objects() + [name] if name not in wanted_objects() else wanted_objects())
        return {"made": name, "path": str(out), "views": len(views)}

    def wanted_objects() -> list[str]:
        """The objects to find from the next frame on: the parameter object (the loop loads them)."""
        return list(dict.fromkeys(params["object"]))

    def set_objects(names: list[str]):
        """The parameter object (under object_lock): what the objects key and make accepted; ids given now."""
        with lock:
            for name in names:
                instance_ids.setdefault(name, len(instance_ids) + 1)
        result = params.set("object", names)
        if not result.successful:
            raise ValueError(result.reason)

    def answer(query, handler):
        """A query answered and dropped at once: the handler checks and accepts, the loop applies."""
        try:
            payload = query.payload.to_string() if query.payload is not None else ""
            result = handler(json.loads(payload) if payload.strip() else None)
            query.reply(query.key_expr, json.dumps(result))
        except Exception as error:  # noqa: BLE001 - the asker gets the error, the node goes on
            query.reply_err(f"{type(error).__name__}: {error}")

    def make_and_answer(query, request):
        """make, in a thread of its own (it waits for views): its answer is its result."""
        try:
            query.reply(query.key_expr, json.dumps(make(request)))
        except Exception as error:  # noqa: BLE001 - the asker gets the error, the node goes on
            query.reply_err(f"{type(error).__name__}: {error}")

    def on_cameras_query(query):
        answer(query, on_cameras)

    def on_objects_query(query):
        try:
            payload = query.payload.to_string() if query.payload is not None else ""
            request = json.loads(payload) if payload.strip() else None
        except json.JSONDecodeError as error:
            query.reply_err(f"{type(error).__name__}: {error}")
            return
        if isinstance(request, dict) and set(request) == {"make"}:
            threading.Thread(target=make_and_answer, args=(query, request["make"]), name="make", daemon=True).start()
        else:
            answer(query, on_objects)

    def on_cameras(request):
        """A camera to start or stop, checked and accepted (the loop starts or stops it); the cameras then."""
        with lock:
            if request is not None:
                if not isinstance(request, dict) or set(request) - {"start", "stop"} or len(request) != 1:
                    raise ValueError('{"start": SPEC} or {"stop": CAMERA} expected')
                if "start" in request:
                    spec = CameraSpec.parse(str(request["start"]))
                    topics.check_name(spec.camera)
                    if spec.camera in accepted:
                        raise ValueError(f"{spec.camera}: already read (as {accepted[spec.camera]}); stop it first")
                    accepted[spec.camera] = spec
                    camera_changes.append(("start", spec))
                else:
                    name = str(request["stop"])
                    if name not in accepted:
                        raise ValueError(f"{name}: not read; read are {sorted(accepted)}")
                    del accepted[name]
                    camera_changes.append(("stop", name))
            return {"cameras": [str(s) for s in accepted.values()]}

    def on_objects(request):
        """An object to add or remove, checked and accepted (the loop loads or drops it); the objects then."""
        with object_lock:
            wanted = wanted_objects()
            if request is not None:
                if not isinstance(request, dict) or set(request) - {"add", "remove"} or len(request) != 1:
                    raise ValueError('{"add": NAME}, {"remove": NAME} or {"make": {...}} expected')
                if "add" in request:
                    name = str(request["add"])
                    objects.folder(objects_dir, name)
                    if name not in objects.names(objects_dir):
                        raise FileNotFoundError(f"{name}: not in {objects_dir}")
                    wanted = wanted if name in wanted else wanted + [name]
                else:
                    name = str(request["remove"])
                    if name not in wanted:
                        raise ValueError(f"{name}: not loaded")
                    wanted = [n for n in wanted if n != name]
                set_objects(wanted)
            with lock:
                return {"objects": {n: instance_ids[n] for n in wanted}, "available": objects.names(objects_dir)}

    def apply_cameras():
        """The cameras accepted on the cameras key, started or stopped (in this thread, between frames)."""
        with lock:
            changes = list(camera_changes)
            camera_changes.clear()
        for what, value in changes:
            if what == "start":
                start(value)
            else:
                stop(value)

    def apply_parameters():
        """The live parameters' values now, in this thread: the thresholds at once; a new object list (the parameter
        object, or the objects key's add, remove, make) loads the new objects and drops the others."""
        for name in LIVE:
            setattr(args, name, params[name])
        wanted = wanted_objects()
        if sorted(wanted) == applied[0]:
            return
        with lock:
            now = set(loaded)
        failed = []
        for name in wanted:
            if name not in now:
                try:
                    add(name)
                except (FileNotFoundError, ValueError) as error:  # gone since the check: not found
                    print(f"xblick-nemo: object {name}: {error}", flush=True)
                    failed.append(name)
        for name in now - set(wanted):
            remove(name)
        applied[0] = sorted(set(wanted) - set(failed))
        if failed:
            with object_lock:
                set_objects([n for n in wanted_objects() if n not in failed])
        print(f"xblick-nemo: objects {applied[0]}", flush=True)

    applied = [sorted(loaded)]
    # answered at once in Zenoh's threads from the state kept here; the loop applies what they accepted
    cameras_queryable = session.declare_queryable(topics.check_name(f"{args.node}/{CAMERAS}"), on_cameras_query)
    objects_queryable = session.declare_queryable(topics.check_name(f"{args.node}/{OBJECTS}"), on_objects_query)
    print(f"xblick-nemo: objects {sorted(loaded)} of {objects_dir}; {args.node}/{CAMERAS}, {args.node}/{OBJECTS}; "
          f"model on {args.device}", flush=True)

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
        xblick.rgbd.Distorter), the 2D boxes from them."""
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
            apply_cameras()
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
                    for reader in camera.inputs.readers.values():  # --local cuda without a CUDA ring: an error
                        check_local(reader.subscriber, args.local)
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
            # replies(): Zenoh does not always end a query at its timeout (the core's AGENTS.md, "Traps")
            for reply in replies(session, key, max(1.0, end - time.monotonic()), payload=payload):
                if reply.ok is not None:
                    return json.loads(reply.ok.payload.to_string())
                raise SystemExit(f"{key}: {reply.err.payload.to_string()}")
            time.sleep(0.2)
    finally:
        session.close()
    raise SystemExit(f"{key}: no answer within {timeout} s (is xblick-nemo running?)")


def objects_client(argv=None) -> int:
    """xblick-nemo-objects [add NAME | remove NAME | make CAMERA NAME (--label L | --sam-id ID) | cameras [start
    SPEC | stop CAMERA]] [--node nemo]: the node's objects (loaded, available) or cameras, or changes them."""
    parser = argparse.ArgumentParser(prog="xblick-nemo-objects", description=objects_client.__doc__)
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
