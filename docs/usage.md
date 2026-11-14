# Usage

This page lists every option of the node and every command of its clients. It explains how you make objects and how
you change objects and cameras while the node works. The camera names on this page are examples.

## Objects

An object is a directory `<objects-dir>/NeMO/<name>/`. You make it in one of two ways.

### From image files

```bash
.venv/bin/xblick-nemo-make torch --objects-dir ~/.local/share/xblick/objects \
    --images a.png b.png c.png d.png --masks a_mask.png b_mask.png c_mask.png d_mask.png
```

Take the images from around the object. Give one mask per image, in which non-zero pixels are the object.

| Option | Effect |
| --- | --- |
| `NAME` | the object's name |
| `--objects-dir DIR` | where the node stores objects (required) |
| `--images FILE ...` | the images (required) |
| `--masks FILE ...` | one mask per image |
| `--boxes X0,Y0,X1,Y1 ...` | one box in pixels per image, as the alternative to `--masks` |
| `--device NAME` | the GPU, default `cuda` |

### From a camera

```bash
.venv/bin/xblick-nemo-objects make cam/azurekinect mug --label mug --views 4 --min-angle 15
```

The node collects the next views of the camera and takes the object's mask in each view from a segmenter. It then
makes, saves and adds the object. Three conditions apply.

1. The node reads the camera with a segmenter, for example
   `--camera cam/azurekinect:depth=depth_to_color:segmenter=sam3`. The
   segmenter is an xblick-sam3 instance on the camera's colour stream.
2. You select the instance with `--label` or with `--sam-id`. With a label, the node takes the largest instance of
   that label.
3. The views are at least `--min-angle` degrees apart, seen from the object. Move the camera or the object. The node
   reads the camera's pose from `<camera>/tf`. Without motion the command ends with a timeout.

The node makes the object in a thread of its own, so it continues to process frames.

## Start the node

```bash
.venv/bin/xblick-nemo --objects-dir ~/.local/share/xblick/objects --object torch \
    --camera cam/rc_visard:color=left:depth=ffs/depth
```

The node needs a Zenoh router on the host. A camera is a colour stream `<CAMERA>/<color>` in `rgb8` or `bgr8` and
a depth stream `<CAMERA>/<depth>` in `16UC1` millimetres. The depth must be in the colour camera's geometry.

| Camera | Spec |
| --- | --- |
| Roboception rc_visard with the depth of xblick-fast-foundationstereo | `cam/rc_visard:color=left:depth=ffs/depth` |
| Azure Kinect | `cam/azurekinect:depth=depth_to_color` |
| a camera with a segmenter for `make` | `cam/azurekinect:depth=depth_to_color:segmenter=sam3` |

## Options of the node

| Option | Default | Effect |
| --- | --- | --- |
| `--objects-dir DIR` | required | The directory of the objects. The node expands `~` and `${VAR}` in it. |
| `--object NAME` | none | Finds this object from the start. You can repeat the option. Live: it changes while the node runs. |
| `--camera SPEC` | none | Reads this camera from the start. You can repeat the option. |
| `--node NAME` | `nemo` | Sets the node's part of the topics. |
| `--segments` | off | Also publishes `instances` and `detections_2d`. |
| `--autocast` | `bfloat16` | Selects the decoder's number format: `bfloat16` or `none` for float32. |
| `--mask-threshold X` | `0.5` | Sets the threshold of NeMO's mask. Live: it changes while the node runs. |
| `--conf-threshold X` | `1.2` | Sets the lowest confidence of a point for the pose fit. The value comes from NeMO's examples. Live: it changes while the node runs. |
| `--inlier-distance X` | `0.01` | Sets the inlier distance of the pose fit in metres. Live: it changes while the node runs. |
| `--min-inliers N` | `50` | Counts an object as lost below this number of inliers. Live: it changes while the node runs. |
| `--min-inlier-ratio X` | `0.3` | Counts an object as lost when fewer than this share of the mask's pixels are inliers. Live: it changes while the node runs. |
| `--min-pixels N` | `200` | Sets the smallest region that a search accepts, in pixels at the depth's resolution. Live: it changes while the node runs. |
| `--detect-every SECONDS` | `0.5` | Sets the shortest time between two searches for a lost object. Live: it changes while the node runs. |
| `--history N` | `10` | Keeps this many frames of each stream for pairing by stamp. |
| `--max-offset-ms X` | half the depth's frame period | Pairs a colour frame with the nearest depth frame within this offset. `0` pairs equal stamps only. |
| `--device NAME` | `cuda` | Selects the GPU. |
| `--max-hz X` | `20` | Takes at most this many colour frames per second and camera. `0` or `none` takes every frame. |
| `--local MODE` | `auto` | Selects the transport for a camera on this host: `auto`, `shm` or `cuda`. |
| `--connect`, `--listen`, `--config` | the host's Zenoh router | Select the Zenoh session, as in every Durchblick program. |

### The frame rate

The node processes about 10 frames a second. `--max-hz` therefore limits each camera to 20 colour frames a second by
default. The node always takes the newest colour frame, and it decodes the depth only for the frames that it takes.
The report states the rate in effect.

### The transport

With `--local auto`, the node reads a camera on the same host from its CUDA ring when that ring is on the node's GPU.
Otherwise it reads the host ring in shared memory. `shm` and `cuda` force one transport. With `cuda`, the node stops
when the camera offers no CUDA ring. The start log and every report state the transport of each stream. Durchblick's
page on [transports](https://github.com/xblick/xblick/blob/main/docs/concepts/transports.md) explains the choices.

## The client at run time

| Command | Effect |
| --- | --- |
| `xblick-nemo-objects` | Lists the loaded and the available objects. |
| `xblick-nemo-objects add NAME` | Loads an object from the objects directory. |
| `xblick-nemo-objects remove NAME` | Removes an object from the node. |
| `xblick-nemo-objects make CAMERA NAME --label LABEL` | Makes an object from the camera's views. `--sam-id ID` replaces the label. |
| `xblick-nemo-objects cameras` | Lists the cameras. |
| `xblick-nemo-objects cameras start SPEC` | Starts a camera. |
| `xblick-nemo-objects cameras stop CAMERA` | Stops a camera. |

The client accepts `--views` (default 4), `--min-angle` (default 15 degrees), `--node`, `--timeout` (default 90 s)
and the session options.

The queryables take the same requests as JSON.

| Queryable | Requests |
| --- | --- |
| `<node>/cameras` | `{"start": SPEC}`, `{"stop": CAMERA}` |
| `<node>/objects` | `{"add": NAME}`, `{"remove": NAME}`, `{"make": {"camera": CAMERA, "name": NAME, "label": LABEL, "views": 4, "min_angle": 15, "timeout": 60}}` |

A `make` request takes `"sam_id": ID` in place of `"label"`.

## As a segmenter for xblick-foundationpose

Start the node with `--segments`. Then give xblick-foundationpose the camera as `--camera CAMERA:segmenter=nemo`.
The node publishes the masks in the colour camera's own image, with the lens distortion of that camera.

## Without a camera

You can feed the node from a simulation, as the benchmark in [accuracy.md](accuracy.md) does. You can also replay a recording
with `xblick play <recording.mcap> --loop`. `xblick rerun` shows the results.

## Tests

```bash
.venv/bin/python -m pytest tests
```

| File | Needs | What it tests |
| --- | --- | --- |
| `test_helpers.py` | CPU | crops and their pixels, Umeyama's fit, the RANSAC with 30 % outliers, the objects' files |
| `test_nemo.py` | a CUDA device and `weights/` | An object from three images of NeMO's mug finds the mug in the fourth image and in a whole image. |

## Parameters

Every option is also a parameter of the node `nemo`. The live ones change while the node runs: `object`,
`mask_threshold`, `conf_threshold`, `inlier_distance`, `min_inliers`, `min_inlier_ratio`, `min_pixels`,
`detect_every`; the tables mark them. The others are read-only and apply at the next start, where `--param
NAME:=VALUE` or `--params-file FILE` set them. `object` is the list of objects to find; the read-only parameter
`library` lists the objects in the objects directory. Durchblick's terminal interface (`xblick`, tab Parameters) and
`xblick param` show them.
