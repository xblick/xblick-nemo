# xblick-nemo

[![xblick core](https://img.shields.io/pypi/v/xblick?label=xblick%20core&color=blue)](https://github.com/xblick/xblick)

This [Durchblick](https://github.com/xblick/xblick) node finds, segments and poses objects that
it knows from a few images. It needs no mesh and no text prompt, and it works on any camera with depth, at about
10 ms per object and frame.

The node wraps [NeMO](https://github.com/DLR-RM/nemo) from DLR (3DV 2026, "Finding NeMO: A Geometry-Aware
Representation of Template Views for Few-Shot Perception"). NeMO's own pose comes from one image and has no scale.
The node fits NeMO's object coordinates to the measured depth, so its poses are metric.

## What it publishes

| Topic | Message type | Content |
| --- | --- | --- |
| `<camera>/nemo/detections_3d` | `vision_msgs/Detection3DArray` | One detection per object, stamped as the frame: the object's name, its pose in the colour camera's frame, and its box. |
| `<camera>/nemo/instances` | image, `mono16` | With `--segments`: the instance id of each pixel, in the colour camera's own image. |
| `<camera>/nemo/detections_2d` | `vision_msgs/Detection2DArray` | With `--segments`: the 2D detections, as xblick-sam3 publishes them. |
| `<camera>/diagnostics` | `diagnostic_msgs/DiagnosticArray` | The node's report with the frame rate in effect and the transport of each stream. |
| `nemo/cameras` | queryable (JSON) | Starts and stops cameras at run time. |
| `nemo/objects` | queryable (JSON) | Lists, adds, removes and makes objects at run time. |

With `--segments`, xblick-foundationpose can use this node as its segmenter (`--camera CAMERA:segmenter=nemo`).

## Requirements

- An NVIDIA GPU with CUDA.
- `uv`, `git` and Python 3.12.
- Durchblick's Python package `xblick` 0.5 or newer, below 0.6. The setup installs it from GitHub.
- A camera with a colour stream and a depth stream in the colour camera's geometry.
- NeMO's code and checkpoint are under the MIT licence, except five files of its decoder heads, which are Naver's
  under CC BY-NC-SA 4.0 (non-commercial use only). The node uses them. DINOv2's code is under Apache-2.0.

## Install

```bash
tools/setup.sh
```

The script fetches NeMO's code and checkpoint, DINOv2's code and the Python environment in `.venv`. Then it runs
`xblick-nemo --download-only`, which joins the checkpoint's shards from the submodule into `weights/checkpoint.pth`
and checks its SHA-256. You can run the script again, and it then does only what changed. After the setup, the node
needs no network.

The node itself also joins a missing checkpoint at its start, before it loads the model. It prints NeMO's terms, the
file and its size first. The checkpoint comes from the submodule, so this step needs no network. The submodule and
DINOv2's code come only from `tools/setup.sh`. A preparation script makes the checkpoint without a start:

```bash
.venv/bin/xblick-nemo --download-only
```

`--download-only` needs no camera, no Zenoh router and no GPU, and it exits with 0 when the checkpoint is complete.
A machine that must never start such a step on its own, such as a robot, starts the node with `--no-download`. The
node then ends at once when the checkpoint is missing, and it names the file.

## Run

First make an object from a few images of it and their masks. Then start the node on a camera:

```bash
.venv/bin/xblick-nemo-make torch --objects-dir ~/.local/share/xblick/objects \
    --images a.png b.png c.png d.png --masks a_mask.png b_mask.png c_mask.png d_mask.png
.venv/bin/xblick-nemo --objects-dir ~/.local/share/xblick/objects --object torch \
    --camera cam/rc_visard:color=left:depth=ffs/depth
```

The camera name `cam/rc_visard` is an example. A camera spec has the form
`CAMERA[:color=NAME][:depth=NAME][:segmenter=NAME]`, with the defaults `color` and `depth`.

The options that most users change:

| Option | Default | Effect |
| --- | --- | --- |
| `--object NAME` | none | Finds this object from the start. Repeat it for each object. Live: it changes while the node runs. |
| `--camera SPEC` | none | Reads this camera from the start. Repeat it for more cameras. |
| `--segments` | off | Also publishes `instances` and `detections_2d`. |
| `--max-hz` | `20` | Limits the colour frames per second and camera. `0` or `none` takes every frame. |
| `--local` | `auto` | Selects the transport for a camera on this host: `auto`, `shm` or `cuda`. |
| `--no-download` | off | Ends the node at once when the checkpoint is missing. The node makes nothing. |
| `--download-only` | off | Makes the missing checkpoint, checks it and ends. `--objects-dir` may be left out. |

You change objects and cameras at run time with one client:

```bash
.venv/bin/xblick-nemo-objects                    # lists the loaded and the available objects
.venv/bin/xblick-nemo-objects make cam/azurekinect mug --label mug --views 4 --min-angle 15
```

The `make` command builds an object from the next views of a camera. That camera needs a segmenter for the masks,
for example `--camera cam/azurekinect:depth=depth_to_color:segmenter=sam3`.

Every option is also a parameter of the node `nemo`, which Durchblick's terminal interface (`xblick`, tab Parameters)
and `xblick param` show. The objects to find are the parameter `object`, a list of object names. A new list
takes effect while the node runs: the node loads the new objects and drops the others, and it refuses a name that is
not in the objects directory, with the reason. The read-only parameter `library` lists the objects in the objects
directory, made ones too, as `xblick-nemo-objects` does. Objects added, removed or made with `xblick-nemo-objects` show in
the parameter. The thresholds, `min_pixels` and `detect_every` also change while the node runs. The other options
apply at the next start, where `--param NAME:=VALUE` or `--params-file FILE` set them.

[docs/usage.md](docs/usage.md) lists every option and command.

## Why it is fast

The node needs 10 ms per object and frame. NeMO's own code needs 43 to 93 ms for the same query at the same accuracy.

| Change | Effect |
| --- | --- |
| The node fits the pose with RANSAC on the GPU and tests all hypotheses at once. | The pose takes 4 ms. NeMO's own pose step uses RANSAC on the CPU. |
| The decoder computes in bfloat16. | A search of the whole frame takes 61 ms per object, down from 160 ms. |
| The node decodes all tracked objects of a frame in one batch. | Seven objects keep the camera's 10 frames a second. |
| The node searches the whole frame only for lost objects, at most every 0.5 s. | A tracked object costs one small crop per frame. |

[docs/performance.md](docs/performance.md) explains the terms and has the measurements.

## Limits

- We tested the node in the simulation only (status of 2026-11-01, version 0.1.0). We have not tested it on a robot.
- The node needs depth. Without depth, NeMO's pose has no scale.
- The decoder does not use TensorRT or CUDA graphs.
- NeMO's masks overlap the true silhouettes with an IoU of 0.59 to 0.92.

[docs/accuracy.md](docs/accuracy.md) has the benchmark.
- We joined the real shards with the node, and the result matched the checkpoint's SHA-256. We have not run
  `tools/setup.sh` from scratch with this version.

## Documentation

| Page | Content |
| --- | --- |
| [docs/usage.md](docs/usage.md) | all options, objects, cameras, the `make` command, tests |
| [docs/design.md](docs/design.md) | neural memory objects, the steps per frame, the metric pose, how the setup pins NeMO and DINOv2 |
| [docs/accuracy.md](docs/accuracy.md) | the benchmark against a mesh pipeline, live results |
| [docs/performance.md](docs/performance.md) | timings and what makes the node fast |

Durchblick's pages explain the terms that these pages use:
[streams and topics](https://github.com/xblick/xblick/blob/main/docs/concepts/streams-and-topics.md),
[transports](https://github.com/xblick/xblick/blob/main/docs/concepts/transports.md),
[frames on the GPU](https://github.com/xblick/xblick/blob/main/docs/concepts/gpu-frames.md) and
[pairing streams](https://github.com/xblick/xblick/blob/main/docs/guides/pairing-streams.md).

## Licence and credits

NeMO is the work of DLR. Its code and checkpoint are under the MIT licence. Five files of its decoder heads
(`third_party/nemo/src/nemolib/modules/heads/`) carry Naver's header with CC BY-NC-SA 4.0, for non-commercial use
only, and the node uses them: check whether you may use the node for your purpose. DINOv2 provides the code of NeMO's
backbone and is under Apache-2.0. `tools/setup.sh` fetches both, and this repository copies nothing of them.

The code of this repository is licensed under the Apache License 2.0 ([LICENSE](LICENSE)). [NOTICE](NOTICE) names
the copyright holder and the licences of NeMO and DINOv2.
