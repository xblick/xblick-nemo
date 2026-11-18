# xblick-nemo

[![xblick core](https://img.shields.io/pypi/v/xblick?label=xblick%20core&color=blue)](https://github.com/xblick/xblick)
[![tests](https://github.com/xblick/xblick-nemo/actions/workflows/ci.yml/badge.svg)](https://github.com/xblick/xblick-nemo/actions/workflows/ci.yml)
[![licence](https://img.shields.io/badge/licence-Apache--2.0-blue)](LICENSE)
[![device](https://img.shields.io/badge/needs-NVIDIA%20GPU-76b900)](README.md)
[![model](https://img.shields.io/badge/model-non--commercial%20use%20only-orange)](NOTICE)

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
`xblick-nemo --download-only`, which joins the checkpoint's shards from the submodule into `~/.cache/xblick/nemo/checkpoint.pth`
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

### Where the weights are

The node keeps its weights in its cache directory, `~/.cache/xblick/nemo/`. `XBLICK_HOME=/dir` moves it to `/dir/cache/nemo/`,
and `XDG_CACHE_HOME` moves it as well. `--weights-dir DIR` names another directory. The place is the same for a
checkout and for a pip install. [Where files are](https://github.com/xblick/xblick/blob/main/docs/reference/files.md) lists the places, and it shows how to copy the cache to a
PC without internet.

Up to version 0.2.2, the weights were in the checkout's `weights/`. The node does not read them there. At its start it
prints the command that moves them. In the checkout it is:

```bash
mkdir -p ~/.cache/xblick && mv weights ~/.cache/xblick/nemo
```

Run `tools/setup.sh` first. It moves DINOv2's code out of `weights/` into `third_party/dinov2`, so that the `mv`
carries only the checkpoint.

## Run

First make an object from a few images of it and their masks. Then start the node on a camera:

```bash
.venv/bin/xblick-nemo-make torch \
    --images a.png b.png c.png d.png --masks a_mask.png b_mask.png c_mask.png d_mask.png
.venv/bin/xblick-nemo --object torch --camera cam/rc_visard:color=left:depth=ffs/depth
```

The objects go to the node's data directory, `~/.local/share/xblick/nemo/`, unless `--objects-dir DIR` names another
one. Objects that you made with an earlier version stay where you made them, for example in
`~/.local/share/xblick/objects` as the earlier example had it. Use them with `--objects-dir`, or move them:

```bash
mkdir -p ~/.local/share/xblick/nemo && mv ~/.local/share/xblick/objects/NeMO ~/.local/share/xblick/nemo/
```

The camera name `cam/rc_visard` is an example. A camera spec has the form
`CAMERA[:color=NAME][:depth=NAME][:segmenter=NAME]`, with the defaults `color` and `depth`.

The options that most users change:

| Option | Default | Effect |
| --- | --- | --- |
| `--objects-dir DIR` | the data directory | Where the objects are, in place of `~/.local/share/xblick/nemo/`. |
| `--object NAME` | none | Finds this object from the start. Repeat it for each object. Live: it changes while the node runs. |
| `--camera SPEC` | none | Reads this camera from the start. Repeat it for more cameras. Without it the node starts with no camera, and you start cameras at run time on `nemo/cameras`. |
| `--segments` | off | Also publishes `instances` and `detections_2d`. |
| `--max-hz` | `20` | Limits the colour frames per second and camera. `0` or `none` takes every frame. |
| `--local` | `auto` | Selects the transport for a camera on this host: `auto`, `shm` or `cuda`. |
| `--no-download` | off | Ends the node at once when the checkpoint is missing. The node makes nothing. |
| `--download-only` | off | Makes the missing checkpoint, checks it and ends. `--objects-dir` may be left out. |
| `--weights-dir DIR` | the cache directory | where the weights are and go, in place of `~/.cache/xblick/nemo/` |

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

- The node needs depth. Without depth, NeMO's pose has no scale.
- The decoder does not use TensorRT or CUDA graphs.
- NeMO's masks overlap the true silhouettes with an IoU of 0.59 to 0.92.

[docs/accuracy.md](docs/accuracy.md) has the benchmark.

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

## Citation

This node uses NeMO (DLR-RM/nemo). If you use it in research, cite their work. The entries below are copied from [its
README](https://github.com/DLR-RM/nemo/blob/76478815baf2c99002cccbee98426773fa4e59cd/README.md):

```bibtex
@inproceedings{jung2026nemo,
  title     = {Finding NeMO: A Geometry-Aware Representation of Template Views for Few-Shot Perception},
  author    = {Jung, Sebastian and Klüpfel, Leonard and Triebel, Rudolph and Durner, Maximilian},
  booktitle = {International Conference on 3D Vision (3DV)},
  year      = {2026},
}
```

NeMO's encoder is DINOv2 (facebookresearch/dinov2). Cite it as its repository asks. The entries below are copied from
[its README](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/README.md):

```bibtex
@misc{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy V. and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and Howes, Russell and Huang, Po-Yao and Xu, Hu and Sharma, Vasu and Li, Shang-Wen and Galuba, Wojciech and Rabbat, Mike and Assran, Mido and Ballas, Nicolas and Synnaeve, Gabriel and Misra, Ishan and Jegou, Herve and Mairal, Julien and Labatut, Patrick and Joulin, Armand and Bojanowski, Piotr},
  journal={arXiv:2304.07193},
  year={2023}
}

@misc{darcet2023vitneedreg,
  title={Vision Transformers Need Registers},
  author={Darcet, Timothée and Oquab, Maxime and Mairal, Julien and Bojanowski, Piotr},
  journal={arXiv:2309.16588},
  year={2023}
}

@misc{jose2024dinov2meetstextunified,
  title={DINOv2 Meets Text: A Unified Framework for Image- and Pixel-Level Vision-Language Alignment},
  author={Cijo Jose and Théo Moutakanni and Dahyun Kang and Federico Baldassarre and Timothée Darcet and Hu Xu and Daniel Li and Marc Szafraniec and Michaël Ramamonjisoa and Maxime Oquab and Oriane Siméoni and Huy V. Vo and Patrick Labatut and Piotr Bojanowski},
  journal={arXiv:2412.16334},
  year={2024}
}
```
