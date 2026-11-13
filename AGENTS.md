# AGENTS.md

xblick-nemo: NeMO (DLR, few-shot "neural memory objects") as a Durchblick node. Objects known from a few images
(no mesh, no text prompt) are found, segmented and posed in every frame of any RGB-D camera. The node fits NeMO's
object coordinates to the measured depth (Sim3). Version 0.2.0, tested in the simulation only. Human documentation:
`README.md` (entry) and `docs/` (usage, design, accuracy, performance). Transport-wide traps (Zenoh sessions
and routers, CUDA rings, looped playback, hanging tests) are in the core repository's AGENTS.md
(`github.com/xblick/xblick`). This file does not repeat the README.

## Commands

```bash
tools/setup.sh                                    # submodule, checkpoint joined + SHA-256, DINOv2 clone, uv sync; idempotent
.venv/bin/python -m pytest tests                  # 5 tests; test_nemo.py needs a CUDA device and weights/
.venv/bin/python -m pytest tests/test_helpers.py  # CPU-only part: crops, Umeyama, RANSAC, object files
.venv/bin/ruff check src tests                    # lint; 120 columns and rules in pyproject.toml, third_party excluded
tools/check_references.sh                         # no internal references (patterns in the script; also in CI)
```

- `tools/setup.sh` needs uv (`UV` names it, default `uv` on PATH) and git.
- Run GPU tests and live checks in the background with `timeout`. Stop them by PID.
- Standalone: `zenohd` on the PC, then `.venv/bin/xblick-nemo --objects-dir DIR --object NAME --camera SPEC`.
  `xblick-nemo-objects` lists and changes objects and cameras at run time. `xblick rerun` shows results.
- Without a camera: a simulation (docs/accuracy.md has the benchmark scene) or `xblick play <recording.mcap> --loop`.

## Rules

- Fail fast. Missing weights, a bad camera spec, an object that does not load: raise with a message that says what
  to do (as `nemo.install_dinov2` and `objects.load` do). No silent defaults.
- Stay offline at run time. `tools/setup.sh` fetches everything at fixed commits/hashes.
- Core pin: `xblick>=0.5.0,<0.6` (the core's Python package, `durchblick` up to 0.3 and `blick` in 0.4; uses
  `xblick.rgbd`: `CameraSpec`, `RGBDInputs`, `FramePreparer`, `collect_views`, `Distorter`; `xblick.paths`,
  `xblick.node`). For a core change needed here: release the core, raise the pin and the rev, `uv lock`.
- Keep options, topics and numbers in README and `docs/` in sync with the code in the same commit. Follow the core's
  `docs/project/writing-docs.md`.
- 120 columns. Match the surrounding style: docstrings say what, comments say why.
- Send every new path argument through `xblick.paths.expand_path` (`${VAR}`, `~`), as `--objects-dir` does.
  Launchers that pass arguments unexpanded need it.

## Traps

| Symptom | Cause | Fix |
| --- | --- | --- |
| NeMO must not be installed as a package | Its package pins plotting libraries the node does not use | NeMO is imported from the submodule; `pyproject.toml` lists only what its model imports (`plotly` too: `nemolib/__init__` imports its visualisation) |
| `weights/checkpoint.pth` fails its SHA-256 after a submodule update | The checkpoint's shards live in the submodule | Check and update `CHECKPOINT_SHA256` in `tools/setup.sh` with every new submodule commit |
| Node contacts GitHub at start, loads pretrained weights | NeMO's backbone calls `torch.hub.load("facebookresearch/dinov2", ...)` | `install_dinov2` redirects it to `weights/dinov2` at `DINOV2_COMMIT` with `pretrained=False` (the checkpoint holds the fine-tuned backbone) and refuses any other hub model. Check offline behaviour by starting the node with the network refused |
| Node crashes on the first compressed subscriber of `instances` | Default JPEG codec on a 16-bit image | The `instances` publisher uses `codec="png"`. It is created with the first frame (colour `camera_info`) and follows camera changes |
| Labels do not fit a distorted camera's image | Masks are found in the undistorted frame | `--segments` maps them back with `xblick.rgbd.Distorter`. Do not publish undistorted labels into a distorted camera's image |
| Depth input of a camera | The node needs depth in the colour camera's geometry | `<camera>/<depth>` is 16UC1 mm in the colour geometry, paired by stamp (`--history` frames). Azure Kinect: `depth_to_color`. rc_visard: `ffs/depth` on `color=left` |
| `make` times out | No segmenter on the camera spec, or no camera motion | `:segmenter=NAME` (an xblick-sam3 instance); views at least `--min-angle` apart seen from the object; camera pose from `<camera>/tf`. `make` has its own thread; the frame loop goes on |
| Two sessions do not see each other | A session given only `--connect` does not listen, and the router does not route between such sessions | Start nodes without endpoints (they find the PC's router) |
| First query after opening a session gets no reply | Queryables need a moment | `query()` repeats until its timeout |

## Layout

| Path | Content |
| --- | --- |
| `src/xblick_nemo/nemo.py` | NeMO: `install_dinov2`, crops, `encode`/`decode`/`detect`, Umeyama + RANSAC `pose` |
| `src/xblick_nemo/node.py` | the node (cameras, tracks, publishing, queryables) and `objects_client` |
| `src/xblick_nemo/objects.py` | neural objects on disk: `<objects-dir>/NeMO/<name>/` |
| `src/xblick_nemo/make.py` | `xblick-nemo-make`: an object from images and masks |
| `third_party/nemo` | NeMO's code and checkpoint shards (submodule, fixed commit) |
| `weights/` | `checkpoint.pth`, `dinov2/` (made by `tools/setup.sh`, not in git) |
| `tests/` | `test_helpers.py` (CPU), `test_nemo.py` (GPU) |

## Defaults

| Item | Value |
| --- | --- |
| Console scripts | `xblick-nemo` (node), `xblick-nemo-objects` (client: list, add, remove, make, cameras), `xblick-nemo-make` (object from image files and masks or boxes) |
| Camera spec | `CAMERA[:color=NAME][:depth=NAME][:segmenter=NAME]`, defaults `color`, `depth` |
| Outputs | `<camera>/<node>/detections_3d`; with `--segments` `<camera>/<node>/instances` (mono16 PNG) and `detections_2d` |
| Queryables | `<node>/cameras` (`start`, `stop`), `<node>/objects` (`add`, `remove`, `make`) |
| `--node` / `--autocast` / `--device` | `nemo` / `bfloat16` / `cuda` |
| `--mask-threshold` / `--conf-threshold` | 0.5 / 1.2 |
| `--inlier-distance` / `--min-inliers` / `--min-inlier-ratio` | 0.01 m / 50 / 0.3 |
| `--min-pixels` / `--detect-every` / `--history` | 200 / 0.5 s / 10 |
| `--max-offset-ms` | half the depth's frame period; 0: equal stamps only |
| `--max-hz` / `--local` | 20 / `auto` (`shm`, `cuda`) |
| Client | `--views` 4, `--min-angle` 15, `--timeout` 90 |
| Other nodes | xblick-sam3 on the same camera for `make`; rc_visard needs xblick-fast-foundationstereo's depth; xblick-foundationpose can consume `--segments` |
