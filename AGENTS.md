# AGENTS.md

xblick-nemo: NeMO (DLR, few-shot "neural memory objects") as a Durchblick node. Objects known from a few images
(no mesh, no text prompt) are found, segmented and posed in every frame of any RGB-D camera. The node fits NeMO's
object coordinates to the measured depth (Sim3). Version 0.2.5, tested in the simulation only. Human documentation:
`README.md` (entry) and `docs/` (usage, design, accuracy, performance). Transport-wide traps (Zenoh sessions
and routers, CUDA rings, looped playback, hanging tests) are in the core repository's AGENTS.md
(`github.com/xblick/xblick`). This file does not repeat the README.

## Commands

```bash
tools/setup.sh                                    # submodule, uv sync, DINOv2 clone, then --download-only; idempotent
.venv/bin/python -m pytest tests                  # test_nemo.py needs a CUDA device and ~/.cache/xblick/nemo/checkpoint.pth
.venv/bin/python -m pytest tests/test_weights.py  # CPU, 4 s: the node's files and its start up to the weights (the download: the core's tests)
.venv/bin/xblick-nemo --download-only             # checkpoint joined from the submodule's shards if missing, SHA-256
.venv/bin/python -m pytest tests/test_helpers.py  # CPU-only part: crops, Umeyama, RANSAC, object files
.venv/bin/ruff check src tests                    # lint; 120 columns and rules in pyproject.toml, third_party excluded
tools/check_references.sh                         # no internal references (patterns in the script; also in CI)
```

- `tools/setup.sh` needs uv (`UV` names it, default `uv` on PATH) and git.
- `--no-download`: a start with a missing checkpoint ends with one line (exit 1). Excludes `--download-only`.
- Nothing of the weights comes from the network: "download" here is the join of the submodule's shards
  (`weights.Shards`). The submodule and DINOv2's clone stay in `tools/setup.sh`; never run git at run time.
- Run GPU tests and live checks in the background with `timeout`. Stop them by PID.
- Standalone: `zenohd` on the PC, then `.venv/bin/xblick-nemo --object NAME --camera SPEC` (objects in the data directory).
  `xblick-nemo-objects` lists and changes objects and cameras at run time. `xblick rerun` shows results.
- Without a camera: a simulation (docs/accuracy.md has the benchmark scene) or `xblick play <recording.mcap> --loop`.

## Rules

- Files: the weights are in the node's cache directory (`xblick.paths.cache_dir("nemo")`, `~/.cache/xblick/nemo`), never in
  the checkout. `weights.STORE = Store(PROG, node=NODE, ..., former=[FORMER])`, `Weight` paths relative to its root;
  `store = weights.store_for(parser, args)` (`--weights-dir`) for `download_only`, `ensure` and `fail`, with
  `options=weights.options(args)`; model code takes its paths from `store.root`. `FORMER` (the checkout's `weights/`) is never
  read: the store prints the command that moves it. The core's `docs/reference/files.md` has the rule. The objects default to `xblick.paths.data_dir("nemo")` (`objects_dir_of`); DINOv2 is code in `third_party/dinov2`.
- Fail fast. Missing code from the setup (`weights.REQUIRED`, `install_dinov2`), a bad camera spec, an object that does not load: raise with a message that says what
  to do (as `nemo.install_dinov2` and `objects.load` do). No silent defaults.
- `--camera` is optional: a documented exception to the core's rule "`--camera` explicit and required". The node takes
  its cameras at the start or at run time on `<node>/cameras`, and never guesses one. Every camera it reads is named,
  on the command line or in a request.
- Queryables answer in Zenoh's thread at once, from the state that `lock` guards (`accepted`, the parameter
  `object`). `start`, `stop`, `add` and `remove` are checked and accepted there, and the loop applies them
  (`apply_cameras`, `apply_parameters`) between frames. Only `make` answers later, in a thread of its own, with its
  result. Never answer a query from the frame loop: a query still open at the caller's timeout gets a `Timeout` reply,
  which a sender takes for a second node.
- Stay offline at run time. `tools/setup.sh` fetches the code at fixed commits. `weights.py` is the one place of the
  checkpoint (shards, `CHECKPOINT_SHA256`, size, `TERMS`); `weights.STORE.ensure` runs at the start (node and
  `xblick-nemo-make`) before the model loads. Checkpoint there: no file written (`test_weights.py` checks it).
- The join: `checkpoint.pth.part`, renamed after the SHA-256 check, under the lock `~/.cache/xblick/nemo/.download.lock`. Never let
  a partial or unchecked file get its final name. `download_only` is in `leave_out` of `declare_options`;
  `no_download` is a read-only parameter. `xblick.weights.parse_args` lifts `required` options for
  `--download-only` (spelled out).
- Core pin: `xblick>=0.5.11,<0.6` (the core's Python package, `durchblick` up to 0.3 and `blick` in 0.4; uses
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
| `~/.cache/xblick/nemo/checkpoint.pth` fails its SHA-256 after a submodule update | The checkpoint's shards live in the submodule | Check and update `CHECKPOINT_SHA256` (and `CHECKPOINT_SIZE`) in `weights.py` with every new submodule commit |
| Node contacts GitHub at start, loads pretrained weights | NeMO's backbone calls `torch.hub.load("facebookresearch/dinov2", ...)` | `install_dinov2` redirects it to `third_party/dinov2` at `DINOV2_COMMIT` with `pretrained=False` (the checkpoint holds the fine-tuned backbone) and refuses any other hub model. Check offline behaviour by starting the node with the network refused |
| Node crashes on the first compressed subscriber of `instances` | Default JPEG codec on a 16-bit image | The `instances` publisher uses `codec="png"`. It is created with the first frame (colour `camera_info`) and follows camera changes |
| Labels do not fit a distorted camera's image | Masks are found in the undistorted frame | `--segments` maps them back with `xblick.rgbd.Distorter`. Do not publish undistorted labels into a distorted camera's image |
| Depth input of a camera | The node needs depth in the colour camera's geometry | `<camera>/<depth>` is 16UC1 mm in the colour geometry, paired by stamp (`--history` frames). Azure Kinect: `depth_to_color`. rc_visard: `ffs/depth` on `color=left` |
| `make` times out | No segmenter on the camera spec, or no camera motion | `:segmenter=NAME` (an xblick-sam3 instance); views at least `--min-angle` apart seen from the object; camera pose from `<camera>/tf_static` or `<camera>/tf`. `make` has its own thread; the frame loop goes on |
| Two sessions do not see each other | A session given only `--connect` does not listen, and the router does not route between such sessions | Start nodes without endpoints (they find the PC's router) |
| First query after opening a session gets no reply | Queryables need a moment | `query()` repeats until its timeout |

## Layout

| Path | Content |
| --- | --- |
| `src/xblick_nemo/nemo.py` | NeMO: `install_dinov2`, crops, `encode`/`decode`/`detect`, Umeyama + RANSAC `pose` |
| `src/xblick_nemo/node.py` | the node (cameras, tracks, publishing, queryables) and `objects_client` |
| `src/xblick_nemo/objects.py` | neural objects on disk: `<objects-dir>/NeMO/<name>/` |
| `src/xblick_nemo/weights.py` | the checkpoint (`Shards`, SHA-256), `REQUIRED`, `TERMS`, `STORE` (the core's `xblick.weights.Store`: the download, `--download-only`/`--no-download`) |
| `src/xblick_nemo/make.py` | `xblick-nemo-make`: an object from images and masks |
| `third_party/nemo` | NeMO's code and checkpoint shards (submodule, fixed commit) |
| `~/.cache/xblick/nemo/` | `checkpoint.pth` (joined by the node), `.download.lock`: the node's cache directory. The checkout's `weights/` is the former place. |
| `third_party/dinov2` | DINOv2's code, a clone at `DINOV2_COMMIT` (`tools/setup.sh`); not in git |
| `~/.local/share/xblick/nemo/` | the default `--objects-dir`: the node's data directory (`xblick.paths.data_dir`) |
| `tests/` | `test_helpers.py`, `test_weights.py` (CPU), `test_nemo.py` (GPU) |

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
| `--max-offset-ms` | 20 (fixed); 0: equal stamps only |
| `--max-hz` / `--local` | 20 / `auto` (`shm`, `cuda`) |
| Client | `--views` 4, `--min-angle` 15, `--timeout` 90 |
| Other nodes | xblick-sam3 on the same camera for `make`; rc_visard needs xblick-fast-foundationstereo's depth; xblick-foundationpose can consume `--segments` |
