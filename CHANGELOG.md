# Changelog

## 0.2.6 (2026-11-18)

The Durchblick core 0.5.14.

### Changed

- The node needs `xblick>=0.5.14,<0.6`; uv.lock fixes xblick 0.5.14 (was 0.5.11). No other package changed.
- A parameter set whose deadline has passed when the node comes to act on it is refused, and the caller raises
  `Expired`. A set meant for another node on the same key is refused (core 0.5.13).
- The rings the node publishes record their creator, so `xblick clean --remove` removes them after the node was
  killed (core 0.5.12).

### Fixed

- Closing no longer raises "Already mutably borrowed" in the thread of the program's declaration (core 0.5.13).

Compatible with earlier clients: nothing on the wire changed.

## 0.2.5 (2026-11-15)

A fixed camera's pose for `make`.

### Changed

- The node needs `xblick>=0.5.11,<0.6`. Its `collect_views` looks the camera's pose up on `<camera>/tf_static` and
  `<camera>/tf`, so `make` works with a fixed camera (`xblick cam --pose`, its pose on `tf_static`) too. The object
  then has to move between the views.
- `--max-offset-ms` takes its default, 20 ms, from the core (`history.PAIR_MAX_OFFSET_NS`).

Compatible with earlier clients: nothing on the wire changed.

## 0.2.4 (2026-11-15)

Queries answered at once, a fixed pairing offset, a fixed GPU, and the MPS and transport checks of the other nodes.

### Changed

- Changed default: `--max-offset-ms` is a fixed 20 ms, no longer half the depth stream's measured frame period. It pairs
  the colour and depth of one capture of the Azure Kinect (up to 17 ms apart) and of the rc_visard (equal stamps). A
  colour frame without a depth frame that close is skipped and counted, as before.
- `--device cuda` is GPU 0, `cuda:N` GPU N, no longer the current CUDA device. Another value, or a GPU that cannot be
  used, ends the node with one line.
- The start line says where the model computes (`model on cuda:0`).
- `--camera` is optional by design: the node starts without a camera and takes its cameras on `nemo/cameras` (README).
- README: how to cite NeMO and DINOv2.

### Fixed

- The queryables `cameras` and `objects` answer at once, also while the model works. `start`, `stop`, `add` and `remove`
  apply from the next frame on; `make` answers with its result.
- With `CUDA_MPS_PIPE_DIRECTORY` set the node stops at the start when no MPS daemon runs.
- `--local cuda` stops the node when a camera on this host offers no CUDA ring.
- `xblick-nemo-objects` no longer waits past its timeout when Zenoh does not end a query.

Compatible with earlier clients: nothing on the wire changed.

## 0.2.3 (2026-11-14)

- The weights moved out of the checkout into the node's cache directory, `~/.cache/xblick/nemo/`. `XBLICK_HOME` and
  `XDG_CACHE_HOME` move it, and `--weights-dir DIR` names another directory. Weights at the former place, the
  checkout's `weights/`, are not used. The node prints the command that moves them. In the checkout it is
  `mkdir -p ~/.cache/xblick && mv weights ~/.cache/xblick/nemo`.
- `--weights-dir DIR` is a new start option and a read-only parameter.
- The message about a missing file names the full download command.
- The node needs `xblick>=0.5.5,<0.6`.
- DINOv2's code moved out of `weights/` into the checkout's `third_party/dinov2`, as code from `tools/setup.sh`. The
  script moves an existing `weights/dinov2` there. Run `tools/setup.sh` before you move the checkpoint, so that the
  `mv` carries only the checkpoint into the cache directory.
- `--objects-dir` is no longer required. Its default is the node's data directory, `~/.local/share/xblick/nemo/`
  (`XBLICK_HOME` and `XDG_DATA_HOME` move it). An explicit `--objects-dir` works as before. Objects made earlier stay
  where you made them, for example in `~/.local/share/xblick/objects/NeMO/` as the README's example had it. Keep them
  in use with `--objects-dir ~/.local/share/xblick/objects`, or move them:
  `mkdir -p ~/.local/share/xblick/nemo && mv ~/.local/share/xblick/objects/NeMO ~/.local/share/xblick/nemo/`.
- `xblick-nemo-make` has the same `--objects-dir` default and a `--weights-dir DIR`.
- `NeMO` takes the checkpoint's directory (`weights.STORE.root`, or `--weights-dir`'s).

## 0.2.2 (2026-11-14)

- The download of the weights is the core's `xblick.weights` (xblick 0.5.3). The node's own copy of that code is
  gone. `weights.py` keeps the node's files, its terms notice and its `STORE`. The options, the messages, and the
  files and their places stay as they were.
- The node needs `xblick>=0.5.3,<0.6`.
- `tests/test_weights.py` keeps the node's own checks. The core's tests cover the download itself.

## 0.2.1 (2026-11-14)

- Licence statement corrected: five files of NeMO's decoder heads are Naver's under CC BY-NC-SA 4.0 (non-commercial use
  only), inside the MIT-licensed NeMO repository. README, NOTICE and the notice printed before the checkpoint is made
  say so.
- The node and `xblick-nemo-make` join a missing checkpoint from the submodule's shards at their start, before the
  model loads, and check its SHA-256. They print NeMO's terms, the file and its size first. With the checkpoint there
  they write no file. Nothing comes from the network: the submodule and DINOv2's code stay with `tools/setup.sh`.
- `--download-only` makes and checks the checkpoint, then exits 0. It needs no camera, Zenoh router or GPU, and
  `--objects-dir` may be left out. `tools/setup.sh` calls it after the submodule, `uv sync` and DINOv2.
- `--no-download` ends a start at once when the checkpoint is missing and names it. It is a read-only parameter.
- `weights.py` is the one place of the checkpoint (shards, SHA-256 moved from `tools/setup.sh`, size).
- The join goes to `checkpoint.pth.part` under the lock `weights/.download.lock` and gets its final name after the
  SHA-256 check. Without the submodule the node names `tools/setup.sh`.
- The joined checkpoint is also checked against its exact size, before the SHA-256.

## 0.2.0 (2026-11-13)

Every name in code, on the wire and on disk is `xblick` now. Only the prose name of the project stays Durchblick.
The node needs the core `xblick` 0.5 and does not work with programs of Durchblick 0.4 (`blick`) or older: run every
program of a setup at 0.5.

| What | 0.1.0 | 0.2.0 |
| --- | --- | --- |
| Repository | github.com/xblick/durchblick-nemo | github.com/xblick/xblick-nemo |
| Distribution | `durchblick-nemo` | `xblick-nemo` |
| Python package | `durchblick_nemo` | `xblick_nemo` |
| Commands | `durchblick-nemo`, `durchblick-nemo-objects`, `durchblick-nemo-make` | `xblick-nemo`, `xblick-nemo-objects`, `xblick-nemo-make` |
| Program name in log lines, the logger and reports | `durchblick-nemo` | `xblick-nemo` |
| Core dependency | `blick>=0.4,<0.5`, git main of github.com/xblick/durchblick | `xblick>=0.5.0,<0.6`, github.com/xblick/xblick at 45908e6 |
| Core imports and command | `import blick`, `blick rerun`, `blick play`, `blick param` | `import xblick`, `xblick rerun`, `xblick play`, `xblick param` |
| Links to the core's docs | github.com/xblick/durchblick/blob/main/docs/... | github.com/xblick/xblick/blob/main/docs/... |
| Objects directory in the docs | `~/.local/share/durchblick/objects` | `~/.local/share/xblick/objects` |
| Marker attribute on the patched `torch.hub.load` | `durchblick_redirect` | `xblick_redirect` |
| Other nodes named in the docs | `durchblick-sam3`, `durchblick-foundationpose`, `durchblick-fast-foundationstereo` | `xblick-sam3`, `xblick-foundationpose`, `xblick-fast-foundationstereo` |
| CI | secret `DURCHBLICK_READ_TOKEN` to read the private core | none |

## 0.1.0

The node as `durchblick-nemo` (Python package `durchblick_nemo`) on the core `blick` 0.4.
