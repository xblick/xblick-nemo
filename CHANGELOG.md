# Changelog

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
