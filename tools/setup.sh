#!/bin/bash
# Sets xblick-nemo up from scratch or updates it:
#
# * NeMO's code and checkpoint shards: the submodule third_party/nemo (a fixed commit)
# * the venv exactly as uv.lock fixes it (.venv)
# * DINOv2's code at a fixed commit in weights/dinov2 (NeMO's backbone; its weights are in NeMO's checkpoint)
# * NeMO's checkpoint, through xblick-nemo --download-only (weights.py): the submodule's shards joined into
#   weights/checkpoint.pth and checked against its SHA-256. The node joins it at its start too.
#
#     tools/setup.sh
#
# Needs uv (UV names it, default uv on PATH) and git. Run again, it does only what changed.
set -euo pipefail
cd "$(dirname "$0")/.."
UV=${UV:-uv}
command -v "$UV" > /dev/null || { echo "no uv: $UV (set UV to the uv binary)" >&2; exit 1; }

git submodule update --init third_party/nemo
"$UV" sync --frozen
mkdir -p weights
DINOV2_COMMIT=$(.venv/bin/python -c "from xblick_nemo.nemo import DINOV2_COMMIT; print(DINOV2_COMMIT)")
if [[ ! -d weights/dinov2 ]]; then
    git clone --quiet https://github.com/facebookresearch/dinov2.git weights/dinov2.tmp
    git -C weights/dinov2.tmp checkout --quiet "$DINOV2_COMMIT"
    mv weights/dinov2.tmp weights/dinov2
fi
[[ "$(git -C weights/dinov2 rev-parse HEAD)" == "$DINOV2_COMMIT" ]] || {
    echo "weights/dinov2: not at $DINOV2_COMMIT; remove it and run again" >&2; exit 1; }
.venv/bin/xblick-nemo --download-only
echo "xblick-nemo: set up"
