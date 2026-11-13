#!/bin/bash
# Sets durchblick-nemo up from scratch or updates it:
#
# * NeMO's code and checkpoint: the submodule third_party/nemo (a fixed commit; its checkpoint as shards in git, joined
#   into weights/checkpoint.pth and checked against its SHA-256)
# * DINOv2's code at a fixed commit in weights/dinov2 (NeMO's backbone; its weights are in NeMO's checkpoint)
# * the venv exactly as uv.lock fixes it (.venv)
#
#     tools/setup.sh
#
# Needs uv (UV names it, default uv on PATH) and git. Run again, it does only what changed.
set -euo pipefail
cd "$(dirname "$0")/.."
UV=${UV:-uv}
command -v "$UV" > /dev/null || { echo "no uv: $UV (set UV to the uv binary)" >&2; exit 1; }
CHECKPOINT_SHA256=546f9c1d52d0c9688c0f9b076d754f01861a93f6d85c42ecf92171560b0df780

git submodule update --init third_party/nemo
mkdir -p weights
if [[ ! -f weights/checkpoint.pth ]]; then
    cat third_party/nemo/checkpoints/part_* > weights/checkpoint.pth.tmp
    mv weights/checkpoint.pth.tmp weights/checkpoint.pth
fi
echo "$CHECKPOINT_SHA256  weights/checkpoint.pth" | sha256sum --check --quiet || {
    echo "weights/checkpoint.pth: not NeMO's checkpoint of third_party/nemo's commit; remove it and run again" >&2; exit 1; }
"$UV" sync --frozen
DINOV2_COMMIT=$(.venv/bin/python -c "from durchblick_nemo.nemo import DINOV2_COMMIT; print(DINOV2_COMMIT)")
if [[ ! -d weights/dinov2 ]]; then
    git clone --quiet https://github.com/facebookresearch/dinov2.git weights/dinov2.tmp
    git -C weights/dinov2.tmp checkout --quiet "$DINOV2_COMMIT"
    mv weights/dinov2.tmp weights/dinov2
fi
[[ "$(git -C weights/dinov2 rev-parse HEAD)" == "$DINOV2_COMMIT" ]] || {
    echo "weights/dinov2: not at $DINOV2_COMMIT; remove it and run again" >&2; exit 1; }
echo "durchblick-nemo: set up"
