#!/bin/bash
# Sets xblick-nemo up from scratch or updates it:
#
# * NeMO's code and checkpoint shards: the submodule third_party/nemo (a fixed commit)
# * the venv exactly as uv.lock fixes it (.venv; Python 3.12, the core xblick from PyPI, torch with CUDA 13)
# * DINOv2's code at a fixed commit in third_party/dinov2, a clone (NeMO's backbone; its weights are in NeMO's
#   checkpoint). A clone in weights/dinov2 (before 0.2.3) is moved there.
# * NeMO's checkpoint, through xblick-nemo --download-only (weights.py): the submodule's shards joined into
#   checkpoint.pth of the node's cache directory (~/.cache/xblick/nemo) and checked against its SHA-256. The node
#   joins it at its start too.
#
#     tools/setup.sh
#
# Needs uv (UV names it, default uv on PATH) and git: without them it stops with the command that installs them.
# Without an NVIDIA GPU with driver 580 or newer (CUDA 13) it says so and goes on: the node needs one to start,
# --download-only does not. It never asks and never uses sudo. Run again, it does only what changed.
set -euo pipefail

PROG=xblick-nemo
URL=https://github.com/xblick/xblick-nemo.git
MIN_DRIVER=580  # NVIDIA's minimum Linux driver for CUDA 13.0, which the cu130 torch is built for

usage() {
    cat <<'EOF'
Usage: tools/setup.sh

Sets xblick-nemo up in this clone, or updates it. It fetches NeMO's code and checkpoint shards (the submodule
third_party/nemo), creates .venv exactly as uv.lock fixes it (Python 3.12, PyTorch for CUDA 13, the core xblick
from PyPI), clones DINOv2's code at a fixed commit into third_party/dinov2 and joins NeMO's checkpoint in the node's
cache directory (~/.cache/xblick/nemo). A second call does only what changed. It never asks and never uses sudo.

Needs git and uv (UV=/path/to/uv names another uv). The node needs an NVIDIA GPU with driver 580 or newer.

  -h, --help  this text
EOF
}

die() {
    echo "setup.sh: $*" >&2
    exit 1
}

for arg in "$@"; do
    case $arg in
        -h | --help) usage; exit 0 ;;
        *) usage >&2; die "unknown option: $arg" ;;
    esac
done

cd "$(dirname "$0")/.."
UV=${UV:-uv}

# ------------------------------------------------------------------------------------------------- prerequisites

[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] ||
    die "the node needs Linux on x86-64 (the core's wheels), this is $(uname -s) $(uname -m)"
command -v git > /dev/null || die "no git: sudo apt install git"
command -v "$UV" > /dev/null ||
    die "no uv ($UV): install it with  wget -qO- https://astral.sh/uv/install.sh | sh  and open a new terminal"
git rev-parse --is-inside-work-tree > /dev/null 2>&1 ||
    die "$PWD is not a git clone (a ZIP download has no submodule): git clone --recurse-submodules $URL"

if ! command -v nvidia-smi > /dev/null; then
    echo "$PROG: no NVIDIA driver found (no nvidia-smi). The node needs an NVIDIA GPU with driver $MIN_DRIVER or" \
        "newer to start (Ubuntu: sudo apt install nvidia-driver-$MIN_DRIVER, then restart). The setup goes on."
elif ! driver=$(timeout 30 nvidia-smi --query-gpu=driver_version --format=csv,noheader 2> /dev/null) ||
        [[ -z $driver ]]; then
    echo "$PROG: nvidia-smi failed or lists no GPU. The node needs an NVIDIA GPU with driver $MIN_DRIVER or newer" \
        "to start. The setup goes on."
else
    driver=$(cut -d. -f1 <<< "$driver" | sort -n | head -1 | tr -d ' ')
    ((driver >= MIN_DRIVER)) ||
        echo "$PROG: NVIDIA driver $driver. PyTorch for CUDA 13 needs $MIN_DRIVER or newer (Ubuntu: sudo apt install" \
            "nvidia-driver-$MIN_DRIVER, then restart). The setup goes on."
fi

# ------------------------------------------------------------------------------------------------- the setup

cat <<EOF
$PROG: setting up $PWD. The first call downloads:
  * NeMO's code and checkpoint shards, the submodule third_party/nemo (about 870 MB, github.com/DLR-RM/nemo)
  * .venv as uv.lock fixes it: Python 3.12, the core xblick, and PyTorch for CUDA 13 with NVIDIA's CUDA libraries
    (NVIDIA's licence, License.txt in each nvidia package); about 3 GB, 6 GB on disk
  * DINOv2's code into third_party/dinov2 (github.com/facebookresearch/dinov2, Apache License 2.0)
NeMO's code and checkpoint are under the MIT licence, except five files of its decoder heads, which are Naver's under
CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/): non-commercial use only. The node uses them.
EOF
git submodule update --init third_party/nemo
"$UV" sync --frozen
DINOV2_COMMIT=$(.venv/bin/python -c "from xblick_nemo.nemo import DINOV2_COMMIT; print(DINOV2_COMMIT)")
if [[ -d weights/dinov2 && ! -e third_party/dinov2 ]]; then
    mv weights/dinov2 third_party/dinov2  # code, not weights: out of the checkpoint's former place
fi
if [[ ! -d third_party/dinov2 ]]; then
    rm -rf third_party/dinov2.tmp  # what an interrupted clone of an earlier call left
    git clone --quiet https://github.com/facebookresearch/dinov2.git third_party/dinov2.tmp
    git -C third_party/dinov2.tmp checkout --quiet "$DINOV2_COMMIT"
    mv third_party/dinov2.tmp third_party/dinov2
fi
[[ "$(git -C third_party/dinov2 rev-parse HEAD)" == "$DINOV2_COMMIT" ]] ||
    die "third_party/dinov2: not at $DINOV2_COMMIT; remove it and run again"
.venv/bin/xblick-nemo --download-only
echo "$PROG: set up. The first step with the node: README.md, Install, step 4."
