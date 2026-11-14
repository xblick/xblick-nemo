"""The node's weights: which files it reads, where they come from, and how they are made.

This module is the one list of them: the node's start, xblick-nemo-make and ``tools/setup.sh`` (through
``xblick-nemo --download-only``) use it. ``STORE.ensure`` touches neither the disk nor the network when every file is
there. A missing file is made under a temporary name in its directory (``<name>.part``), checked and only then
renamed, all under a file lock, so that two nodes started at once make it once and never read half a file.

* ``weights/checkpoint.pth``: NeMO's checkpoint, joined from its shards in the submodule third_party/nemo (git keeps
  it so: ``Shards``) and checked by its SHA-256. Nothing comes from the network here: the submodule and DINOv2's
  code (weights/dinov2) are code, which ``tools/setup.sh`` fetches with git (``REQUIRED``: the shards first).
* ``needed()``: the checkpoint (every start reads it)

The download itself (lock, ``.part`` files, checks, progress, ``--download-only``/``--no-download``) is the core's
``xblick.weights``; this module keeps the node's files, its terms notice and its ``STORE``.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from xblick.weights import CHUNK, Store, Weight, WeightsError

from .nemo import REPO, WEIGHTS

PROG = "xblick-nemo"
SHARDS = REPO / "third_party" / "nemo" / "checkpoints"  # part_aa, part_ab, ...: the checkpoint in git
REQUIRED = ((SHARDS, "NeMO's checkpoint shards, the submodule third_party/nemo"),)  # (path, what): from tools/setup.sh
CHECKPOINT_SHA256 = "546f9c1d52d0c9688c0f9b076d754f01861a93f6d85c42ecf92171560b0df780"  # of the submodule's commit
CHECKPOINT_SIZE = 823_730_274
TERMS = """\
NeMO is the work of DLR, under its terms, and you are the one who uses it:
  * NeMO's code and checkpoint (github.com/DLR-RM/nemo, the submodule third_party/nemo) are under the MIT licence.
  * Five files of its decoder heads (nemolib/modules/heads) are Naver's, under CC BY-NC-SA 4.0: non-commercial
    use only. Whether you may use the node for your purpose is yours to check."""


def needed() -> list[Weight]:
    """The checkpoint every start reads."""
    return [Weight(WEIGHTS / "checkpoint.pth", Shards(SHARDS, "part_*"), CHECKPOINT_SIZE, CHECKPOINT_SHA256)]


@dataclass(frozen=True)
class Shards:
    """A file joined from its shards (``pattern`` in ``directory``, in the order of their names): NeMO's checkpoint
    as the submodule keeps it in git."""

    directory: Path
    pattern: str

    def describe(self) -> str:
        return f"the shards {STORE.short(self.directory)}/{self.pattern} (joined)"

    def fetch(self, tmp: Path, progress, store: Store) -> str:
        parts = sorted(self.directory.glob(self.pattern))
        if not parts:
            raise WeightsError(f"{store.short(self.directory)}: no {self.pattern}: run tools/setup.sh")
        total, done, digest = sum(p.stat().st_size for p in parts), 0, hashlib.sha256()
        progress(0, total)
        with open(tmp, "wb") as f:
            for part in parts:
                with open(part, "rb") as source:
                    while chunk := source.read(CHUNK):
                        f.write(chunk)
                        digest.update(chunk)
                        done += len(chunk)
                        progress(done, total)
            f.flush()
            os.fsync(f.fileno())
        return digest.hexdigest()


STORE = Store(PROG, WEIGHTS, TERMS, REQUIRED, repo=REPO)  # the download (xblick.weights), its lock in WEIGHTS
