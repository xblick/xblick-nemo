"""The node's objects on disk: ``<objects-dir>/NeMO/<name>/`` with ``nemo.pt`` (the neural object: its features and
surface points, NeMO's frame), ``templates.png`` (the template crops it was made from, side by side) and
``made.json`` (where they came from). Made by ``xblick-nemo-make`` (images) or the node (a camera's views)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATASET = "NeMO"
NAME = re.compile(r"^[A-Za-z0-9_-]+$")
FORMAT = 1  # of nemo.pt


@dataclass
class NeuralObject:
    name: str
    features: object  # (N, C) torch
    points: object  # (N, 3) torch: its surface points, NeMO's frame
    lo: np.ndarray  # the surface points' bounding box in NeMO's frame
    hi: np.ndarray


def folder(objects_dir: Path, name: str) -> Path:
    if not NAME.match(name):
        raise ValueError(f"name {name!r}: letters, digits, _ and -")
    return Path(objects_dir) / DATASET / name


def save(objects_dir: Path, name: str, features, points, crops, record: dict) -> Path:
    """Writes the object (crops: (T, 3, 224, 224) in [0, 1]); returns its folder."""
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    out = folder(objects_dir, name)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"format": FORMAT, "features": features.detach().cpu(), "points": points.detach().cpu()},
               out / "nemo.pt")
    strip = (crops.clamp(0, 1) * 255).round().to(torch.uint8).permute(2, 0, 3, 1).reshape(crops.shape[2], -1, 3)
    Image.fromarray(strip.cpu().numpy()).save(out / "templates.png")
    (out / "made.json").write_text(json.dumps({"name": name, "templates": int(len(crops)), **record}, indent=1))
    return out


def load(objects_dir: Path, name: str, device) -> NeuralObject:
    import torch  # noqa: PLC0415

    path = folder(objects_dir, name) / "nemo.pt"
    if not path.is_file():
        raise FileNotFoundError(f"{path}: no such object (xblick-nemo-make makes one)")
    data = torch.load(path, map_location=device, weights_only=True)
    if data.get("format") != FORMAT:
        raise ValueError(f"{path}: format {data.get('format')}, this node reads {FORMAT}")
    points = data["points"]
    p = points.cpu().numpy()
    return NeuralObject(name, data["features"], points, np.percentile(p, 2, axis=0), np.percentile(p, 98, axis=0))


def names(objects_dir: Path) -> list[str]:
    root = Path(objects_dir) / DATASET
    return sorted(p.parent.name for p in root.glob("*/nemo.pt")) if root.is_dir() else []
