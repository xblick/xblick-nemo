"""xblick-nemo-make: a neural object from images of the object (a few, from around it), without a camera stream.

    xblick-nemo-make NAME --images A.jpg B.jpg ... [--objects-dir DIR] [--weights-dir DIR]
                         [--masks A.png B.png ... | --boxes X0,Y0,X1,Y1 ...]

Each image is cropped square around its mask's or box's extent (1.4 times its longer side), or, with neither, to its
centred square (as NeMO's own capture does: the object in the middle, filling much of the image). Written to
``<DIR>/NeMO/NAME/`` (objects.py); the node loads it with ``--object NAME`` or at run time (xblick-nemo-objects add
NAME). From a camera's stream instead: xblick-nemo-objects make (the node collects the views). A missing checkpoint
is joined from its shards first, as the node does at its start (weights.py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="xblick-nemo-make", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name")
    parser.add_argument("--objects-dir", default=None,
                        help="where the objects go (default: the node's data directory, ~/.local/share/xblick/nemo)")
    parser.add_argument("--weights-dir", default=None, metavar="DIR",
                        help="where NeMO's checkpoint is (default: the node's cache directory, ~/.cache/xblick/nemo)")
    parser.add_argument("--images", nargs="+", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--masks", nargs="+", help="per image a mask (non-zero: the object)")
    group.add_argument("--boxes", nargs="+", metavar="X0,Y0,X1,Y1", help="per image the object's box [px]")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    given = args.masks or args.boxes
    if given is not None and len(given) != len(args.images):
        parser.error(f"{len(args.images)} images, {len(given)} masks or boxes")
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from xblick import weights as xblick_weights  # noqa: PLC0415

    from . import (
        objects,  # noqa: PLC0415
        weights,  # noqa: PLC0415
    )
    from .nemo import NeMO, crop_images, square_box  # noqa: PLC0415
    from .node import objects_dir_of  # noqa: PLC0415

    objects_dir = objects_dir_of(args)
    objects.folder(objects_dir, args.name)  # checks the name before the model loads
    store = weights.store_for(parser, args)
    try:
        store.ensure(weights.needed(), options=weights.options(args))
    except xblick_weights.WeightsError as error:
        return store.fail(error)
    device = torch.device(args.device)
    crops = []
    for i, path in enumerate(args.images):
        rgb = np.asarray(Image.open(path).convert("RGB"))
        h, w = rgb.shape[:2]
        if args.masks:
            box = square_box(mask=np.asarray(Image.open(args.masks[i])) > 0)
        elif args.boxes:
            box = square_box(box=[float(v) for v in args.boxes[i].split(",")])
        else:
            side = min(h, w)
            box = ((w - side) / 2, (h - side) / 2, side)
        crops.append(crop_images(torch.as_tensor(rgb, device=device).float() / 255.0, [box], mode="bicubic")[0])
    crops = torch.stack(crops).clamp(0, 1)
    nemo = NeMO(device, store.root)
    features, points = nemo.encode(crops)
    out = objects.save(objects_dir, args.name, features, points, crops,
                       {"source": "images", "images": [str(Path(p).resolve()) for p in args.images],
                        "crop": "masks" if args.masks else "boxes" if args.boxes else "centred square"})
    print(f"xblick-nemo-make: {args.name} from {len(crops)} images -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
