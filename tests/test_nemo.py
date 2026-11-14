"""NeMO on its own example (third_party/nemo/assets/mug): a neural object from three of the mug's images finds the mug
in the fourth (its mask in the middle of the crop, not at its border) and in a whole image. Needs a CUDA device and
the checkpoint in the node's cache directory (xblick-nemo --download-only)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from xblick_nemo.weights import STORE

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not (STORE.root / "checkpoint.pth").is_file(),
                                reason="needs a CUDA device and the checkpoint (xblick-nemo --download-only)")
MUG = Path(__file__).resolve().parents[1] / "third_party" / "nemo" / "assets" / "mug"


def test_nemo_finds_the_mug():
    from PIL import Image

    from xblick_nemo.nemo import NeMO, crop_images

    images = [torch.as_tensor(np.asarray(Image.open(p).convert("RGB")), device="cuda")
              for p in sorted(MUG.glob("*.jpg"))]
    crops = torch.cat([crop_images(i.float() / 255.0, [(0, 0, min(i.shape[:2]))], mode="bicubic") for i in images])
    nemo = NeMO("cuda", STORE.root, autocast="bfloat16")
    features, points = nemo.encode(crops[:3].clamp(0, 1))
    assert points.abs().max() <= 1.5  # NeMO's normalised frame
    d = nemo.decode(crops[3:], features)
    mask = (d.mask[0] > 0.5).cpu().numpy()
    assert 0.03 < mask.mean() < 0.7, mask.mean()
    ys, xs = np.nonzero(mask)
    assert 40 < xs.mean() < 184 and 40 < ys.mean() < 184  # the mug, not the crop's border
    small = torch.nn.functional.interpolate(images[3].permute(2, 0, 1)[None].float(), size=(640, 640), mode="area")
    probs = nemo.detect(small[0].permute(1, 2, 0).round().to(torch.uint8), features[None])  # a camera's size
    assert probs.shape == (1, 640, 640) and float(probs.max()) > 0.5
