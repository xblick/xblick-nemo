"""The crops, the pose fit and the objects' files (nemo.py, objects.py); on the CPU."""

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from durchblick_nemo import objects
from durchblick_nemo.nemo import CROP, crop_images, crop_pixels, sim3_ransac, square_box, umeyama_batched


def test_square_box_around_a_mask():
    mask = np.zeros((100, 200), bool)
    mask[20:40, 50:110] = True  # 60 x 20 px
    x0, y0, side = square_box(mask=mask, scale=1.5)
    assert side == pytest.approx(90.0) and (x0 + side / 2, y0 + side / 2) == pytest.approx((80.0, 30.0))
    assert square_box(box=(0, 0, 10, 10))[2] == 64.0  # at least min_side


def test_a_crop_samples_where_crop_pixels_says():
    image = torch.zeros(100, 120)
    image[30, 70] = 1.0
    box = (40.0, 10.0, 112.0)  # 0.5 image px per crop px
    crop = crop_images(image, [box], mode="nearest")[0, 0]
    u, v = crop_pixels(box)
    hit = crop > 0.5
    assert hit.any()
    assert np.round(u[hit.numpy()]).astype(int).tolist() == [70] * int(hit.sum())
    assert np.round(v[hit.numpy()]).astype(int).tolist() == [30] * int(hit.sum())
    assert crop.shape == (CROP, CROP)


def test_umeyama_and_ransac_recover_a_similarity():
    g = torch.Generator().manual_seed(0)
    x = torch.rand(400, 3, generator=g) * 2 - 1
    R = torch.as_tensor(Rotation.from_rotvec([0.3, -0.8, 0.5]).as_matrix(), dtype=torch.float32)
    t = torch.tensor([0.1, -0.2, 0.9])
    y = 0.07 * x @ R.T + t
    s1, R1, t1 = umeyama_batched(x[None], y[None], torch.ones(1, 400))
    assert float(s1[0]) == pytest.approx(0.07, rel=1e-4)
    torch.testing.assert_close(R1[0], R, atol=1e-4, rtol=0)
    y[:120] += torch.rand(120, 3, generator=g) * 0.2 + 0.05  # 30 % outliers
    T, inliers = sim3_ransac(x, y, threshold=0.005, generator=g)
    assert inliers == 280
    np.testing.assert_allclose(T[:3, :3], 0.07 * R.numpy(), atol=1e-4)
    np.testing.assert_allclose(T[:3, 3], t.numpy(), atol=1e-4)


def test_objects_save_and_load(tmp_path):
    features, points = torch.randn(50, 8), torch.rand(50, 3) * 2 - 1
    out = objects.save(tmp_path, "pink-tape", features, points, torch.rand(3, 3, CROP, CROP), {"source": "test"})
    assert (out / "templates.png").is_file() and objects.names(tmp_path) == ["pink-tape"]
    obj = objects.load(tmp_path, "pink-tape", "cpu")
    torch.testing.assert_close(obj.features, features)
    assert (obj.lo < obj.hi).all()
    with pytest.raises(ValueError, match="letters"):
        objects.folder(tmp_path, "a b")
    with pytest.raises(FileNotFoundError):
        objects.load(tmp_path, "missing", "cpu")
