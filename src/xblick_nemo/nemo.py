"""NeMO (github.com/DLR-RM/nemo, DLR, 3DV 2026: "Finding NeMO", Neural Memory Objects): a neural object from a few
images of it (``NeMO.encode``); for a new image of it, cropped around it, the object's mask and per pixel its surface
point in the object's frame (``NeMO.decode``). The code: third_party/nemo (a fixed commit, imported from there, not
installed: its package pins plotting libraries the node does not use); the checkpoint (its shards in the submodule,
joined into weights/checkpoint.pth) and DINOv2's code at a fixed commit (weights/dinov2) by tools/setup.sh.

What the node adds (``pose``): NeMO's object frame is normalised (its points within [-1, 1]^3) and its own pose
estimate is PnP from the image alone, up to that unknown scale; with the measured depth of the mask's pixels the pose
is a similarity transform camera = s R x + t (3-point RANSAC with Umeyama's fit, all hypotheses at once on the GPU,
refit on the inliers), metric. ``detect`` finds an object in a whole image as NeMO's example does (128 px patches,
stride 64, every patch decoded, the masks averaged).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CODE = REPO / "third_party" / "nemo" / "src"
WEIGHTS = REPO / "weights"
DINOV2_REPO = "facebookresearch/dinov2"
DINOV2_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"  # facebookresearch/dinov2
CROP = 224  # NeMO's input size
SURFACE_POINTS = 1500  # sample points of a neural object (NeMO's examples)


def install_dinov2(dinov2_dir: Path):
    """NeMO's backbone calls ``torch.hub.load("facebookresearch/dinov2", name)`` (GitHub's main branch at run time, and
    its pretrained checkpoint): redirected to the clone at a fixed commit, without the pretrained checkpoint (NeMO's
    checkpoint holds its own, fine-tuned, backbone weights). Any other hub model is refused."""
    import torch  # noqa: PLC0415

    if not (dinov2_dir / "hubconf.py").is_file():
        raise FileNotFoundError(f"{dinov2_dir}/hubconf.py: missing (tools/setup.sh clones DINOv2)")
    hub_load = torch.hub.load
    if getattr(hub_load, "xblick_redirect", False):
        return

    def load(repo_or_dir, model, *args, source="github", **kwargs):
        if source != "github" or repo_or_dir != DINOV2_REPO:
            raise RuntimeError(f"torch.hub.load({repo_or_dir!r}, {model!r}, source={source!r}): only {DINOV2_REPO} "
                               "from GitHub is redirected (to weights/dinov2)")
        return hub_load(str(dinov2_dir), model, *args, source="local", pretrained=False, **kwargs)

    load.xblick_redirect = True
    torch.hub.load = load


def square_box(mask: np.ndarray | None = None, box=None, scale: float = 1.4, min_side: float = 64.0) -> tuple:
    """The square crop (x0, y0, side) around a mask's or a box's (x0, y0, x1, y1, pixels) extent, ``scale`` times its
    longer side, at least ``min_side``."""
    if mask is not None:
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            raise ValueError("empty mask")
        box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    x0, y0, x1, y1 = (float(v) for v in box)
    side = max(scale * max(x1 - x0, y1 - y0), min_side)
    return ((x0 + x1) - side) / 2, ((y0 + y1) - side) / 2, side


def crop_images(image, boxes, size: int = CROP, mode: str = "bilinear"):
    """``image`` (H, W, C) or (H, W) torch tensor, ``boxes`` [(x0, y0, side), ...] (pixel edges at integers) -> (K, C,
    size, size) float crops; 0 outside the image."""
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    h, w = image.shape[:2]
    x = (image[..., None] if image.ndim == 2 else image).permute(2, 0, 1)[None].float()
    b = torch.as_tensor(boxes, dtype=torch.float32, device=image.device).reshape(-1, 3)
    c = (torch.arange(size, device=image.device, dtype=torch.float32) + 0.5) / size
    gx = (b[:, :1] + c[None] * b[:, 2:]) / w * 2 - 1  # (K, size)
    gy = (b[:, 1:2] + c[None] * b[:, 2:]) / h * 2 - 1
    grid = torch.stack([gx[:, None, :].expand(-1, size, -1), gy[:, :, None].expand(-1, -1, size)], -1)
    return F.grid_sample(x.expand(len(b), -1, -1, -1), grid, mode=mode, padding_mode="zeros", align_corners=False)


def crop_pixels(box, size: int = CROP):
    """The image coordinates (u, v), (size, size) each, of a crop's pixel centres."""
    x0, y0, side = box
    c = (np.arange(size) + 0.5) * side / size - 0.5
    return np.meshgrid(x0 + c, y0 + c)


def umeyama_batched(x, y, weights):
    """Similarity transforms y = s R x + t of point sets ``x``, ``y`` (B, N, 3) with ``weights`` (B, N): s (B,),
    R (B, 3, 3), t (B, 3)."""
    import torch  # noqa: PLC0415

    wsum = weights.sum(1, keepdim=True).clamp_min(1e-9)
    w = (weights / wsum)[..., None]
    mx, my = (w * x).sum(1), (w * y).sum(1)
    xc, yc = x - mx[:, None], y - my[:, None]
    cov = (w * yc).transpose(1, 2) @ xc  # (B, 3, 3)
    U, S, Vt = torch.linalg.svd(cov)
    d = torch.sign(torch.linalg.det(U @ Vt))
    D = torch.ones_like(S)
    D[:, 2] = d
    R = U @ torch.diag_embed(D) @ Vt
    var = (w[..., 0] * (xc ** 2).sum(-1)).sum(1).clamp_min(1e-12)
    s = (S * D).sum(1) / var
    return s, R, my - s[:, None] * (R @ mx[..., None])[..., 0]


def sim3_ransac(x, y, hypotheses: int = 512, threshold: float = 0.01, generator=None):
    """The similarity transform y = s R x + t with the most inliers (|y - (s R x + t)| < ``threshold``) among
    ``hypotheses`` fits of 3 random correspondences, refit on its inliers twice: (4 x 4 [sR | t] float64 numpy, inlier
    count). ``x`` NeMO's points, ``y`` the camera's [m], (N, 3) torch tensors on one device."""
    import torch  # noqa: PLC0415

    n = len(x)
    if n < 3:
        raise ValueError(f"{n} correspondences: at least 3")
    idx = torch.randint(0, n, (hypotheses, 3), device=x.device, generator=generator)
    s, R, t = umeyama_batched(x[idx], y[idx], torch.ones(hypotheses, 3, device=x.device))
    ok = torch.isfinite(s) & (s > 0)
    pred = s[:, None, None] * (x[None] @ R.transpose(1, 2)) + t[:, None]  # (H, N, 3)
    inliers = ((pred - y[None]).norm(dim=-1) < threshold) & ok[:, None]
    best = inliers.sum(1).argmax()
    mask = inliers[best]
    for _ in range(2):
        s1, R1, t1 = umeyama_batched(x[None], y[None], mask[None].float())
        mask = ((s1[0] * x @ R1[0].T + t1[0]) - y).norm(dim=-1) < threshold
    T = np.eye(4)
    T[:3, :3] = (s1[0] * R1[0]).double().cpu().numpy()
    T[:3, 3] = t1[0].double().cpu().numpy()
    return T, int(mask.sum())


@dataclass
class Decoded:
    """NeMO's prediction for crops: per pixel the surface point in the object's frame (K, 224, 224, 3), its confidence
    (K, 224, 224) and the visible (modal) mask's probability (K, 224, 224)."""

    points: object
    confidence: object
    mask: object


@dataclass
class Pose:
    """An object's pose from one crop: camera = scale R x + t (x in NeMO's frame), the inliers, the mask's pixels."""

    matrix: np.ndarray  # 4 x 4 [s R | t]
    scale: float
    inliers: int
    mask_pixels: int


class NeMO:
    """NeMO's network on ``device``, loaded once; ``autocast``: run it in bfloat16 (``None``: float32)."""

    def __init__(self, device="cuda", weights: Path = WEIGHTS, autocast: str | None = None):
        import torch  # noqa: PLC0415

        checkpoint = weights / "checkpoint.pth"
        for path in (CODE / "nemolib", checkpoint, weights / "dinov2"):
            if not path.exists():
                raise FileNotFoundError(f"{path}: missing (git submodule update --init; tools/setup.sh)")
        install_dinov2(weights / "dinov2")
        if str(CODE) not in sys.path:
            sys.path.insert(0, str(CODE))
        from nemolib.model import Model  # noqa: PLC0415

        self.device = torch.device(device)
        self.model = Model.from_checkpoint(str(checkpoint), device=self.device).eval()
        if autocast not in (None, "bfloat16"):
            raise ValueError(f"autocast {autocast!r}: None or bfloat16")
        self.dtype = torch.bfloat16 if autocast else None

    def _autocast(self):
        import contextlib  # noqa: PLC0415

        import torch  # noqa: PLC0415

        if self.dtype is None:
            return contextlib.nullcontext()
        return torch.autocast("cuda", dtype=self.dtype)

    def encode(self, crops, seed: int = 0):
        """A neural object from template crops (T, 3, 224, 224) in [0, 1]: its features (N, C) and surface points
        (N, 3) in its frame."""
        import torch  # noqa: PLC0415

        g = torch.Generator(device=self.device).manual_seed(seed)
        samples = torch.rand(1, SURFACE_POINTS, 3, device=self.device, generator=g) * 2 - 1
        with torch.no_grad():
            nemo = self.model.encode_images(crops[None].float(), samples)  # needs float32: autograd of the field
            features = nemo["features_3d"] + self.model.point_encoder(nemo["surface_points"])
        return features[0], nemo["surface_points"][0]

    def decode(self, crops, features) -> Decoded:
        """Crops (B, K, 3, 224, 224) in [0, 1] of B objects with their features (B, N, C) -> Decoded with (B, K, ...)
        fields (one object: crops (K, 3, 224, 224), features (N, C) -> (K, ...))."""
        import torch  # noqa: PLC0415

        single = crops.ndim == 4
        if single:
            crops, features = crops[None], features[None]
        with torch.no_grad(), self._autocast():
            out = self.model.decode_images(crops.float(), features)
        d = Decoded(out["pts3d"].float(), out["conf"].float(), out["mask"].float())
        if single:
            d = Decoded(d.points[0], d.confidence[0], d.mask[0])
        return d

    @staticmethod
    def pose(decoded_points, confidence, mask, depth, K: np.ndarray, box, conf_threshold: float = 1.2,
             mask_threshold: float = 0.5, threshold: float = 0.01, generator=None) -> Pose | None:
        """The pose from one decoded crop (points (224, 224, 3), confidence, mask (224, 224)) and the frame's depth
        (H, W) [m] (torch, on the decoded crop's device): the crop's confident mask pixels with depth, their NeMO
        points against their camera points. None with fewer than 10 such pixels."""
        import torch  # noqa: PLC0415

        dev = depth.device
        x0, y0, side = box
        c = (torch.arange(CROP, device=dev, dtype=torch.float32) + 0.5) * (side / CROP) - 0.5
        vv, uu = torch.meshgrid(y0 + c, x0 + c, indexing="ij")  # the crop's pixel centres in the image
        h, w = depth.shape
        ui, vi = uu.round().long(), vv.round().long()
        inside = (ui >= 0) & (vi >= 0) & (ui < w) & (vi < h)
        z = torch.where(inside, depth[vi.clamp(0, h - 1), ui.clamp(0, w - 1)], torch.zeros((), device=dev))
        selected = (mask > mask_threshold) & (confidence > conf_threshold) & (z > 0)
        n = int(selected.sum())
        if n < 10:
            return None
        cam = torch.stack([(uu - K[0, 2]) * z / K[0, 0], (vv - K[1, 2]) * z / K[1, 1], z], -1)
        T, inliers = sim3_ransac(decoded_points[selected], cam[selected], threshold=threshold, generator=generator)
        return Pose(T, float(np.cbrt(np.linalg.det(T[:3, :3]))), inliers, n)

    def detect(self, image, features, patch: int = 128, stride: int = 64, chunk: int = 64):
        """Each object's visible-mask probability over the whole ``image`` (H, W, 3) uint8 torch: every ``patch``-px
        patch (``stride`` apart) decoded, the probabilities averaged where patches overlap (NeMO's
        run_detection_on_video.py), ``chunk`` patches at a time. ``features`` (B, N, C) -> (B, H, W)
        probabilities."""
        import math  # noqa: PLC0415

        import torch  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415

        h, w = image.shape[:2]
        nh, nw = math.ceil(max(h - patch, 0) / stride) + 1, math.ceil(max(w - patch, 0) / stride) + 1
        boxes = [(j * stride, i * stride, patch) for i in range(nh) for j in range(nw)]
        b = features.shape[0]
        masks = []
        for k in range(0, len(boxes), chunk):
            crops = crop_images(image.float() / 255.0, boxes[k:k + chunk])  # (P, 3, 224, 224)
            masks.append(self.decode(crops[None].expand(b, -1, -1, -1, -1), features).mask)
        mask = torch.cat(masks, 1)
        probs = F.interpolate(mask.reshape(b * len(boxes), 1, CROP, CROP), size=(patch, patch), mode="bilinear",
                              align_corners=False).reshape(b, len(boxes), patch * patch).transpose(1, 2)
        th, tw = (nh - 1) * stride + patch, (nw - 1) * stride + patch
        total = F.fold(probs, (th, tw), patch, stride=stride)
        count = F.fold(torch.ones_like(probs[:1]), (th, tw), patch, stride=stride)
        return (total / count)[:, 0, :h, :w]
