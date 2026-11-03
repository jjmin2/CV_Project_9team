"""Sobel gradients, normal-reference scores, and candidate defect masks."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np


@dataclass(frozen=True)
class SobelConfig:
    blur_size: int = 5
    blur_sigma: float = 1.0
    epsilon: float = 0.01
    pixel_quantile: float = 0.995
    image_quantile: float = 0.99
    opening_size: int = 0
    closing_size: int = 3
    min_area: int = 20

    def __post_init__(self):
        for name in ("blur_size", "opening_size", "closing_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0 or (value and value % 2 == 0):
                raise ValueError(f"{name} must be zero or a positive odd integer")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if not np.isfinite(self.blur_sigma) or self.blur_sigma < 0:
            raise ValueError("blur_sigma must be finite and nonnegative")
        if not isinstance(self.min_area, int) or self.min_area < 1:
            raise ValueError("min_area must be a positive integer")
        for q in (self.pixel_quantile, self.image_quantile):
            if not 0 < q <= 1:
                raise ValueError("quantiles must be in (0, 1]")


def as_rgb(image):
    """Accept the team's CHW tensor [0,1], or an HWC RGB NumPy array.

    uint8 arrays are divided by 255. Floating arrays must already be in [0,1].
    No resize or per-image min/max normalization is performed here.
    """
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("Expected one RGB tensor shaped (3,H,W), without a batch axis")
        image = np.moveaxis(image, 0, -1)
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) < 2:
        raise ValueError("Expected an RGB array shaped (H,W,3), at least 2x2")
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    elif np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float32)
    else:
        raise ValueError("Use uint8 [0,255] or floating RGB [0,1]")
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("RGB values must be finite and in [0,1]")
    return np.ascontiguousarray(image)


def sobel_manual(gray):
    """Apply both 3x3 Sobel kernels explicitly with NumPy shifts.

    reflect padding matches OpenCV BORDER_REFLECT_101. The kernel is applied
    without flipping, following OpenCV's correlation/sign convention.
    """
    gray = np.asarray(gray, dtype=np.float32)
    if gray.ndim != 2 or min(gray.shape) < 2 or not np.isfinite(gray).all():
        raise ValueError("Expected a finite 2D image, at least 2x2")
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = kx.T
    padded = np.pad(gray, 1, mode="reflect")
    height, width = gray.shape
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    for row in range(3):
        for col in range(3):
            patch = padded[row:row + height, col:col + width]
            gx += kx[row, col] * patch
            gy += ky[row, col] * patch
    magnitude = np.sqrt(gx * gx + gy * gy)
    return gx, gy, magnitude


def postprocess_mask(raw_mask, config):
    """Morphology followed by 8-connected component area filtering."""
    mask = (np.asarray(raw_mask) != 0).astype(np.uint8)
    for size, operation in ((config.opening_size, cv2.MORPH_OPEN),
                            (config.closing_size, cv2.MORPH_CLOSE)):
        if size > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            mask = cv2.morphologyEx(mask, operation, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(count, dtype=np.uint8)
    regions = []
    for i in range(1, count):
        x, y, width, height, area = map(int, stats[i])
        if area >= config.min_area:
            keep[i] = 1
            regions.append({"x": x, "y": y, "width": width, "height": height, "area": area})
    return keep[labels], regions


class SobelDetector:
    """fit(normal train) -> calibrate(normal validation) -> predict(test)."""

    def __init__(self, config=None):
        self.config = config or SobelConfig()
        self.mean = None
        self.std = None
        self.pixel_threshold = None
        self.image_threshold = None
        self.train_count = 0
        self.val_count = 0

    def gradients(self, image):
        rgb = as_rgb(image)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if self.config.blur_size > 1:
            gray = cv2.GaussianBlur(gray, (self.config.blur_size,) * 2,
                                    self.config.blur_sigma, borderType=cv2.BORDER_REFLECT_101)
        return sobel_manual(gray)

    def fit(self, normal_images):
        """Estimate per-pixel population mean/std using normal training images.

        Welford accumulation avoids retaining every gradient map in memory.
        Calling fit again clears the previous model and thresholds.
        """
        self.mean = self.std = None
        self.pixel_threshold = self.image_threshold = None
        self.train_count = self.val_count = 0
        mean = m2 = None
        for image in normal_images:
            magnitude = self.gradients(image)[2].astype(np.float64)
            if mean is None:
                mean = np.zeros_like(magnitude)
                m2 = np.zeros_like(magnitude)
            if magnitude.shape != mean.shape:
                raise ValueError("All training images must have the same dimensions")
            self.train_count += 1
            delta = magnitude - mean
            mean += delta / self.train_count
            m2 += delta * (magnitude - mean)
        if self.train_count < 2:
            raise ValueError("At least two normal training images are required")
        self.mean = mean.astype(np.float32)
        self.std = np.sqrt(np.maximum(m2 / self.train_count, 0)).astype(np.float32)
        return self

    def score(self, image):
        if self.mean is None:
            raise RuntimeError("Call fit before scoring")
        gx, gy, magnitude = self.gradients(image)
        if magnitude.shape != self.mean.shape:
            raise ValueError(f"Expected image size {self.mean.shape}, got {magnitude.shape}")
        score_map = np.abs(magnitude - self.mean) / (self.std + self.config.epsilon)
        return gx, gy, magnitude, score_map

    def calibrate(self, normal_images):
        """Fix pixel and image thresholds using only held-out normal images.

        Pixel threshold: pooled pixel quantile. Image threshold: quantile of
        each image's maximum score. These are different decision thresholds.
        """
        self.pixel_threshold = self.image_threshold = None
        maps = [self.score(image)[3] for image in normal_images]
        if not maps:
            raise ValueError("Normal validation images are required")
        self.val_count = len(maps)
        self.pixel_threshold = float(np.quantile(np.stack(maps), self.config.pixel_quantile,
                                                 method="higher"))
        self.image_threshold = float(np.quantile([m.max() for m in maps],
                                                 self.config.image_quantile, method="higher"))
        return self

    def predict(self, image):
        if self.pixel_threshold is None or self.image_threshold is None:
            raise RuntimeError("Call fit and calibrate before prediction")
        start = perf_counter()
        gx, gy, magnitude, score_map = self.score(image)
        raw_mask = (score_map > self.pixel_threshold).astype(np.uint8)
        mask, regions = postprocess_mask(raw_mask, self.config)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_score = float(score_map.max())
        return {
            "gx": gx, "gy": gy, "edge_map": magnitude, "score_map": score_map,
            "raw_mask": raw_mask, "mask": mask, "regions": regions, "contours": contours,
            "image_score": image_score, "is_anomaly": image_score > self.image_threshold,
            "elapsed_ms": (perf_counter() - start) * 1000,
        }

    def save(self, path):
        if self.pixel_threshold is None or self.mean is None:
            raise RuntimeError("Only a fitted and calibrated model can be saved")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "version": 1, "config": asdict(self.config), "train_count": self.train_count,
            "val_count": self.val_count, "pixel_threshold": self.pixel_threshold,
            "image_threshold": self.image_threshold,
        }
        with path.open("wb") as output:
            np.savez_compressed(output, mean=self.mean, std=self.std,
                                metadata=json.dumps(metadata))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"]))
            if metadata["version"] != 1:
                raise ValueError("Unsupported Sobel model version")
            model = cls(SobelConfig(**metadata["config"]))
            model.mean = saved["mean"].astype(np.float32)
            model.std = saved["std"].astype(np.float32)
        if (model.mean.ndim != 2 or model.mean.shape != model.std.shape
                or not np.isfinite(model.mean).all() or not np.isfinite(model.std).all()
                or (model.std < 0).any()):
            raise ValueError("Invalid normal statistics")
        for name in ("train_count", "val_count", "pixel_threshold", "image_threshold"):
            setattr(model, name, metadata[name])
        if not all(np.isfinite(t) and t >= 0
                   for t in (model.pixel_threshold, model.image_threshold)):
            raise ValueError("Invalid thresholds")
        return model
