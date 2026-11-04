"""Two anomaly detectors with the SAME interface:

    detector.fit(normal_images: list[np.ndarray])   # resized BGR images, normal ones only
    detector.score_map(img_bgr) -> np.ndarray       # float32, shape (H, W) equal to the input image
    detector.score_range, detector.default_threshold, detector.name

As long as this interface is kept, app.py does not need to change when the internals are replaced.

Optional statistics cache (so the app does not re-estimate statistics on every start):
    detector.supports_cache = True        # opt in
    detector.cache_version = "1"          # bump when the saved statistics change meaning
    detector.save(path)                   # write the fitted statistics to path (a .npz file)
    detector.load(path)                   # restore them; no fit() needed afterwards
If supports_cache is False (or missing) the detector is simply fitted on every start.

Sobel: SobelDetector (below) is a thin adapter around the team's own module sobel_detector.py
(written by the Sobel owner, kept here UNCHANGED). The app uses that module for the Sobel gradients and
the normal mean/std statistics. Thresholds and postprocessing come from the shared OpenCV pipeline
in common.py, so Sobel and PaDiM are compared under identical conditions.
LocalSobelDetector is the earlier from-scratch version, kept only as an independent reference.
"""
from __future__ import annotations

import cv2
import numpy as np

from padim import PaDiMDetector  # 실제 ResNet18 기반 PaDiM을 공통 인터페이스로 제공한다.

from sobel_detector import SobelDetector as TeamSobelDetector  # team module, source: branch sobel-module


# ==========================================================================
# Earlier from-scratch Sobel (reference only; the app uses the team module via SobelDetector below)
# ==========================================================================
SOBEL_KX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
SOBEL_KY = SOBEL_KX.T.copy()


def conv3x3(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """3x3 correlation with reflected borders (same convention as cv2.Sobel)."""
    p = np.pad(img, 1, mode="reflect")
    h, w = img.shape
    out = np.zeros((h, w), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            out += kernel[i, j] * p[i : i + h, j : j + w]
    return out


def sobel_gray(img_bgr: np.ndarray) -> np.ndarray:
    """Gaussian-smoothed grayscale image: the input of the Sobel filter."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.GaussianBlur(gray, (5, 5), 1.0)


def sobel_gradients(img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = sobel_gray(img_bgr)
    gx, gy = conv3x3(g, SOBEL_KX), conv3x3(g, SOBEL_KY)
    return gx, gy, np.sqrt(gx**2 + gy**2)


class LocalSobelDetector:
    """Earlier version: candidate score S = |G - mu_G| / (sigma_G + eps) on a 0..255 grayscale image.
    Not used by the app any more (see SobelDetector below); kept to cross-check the team module."""

    name = "Sobel"
    score_range = 100.0       # upper end of the threshold sliders (scores on defects can exceed 100)
    default_threshold = 3.0   # TODO: fix with the normal validation images
    eps = 1.0                 # numerical-stability constant, TODO: tune on the data
    is_placeholder = False
    supports_cache = True     # statistics can be saved/loaded (see save/load below)
    cache_version = "1"       # bump when the statistics change meaning (e.g. different smoothing)

    def __init__(self) -> None:
        self.mu: np.ndarray | None = None
        self.sigma: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return self.mu is not None

    def fit(self, normal_images: list[np.ndarray]) -> None:
        g = np.stack([sobel_gradients(im)[2] for im in normal_images])
        self.mu, self.sigma = g.mean(axis=0), g.std(axis=0)

    def save(self, path: str) -> None:
        if not self.fitted:
            raise RuntimeError("Nothing to save: call fit() first.")
        np.savez(path, mu=self.mu, sigma=self.sigma)

    def load(self, path: str) -> None:
        with np.load(path) as z:
            self.mu, self.sigma = z["mu"], z["sigma"]

    def edges(self, img_bgr: np.ndarray) -> np.ndarray:
        """Raw edge map G (intermediate output, NOT a defect probability)."""
        return sobel_gradients(img_bgr)[2]

    def score_map(self, img_bgr: np.ndarray) -> np.ndarray:
        g = self.edges(img_bgr)
        if self.fitted:
            s = np.abs(g - self.mu) / (self.sigma + self.eps)
        else:
            # Temporary fallback for demos when there are no normal images: normalize G by its own standard deviation.
            s = g / (float(g.std()) + self.eps)
        return s.astype(np.float32)


# ==========================================================================
# Sobel used by the app: adapter around the team module (sobel_detector.py)
# ==========================================================================
class SobelDetector:
    """The team's Sobel module behind the app's detector interface.

    What comes from the team module: the grayscale + Gaussian blur + 3x3 Sobel gradients computed with
    NumPy, the per-pixel normal mean/std (fit), and the score |G - mean| / (std + epsilon) (score).
    What does NOT come from it: its calibrate()/predict() thresholds and mask postprocessing. The app
    and evaluate.py apply the same thresholds-from-validation and OpenCV postprocessing to every method.

    Conversions done here: the app passes resized BGR uint8 images, the team module expects RGB.
    The team module needs at least 2 normal images and all images must have the same size.
    """

    name = "Sobel"
    score_range = 200.0       # upper end of the threshold sliders (scores on defects can exceed 100)
    default_threshold = 3.0   # TODO: fix with the normal validation images (evaluate.py does this)
    is_placeholder = False
    supports_cache = True     # mean/std can be saved/loaded (see save/load below)
    cache_version = "team-1"  # bump when the statistics change meaning (e.g. other blur or epsilon)

    def __init__(self, config=None) -> None:
        self.core = TeamSobelDetector(config)  # config: sobel_detector.SobelConfig or None for defaults

    @property
    def fitted(self) -> bool:
        return self.core.mean is not None

    @staticmethod
    def _rgb(img_bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def fit(self, normal_images: list[np.ndarray]) -> None:
        self.core.fit(self._rgb(im) for im in normal_images)

    def save(self, path: str) -> None:
        if not self.fitted:
            raise RuntimeError("Nothing to save: call fit() first.")
        np.savez(path, mean=self.core.mean, std=self.core.std, train_count=np.int64(self.core.train_count))

    def load(self, path: str) -> None:
        with np.load(path, allow_pickle=False) as z:
            mean = z["mean"].astype(np.float32)
            std = z["std"].astype(np.float32)
            count = int(z["train_count"])
        if mean.ndim != 2 or mean.shape != std.shape or not np.isfinite(mean).all() \
                or not np.isfinite(std).all() or (std < 0).any():
            raise ValueError("Invalid normal statistics in the cache file")
        self.core.mean, self.core.std, self.core.train_count = mean, std, count
        self.core.pixel_threshold = self.core.image_threshold = None  # the team module's own thresholds are not used
        self.core.val_count = 0

    def edges(self, img_bgr: np.ndarray) -> np.ndarray:
        """Raw Sobel edge strength (intermediate output, NOT a defect probability)."""
        return self.core.gradients(self._rgb(img_bgr))[2]

    def score_map(self, img_bgr: np.ndarray) -> np.ndarray:
        rgb = self._rgb(img_bgr)
        if self.fitted:
            s = self.core.score(rgb)[3]
        else:
            # Temporary fallback for demos when there are no normal images: normalize G by its own standard deviation.
            g = self.core.gradients(rgb)[2]
            s = g / (float(g.std()) + self.core.config.epsilon)
        return s.astype(np.float32)
