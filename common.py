"""Shared functions for both methods: preprocessing, OpenCV postprocessing, IoU, display helpers.

Kept independent of Streamlit so it can be reused by the evaluation and benchmark scripts
(AUROC, IoU, CPU timing) without opening the UI.
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

INPUT_SIZE = 256  # common input resolution for both methods


# --------------------------------------------------------------------------
# Reading and preprocessing
# --------------------------------------------------------------------------
def load_bgr(file_bytes: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not read the image. Please upload a valid PNG/JPG file.")
    return img


def load_mask(file_bytes: bytes) -> np.ndarray:
    """Ground-truth mask -> binary 0/1 array."""
    m = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise ValueError("Could not read the mask. Please upload a valid PNG/JPG file.")
    return (m > 0).astype(np.uint8)


def preprocess(img_bgr: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    return cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)


def preprocess_mask(mask: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    return cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)


# --------------------------------------------------------------------------
# OpenCV postprocessing (shared by PaDiM and Sobel)
# --------------------------------------------------------------------------
@dataclass
class PostConfig:
    blur_sigma: float = 4.0   # smooth the score map before thresholding
    threshold: float = 3.0    # per-pixel score threshold (to be fixed from normal data)
    morph_kernel: int = 5     # kernel size of the morphological opening/closing
    min_area: int = 50        # minimum region area (pixels)


@dataclass
class Localization:
    raw_mask: np.ndarray                      # after thresholding, BEFORE postprocessing
    mask: np.ndarray                          # AFTER morphology + removal of small regions
    contours: list = field(default_factory=list)


def postprocess(score_map: np.ndarray, cfg: PostConfig) -> Localization:
    smooth = (
        cv2.GaussianBlur(score_map, (0, 0), cfg.blur_sigma)
        if cfg.blur_sigma > 0
        else score_map
    )
    raw = (smooth >= cfg.threshold).astype(np.uint8)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.morph_kernel, cfg.morph_kernel))
    m = cv2.morphologyEx(raw, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    clean = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= cfg.min_area:
            clean[labels == i] = 1

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return Localization(raw_mask=raw, mask=clean, contours=list(contours))


# --------------------------------------------------------------------------
# Run one method and time it
# --------------------------------------------------------------------------
@dataclass
class MethodResult:
    score_map: np.ndarray
    image_score: float
    is_defective: bool
    loc: Localization
    ms: float  # 공통 전처리부터 점수·판정 계산과 후처리까지의 총 경과 시간
    preprocess_ms: float
    score_ms: float
    postprocess_ms: float
    input_image: np.ndarray


def run_method(detector, img_bgr: np.ndarray, cfg: PostConfig, image_threshold: float,
               size: int = INPUT_SIZE) -> MethodResult:
    """파일 디코딩이 끝난 원본부터 전처리·점수·후처리를 같은 범위로 측정한다."""
    t0 = time.perf_counter()
    image = preprocess(img_bgr, size)
    t1 = time.perf_counter()
    score_map = detector.score_map(image)
    image_score = float(score_map.max())
    is_defective = image_score >= image_threshold
    t2 = time.perf_counter()
    loc = postprocess(score_map, cfg)
    t3 = time.perf_counter()
    return MethodResult(score_map, image_score, is_defective, loc, (t3 - t0) * 1000,
                        (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000, image)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def iou(pred: np.ndarray, gt: np.ndarray) -> float | None:
    """IoU = intersection / union. Returns None when both masks are empty (IoU undefined)."""
    p, g = pred.astype(bool), gt.astype(bool)
    union = np.logical_or(p, g).sum()
    if union == 0:
        return None
    return float(np.logical_and(p, g).sum() / union)


# --------------------------------------------------------------------------
# Display (returns BGR or uint8 grayscale images)
# --------------------------------------------------------------------------
def heatmap_overlay(img_bgr: np.ndarray, score_map: np.ndarray, vmax: float, alpha: float = 0.5) -> np.ndarray:
    norm = np.clip(score_map / max(vmax, 1e-6), 0, 1)
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 1 - alpha, color, alpha, 0)


def draw_contours(img_bgr: np.ndarray, contours: list) -> np.ndarray:
    out = img_bgr.copy()
    cv2.drawContours(out, contours, -1, (0, 255, 0), 2)
    return out


def mask_to_image(mask: np.ndarray) -> np.ndarray:
    return (mask.astype(np.uint8) * 255)


def gray_to_image(x: np.ndarray) -> np.ndarray:
    """Normalize a continuous map to 0-255, for display only."""
    x = x.astype(np.float32)
    x = (x - x.min()) / max(float(x.max() - x.min()), 1e-6)
    return (x * 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Files, normal images, statistics cache, MVTec AD folder layout
# (used by the dataset browser in app.py, the statistics cache, and benchmark.py / evaluate.py)
# --------------------------------------------------------------------------
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
CACHE_VERSION = "1"  # bump if the cache file layout or key recipe changes


def clean_path(p: str) -> str:
    """Strip spaces and the quotes Windows adds with 'Copy as path'."""
    p = (p or "").strip().strip('"').strip("'").strip()
    return os.path.normpath(p) if p else ""


def list_images(folder: str) -> list[str]:
    """Sorted image paths directly inside folder (empty list if it does not exist)."""
    if not folder or not os.path.isdir(folder):
        return []
    names = [n for n in os.listdir(folder) if n.lower().endswith(IMAGE_EXTS)]
    return [os.path.join(folder, n) for n in sorted(names)]


def read_bgr(path: str) -> np.ndarray | None:
    """Read an image file. np.fromfile + imdecode so non-ASCII paths also work on Windows."""
    try:
        return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    except (OSError, cv2.error):
        return None


def read_mask_file(path: str) -> np.ndarray | None:
    """Read a ground-truth mask file -> binary 0/1 array (None if unreadable)."""
    try:
        m = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)
    except (OSError, cv2.error):
        return None
    return None if m is None else (m > 0).astype(np.uint8)


def load_normal_images(folder: str, size: int, max_images: int) -> tuple[list[np.ndarray], list[str]]:
    """Read and resize up to max_images normal images. Returns (images, paths actually used)."""
    imgs, used = [], []
    for p in list_images(folder)[:max_images]:
        im = read_bgr(p)
        if im is not None:
            imgs.append(preprocess(im, size))
            used.append(p)
    return imgs, used


def stats_signature(paths: list[str], size: int, tag: str = "") -> str:
    """Short hash of (normal image set, resolution, detector tag).

    Changes when any image is added, removed or modified, or when the resolution changes,
    so a stale cache is never reused."""
    h = hashlib.sha1()
    h.update(f"{CACHE_VERSION}|{tag}|{size}|{len(paths)}".encode())
    for p in paths:
        st = os.stat(p)
        # 같은 이름·크기·수정 시간을 가진 다른 카테고리의 캐시 충돌을 방지한다.
        h.update(f"|{os.path.abspath(p)}|{st.st_size}|{st.st_mtime_ns}".encode("utf-8", "surrogatepass"))
    return h.hexdigest()[:16]


def fit_or_load(det, imgs: list[np.ndarray], paths: list[str], size: int, cache_dir: str) -> dict:
    """Load the detector's saved statistics if a matching cache file exists, otherwise fit and save.

    Returns {"source": "none" | "cache" | "fit" | "fit+saved", "seconds": float}.
    A detector opts in with det.supports_cache = True and det.save(path) / det.load(path)."""
    if not imgs:
        return {"source": "none", "seconds": 0.0}
    t0 = time.perf_counter()
    cache_file = None
    if getattr(det, "supports_cache", False):
        tag = f"{det.name}-{getattr(det, 'cache_version', '1')}"
        cache_file = os.path.join(cache_dir, f"{det.name}_{stats_signature(paths, size, tag)}.npz")
        if os.path.isfile(cache_file):
            try:
                det.load(cache_file)
                return {"source": "cache", "seconds": time.perf_counter() - t0}
            except Exception:
                pass  # unreadable or incompatible cache file: refit below
    det.fit(imgs)
    seconds = time.perf_counter() - t0
    source = "fit"
    if cache_file is not None:
        tmp = cache_file + ".tmp.npz"
        try:
            os.makedirs(cache_dir, exist_ok=True)
            det.save(tmp)
            os.replace(tmp, cache_file)  # atomic: never leaves a half-written cache file
            source = "fit+saved"
        except Exception:
            pass
    return {"source": source, "seconds": seconds}


@dataclass
class MvtecCategory:
    root: str
    train_good: str | None   # <root>/train/good if it exists
    test: dict               # defect type -> list of image paths ("good" first)


def scan_mvtec(root: str) -> MvtecCategory | None:
    """Recognize the MVTec AD layout: <root>/train/good, <root>/test/<type>/, <root>/ground_truth/<type>/."""
    test_dir = os.path.join(root, "test") if root else ""
    if not test_dir or not os.path.isdir(test_dir):
        return None
    types = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    types.sort(key=lambda t: (t != "good", t))
    test = {t: list_images(os.path.join(test_dir, t)) for t in types}
    test = {t: v for t, v in test.items() if v}
    if not test:
        return None
    tg = os.path.join(root, "train", "good")
    return MvtecCategory(root, tg if os.path.isdir(tg) else None, test)


def mvtec_mask_path(cat: MvtecCategory, defect: str, img_path: str) -> str | None:
    """Ground-truth mask of a test image (MVTec names it <stem>_mask.png). None for 'good' or if missing."""
    if defect == "good":
        return None
    gt_dir = os.path.join(cat.root, "ground_truth", defect)
    stem = os.path.splitext(os.path.basename(img_path))[0]
    for name in (f"{stem}_mask.png", f"{stem}_mask.jpg", f"{stem}.png"):
        p = os.path.join(gt_dir, name)
        if os.path.isfile(p):
            return p
    return None
