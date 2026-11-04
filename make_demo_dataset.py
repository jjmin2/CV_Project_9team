"""Create a tiny SYNTHETIC dataset in the MVTec AD folder layout.

It exists only to try the app and benchmark.py before the real MVTec AD data is available.
Anything measured on it is NOT an experimental result for the project.

Usage:
    python make_demo_dataset.py                 # writes demo_data/toy
    python make_demo_dataset.py D:\\data\\toy    # or any folder you like

Layout produced (same as MVTec AD):
    toy/train/good/000.png ...
    toy/test/good/000.png ...            toy/test/scratch/000.png ...     toy/test/blob/000.png ...
    toy/ground_truth/scratch/000_mask.png ...    toy/ground_truth/blob/000_mask.png ...
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

SIZE = 320
N_TRAIN, N_TEST = 30, 6


def save_png(path: str, img: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"Could not encode {path}")
    buf.tofile(path)  # tofile() also works for non-ASCII paths on Windows


def normal_image(rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    img = 110 + 25 * (xx / SIZE) + 10 * np.sin(yy / 40.0)
    img = np.repeat(img[..., None], 3, axis=2)
    cv2.circle(img, (SIZE // 2, SIZE // 2), SIZE // 4, (60, 60, 60), 4)  # fixed "object" outline
    img += rng.normal(0, 3.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def add_scratch(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros(img.shape[:2], np.uint8)
    x1, y1 = (int(v) for v in rng.integers(30, SIZE - 30, 2))
    ang, length = rng.uniform(0, np.pi), int(rng.integers(60, 120))
    x2 = int(np.clip(x1 + length * np.cos(ang), 5, SIZE - 5))
    y2 = int(np.clip(y1 + length * np.sin(ang), 5, SIZE - 5))
    cv2.line(img, (x1, y1), (x2, y2), (225, 225, 225), 3)
    cv2.line(mask, (x1, y1), (x2, y2), 255, 7)
    return mask


def add_blob(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros(img.shape[:2], np.uint8)
    cx, cy = (int(v) for v in rng.integers(50, SIZE - 50, 2))
    axes = (int(rng.integers(12, 28)), int(rng.integers(8, 20)))
    angle = int(rng.integers(0, 180))
    cv2.ellipse(img, (cx, cy), axes, angle, 0, 360, (35, 35, 35), -1)
    cv2.ellipse(mask, (cx, cy), (axes[0] + 2, axes[1] + 2), angle, 0, 360, 255, -1)
    return mask


def main() -> None:
    root = os.path.normpath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join("demo_data", "toy")
    rng = np.random.default_rng(0)

    for i in range(N_TRAIN):
        save_png(os.path.join(root, "train", "good", f"{i:03d}.png"), normal_image(rng))
    for i in range(N_TEST):
        save_png(os.path.join(root, "test", "good", f"{i:03d}.png"), normal_image(rng))
    for name, add in (("scratch", add_scratch), ("blob", add_blob)):
        for i in range(N_TEST):
            img = normal_image(rng)
            mask = add(img, rng)
            save_png(os.path.join(root, "test", name, f"{i:03d}.png"), img)
            save_png(os.path.join(root, "ground_truth", name, f"{i:03d}_mask.png"), mask)

    root_abs = os.path.abspath(root)
    print(f"Synthetic demo dataset written to: {root_abs}")
    print(f"  train/good: {N_TRAIN} images | test/good: {N_TEST} | test/scratch: {N_TEST} | test/blob: {N_TEST}")
    print("Use it in the app: paste the folder above into 'MVTec AD category folder' (sidebar).")
    print(f'Benchmark it:      python benchmark.py --category "{root_abs}"')


if __name__ == "__main__":
    main()
