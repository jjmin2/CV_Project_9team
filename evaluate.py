"""Evaluate PaDiM and Sobel on a whole MVTec AD category, following the protocol in the proposal.

Usage:
    python evaluate.py --category "D:\\mvtec\\bottle"
    python evaluate.py --category "D:\\mvtec\\bottle" --size 256 --seed 0 --out-dir evaluation_output

Protocol (nothing here looks at the test set to choose a setting):
  1. train/good is split once (fixed --seed) into a STATISTICS set and a VALIDATION set.
     Both methods use exactly the same split, images, resolution and evaluation region.
  2. Each method estimates its normal statistics (fit) on the STATISTICS set only.
  3. Thresholds are fixed on the VALIDATION images only (normal images):
       image threshold = --image-quantile of the validation image scores   (image score = max of the score map)
       pixel threshold = --pixel-quantile of the validation pixel scores   (scores after Gaussian smoothing)
     Limitation: normal-only validation data cannot say how defective pixels score, so these are
     false-alarm-controlled thresholds, not thresholds optimized for IoU.
  4. The test set (test/good + every defect type, with ground-truth masks) is scored once.

Reported (overall and per defect type):
  image AUROC, pixel AUROC (raw and smoothed score map), image TPR/FPR at the fixed threshold,
  mean/median IoU on defective images before and after postprocessing, and the false-positive area
  ratio on normal images. Normal images whose predicted mask is empty have an undefined IoU and are
  counted separately. Timing is measured by benchmark.py, not here.

Memory: pixel AUROC needs roughly 70 MB per million test pixels (about 0.7 GB for a 150-image
test set at 256x256). On a small laptop lower --size or use --max-test-per-type.

Outputs in --out-dir: summary.csv, per_image.csv (use it to pick failure cases), split.json.
A method that is still a placeholder is skipped (its numbers would be meaningless) unless
--include-placeholders is given.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import sys
import time

import cv2
import numpy as np

from common import (
    INPUT_SIZE,
    PostConfig,
    clean_path,
    iou,
    list_images,
    mvtec_mask_path,
    postprocess,
    preprocess,
    preprocess_mask,
    read_bgr,
    read_mask_file,
    scan_mvtec,
)
from detectors import PaDiMDetector, SobelDetector


# --------------------------------------------------------------------------
# Small numeric helpers
# --------------------------------------------------------------------------
def auroc(scores, labels) -> float | None:
    """Area under the ROC curve (rank / Mann-Whitney form, ties get the average rank).
    Returns None when only one class is present."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel().astype(bool)
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    avg_rank = np.cumsum(counts) - (counts - 1) / 2.0  # 1-based average rank of each distinct value
    ranks = avg_rank[inv]
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def smooth(score_map: np.ndarray, sigma: float) -> np.ndarray:
    """Same Gaussian smoothing that postprocess() applies before thresholding."""
    return cv2.GaussianBlur(score_map, (0, 0), sigma) if sigma > 0 else score_map


def mean_or_none(v: list[float]) -> float | None:
    return statistics.fmean(v) if v else None


def median_or_none(v: list[float]) -> float | None:
    return statistics.median(v) if v else None


def fmt(v, nd: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


# --------------------------------------------------------------------------
# Data: split of the normal images and the test items
# --------------------------------------------------------------------------
def split_normal(paths: list[str], val_fraction: float, seed: int, max_stats: int | None):
    """Fixed random split of train/good into (statistics set, validation set)."""
    order = np.random.default_rng(seed).permutation(len(paths))
    n_val = min(max(1, int(round(len(paths) * val_fraction))), len(paths) - 1)
    val = [paths[i] for i in sorted(order[:n_val])]
    stats = [paths[i] for i in sorted(order[n_val:])]
    if max_stats:
        stats = stats[:max_stats]
    return stats, val


def read_resized(paths: list[str], size: int):
    imgs, ok = [], []
    for p in paths:
        im = read_bgr(p)
        if im is not None:
            imgs.append(preprocess(im, size))
            ok.append(p)
    return imgs, ok


def load_test_items(category, size: int, max_per_type: int | None):
    """One dict per test image: defect type, label (0 normal / 1 defective), resized image, resized GT mask."""
    items, missing_mask, unreadable = [], 0, 0
    for defect, paths in category.test.items():
        for p in (paths[:max_per_type] if max_per_type else paths):
            raw = read_bgr(p)
            if raw is None:
                unreadable += 1
                continue
            img = preprocess(raw, size)
            if defect == "good":
                gt = np.zeros(img.shape[:2], np.uint8)  # normal image: the correct mask is empty
            else:
                mp = mvtec_mask_path(category, defect, p)
                m = read_mask_file(mp) if mp else None
                gt = preprocess_mask(m, size) if m is not None else None
                if gt is None:
                    missing_mask += 1
            items.append({"defect": defect, "path": p, "label": 0 if defect == "good" else 1, "img": img, "gt": gt})
    return items, missing_mask, unreadable


# --------------------------------------------------------------------------
# One method: fit on the statistics set, fix thresholds on validation, score the test set
# --------------------------------------------------------------------------
def run_detector(det, args, stats_imgs, val_imgs, items) -> dict:
    t0 = time.perf_counter()
    det.fit(stats_imgs)
    fit_s = time.perf_counter() - t0

    sigma = args.blur_sigma
    val_maps = [det.score_map(im) for im in val_imgs]
    img_thr = float(np.quantile([float(m.max()) for m in val_maps], args.image_quantile))
    pix_thr = float(np.quantile(np.concatenate([smooth(m, sigma).ravel() for m in val_maps]), args.pixel_quantile))
    cfg = PostConfig(blur_sigma=sigma, threshold=pix_thr, morph_kernel=args.morph_kernel, min_area=args.min_area)

    rows, raw_maps, smooth_maps = [], [], []
    for it in items:
        sm = det.score_map(it["img"])
        raw_maps.append(sm)
        smooth_maps.append(smooth(sm, sigma))
        loc = postprocess(sm, cfg)
        score = float(sm.max())  # image score = maximum of the continuous score map
        gt = it["gt"]
        rows.append({
            "method": det.name,
            "defect_type": it["defect"],
            "file": os.path.basename(it["path"]),
            "label": it["label"],
            "image_score": score,
            "predicted_defective": int(score >= img_thr),
            "iou_raw": iou(loc.raw_mask, gt) if gt is not None else None,
            "iou_after": iou(loc.mask, gt) if gt is not None else None,
            "area_ratio_raw": float(loc.raw_mask.mean()),
            "area_ratio_after": float(loc.mask.mean()),
            "gt_area_ratio": None if gt is None else float(gt.mean()),
        })
    return {"fit_s": fit_s, "img_thr": img_thr, "pix_thr": pix_thr, "rows": rows,
            "raw": raw_maps, "smooth": smooth_maps}


def compute_metrics(items, res, idx: list[int]) -> dict:
    """Metrics over the test items listed in idx (a mix of normal and defective images)."""
    rows = res["rows"]
    labels = np.array([items[i]["label"] for i in idx])
    scores = np.array([rows[i]["image_score"] for i in idx])
    pred = np.array([rows[i]["predicted_defective"] for i in idx])
    m: dict = {"image AUROC": auroc(scores, labels)}

    with_gt = [i for i in idx if items[i]["gt"] is not None]
    if with_gt:
        gt_all = np.concatenate([items[i]["gt"].ravel() for i in with_gt])
        m["pixel AUROC (raw score map)"] = auroc(np.concatenate([res["raw"][i].ravel() for i in with_gt]), gt_all)
        m["pixel AUROC (smoothed score map)"] = auroc(np.concatenate([res["smooth"][i].ravel() for i in with_gt]), gt_all)
    else:
        m["pixel AUROC (raw score map)"] = m["pixel AUROC (smoothed score map)"] = None

    m["image TPR (defective flagged)"] = float(pred[labels == 1].mean()) if (labels == 1).any() else None
    m["image FPR (normal flagged)"] = float(pred[labels == 0].mean()) if (labels == 0).any() else None

    defective = [i for i in idx if items[i]["label"] == 1 and rows[i]["iou_after"] is not None]
    m["IoU before postprocessing (mean)"] = mean_or_none([rows[i]["iou_raw"] for i in defective])
    m["IoU after postprocessing (mean)"] = mean_or_none([rows[i]["iou_after"] for i in defective])
    m["IoU after postprocessing (median)"] = median_or_none([rows[i]["iou_after"] for i in defective])

    normal = [i for i in idx if items[i]["label"] == 0]
    m["false-positive area ratio, normal images (after)"] = mean_or_none([rows[i]["area_ratio_after"] for i in normal])
    m["false-positive area ratio, normal images (before)"] = mean_or_none([rows[i]["area_ratio_raw"] for i in normal])
    m["normal images with empty mask (IoU undefined)"] = sum(rows[i]["area_ratio_after"] == 0 for i in normal) if normal else None
    return m


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate PaDiM and Sobel on one MVTec AD category.")
    ap.add_argument("--category", required=True, help="MVTec AD category folder (train/, test/, ground_truth/)")
    ap.add_argument("--size", type=int, default=INPUT_SIZE, help="input resolution (default %(default)s)")
    ap.add_argument("--seed", type=int, default=0, help="seed of the train/good split (default %(default)s)")
    ap.add_argument("--val-fraction", type=float, default=0.2, help="share of train/good kept for validation")
    ap.add_argument("--max-normal", type=int, default=None, help="cap on statistics images (default: all)")
    ap.add_argument("--image-quantile", type=float, default=0.99, help="validation quantile for the image threshold")
    ap.add_argument("--pixel-quantile", type=float, default=0.999, help="validation quantile for the pixel threshold")
    ap.add_argument("--blur-sigma", type=float, default=4.0, help="Gaussian sigma of the postprocessing")
    ap.add_argument("--morph-kernel", type=int, default=5, help="morphology kernel size of the postprocessing")
    ap.add_argument("--min-area", type=int, default=50, help="minimum region area (px) kept by the postprocessing")
    ap.add_argument("--max-test-per-type", type=int, default=None, help="quick runs: cap test images per defect type")
    ap.add_argument("--include-placeholders", action="store_true", help="also run methods that are still placeholders")
    ap.add_argument("--out-dir", default="evaluation_output", help="folder for summary.csv, per_image.csv, split.json")
    args = ap.parse_args()

    if not (0.0 < args.val_fraction < 1.0):
        sys.exit("--val-fraction must be between 0 and 1")
    for q in (args.image_quantile, args.pixel_quantile):
        if not (0.0 < q <= 1.0):
            sys.exit("--image-quantile and --pixel-quantile must be in (0, 1]")

    category = scan_mvtec(clean_path(args.category))
    if category is None:
        sys.exit(f"Not an MVTec AD category folder (expected test/<type>/ with images): {args.category}")
    if not category.train_good:
        sys.exit("The category folder has no train/good folder.")
    normal_paths = list_images(category.train_good)
    if len(normal_paths) < 3:
        sys.exit("Need at least 3 images in train/good to make a statistics/validation split.")

    detectors, skipped = [], []
    for d in (PaDiMDetector(), SobelDetector()):
        if d.is_placeholder and not args.include_placeholders:
            skipped.append(d.name)
        else:
            detectors.append(d)
    if not detectors:
        sys.exit("Nothing to evaluate: every method is still a placeholder.")

    stats_paths, val_paths = split_normal(normal_paths, args.val_fraction, args.seed, args.max_normal)
    stats_imgs, stats_ok = read_resized(stats_paths, args.size)
    val_imgs, val_ok = read_resized(val_paths, args.size)
    if not stats_imgs or not val_imgs:
        sys.exit("Could not read the normal images of the split.")
    items, missing_mask, unreadable = load_test_items(category, args.size, args.max_test_per_type)
    if not items:
        sys.exit("No readable test images.")

    n_normal = sum(it["label"] == 0 for it in items)
    n_defect = len(items) - n_normal
    print(f"Python {platform.python_version()} | OpenCV {cv2.__version__} | NumPy {np.__version__}")
    print(f"Category: {os.path.basename(os.path.normpath(category.root))} | input size {args.size} | seed {args.seed}")
    print(f"train/good split: {len(stats_imgs)} statistics + {len(val_imgs)} validation images")
    print(f"test set: {n_normal} normal + {n_defect} defective images "
          f"({', '.join(f'{t}: {len(v)}' for t, v in category.test.items())})")
    print(f"postprocessing: sigma {args.blur_sigma}, kernel {args.morph_kernel}, min area {args.min_area}")
    print(f"thresholds from validation: image quantile {args.image_quantile}, pixel quantile {args.pixel_quantile}")
    if missing_mask:
        print(f"WARNING: {missing_mask} defective image(s) have no ground-truth mask; they count for image-level "
              f"metrics only.")
    if unreadable:
        print(f"WARNING: {unreadable} test image(s) could not be read and were skipped.")
    for name in skipped:
        print(f"Skipping {name}: it is still a placeholder (use --include-placeholders to run it anyway).")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "split.json"), "w", encoding="utf-8") as f:
        json.dump({
            "category": os.path.basename(os.path.normpath(category.root)), "seed": args.seed,
            "val_fraction": args.val_fraction, "size": args.size,
            "statistics_images": [os.path.basename(p) for p in stats_ok],
            "validation_images": [os.path.basename(p) for p in val_ok],
        }, f, indent=1)

    good_idx = [i for i, it in enumerate(items) if it["label"] == 0]
    summary_rows, per_image_rows = [], []
    for det in detectors:
        res = run_detector(det, args, stats_imgs, val_imgs, items)
        per_image_rows += res["rows"]
        print(f"\n=== {det.name} ===  (fit {res['fit_s']:.2f} s)")
        if det.is_placeholder:
            print("NOTE: this method is a placeholder, so the numbers below are NOT meaningful.")
        print(f"thresholds fixed on validation images: image {res['img_thr']:.4f} | pixel {res['pix_thr']:.4f}")

        subsets = [("overall", list(range(len(items))))]
        for t in category.test:
            if t != "good":
                subsets.append((t, good_idx + [i for i, it in enumerate(items) if it["defect"] == t]))
        overall = None
        for name, idx in subsets:
            m = compute_metrics(items, res, idx)
            if name == "overall":
                overall = m
            for metric, v in m.items():
                summary_rows.append([det.name, name, metric, "" if v is None else round(float(v), 6), len(idx)])
        summary_rows += [[det.name, "overall", "image threshold (from validation)", round(res["img_thr"], 6), len(val_imgs)],
                         [det.name, "overall", "pixel threshold (from validation)", round(res["pix_thr"], 6), len(val_imgs)],
                         [det.name, "overall", "statistics estimation time (s)", round(res["fit_s"], 4), len(stats_imgs)]]

        width = max(len(k) for k in overall)
        for metric, v in overall.items():
            print(f"  {metric:<{width}}  {fmt(v) if not isinstance(v, int) else v}")

        if len(subsets) > 2:
            print("\n  per defect type (each against all normal test images):")
            print(f"  {'type':<14} {'n':>4} {'image AUROC':>12} {'pixel AUROC':>12} {'TPR':>7} {'IoU after':>10}")
            for name, idx in subsets[1:]:
                m = compute_metrics(items, res, idx)
                n_def = sum(items[i]["label"] == 1 for i in idx)
                print(f"  {name:<14} {n_def:>4} {fmt(m['image AUROC']):>12} {fmt(m['pixel AUROC (smoothed score map)']):>12} "
                      f"{fmt(m['image TPR (defective flagged)']):>7} {fmt(m['IoU after postprocessing (mean)']):>10}")

    with open(os.path.join(args.out_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "subset", "metric", "value", "n_images"])
        w.writerows(summary_rows)
    cols = ["method", "defect_type", "file", "label", "image_score", "predicted_defective", "iou_raw", "iou_after",
            "area_ratio_raw", "area_ratio_after", "gt_area_ratio"]
    with open(os.path.join(args.out_dir, "per_image.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in per_image_rows:
            w.writerow([("" if r[c] is None else (round(r[c], 6) if isinstance(r[c], float) else r[c])) for c in cols])

    print("\nThresholds were fixed on normal validation images only; the test set was not used for tuning.")
    print("Limitation: normal-only validation data cannot reflect how defective pixels score.")
    print(f"Saved to: {os.path.abspath(args.out_dir)}  (summary.csv, per_image.csv, split.json)")


if __name__ == "__main__":
    main()
