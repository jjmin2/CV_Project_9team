"""CPU timing benchmark for DefectVision: batch size 1, repeated runs after warm-up.

Usage:
    python benchmark.py --category "D:\\mvtec\\bottle"
    python benchmark.py --category "D:\\mvtec\\bottle" --size 256 --n-images 6 --warmup 5 --repeats 30

Each stage is measured separately, as the proposal requires (milliseconds per image unless noted):
  statistics estimation (s)   one-off fit on the normal images (always a fresh fit; the cache is not used)
  file read                   reading one image file from disk (one read per image; the OS cache may apply)
  resize                      resizing to the common input size
  score                       detector.score_map(): feature extraction / gradients and scoring
  postprocess                 OpenCV postprocessing: blur, threshold, morphology, connected components, contours
  total = score + postprocess this is what the app shows as "CPU time"
  render                      OpenCV drawing for display only (heatmap, contours, mask); Streamlit and
                              browser rendering time are NOT included

Settings used: default postprocessing parameters. Run it on the same machine and with the same
--size / --threads for both methods. Results are written to a CSV file (default benchmark_results.csv).
"""
from __future__ import annotations

import argparse
import csv
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
    draw_contours,
    heatmap_overlay,
    load_normal_images,
    mask_to_image,
    postprocess,
    preprocess,
    read_bgr,
    scan_mvtec,
)
from detectors import PaDiMDetector, SobelDetector


def ms(t0: float, t1: float) -> float:
    return (t1 - t0) * 1000.0


def summarize(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def pick_test_images(category, n: int) -> list[str]:
    """n test images spread evenly over the whole test set (normal and defective)."""
    items = [p for paths in category.test.values() for p in paths]
    if n >= len(items):
        return items
    idx = np.unique(np.linspace(0, len(items) - 1, n).round().astype(int))
    return [items[i] for i in idx]


def main() -> None:
    ap = argparse.ArgumentParser(description="CPU timing benchmark for PaDiM vs Sobel.")
    ap.add_argument("--category", required=True, help="MVTec AD category folder (contains train/ and test/)")
    ap.add_argument("--size", type=int, default=INPUT_SIZE, help="input resolution (default %(default)s)")
    ap.add_argument("--max-normal", type=int, default=100, help="max normal images for the statistics")
    ap.add_argument("--n-images", type=int, default=6, help="number of test images to time")
    ap.add_argument("--warmup", type=int, default=5, help="warm-up runs per image and method (not counted)")
    ap.add_argument("--repeats", type=int, default=30, help="timed runs per image and method")
    ap.add_argument("--threads", type=int, default=None, help="OpenCV thread count (default: OpenCV's own)")
    ap.add_argument("--out", default="benchmark_results.csv", help="CSV output file")
    args = ap.parse_args()
    if args.warmup < 0 or args.repeats < 1 or args.n_images < 1:
        sys.exit("--warmup must be >= 0, --repeats and --n-images must be >= 1")

    if args.threads is not None:
        cv2.setNumThreads(args.threads)

    category = scan_mvtec(clean_path(args.category))
    if category is None:
        sys.exit(f"Not an MVTec AD category folder (expected test/<type>/ with images): {args.category}")
    if not category.train_good:
        sys.exit("The category folder has no train/good folder to estimate normal statistics from.")

    print(f"Python {platform.python_version()} | OpenCV {cv2.__version__} | NumPy {np.__version__}")
    print(f"CPU: {platform.processor() or platform.machine()} | logical cores: {os.cpu_count()} "
          f"| OpenCV threads: {cv2.getNumThreads()}")
    print(f"Input size {args.size} | warm-up {args.warmup} | timed repeats {args.repeats} | batch size 1")

    imgs, used = load_normal_images(category.train_good, args.size, args.max_normal)
    if not imgs:
        sys.exit("No readable images in train/good.")
    test_paths = pick_test_images(category, args.n_images)
    print(f"{len(imgs)} normal images for statistics, {len(test_paths)} test images timed\n")

    detectors = [PaDiMDetector(), SobelDetector()]
    cfg = PostConfig()

    # 1) one-off statistics estimation (always a fresh fit)
    fit_s = {}
    for d in detectors:
        t0 = time.perf_counter()
        d.fit(imgs)
        fit_s[d.name] = time.perf_counter() - t0

    # 2) per-image timing
    shared = {"file read (ms)": [], "resize (ms)": []}
    per = {d.name: {"score (ms)": [], "postprocess (ms)": [], "total = score + postprocess (ms)": [],
                    "render, OpenCV drawing only (ms)": []} for d in detectors}
    for path in test_paths:
        t0 = time.perf_counter()
        raw = read_bgr(path)
        t1 = time.perf_counter()
        if raw is None:
            print(f"Skipping unreadable image: {path}")
            continue
        img = preprocess(raw, args.size)
        t2 = time.perf_counter()
        shared["file read (ms)"].append(ms(t0, t1))
        shared["resize (ms)"].append(ms(t1, t2))

        for d in detectors:
            rec = per[d.name]
            for i in range(args.warmup + args.repeats):
                a = time.perf_counter()
                score_map = d.score_map(img)
                b = time.perf_counter()
                loc = postprocess(score_map, cfg)
                c = time.perf_counter()
                heatmap_overlay(img, score_map, d.score_range)
                draw_contours(img, loc.contours)
                mask_to_image(loc.mask)
                e = time.perf_counter()
                if i >= args.warmup:
                    rec["score (ms)"].append(ms(a, b))
                    rec["postprocess (ms)"].append(ms(b, c))
                    rec["total = score + postprocess (ms)"].append(ms(a, c))
                    rec["render, OpenCV drawing only (ms)"].append(ms(c, e))

    # 3) report
    rows = []  # (scope, metric, stats)
    for d in detectors:
        rows.append((d.name, "statistics estimation (s)", summarize([fit_s[d.name]])))
    for metric, vals in shared.items():
        if vals:
            rows.append(("shared", metric, summarize(vals)))
    for d in detectors:
        for metric, vals in per[d.name].items():
            if vals:
                rows.append((d.name, metric, summarize(vals)))

    print(f"{'Scope':<8} {'Metric':<36} {'mean':>9} {'median':>9} {'std':>9} {'min':>9} {'max':>9} {'n':>6}")
    print("-" * 100)
    for scope, metric, s in rows:
        print(f"{scope:<8} {metric:<36} {s['mean']:9.3f} {s['median']:9.3f} {s['std']:9.3f} "
              f"{s['min']:9.3f} {s['max']:9.3f} {s['n']:6d}")
    for d in detectors:
        if d.is_placeholder:
            print(f"\nNote: {d.name} is still a placeholder, so its timings are NOT meaningful yet.")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scope", "metric", "mean", "median", "std", "min", "max", "n"])
        for scope, metric, s in rows:
            w.writerow([scope, metric] + [round(s[k], 4) for k in ("mean", "median", "std", "min", "max")] + [s["n"]])
    print(f"\nSaved: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
