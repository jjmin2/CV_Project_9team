"""동일 CPU에서 전처리부터 후처리까지 배치 1의 반복 경과 시간을 측정한다.

실행: python benchmark.py --category D:/mvtec/bottle --threads 2
통계 준비·임계값 보정·파일 읽기·OpenCV 그림 생성은 총 추론 시간에서 제외해 별도 기록한다.
실제 성능 비교는 사용자가 준비한 원본 데이터로 별도 수행한다.
"""

import argparse
import csv
import json
from pathlib import Path
import platform
import statistics
import time

import cv2
import numpy as np
import torch

from common import PostConfig, draw_contours, heatmap_overlay, mask_to_image, read_bgr, run_method, scan_mvtec
from data_split import category_split, read_split_images, split_digest
from detectors import PaDiMDetector, SobelDetector
from padim_preview import calibrate_maps


def summarize(values):
    return {"mean": statistics.fmean(values), "median": statistics.median(values),
            "std": statistics.pstdev(values), "min": min(values), "max": max(values), "n": len(values)}


def measure_runs(detector, raw, cfg, image_threshold, size=256, warmup=5, repeats=30):
    """리사이즈도 매번 실행하고, 공통 실행 함수가 반환한 단계별 시간을 모은다."""
    if warmup < 0 or repeats < 1:
        raise ValueError("워밍업은 0 이상, 반복 횟수는 1 이상이어야 합니다.")
    records = {key: [] for key in ("preprocess (ms)", "score (ms)", "postprocess (ms)",
                                   "total = preprocess + score + postprocess (ms)",
                                   "render, OpenCV drawing only (ms)")}
    for i in range(warmup + repeats):
        result = run_method(detector, raw, cfg, image_threshold, size)
        start = time.perf_counter()
        heatmap_overlay(result.input_image, result.score_map, max(2 * cfg.threshold, 1e-6))
        draw_contours(result.input_image, result.loc.contours)
        mask_to_image(result.loc.mask)
        render_ms = (time.perf_counter() - start) * 1000
        if i >= warmup:
            for key, value in zip(records, (result.preprocess_ms, result.score_ms, result.postprocess_ms, result.ms, render_ms)):
                records[key].append(value)
    return records


def main():
    parser = argparse.ArgumentParser(description="전처리~후처리 공통 CPU 측정")
    parser.add_argument("--category", required=True, help="원본 MVTec 카테고리 폴더")
    parser.add_argument("--split-file", default=None, help="공통 분할 파일, 생략 시 splits/카테고리.json")
    parser.add_argument("--size", type=int, default=256, choices=[256], help="공통 입력 크기")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-normal", type=int, default=None, help="기존 호환용: 고정 통계 목록을 줄일 수 없습니다")
    parser.add_argument("--n-images", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--threads", type=int, default=2, help="OpenCV와 PyTorch에 공통 적용")
    parser.add_argument("--out", default="benchmark_results.csv")
    args = parser.parse_args()
    if args.warmup < 0 or min(args.repeats, args.n_images, args.threads) < 1:
        parser.error("반복 횟수·이미지 수·스레드는 1 이상, 워밍업은 0 이상이어야 합니다.")
    cv2.setNumThreads(args.threads)
    torch.set_num_threads(args.threads)
    category = scan_mvtec(args.category)
    if category is None:
        parser.error("MVTec 카테고리 폴더를 확인하세요.")
    split = category_split(category, args.split_file, args.seed, args.val_fraction)
    if args.max_normal is not None and args.max_normal < len(split["statistics_images"]):
        parser.error("최종 분할의 통계 이미지 수를 줄일 수 없습니다. --max-normal을 생략하세요.")
    normal, _ = read_split_images(category.root, split, "statistics_images")
    validation, _ = read_split_images(category.root, split, "validation_images")
    paths = [str(Path(category.root) / p) for p in split["test_images"]]
    if not paths:
        parser.error("고정 분할에 테스트 이미지가 없습니다.")
    ids = np.unique(np.linspace(0, len(paths) - 1, min(args.n_images, len(paths))).round().astype(int))
    test_paths = [paths[i] for i in ids]
    rows = []
    detectors, settings = [PaDiMDetector(), SobelDetector()], {}
    for det in detectors:
        start = time.perf_counter()
        det.fit(normal)
        rows.append((det.name, "statistics preparation (s)", summarize([time.perf_counter() - start])))
        start = time.perf_counter()
        thresholds = calibrate_maps(det.score_map(image) for image in validation)
        rows.append((det.name, "threshold calibration (s)", summarize([time.perf_counter() - start])))
        settings[det.name] = (PostConfig(threshold=thresholds["pixel"]), thresholds["image"])
    records = {det.name: {} for det in detectors}
    reads = []
    for path in test_paths:
        start = time.perf_counter()
        raw = read_bgr(path)
        reads.append((time.perf_counter() - start) * 1000)
        if raw is None:
            raise ValueError(f"고정 테스트 이미지를 읽을 수 없습니다: {path}")
        for det in detectors:
            cfg, threshold = settings[det.name]
            result = measure_runs(det, raw, cfg, threshold, args.size, args.warmup, args.repeats)
            for metric, values in result.items():
                records[det.name].setdefault(metric, []).extend(values)
    rows.append(("shared", "file read + decode (ms)", summarize(reads)))
    for name, metrics in records.items():
        rows.extend((name, metric, summarize(values)) for metric, values in metrics.items())
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["scope", "metric", "mean", "median", "std", "min", "max", "n"])
        for name, metric, stats in rows:
            writer.writerow([name, metric, *[stats[key] for key in ("mean", "median", "std", "min", "max", "n")]])
            print(f"{name} | {metric} | 평균 {stats['mean']:.3f} | 중앙값 {stats['median']:.3f} | n={stats['n']}")
    metadata = {"split_id": split_digest(split), "split": split, "test_images": [split["test_images"][i] for i in ids],
                "python": platform.python_version(), "torch": torch.__version__, "opencv": cv2.__version__,
                "cpu": platform.processor(), "torch_threads": torch.get_num_threads(), "opencv_threads": cv2.getNumThreads(),
                "warmup": args.warmup, "repeats": args.repeats, "batch_size": 1, "image_size": args.size,
                "total_scope": "공통 전처리 + 특징/엣지 및 점수·판정 계산 + 후처리",
                "excluded": "통계 준비, 임계값 보정, 파일 읽기·디코딩, 화면 렌더링"}
    target.with_suffix(".metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"분할 ID: {metadata['split_id']} · CSV 저장: {target}")


if __name__ == "__main__":
    main()
