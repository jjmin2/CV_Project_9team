"""업로드한 이미지의 실제 추론과 정상 데이터 기반 모델 준비를 담당한다."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common import (
    PostConfig, fit_or_load, load_bgr, load_mask, preprocess, preprocess_mask,
    run_method, scan_mvtec, stats_signature,
)
from data_split import category_split, read_split_images, split_digest
from detectors import PaDiMDetector, SobelDetector
from padim_preview import calibrate_maps


ROOT = Path(__file__).resolve().parent
DEFAULT_CATEGORY = ROOT / "bottle"
CACHE_DIR = ROOT / "stats_cache"


def model_signature(category_root=DEFAULT_CATEGORY, regularization=0.01):
    """정상 통계·검증 파일과 설정이 바뀌면 준비된 모델을 갱신한다."""
    category = scan_mvtec(str(category_root))
    if category is None:
        raise ValueError("bottle 원본의 train/good, test 폴더를 확인하세요.")
    split = category_split(category)
    paths = [str(Path(category.root) / p) for p in split["statistics_images"] + split["validation_images"]]
    version = PaDiMDetector(regularization=regularization).cache_version
    return stats_signature(paths, 256, f"upload-1|{split_digest(split)}|{version}")


def prepare_models(category_root=DEFAULT_CATEGORY, regularization=0.01, cache_dir=CACHE_DIR, split_path=None):
    """정상 학습·검증만 사용하며 테스트 이미지와 정답은 모델 준비에 읽지 않는다."""
    category = scan_mvtec(str(category_root))
    if category is None:
        raise ValueError("bottle 원본 데이터 폴더를 확인하세요.")
    split = category_split(category, split_path)
    normal, paths = read_split_images(category.root, split, "statistics_images")
    validation, _ = read_split_images(category.root, split, "validation_images")
    detectors = [PaDiMDetector(regularization=regularization), SobelDetector()]
    thresholds, configs, sources = {}, {}, {}
    for detector in detectors:
        sources[detector.name] = fit_or_load(detector, normal, paths, 256, str(cache_dir))
        thresholds[detector.name] = calibrate_maps(detector.score_map(image) for image in validation)
        configs[detector.name] = PostConfig(threshold=thresholds[detector.name]["pixel"])
    return {"detectors": detectors, "thresholds": thresholds, "configs": configs,
            "sources": sources, "split": split, "split_id": split_digest(split)}


@dataclass
class Inspection:
    image: np.ndarray
    ground_truth: np.ndarray | None
    results: dict
    original_shape: tuple


def decode_upload(image_bytes, mask_bytes=None):
    """파일명 대신 업로드 바이트를 읽으며 정답 마스크의 원본 크기를 확인한다."""
    raw = load_bgr(image_bytes)
    mask = load_mask(mask_bytes) if mask_bytes is not None else None
    if mask is not None and mask.shape != raw.shape[:2]:
        raise ValueError("정답 마스크는 제품 원본 이미지와 같은 가로·세로 크기여야 합니다.")
    return raw, mask


def analyze_upload(image_bytes, models, mask_bytes=None):
    """처음 보는 이미지도 같은 통계와 고정 임계값으로 두 방법에 전달한다.

    정답 마스크는 비교 표시에만 사용하며 점수·판정에는 사용하지 않는다.
    파일 읽기·디코딩은 모델별 전처리~후처리 시간에 포함하지 않는다.
    """
    raw, mask = decode_upload(image_bytes, mask_bytes)
    results = {}
    for detector in models["detectors"]:
        name = detector.name
        results[name] = run_method(detector, raw, models["configs"][name], models["thresholds"][name]["image"])
    return Inspection(preprocess(raw, 256), preprocess_mask(mask, 256) if mask is not None else None,
                      results, raw.shape[:2])
