"""원본 이미지 없이 저장된 특징으로 PaDiM을 학습·검증·시각화하는 보조 함수."""

from pathlib import Path

import cv2
import numpy as np
import torch

from common import stats_signature
from padim import PaDiMDetector
from data_split import get_or_create_split, image_id, split_digest


def load_features(path, normal_only=False):
    """팀 노트북의 저장 형식과 채널·라벨·마스크 크기를 확인한다."""
    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    required = {"emb", "idx", "labels", "masks", "paths"}
    if not isinstance(saved, dict) or not required.issubset(saved):
        raise ValueError("특징 파일에는 emb, idx, labels, masks, paths가 필요합니다.")
    emb, labels, masks = saved["emb"], saved["labels"], saved["masks"]
    if not isinstance(emb, torch.Tensor) or emb.ndim != 4 or min(emb.shape) < 1:
        raise ValueError("특징은 비어 있지 않은 (N,D,H,W) 텐서여야 합니다.")
    n, channels, _, _ = emb.shape
    saved["idx"] = PaDiMDetector._indices(saved["idx"], channels)
    if not isinstance(labels, torch.Tensor) or labels.shape != (n,) or not ((labels == 0) | (labels == 1)).all():
        raise ValueError("각 특징에 정상 0 또는 불량 1 라벨이 필요합니다.")
    if normal_only and (labels != 0).any():
        raise ValueError("정상 학습 특징 파일에 불량 라벨이 포함되어 있습니다.")
    if not isinstance(masks, torch.Tensor) or masks.ndim != 4 or masks.shape[:2] != (n, 1) or min(masks.shape[2:]) < 1:
        raise ValueError("정답 마스크는 (N,1,H,W) 텐서여야 합니다.")
    if not torch.isfinite(masks).all():
        raise ValueError("정답 마스크에 유한하지 않은 값이 있습니다.")
    if len(saved["paths"]) != n or not all(isinstance(p, str) for p in saved["paths"]):
        raise ValueError("특징 수와 원본 이미지 경로 수가 일치해야 합니다.")
    return saved


def prepare_preview(train_path, test_path, cache_dir, regularization=0.01, seed=0, split_path=None):
    """정상 특징만 분할하여 학습하고, 테스트 특징은 시각화용으로 보관한다."""
    train = load_features(train_path, normal_only=True)
    test = load_features(test_path)
    if tuple(train["emb"].shape[1:]) != tuple(test["emb"].shape[1:]) or not torch.equal(train["idx"], test["idx"]):
        raise ValueError("학습·테스트 특징의 크기 또는 채널 순서가 다릅니다.")
    n = len(train["emb"])
    if n < 3:
        raise ValueError("정상 특징은 통계 추정과 검증을 위해 최소 세 장 필요합니다.")
    parts = train["paths"][0].replace("\\", "/").split("/")
    train_pos = parts.index("train")
    category = parts[train_pos - 1] if train_pos > 1 else Path(train_path).stem.replace("_train_features", "")
    split = get_or_create_split(category, train["paths"], test["paths"], split_path, seed)
    train_lookup = {image_id(p): i for i, p in enumerate(train["paths"])}
    test_lookup = {image_id(p): i for i, p in enumerate(test["paths"])}
    stats_indices = [train_lookup[p] for p in split["statistics_images"]]
    val_indices = [train_lookup[p] for p in split["validation_images"]]
    # 공통 테스트 목록에 없는 추가 표본은 최종 비교에 자동 편입하지 않는다.
    test_indices = [test_lookup[p] for p in split["test_images"]]
    test = {**test, "emb": test["emb"][test_indices], "labels": test["labels"][test_indices],
            "masks": test["masks"][test_indices], "paths": [test["paths"][i] for i in test_indices]}
    n_val = len(val_indices)
    detector = PaDiMDetector(train_pt_path=train_path, regularization=regularization)
    tag = f"{detector.cache_version}|fixed-split-{split_digest(split)}"
    signature = stats_signature([str(train_path)], 256, tag)
    cache_file = Path(cache_dir) / f"PaDiM_preview_{signature}.npz"
    source = "정상 특징으로 통계 계산"
    if cache_file.is_file():
        try:
            detector.load(cache_file)
            if not torch.equal(detector.idx, train["idx"]) or detector.train_count != len(stats_indices) \
                    or detector.regularization != regularization:
                raise ValueError("캐시 설정이 현재 학습 설정과 다릅니다.")
            source = "저장된 통계 재사용"
        except (ValueError, RuntimeError, OSError, EOFError):
            detector = PaDiMDetector(train_pt_path=train_path, regularization=regularization)
    if not detector.fitted:
        detector.fit_features(train["emb"][stats_indices], train["idx"])
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_file.with_suffix(".tmp.npz")
        detector.save(temporary)
        temporary.replace(cache_file)
    return {
        "detector": detector,
        "validation": train["emb"][val_indices],
        "test": test,
        "signature": signature,
        "source": source,
        "n_stats": len(stats_indices),
        "n_val": n_val,
        "seed": seed,
        "split": split,
    }


def calibrate_maps(maps, sigma=4.0):
    """정상 검증 점수만 사용해 이미지 99%, 평활화 픽셀 99.9% 분위수를 구한다."""
    maps = list(maps)
    if not maps:
        raise ValueError("임계값 보정에는 정상 검증 점수가 필요합니다.")
    smoothed = [cv2.GaussianBlur(m, (0, 0), sigma) if sigma > 0 else m for m in maps]
    return {
        "image": float(np.quantile([float(m.max()) for m in maps], 0.99)),
        "pixel": float(np.quantile(np.concatenate([m.ravel() for m in smoothed]), 0.999)),
    }


def calibrate_preview(bundle):
    """저장 특징을 256×256 점수 맵으로 복원한 후 공통 기준으로 보정한다."""
    detector = bundle["detector"]
    return calibrate_maps(detector.score_features(v[None], output_size=(256, 256)) for v in bundle["validation"])


def test_type(path):
    """Colab 경로와 Windows 경로 모두에서 결함 종류를 읽는다."""
    return path.replace("\\", "/").split("/")[-2]


def original_path(saved_path, category_root=""):
    """원래 경로 또는 사용자가 지정한 카테고리 폴더에서 원본을 찾는다."""
    direct = Path(saved_path)
    if direct.is_file():
        return direct
    if category_root:
        parts = saved_path.replace("\\", "/").split("/")
        # 특징 파일의 경로 전체를 이어 붙이지 않고 파일명과 결함 폴더만 사용한다.
        if len(parts) >= 2 and parts[-2] not in (".", "..") and parts[-1] not in (".", ".."):
            candidate = Path(category_root) / "test" / parts[-2] / parts[-1]
            if candidate.is_file():
                return candidate
    return None
