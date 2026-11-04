"""CPU용 PaDiM: 위치별 정상 분포와 Mahalanobis 거리 기반 이상 탐지.

필요 패키지는 torch, torchvision, numpy, opencv-python이다.
기존 특징 추출기는 이미지를 256×256으로 변환하므로 통계의 특징 격자는
64×64이다. 최종 점수 맵만 전달받은 이미지의 크기로 복원한다.
이미지 입력은 BGR uint8 배열이며 점수는 확률이 아닌 거리이다.

사용 예시::

    from padim import PaDiMDetector

    detector = PaDiMDetector()
    detector.fit_from_file()  # 정상 학습 특징과 채널 인덱스를 함께 읽는다.
    score_map = detector.score_map(image_bgr)
    detector.save("padim_statistics.pt")

또는 정상 BGR 이미지들을 detector.fit(normal_images)에 전달한다.
저장된 통계는 새 객체에서 detector.load(path)로 복원한다.
이미지 추론에는 기존 특징 추출기와 사전학습 ResNet18 가중치가 필요하다.
통계를 복원하면 저장된 채널 인덱스로 추출기를 만들 수 있다.
통계 계산 자체는 가중치가 필요 없다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import hashlib

import cv2
import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_FEATURE_PATH = Path(__file__).resolve().parent / "feature" / "bottle_train_features.pt"


class PaDiMDetector:
    """정상 특징의 표본 공분산을 정규화하고 Cholesky 분해로 거리를 계산한다.

    공분산의 역행렬을 직접 저장하는 대신 하삼각 인자를 저장한다.
    이는 같은 Mahalanobis 거리를 삼각 연립방정식으로 계산하기 위함이다.
    임계값은 정상 검증 데이터로 별도 보정해야 하며 기본값은 임시값이다.
    """

    name = "PaDiM"
    score_range = 50.0
    default_threshold = 20.0
    is_placeholder = False
    # 기존 공통 캐시는 확장자가 .npz여도 전달된 경로를 그대로 사용한다.
    supports_cache = True

    def __init__(
        self,
        train_pt_path: str | Path = DEFAULT_FEATURE_PATH,
        regularization: float = 0.01,
        spatial_batch_size: int = 64,
        feature_extractor=None,
    ) -> None:
        if not np.isfinite(regularization) or regularization <= 0:
            raise ValueError("공분산 정규화 계수는 유한한 양수여야 합니다.")
        if isinstance(spatial_batch_size, bool) or not isinstance(spatial_batch_size, int) or spatial_batch_size < 1:
            raise ValueError("공간 배치 크기는 양의 정수여야 합니다.")
        self.train_pt_path = Path(train_pt_path)
        self.regularization = float(regularization)
        self.spatial_batch_size = spatial_batch_size
        self._extractor = feature_extractor
        self.mean = None
        self.cholesky = None
        self.idx = None
        self.feature_shape = None
        self.train_count = 0

    @property
    def fitted(self) -> bool:
        """정상 통계가 준비되었는지 반환한다."""
        return self.mean is not None

    def _channel_indices(self):
        """통계·추출기의 인덱스를 우선하고, 없으면 팀의 고정 시드를 사용한다."""
        if self.idx is not None:
            return self.idx
        if self._extractor is not None:
            return torch.as_tensor(self._extractor.idx).detach().cpu()
        if self.train_pt_path.is_file():
            saved = torch.load(self.train_pt_path, map_location="cpu", weights_only=True, mmap=True)
            return torch.as_tensor(saved["idx"]).clone()
        return torch.randperm(448, generator=torch.Generator().manual_seed(1024))[:100]

    @property
    def cache_version(self):
        """가중치·전처리·채널 순서·정규화 설정이 다른 캐시를 구분한다."""
        idx = self._channel_indices()
        idx = self._indices(idx, idx.numel())
        digest = hashlib.sha256(idx.numpy().tobytes()).hexdigest()[:16]
        return f"padim-resnet18-imagenet-v1-256-cholesky-2-{self.regularization!r}-{digest}"

    @staticmethod
    def _indices(idx, channels: int) -> torch.Tensor:
        values = torch.as_tensor(idx).detach().cpu()
        if values.ndim != 1 or values.numel() != channels:
            raise ValueError("채널 인덱스 수와 특징의 채널 수가 일치해야 합니다.")
        if values.dtype == torch.bool or values.is_floating_point() or values.is_complex():
            raise ValueError("채널 인덱스는 정수여야 합니다.")
        values = values.to(torch.int64).clone()
        if channels < 1 or (values < 0).any() or (values >= 448).any() or values.unique().numel() != channels:
            raise ValueError("채널 인덱스는 0부터 447 사이의 중복 없는 정수여야 합니다.")
        return values

    def _get_extractor(self):
        if self._extractor is None:
            # 특징 파일로만 학습할 때는 모델 생성과 가중치 다운로드를 하지 않는다.
            from feature_extractor import FeatureExtractor

            self._extractor = FeatureExtractor(idx=self._channel_indices())
        if self.idx is not None:
            current = self._indices(self._extractor.idx, self.idx.numel())
            if not torch.equal(current, self.idx):
                raise ValueError("특징 추출기의 채널 인덱스가 학습 통계와 다릅니다.")
        return self._extractor

    @staticmethod
    def _image(image_bgr: np.ndarray) -> np.ndarray:
        image = np.asarray(image_bgr)
        if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1 or image.dtype != np.uint8:
            raise ValueError("이미지는 (높이, 너비, 3) 형태의 BGR uint8 배열이어야 합니다.")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def fit(self, normal_images: Iterable[np.ndarray]):
        """정상 BGR 이미지에서 특징을 추출하여 위치별 분포를 학습한다.

        한 번에 한 장씩 특징을 추출한다. 모든 특징은 CPU에 보관하므로
        이미 저장된 정상 특징이 있다면 fit_from_file 사용을 권장한다.
        """
        extractor = self._get_extractor()
        embeddings = []
        with torch.no_grad():
            for image in normal_images:
                embeddings.append(extractor.from_rgb(self._image(image)).detach().cpu())
        if len(embeddings) < 2:
            raise ValueError("정상 학습 이미지는 최소 두 장 필요합니다.")
        return self.fit_features(torch.cat(embeddings, dim=0), extractor.idx)

    def fit_from_file(self, path: str | Path | None = None):
        """팀 노트북이 저장한 emb·idx·labels로 정상 통계를 학습한다.

        labels에 불량 라벨이 있으면 거부하여 테스트 특징의 오사용을 막는다.
        전달된 파일은 이후 이미지 특징 추출기를 생성할 때도 사용한다.
        """
        source = Path(path) if path is not None else self.train_pt_path
        saved = torch.load(source, map_location="cpu", weights_only=True)
        if not isinstance(saved, dict) or not {"emb", "idx", "labels"}.issubset(saved):
            raise ValueError("특징 파일에는 emb, idx, labels가 있어야 합니다.")
        features = torch.as_tensor(saved["emb"])
        labels = torch.as_tensor(saved["labels"])
        if features.ndim != 4 or labels.ndim != 1 or labels.numel() != features.shape[0] or (labels != 0).any():
            raise ValueError("정상 라벨 0만 포함하고 특징 수와 라벨 수가 같은 학습 파일이 필요합니다.")
        self.fit_features(features, saved["idx"])
        self.train_pt_path = source
        # 이전 특징 파일로 만든 추출기를 재사용하지 않는다.
        self._extractor = None
        return self

    def fit_features(self, normal_features, idx):
        """정상 특징 (N,D,H,W)과 해당 채널 인덱스로 통계를 계산한다.

        공분산은 N-1로 나누는 표본 공분산이며, 각 위치에 λI를 더한다.
        계산 중에는 float64를 사용하고 최종 평균과 인자는 float32로 저장한다.
        공간 위치를 나누어 처리해 큰 공분산 임시 배열 생성을 제한한다.
        """
        features = torch.as_tensor(normal_features).detach().cpu()
        if features.ndim != 4 or features.shape[0] < 2 or min(features.shape[1:]) < 1:
            raise ValueError("정상 특징은 N이 2 이상인 (N,D,H,W) 형태여야 합니다.")
        if features.is_complex() or features.dtype == torch.bool:
            raise ValueError("정상 특징은 실수 값이어야 합니다.")
        n, channels, height, width = features.shape
        selected = self._indices(idx, channels)
        count = height * width
        flattened = features.reshape(n, channels, count)
        means = torch.empty((count, channels), dtype=torch.float32)
        factors = torch.empty((count, channels, channels), dtype=torch.float32)
        identity = torch.eye(channels, dtype=torch.float64)
        for start in range(0, count, self.spatial_batch_size):
            end = min(start + self.spatial_batch_size, count)
            block = flattened[:, :, start:end].permute(2, 0, 1).to(torch.float64)
            if not torch.isfinite(block).all():
                raise ValueError("정상 특징에 NaN 또는 무한대가 포함되어 있습니다.")
            mean = block.mean(dim=1)
            centered = block - mean[:, None, :]
            covariance = centered.transpose(1, 2) @ centered / (n - 1)
            covariance = covariance + self.regularization * identity
            factor = torch.linalg.cholesky(covariance)
            means[start:end] = mean.to(torch.float32)
            factors[start:end] = factor.to(torch.float32)
        self._validate_statistics(means, factors)
        # 실패한 재학습이 기존 통계를 훼손하지 않도록 완료 후 교체한다.
        self.mean, self.cholesky = means, factors
        self.idx = selected
        self.feature_shape = (channels, height, width)
        self.train_count = n
        return self

    @staticmethod
    def _validate_statistics(mean, factors):
        if not torch.isfinite(mean).all():
            raise ValueError("평균 또는 공분산 인자에 유한하지 않은 값이 있습니다.")
        # 검증에서도 큰 공분산 크기의 임시 배열을 한꺼번에 만들지 않는다.
        for block in factors.split(64):
            if not torch.isfinite(block).all():
                raise ValueError("평균 또는 공분산 인자에 유한하지 않은 값이 있습니다.")
            if (torch.diagonal(block, dim1=-2, dim2=-1) <= 0).any():
                raise ValueError("공분산 인자의 대각 원소는 양수여야 합니다.")
            if torch.count_nonzero(torch.triu(block, diagonal=1)):
                raise ValueError("공분산 인자는 하삼각 행렬이어야 합니다.")

    def score_features(self, features, output_size=None) -> np.ndarray:
        """한 장의 특징에서 거리 맵을 구하고, 필요하면 지정 크기로 보간한다."""
        if not self.fitted:
            raise RuntimeError("이상 점수를 계산하기 전에 정상 통계를 학습하거나 불러오세요.")
        values = torch.as_tensor(features).detach().cpu()
        if tuple(values.shape) != (1, *self.feature_shape) or values.is_complex():
            raise ValueError("검사 특징의 형태가 학습 특징과 일치해야 합니다.")
        values = values.to(torch.float32)
        if not torch.isfinite(values).all():
            raise ValueError("검사 특징에 NaN 또는 무한대가 포함되어 있습니다.")
        channels, height, width = self.feature_shape
        vectors = values[0].reshape(channels, -1).T
        distances = torch.empty(height * width, dtype=torch.float32)
        for start in range(0, len(vectors), self.spatial_batch_size):
            end = min(start + self.spatial_batch_size, len(vectors))
            delta = (vectors[start:end] - self.mean[start:end]).unsqueeze(-1)
            # Σ=LLᵀ일 때 거리 제곱은 ||L⁻¹(x-μ)||²와 같다.
            solved = torch.linalg.solve_triangular(self.cholesky[start:end], delta, upper=False)
            distances[start:end] = torch.linalg.vector_norm(solved.squeeze(-1), dim=-1)
        if not torch.isfinite(distances).all():
            raise ValueError("계산된 이상 점수가 표현 범위를 벗어났습니다.")
        grid = distances.reshape(height, width)
        if output_size is not None:
            if len(output_size) != 2 or any(not isinstance(v, int) or v < 1 for v in output_size):
                raise ValueError("출력 크기는 양의 정수 (높이, 너비)여야 합니다.")
            grid = F.interpolate(grid[None, None], size=tuple(output_size), mode="bilinear", align_corners=False)[0, 0]
        return grid.numpy()

    def score_map(self, img_bgr: np.ndarray) -> np.ndarray:
        """BGR 이미지에서 원래 입력 크기의 float32 이상 점수 맵을 반환한다."""
        if not self.fitted:
            raise RuntimeError("이상 점수를 계산하기 전에 정상 통계를 학습하거나 불러오세요.")
        rgb = self._image(img_bgr)
        with torch.no_grad():
            features = self._get_extractor().from_rgb(rgb)
        # 픽셀별 임계값과 평활화는 기존 공통 후처리에서 적용한다.
        return self.score_features(features, output_size=rgb.shape[:2])

    def save(self, path: str | Path) -> None:
        """정상 통계와 채널 인덱스를 전달된 경로에 저장한다."""
        if not self.fitted:
            raise RuntimeError("저장할 정상 통계가 없습니다.")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "version": 1, "mean": self.mean, "cholesky": self.cholesky,
            "idx": self.idx, "feature_shape": self.feature_shape,
            "train_count": self.train_count, "regularization": self.regularization,
        }, target)

    def load(self, path: str | Path):
        """통계를 복원한다. 이미지 추론 시 추출기의 채널 인덱스를 재확인한다."""
        saved = torch.load(path, map_location="cpu", weights_only=True)
        required = {"version", "mean", "cholesky", "idx", "feature_shape", "train_count", "regularization"}
        if not isinstance(saved, dict) or not required.issubset(saved) or saved["version"] != 1:
            raise ValueError("지원하지 않거나 불완전한 PaDiM 통계 파일입니다.")
        shape = tuple(saved["feature_shape"])
        if len(shape) != 3 or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in shape):
            raise ValueError("저장된 특징 크기가 올바르지 않습니다.")
        channels, height, width = shape
        selected = self._indices(saved["idx"], channels)
        mean = torch.as_tensor(saved["mean"]).to(torch.float32)
        factors = torch.as_tensor(saved["cholesky"]).to(torch.float32)
        if mean.shape != (height * width, channels) or factors.shape != (height * width, channels, channels):
            raise ValueError("저장된 정상 통계의 형태가 올바르지 않습니다.")
        count = saved["train_count"]
        reg = saved["regularization"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 2 or not np.isfinite(reg) or reg <= 0:
            raise ValueError("저장된 학습 이미지 수 또는 정규화 계수가 올바르지 않습니다.")
        self._validate_statistics(mean, factors)
        self.mean, self.cholesky = mean, factors
        self.idx, self.feature_shape, self.train_count = selected, shape, count
        self.regularization = float(reg)
        return self
