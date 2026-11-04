"""PaDiM 수학 계산과 저장·복원 검증. 사전학습 가중치나 실제 데이터는 사용하지 않는다.

실행: python -m unittest -v test_padim
필요 패키지: torch, numpy, opencv-python
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from padim import PaDiMDetector


class FakeExtractor:
    """색상 변환과 출력 크기 검증을 위한 작은 특징 추출기 대역."""

    idx = torch.tensor([0, 1])

    def from_rgb(self, image):
        self.received = image.copy()
        return torch.ones((1, 2, 2, 3), dtype=torch.float32)


class PaDiMTests(unittest.TestCase):
    def test_correlated_covariance_and_spatial_statistics(self):
        """채널 간 공분산을 포함한 거리가 독립적인 NumPy 계산과 일치해야 한다."""
        vectors = np.array([
            [[-1, -2], [4, 1]],
            [[1, 2], [6, 5]],
            [[0, 1], [5, 3]],
        ], dtype=np.float32)
        features = torch.from_numpy(vectors.transpose(0, 2, 1).copy()).reshape(3, 2, 1, 2)
        query = np.array([[2, -1], [7, 4]], dtype=np.float32)
        detector = PaDiMDetector(regularization=0.25, spatial_batch_size=1)
        detector.fit_features(features, [3, 7])
        actual = detector.score_features(query.T.copy().reshape(1, 2, 1, 2))
        expected = []
        for position in range(2):
            samples = vectors[:, position, :]
            covariance = np.cov(samples, rowvar=False) + 0.25 * np.eye(2)
            delta = query[position] - samples.mean(axis=0)
            expected.append(np.sqrt(delta @ np.linalg.solve(covariance, delta)))
            factor = detector.cholesky[position].numpy()
            np.testing.assert_allclose(factor @ factor.T, covariance, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(actual.ravel(), expected, rtol=1e-5)

    def test_singular_normal_features_are_regularized(self):
        """완전히 같은 정상 특징도 계산 가능하고 평균에서의 거리는 0이어야 한다."""
        detector = PaDiMDetector(regularization=0.25)
        detector.fit_features(torch.zeros((2, 2, 2, 3)), [0, 1])
        np.testing.assert_array_equal(detector.score_features(torch.zeros((1, 2, 2, 3))), 0)
        expected = np.sqrt(8.0)
        np.testing.assert_allclose(detector.score_features(torch.ones((1, 2, 2, 3))), expected, rtol=1e-6)

    def test_image_color_size_and_restored_statistics(self):
        """통계 복원 후에도 BGR→RGB 변환, 입력 크기 복원, 거리 값이 유지되어야 한다."""
        extractor = FakeExtractor()
        detector = PaDiMDetector(regularization=0.25, feature_extractor=extractor)
        detector.fit_features(torch.zeros((2, 2, 2, 3)), extractor.idx)
        image = np.zeros((7, 11, 3), dtype=np.uint8)
        image[:, :, 0] = 10
        image[:, :, 2] = 30
        first = detector.score_map(image)
        self.assertEqual(first.shape, (7, 11))
        self.assertEqual(first.dtype, np.float32)
        np.testing.assert_array_equal(extractor.received[0, 0], [30, 0, 10])
        np.testing.assert_allclose(first, np.sqrt(8.0), rtol=1e-6)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "statistics.npz"
            detector.save(path)
            restored = PaDiMDetector(feature_extractor=FakeExtractor()).load(path)
            np.testing.assert_array_equal(restored.score_map(image), first)
            self.assertEqual(restored.train_count, 2)
            self.assertEqual(restored.regularization, 0.25)

    def test_feature_file_rejects_defective_labels(self):
        """불량 특징을 정상 통계에 섞는 실수를 거부해야 한다."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "features.pt"
            payload = {"emb": torch.zeros((2, 2, 1, 1)), "idx": torch.tensor([0, 1]),
                       "labels": torch.tensor([0, 1])}
            torch.save(payload, path)
            detector = PaDiMDetector()
            with self.assertRaises(ValueError):
                detector.fit_from_file(path)
            payload["labels"] = torch.zeros(2, dtype=torch.int64)
            torch.save(payload, path)
            detector.fit_from_file(path)
            self.assertTrue(detector.fitted)
            self.assertIsNone(detector._extractor)

    def test_channel_mismatch_and_unfitted_inference(self):
        """채널 순서가 다른 추출기나 학습 전 추론은 명시적으로 거부해야 한다."""
        detector = PaDiMDetector(feature_extractor=FakeExtractor())
        with self.assertRaises(RuntimeError):
            detector.score_features(torch.zeros((1, 2, 2, 3)))
        detector.fit_features(torch.zeros((2, 2, 2, 3)), [1, 0])
        with self.assertRaises(ValueError):
            detector.score_map(np.zeros((7, 11, 3), dtype=np.uint8))

    def test_invalid_training_does_not_replace_previous_statistics(self):
        """손상된 입력으로 재학습이 실패해도 이전 정상 통계를 보존해야 한다."""
        detector = PaDiMDetector()
        detector.fit_features(torch.zeros((2, 2, 1, 1)), [0, 1])
        previous = detector.mean.clone()
        invalid = torch.zeros((2, 2, 1, 1))
        invalid[0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            detector.fit_features(invalid, [0, 1])
        torch.testing.assert_close(detector.mean, previous)
        with self.assertRaises(ValueError):
            detector.fit_features(torch.zeros((1, 2, 1, 1)), [0, 1])
        with self.assertRaises(ValueError):
            detector.fit_features(torch.zeros((2, 2, 1, 1)), [0, 0])

    def test_corrupt_statistics_are_rejected(self):
        """저장 파일의 공분산 인자가 손상되면 복원을 거부해야 한다."""
        detector = PaDiMDetector()
        detector.fit_features(torch.zeros((2, 2, 1, 1)), [0, 1])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "statistics.pt"
            detector.save(path)
            saved = torch.load(path, map_location="cpu", weights_only=True)
            saved["cholesky"][0, 0, 0] = -1
            torch.save(saved, path)
            with self.assertRaises(ValueError):
                PaDiMDetector().load(path)


if __name__ == "__main__":
    unittest.main()
