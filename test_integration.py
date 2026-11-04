"""실제 PaDiM 연결·통계 캐시·저장 특징 미리보기의 통합 검증."""

from pathlib import Path
import os
import tempfile
import unittest

import numpy as np
import torch

from common import fit_or_load, stats_signature
from detectors import PaDiMDetector
from padim import PaDiMDetector as ActualPaDiM
from padim_preview import prepare_preview, calibrate_preview
from test_padim import FakeExtractor


class IntegrationTests(unittest.TestCase):
    def test_detector_import_uses_real_padim(self):
        """앱과 평가 코드의 기존 import가 실제 구현을 가리켜야 한다."""
        self.assertIs(PaDiMDetector, ActualPaDiM)
        self.assertFalse(PaDiMDetector.is_placeholder)

    def test_common_statistics_cache_and_configuration(self):
        """같은 설정만 캐시를 재사용하고 정규화·채널 변경은 다른 키를 써야 한다."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "normal.png"
            path.write_bytes(b"test")
            images = [np.zeros((4, 4, 3), np.uint8) for _ in range(2)]
            first = PaDiMDetector(feature_extractor=FakeExtractor())
            result = fit_or_load(first, images, [str(path)], 256, temporary)
            self.assertEqual(result["source"], "fit+saved")
            second = PaDiMDetector(feature_extractor=FakeExtractor())
            self.assertEqual(fit_or_load(second, images, [str(path)], 256, temporary)["source"], "cache")
            self.assertTrue(second.fitted)
            changed = PaDiMDetector(regularization=0.1, feature_extractor=FakeExtractor())
            self.assertNotEqual(first.cache_version, changed.cache_version)
            other_extractor = FakeExtractor()
            other_extractor.idx = torch.tensor([1, 0])
            self.assertNotEqual(first.cache_version, PaDiMDetector(feature_extractor=other_extractor).cache_version)

    def test_category_path_separates_statistics_caches(self):
        """파일명이 같아도 다른 카테고리의 정상 통계를 섞지 않아야 한다."""
        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for category in ("a", "b"):
                folder = Path(temporary) / category
                folder.mkdir()
                path = folder / "000.png"
                path.write_bytes(b"test")
                os.utime(path, ns=(1700000000000000000, 1700000000000000000))
                paths.append(str(path))
            self.assertNotEqual(stats_signature([paths[0]], 256), stats_signature([paths[1]], 256))

    def test_preview_split_cache_and_test_independence(self):
        """테스트 라벨을 바꿔도 정상 통계·임계값은 바뀌지 않아야 한다."""
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            train_path, test_path = folder / "train.pt", folder / "test.pt"
            rng = torch.Generator().manual_seed(3)
            train = {"emb": torch.randn(10, 2, 2, 3, generator=rng), "idx": torch.tensor([0, 1]),
                     "labels": torch.zeros(10, dtype=torch.int64), "masks": torch.zeros(10, 1, 256, 256),
                     "paths": [f"/train/good/{i}.png" for i in range(10)]}
            test = {"emb": torch.randn(2, 2, 2, 3, generator=rng), "idx": torch.tensor([0, 1]),
                    "labels": torch.tensor([0, 1]), "masks": torch.zeros(2, 1, 256, 256),
                    "paths": ["/test/good/0.png", "/test/scratch/1.png"]}
            torch.save(train, train_path)
            torch.save(test, test_path)
            first = prepare_preview(train_path, test_path, folder / "cache", split_path=folder / "split.json")
            self.assertEqual((first["n_stats"], first["n_val"]), (8, 2))
            thresholds = calibrate_preview(first)
            self.assertTrue(all(np.isfinite(v) for v in thresholds.values()))
            test["labels"] = torch.tensor([1, 0])
            # Windows에서는 메모리 매핑 중인 파일을 덮어쓸 수 없어 별도 파일을 사용한다.
            changed_test_path = folder / "changed_test.pt"
            torch.save(test, changed_test_path)
            second = prepare_preview(train_path, changed_test_path, folder / "cache", split_path=folder / "split.json")
            self.assertEqual(second["source"], "저장된 통계 재사용")
            self.assertEqual(thresholds, calibrate_preview(second))
            torch.testing.assert_close(first["detector"].mean, second["detector"].mean)

    def test_resized_feature_scores_keep_distances(self):
        """크기 복원 시 점수를 이미지별로 0~1 정규화하지 않아야 한다."""
        detector = PaDiMDetector(regularization=0.25)
        detector.fit_features(torch.zeros(2, 2, 2, 3), [0, 1])
        output = detector.score_features(torch.ones(1, 2, 2, 3), output_size=(256, 256))
        self.assertEqual(output.shape, (256, 256))
        np.testing.assert_allclose(output, np.sqrt(8.0), rtol=1e-6)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
