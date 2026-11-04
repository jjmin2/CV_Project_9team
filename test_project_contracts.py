"""공통 분할·CPU 시간 범위·OpenCV 대조에 대한 회귀 검증."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import cv2

from benchmark import measure_runs
from common import PostConfig, run_method
from data_split import get_or_create_split, image_id, validate_split
from verify_sobel import verify_sobel


class SplitTests(unittest.TestCase):
    def test_notebook_dataset_uses_manifest_and_common_preprocessing(self):
        """노트북 데이터셋이 고정 목록과 앱의 RGB·마스크 전처리를 따라야 한다."""
        from mvtec_dataset import MVTecDataset
        from make_demo_dataset import save_png

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            category = root / "bottle"
            raw = np.full((31, 47, 3), [20, 40, 60], np.uint8)
            for i in range(5):
                save_png(str(category / "train" / "good" / f"{i}.png"), raw)
            save_png(str(category / "test" / "good" / "0.png"), raw)
            save_png(str(category / "test" / "scratch" / "1.png"), raw)
            mask = np.zeros((31, 47), np.uint8)
            mask[5:10, 12:20] = 255
            save_png(str(category / "ground_truth" / "scratch" / "1_mask.png"), mask)
            target = root / "split.json"
            train = MVTecDataset(root, "bottle", split_path=target)
            validation = MVTecDataset(root, "bottle", split="val", split_path=target)
            test = MVTecDataset(root, "bottle", split="test", split_path=target)
            self.assertEqual([image_id(p) for p in train.image_paths], train.split_definition["statistics_images"])
            self.assertEqual([image_id(p) for p in validation.image_paths], train.split_definition["validation_images"])
            self.assertFalse(set(train.image_paths) & set(validation.image_paths))
            image, _, _, _ = train[0]
            np.testing.assert_allclose(image[:, 0, 0].numpy(), np.array([60, 40, 20]) / 255, rtol=1e-6)
            _, label, actual_mask, _ = test[1]
            self.assertEqual(label, 1)
            expected = cv2.resize((mask > 0).astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)
            np.testing.assert_array_equal(actual_mask[0].numpy(), expected)

    def test_local_and_colab_reuse_exact_split_with_extra_originals(self):
        """파일 순서·저장 위치·추가 원본이 달라도 최초 분할 목록을 그대로 사용한다."""
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "bottle.json"
            normal = [f"/content/mvtec/bottle/train/good/{i}.png" for i in range(10)]
            test = ["/content/mvtec/bottle/test/good/0.png", "/content/mvtec/bottle/test/scratch/1.png"]
            first = get_or_create_split("bottle", normal, test, target)
            local_normal = ["D:/data/bottle/" + image_id(p) for p in reversed(normal)] + ["D:/data/bottle/train/good/extra.png"]
            local_test = ["D:/data/bottle/" + image_id(p) for p in reversed(test)]
            second = get_or_create_split("bottle", local_normal, local_test, target)
            self.assertEqual(first, second)
            self.assertEqual((len(first["statistics_images"]), len(first["validation_images"])), (8, 2))
            self.assertNotIn("train/good/extra.png", second["normal_images"])
            with self.assertRaises(ValueError):
                get_or_create_split("bottle", local_normal[:-2], local_test, target)
            with self.assertRaises(ValueError):
                get_or_create_split("bottle", normal, test, target, seed=42)

    def test_overlap_and_test_leakage_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "bottle.json"
            split = get_or_create_split("bottle", [f"train/good/{i}.png" for i in range(5)], ["test/good/0.png"], target)
            damaged = json.loads(json.dumps(split))
            damaged["validation_images"] = damaged["statistics_images"][:1]
            with self.assertRaises(ValueError):
                validate_split(damaged)
            with self.assertRaises(ValueError):
                image_id("train/good/../../escape.png")


class TimingTests(unittest.TestCase):
    def test_total_includes_resize_score_and_postprocess(self):
        """독립 시계를 주입해 총 시간이 세 구간의 합인지 정확히 확인한다."""
        class Detector:
            def score_map(self, image):
                self.seen = image.shape
                return np.zeros(image.shape[:2], np.float32)

        detector = Detector()
        raw = np.zeros((73, 91, 3), np.uint8)
        with patch("common.time.perf_counter", side_effect=[10, 10.01, 10.04, 10.06]):
            result = run_method(detector, raw, PostConfig(), 3)
        self.assertEqual(detector.seen, (256, 256, 3))
        self.assertAlmostEqual(result.preprocess_ms, 10)
        self.assertAlmostEqual(result.score_ms, 30)
        self.assertAlmostEqual(result.postprocess_ms, 20)
        self.assertAlmostEqual(result.ms, 60)
        self.assertAlmostEqual(result.ms, result.preprocess_ms + result.score_ms + result.postprocess_ms)

    def test_every_repetition_includes_preprocessing(self):
        """워밍업을 결과에서 제외하되 모든 반복마다 원본을 다시 전처리한다."""
        class Detector:
            def score_map(self, image):
                return np.zeros(image.shape[:2], np.float32)

        from common import preprocess

        with patch("common.preprocess", wraps=preprocess) as resize:
            records = measure_runs(Detector(), np.zeros((40, 60, 3), np.uint8), PostConfig(), 3, warmup=2, repeats=3)
        self.assertEqual(resize.call_count, 5)
        self.assertTrue(all(len(values) == 3 for values in records.values()))
        for i, total in enumerate(records["total = preprocess + score + postprocess (ms)"]):
            self.assertAlmostEqual(total, sum(records[key][i] for key in ("preprocess (ms)", "score (ms)", "postprocess (ms)")))


class SobelTests(unittest.TestCase):
    def test_manual_kernels_and_rgb_pipeline_match_opencv(self):
        results = verify_sobel()
        self.assertEqual(len(results), 26)
        for result in results:
            with self.subTest(case=result["사례"]):
                self.assertTrue(result["통과"], result)


if __name__ == "__main__":
    unittest.main()
