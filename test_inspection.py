"""업로드 바이트 기반 추론과 정답 마스크 분리 검증."""

from pathlib import Path
import unittest

import cv2
import numpy as np
import torch

from common import PostConfig
from inspection import analyze_upload, decode_upload, prepare_models


def png_bytes(image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("테스트 이미지를 PNG로 만들 수 없습니다.")
    return encoded.tobytes()


class PixelDetector:
    """입력 내용 변경이 결과 변경으로 이어지는지 확인하는 결정적 대역."""
    def __init__(self, name):
        self.name = name

    def score_map(self, image):
        return image.mean(axis=2).astype(np.float32) / 255

    def edges(self, image):
        return self.score_map(image)


def fake_models():
    detectors = [PixelDetector("PaDiM"), PixelDetector("Sobel")]
    return {"detectors": detectors, "configs": {d.name: PostConfig(threshold=0.5) for d in detectors},
            "thresholds": {d.name: {"image": 0.5, "pixel": 0.5} for d in detectors},
            "split_id": "test-split", "split": {"statistics_images": ["a", "b"], "validation_images": ["c"]}}


class UploadTests(unittest.TestCase):
    def test_arbitrary_uploaded_pixels_are_scored_without_dataset_lookup(self):
        """데이터셋 파일명이 없는 새 이미지도 입력 내용에 따라 판정되어야 한다."""
        models = fake_models()
        dark = analyze_upload(png_bytes(np.zeros((73, 91, 3), np.uint8)), models)
        bright = analyze_upload(png_bytes(np.full((73, 91, 3), 255, np.uint8)), models)
        self.assertIsNone(dark.ground_truth)
        self.assertEqual(dark.image.shape, (256, 256, 3))
        for name in ("PaDiM", "Sobel"):
            self.assertFalse(dark.results[name].is_defective)
            self.assertTrue(bright.results[name].is_defective)
            self.assertFalse(np.array_equal(dark.results[name].score_map, bright.results[name].score_map))

    def test_ground_truth_never_changes_prediction(self):
        image = png_bytes(np.full((31, 47, 3), 128, np.uint8))
        empty = png_bytes(np.zeros((31, 47), np.uint8))
        full = png_bytes(np.full((31, 47), 255, np.uint8))
        a = analyze_upload(image, fake_models(), empty)
        b = analyze_upload(image, fake_models(), full)
        for name in a.results:
            np.testing.assert_array_equal(a.results[name].score_map, b.results[name].score_map)
            self.assertEqual(a.results[name].is_defective, b.results[name].is_defective)
        self.assertEqual(a.ground_truth.sum(), 0)
        self.assertEqual(b.ground_truth.sum(), 256 * 256)

    def test_invalid_upload_and_incompatible_mask_are_rejected(self):
        with self.assertRaises(ValueError):
            decode_upload(b"invalid image")
        with self.assertRaises(ValueError):
            decode_upload(png_bytes(np.zeros((31, 47, 3), np.uint8)), png_bytes(np.zeros((30, 47), np.uint8)))


@unittest.skipUnless((Path(__file__).parent / "bottle" / "train" / "good").is_dir(), "원본 bottle 데이터가 없습니다.")
class RealBottleTests(unittest.TestCase):
    def test_original_bottle_model_and_uploaded_image(self):
        """실제 정상 데이터로 준비한 두 모델이 업로드 바이트를 추론해야 한다."""
        root = Path(__file__).resolve().parent
        torch.set_num_threads(2)
        cv2.setNumThreads(2)
        models = prepare_models(root / "bottle")
        self.assertEqual((len(models["split"]["statistics_images"]), len(models["split"]["validation_images"])), (150, 38))
        image = (root / "bottle" / "test" / "broken_large" / "000.png").read_bytes()
        result = analyze_upload(image, models)
        normal = analyze_upload((root / "bottle" / "test" / "good" / "000.png").read_bytes(), models)
        for name in ("PaDiM", "Sobel"):
            scores = result.results[name].score_map
            self.assertEqual(scores.shape, (256, 256))
            self.assertTrue(np.isfinite(scores).all())
            self.assertGreater(float(scores.max()), 0)
            self.assertFalse(np.array_equal(scores, normal.results[name].score_map))
        self.assertIsNone(result.ground_truth)


if __name__ == "__main__":
    unittest.main()
