"""업로드→명시적 분석→결과 표시→새 업로드 시 초기화 흐름을 검증한다."""

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest

from test_inspection import fake_models, png_bytes


ROOT = Path(__file__).resolve().parent
SCRIPT = "import app\napp.main()\n"


class UploadScreenTests(unittest.TestCase):
    def test_initial_screen_waits_for_user_and_has_no_dataset_list(self):
        with patch("app.model_signature", return_value="test-model"), patch("app.load_models") as load:
            app = AppTest.from_string(SCRIPT).run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertTrue(app.button(key="analyze_button").disabled)
            self.assertEqual(len(app.selectbox), 0)
            self.assertEqual(len(app.file_uploader), 2)
            self.assertEqual(len(app.metric), 0)
            self.assertTrue({"원본 이미지", "정답 마스크", "PaDiM 결과", "Sobel 결과"}.issubset({x.value for x in app.subheader}))
            load.assert_not_called()

    def test_analysis_is_explicit_and_changed_upload_clears_previous_result(self):
        dark = png_bytes(np.zeros((31, 47, 3), np.uint8))
        bright = png_bytes(np.full((31, 47, 3), 255, np.uint8))
        with patch("app.model_signature", return_value="test-model"), patch("app.load_models", return_value=fake_models()) as load:
            app = AppTest.from_string(SCRIPT).run()
            app.file_uploader(key="product_upload").set_value(("new-product.png", dark, "image/png")).run()
            load.assert_not_called()
            self.assertFalse(app.button(key="analyze_button").disabled)
            app.button(key="analyze_button").click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual(len(app.metric), 6)
            first = app.session_state["inspection_result"]
            self.assertTrue(all(not r.is_defective for r in first.results.values()))
            app.radio(key="result_view").set_value("후처리 전").run()
            self.assertEqual(load.call_count, 1)
            self.assertEqual(len(app.metric), 6)
            app.file_uploader(key="product_upload").set_value(("another-product.png", bright, "image/png")).run()
            self.assertEqual(len(app.metric), 0)
            app.button(key="analyze_button").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(all(r.is_defective for r in app.session_state["inspection_result"].results.values()))
            app.file_uploader(key="product_upload").clear().run()
            self.assertEqual(len(app.metric), 0)
            self.assertTrue(app.button(key="analyze_button").disabled)

    def test_mask_is_optional_and_invalid_mask_does_not_run_inference(self):
        image = png_bytes(np.zeros((31, 47, 3), np.uint8))
        bad_mask = png_bytes(np.zeros((30, 47), np.uint8))
        with patch("app.model_signature", return_value="test-model"), patch("app.load_models", return_value=fake_models()) as load:
            app = AppTest.from_string(SCRIPT).run()
            app.file_uploader(key="product_upload").set_value(("image.png", image, "image/png")).run()
            app.file_uploader(key="mask_upload").set_value(("mask.png", bad_mask, "image/png")).run()
            app.button(key="analyze_button").click().run()
            self.assertTrue(app.error)
            self.assertFalse(app.exception)
            load.assert_not_called()


@unittest.skipUnless((ROOT / "bottle" / "train" / "good").is_dir(), "원본 bottle 데이터가 없습니다.")
class RealUploadScreenTests(unittest.TestCase):
    def test_real_uploaded_product_and_optional_ground_truth(self):
        """저장된 예시 선택이 아닌 실제 업로드 위젯으로 원본 이미지를 분석한다."""
        image = (ROOT / "bottle" / "test" / "broken_large" / "000.png").read_bytes()
        mask = (ROOT / "bottle" / "ground_truth" / "broken_large" / "000_mask.png").read_bytes()
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
        app.file_uploader(key="product_upload").set_value(("uploaded-bottle.png", image, "image/png")).run()
        self.assertEqual(len(app.metric), 0)
        app.button(key="analyze_button").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(len(app.metric), 6)
        first = app.session_state["inspection_result"]
        app.file_uploader(key="mask_upload").set_value(("optional-mask.png", mask, "image/png")).run()
        self.assertEqual(len(app.metric), 0)
        app.button(key="analyze_button").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        for name in first.results:
            np.testing.assert_array_equal(first.results[name].score_map,
                                          app.session_state["inspection_result"].results[name].score_map)
        self.assertTrue(all(m.value != "—" for m in app.metric if m.label == "IoU"))


if __name__ == "__main__":
    unittest.main()
