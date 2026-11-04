"""실제 저장 특징을 사용하는 Streamlit 화면 검증.

원본 비교 경로까지 검증하려면 DEFECTVISION_IMAGE_TEST=1을 지정한다.
해당 검증은 임시 합성 이미지와 실제 ResNet18을 사용하며,
가중치가 없으면 최초 실행에 인터넷 연결이 필요하다.
"""

import os
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from streamlit.testing.v1 import AppTest
from data_split import image_id


ROOT = Path(__file__).resolve().parent


class StreamlitTests(unittest.TestCase):
    @unittest.skipUnless((ROOT / "feature" / "bottle_test_features.pt").is_file(), "실제 특징 파일이 없습니다.")
    def test_preview_image_switch_and_missing_original_guidance(self):
        """결함 종류 변경이 실제 표본을 바꾸고, 원본 없는 비교는 안내를 보여야 한다."""
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(len(app.get("image")), 4)
        self.assertEqual(len(app.metric), 5)
        self.assertTrue({"PaDiM", "Sobel"}.issubset({s.value for s in app.subheader}))
        app.radio(key="preview_mask_view").set_value("후처리 전").run()
        self.assertEqual(len(app.get("image")), 3)
        before = app.selectbox[1].value
        app.selectbox(key="preview_type").select("good").run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        after = app.selectbox[1].value
        self.assertNotEqual(before, after)
        data = torch.load(ROOT / "feature" / "bottle_test_features.pt", map_location="cpu", weights_only=True, mmap=True)
        split = json.loads((ROOT / "splits" / "bottle.json").read_text(encoding="utf-8"))
        labels = {image_id(p): int(label) for p, label in zip(data["paths"], data["labels"])}
        self.assertEqual(labels[split["test_images"][after]], 0)
        app.sidebar.radio[0].set_value("이미지 비교 (PaDiM · Sobel)").run()
        self.assertFalse(app.error)
        self.assertTrue(any("정상 이미지가 최소 세 장" in x.value for x in app.info))

    @unittest.skipUnless(os.environ.get("DEFECTVISION_IMAGE_TEST") == "1", "실제 ResNet18 이미지 검증은 선택 실행합니다.")
    def test_actual_resnet_and_sobel_comparison(self):
        """실제 특징 추출·Sobel·공통 후처리를 거쳐 두 결과가 화면에 나와야 한다."""
        from make_demo_dataset import normal_image, add_scratch, save_png

        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temporary, patch("data_split.SPLIT_DIR", Path(temporary) / "splits"):
            category = Path(temporary) / "toy"
            rng = np.random.default_rng(42)
            for i in range(5):
                save_png(str(category / "train" / "good" / f"{i}.png"), normal_image(rng))
            save_png(str(category / "test" / "good" / "0.png"), normal_image(rng))
            defective = normal_image(rng)
            mask = add_scratch(defective, rng)
            save_png(str(category / "test" / "scratch" / "0.png"), defective)
            save_png(str(category / "ground_truth" / "scratch" / "0_mask.png"), mask)
            # 저장 특징 경로를 먼저 실행하지 않고 이미지 비교 화면만 검증한다.
            script = "import torch\nimport app\ntorch.set_num_threads(2)\napp.run_comparison(0.01)\n"
            app = AppTest.from_string(script, default_timeout=180).run()
            app.sidebar.text_input[0].set_value(str(category)).run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual(len(app.metric), 10)
            self.assertEqual(len(app.dataframe[0].value), 2)
            self.assertEqual(set(app.dataframe[0].value["방법"]), {"PaDiM", "Sobel"})
            table = app.dataframe[0].value
            np.testing.assert_allclose(table["처리 시간 (ms)"], table["전처리 (ms)"] + table["점수 계산 (ms)"] + table["후처리 (ms)"])
            self.assertTrue({"PaDiM", "Sobel"}.issubset({s.value for s in app.subheader}))
            app.radio(key="comparison_mask_view").set_value("후처리 전").run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            app.selectbox[0].select("scratch").run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual(len(app.metric), 10)


if __name__ == "__main__":
    unittest.main()
