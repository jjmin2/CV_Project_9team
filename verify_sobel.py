"""직접 구현한 Sobel을 OpenCV의 동일 커널·경계 조건과 수치 비교한다.

실행: python verify_sobel.py
선택 저장: python verify_sobel.py --out sobel_verification.json
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from sobel_detector import SobelDetector, sobel_manual


def reference(gray):
    """직접 구현과 같은 상관 연산 부호·3×3 커널·반사 경계를 사용한다."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    return gx, gy, np.sqrt(gx * gx + gy * gy)


def verify_sobel():
    """상수·방향별 경사·경계 충격·잡음과 실제 RGB 전처리 경로를 검증한다."""
    rng = np.random.default_rng(42)
    rows = []

    def compare(name, actual, expected):
        errors = [float(np.max(np.abs(a - b))) for a, b in zip(actual, expected)]
        passed = all(np.allclose(a, b, atol=1e-5, rtol=1e-5) for a, b in zip(actual, expected))
        rows.append({"사례": name, "Gx 최대 절대 오차": errors[0], "Gy 최대 절대 오차": errors[1],
                     "엣지 강도 최대 절대 오차": errors[2], "통과": passed})

    for height, width in ((2, 2), (3, 7), (31, 47), (128, 128)):
        yy, xx = np.mgrid[:height, :width].astype(np.float32)
        impulse = np.zeros((height, width), np.float32)
        impulse[0, 0] = 1
        cases = {"상수": np.ones((height, width), np.float32),
                 "수평 경사": xx / max(width - 1, 1), "수직 경사": yy / max(height - 1, 1),
                 "대각 경사": (xx + yy) / max(width + height - 2, 1), "모서리 충격": impulse,
                 "잡음": rng.random((height, width), dtype=np.float32)}
        for name, gray in cases.items():
            compare(f"{height}×{width} {name}", sobel_manual(gray), reference(gray))
    detector = SobelDetector()
    for dtype in ("uint8", "float32"):
        rgb = rng.integers(0, 256, (31, 47, 3), dtype=np.uint8)
        if dtype == "float32":
            rgb = rgb.astype(np.float32) / 255
        normalized = rgb.astype(np.float32) / 255 if dtype == "uint8" else rgb
        gray = cv2.cvtColor(normalized, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 1.0, borderType=cv2.BORDER_REFLECT_101)
        compare(f"RGB {dtype} 전처리 포함", detector.gradients(rgb), reference(gray))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Sobel 직접 구현과 OpenCV 수치 검증")
    parser.add_argument("--out", default=None, help="결과를 저장할 JSON 경로")
    args = parser.parse_args()
    rows = verify_sobel()
    result = {"opencv": cv2.__version__, "numpy": np.__version__, "절대 허용 오차": 1e-5,
              "상대 허용 오차": 1e-5, "커널 크기": 3, "경계": "BORDER_REFLECT_101",
              "전체 통과": all(row["통과"] for row in rows), "사례": rows}
    print(f"Sobel 대조 검증: {sum(r['통과'] for r in rows)}/{len(rows)} 통과")
    for key in ("Gx 최대 절대 오차", "Gy 최대 절대 오차", "엣지 강도 최대 절대 오차"):
        print(f"{key}: {max(r[key] for r in rows):.9g}")
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if not result["전체 통과"]:
        raise SystemExit("OpenCV와 일치하지 않는 Sobel 계산이 있습니다.")


if __name__ == "__main__":
    main()
