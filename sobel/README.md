# Sobel 담당 코드

| 파일 | 역할 |
| --- | --- |
| [루트 Sobel 모듈](../sobel_detector.py) | Sobel 계산, 정상 통계·임계값 설정, 후보 점수 맵·예측 마스크 생성 |
| `README.md` | 기존 전처리 결과와 연결하는 방법 |

## 핵심 구현

- `sobel_manual()`: 3×3 Sobel 커널을 NumPy로 직접 적용해 Gx, Gy, 엣지 강도를 계산합니다. 이 계산에는 `cv2.Sobel()`이나 `cv2.filter2D()`를 사용하지 않습니다.
- `SobelDetector`: 정상 학습 이미지로 위치별 평균·표준편차를 구하고, 정상 검증 이미지로 임계값을 설정한 뒤 결함 후보를 찾습니다.
- `postprocess_mask()`: 기본 설정에서 3×3 closing과 면적 20픽셀 미만 영역 제거를 수행합니다.

## 전처리와 연결하기

1. [루트 Sobel 모듈](../sobel_detector.py)을 팀 노트북의 작업 디렉터리 또는 Python에서 import할 수 있는 위치에 둡니다. 이 폴더의 중복 사본은 제거했으며 프로젝트 루트에서 불러 사용합니다.
2. 기존 전처리 셀을 실행하고, 공통 분할에 따른 학습·검증·테스트 Dataset을 준비합니다.
3. 아래 호출 코드를 별도 셀에 추가합니다. 기존 전처리 클래스는 수정하지 않습니다.

필요한 패키지는 NumPy와 OpenCV입니다.

팀 전처리 결과인 RGB 텐서 `(3, H, W)`, 값 범위 `0~1`을 이미지 **한 장씩** 입력합니다. 학습·검증·검사 이미지의 크기와 전처리 조건은 같아야 합니다.

아래 예시는 기존 전처리로 만든 `train_dataset`, `val_dataset`, `test_dataset`이 이미 있다고 가정합니다. 실제 변수명에 맞춰 사용하세요. 각 Dataset은 팀원의 `MVTecDataset`처럼 `(이미지, 라벨, 정답 마스크, 경로)`를 반환해야 합니다. **학습·검증 Dataset은 서로 분리된 정상 이미지만 포함**해야 합니다. 팀 공통 분할을 그대로 사용하며 테스트 정답으로 통계나 임계값을 정하지 않습니다.

```python
from sobel_detector import SobelDetector

model = SobelDetector()

# 정상 학습 이미지: 위치별 엣지 강도의 평균·표준편차 추정 (최소 2장)
model.fit(train_dataset[i][0] for i in range(len(train_dataset)))

# 정상 검증 이미지: 픽셀·이미지 임계값 설정 (최소 1장)
model.calibrate(val_dataset[i][0] for i in range(len(val_dataset)))

# 검사 이미지 한 장의 결과
image, label, gt_mask, image_path = test_dataset[0]
result = model.predict(image)

score_map = result["score_map"]  # 픽셀별 연속 이상 점수
mask = result["mask"]            # 후처리된 예측 마스크 (0 또는 1)
image_score = result["image_score"]
is_anomaly = result["is_anomaly"]
```

`fit()`과 `calibrate()`는 모델 준비 단계에서 수행하고, 이후 검사할 이미지마다 `predict()`를 호출합니다. `.py` 파일 자체를 실행하면 실험이 시작되는 형태가 아니라, 위처럼 불러 사용하는 모듈입니다.

## 반환값과 판정 기준

| 반환값 | 의미 |
| --- | --- |
| `gx`, `gy`, `edge_map` | x·y 방향 밝기 변화량과 Sobel 엣지 강도, 각각 `(H, W)` |
| `score_map` | 정상 통계와 비교한 이상 점수 `abs(G - mean) / (std + 0.01)`, 기본 설정 기준 |
| `raw_mask`, `mask` | 후처리 전·후 후보 마스크, `(H, W)`의 `uint8` 값 0 또는 1 |
| `image_score`, `is_anomaly` | 점수 맵의 최댓값과 이미지 전체의 이상 여부 |
| `regions`, `contours` | 남은 영역의 위치·면적 정보와 외곽선 |
| `elapsed_ms` | `predict()` 내부 처리 시간. 파일 읽기와 외부 전처리 시간은 제외 |

기본 픽셀 임계값은 정상 검증 이미지의 모든 픽셀 점수 중 99.5백분위수, 이미지 임계값은 각 정상 검증 이미지의 최댓값들 중 99백분위수입니다. 두 임계값은 보정 후 고정되며 검사 사진마다 새로 계산하지 않습니다. 이미지 판정은 후처리 마스크가 아니라 점수 맵의 최댓값을 사용합니다. 점수는 확률이 아닙니다.

모델을 재사용할 때는 정상 통계와 임계값을 저장하고 불러올 수 있습니다.

```python
model.save("sobel_model.npz")
# 다음 실행에서는 fit/calibrate 대신 저장된 모델을 불러옵니다.
model = SobelDetector.load("sobel_model.npz")
```

이 폴더에는 학습된 모델이나 데이터가 포함되어 있지 않습니다. 최초 사용 시 팀의 정상 학습·검증 데이터로 모델을 준비합니다.
