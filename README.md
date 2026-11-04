# DefectVision

MVTec AD를 활용한 산업 제품 이상 탐지·결함 위치 시각화 프로젝트입니다.
실제 ResNet18 기반 PaDiM과 NumPy로 구현한 Sobel을 Streamlit에서 실행합니다.

## 실행

프로젝트 루트의 PowerShell에서 실행합니다. 가상환경이 이미 있으면 생성 단계는 생략합니다.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

브라우저에서 [앱 열기](http://localhost:8501)를 선택합니다.

## 1. 저장 특징 미리보기

기본 실행 모드입니다. 다음 두 파일이 있으면 원본 이미지나 가중치 다운로드 없이
실제 PaDiM 거리 맵·정답 마스크·예측 마스크·IoU를 확인할 수 있습니다.

- [정상 학습 특징](./feature/bottle_train_features.pt)
- [테스트 특징](./feature/bottle_test_features.pt)

파일은 팀 특징 추출 노트북이 저장하는 `emb`, `idx`, `labels`, `masks`, `paths`를 포함해야 합니다.
학습·테스트 특징의 채널 인덱스와 공간 크기가 같아야 하며, 학습 라벨은 정상 0만 허용합니다.

최종 이미지 목록은 [공통 분할 파일](./splits/bottle.json)에 고정되어 있습니다.
현재 정상 188장 중 **통계 추정 150장, 검증 38장**, 테스트 83장을 사용합니다.
테스트는 학습·임계값 보정에 사용하지 않습니다. 기존 특징 파일에 없는 정상 원본은
원본 데이터가 추가되어도 자동으로 학습에 편입하지 않습니다.

앱·특징 미리보기·두 노트북·평가 및 시간 측정 도구는 모두 이 목록을 재사용합니다.
분할은 파일 경로 순서나 Colab/Windows 저장 위치에 영향을 받지 않습니다.
목록의 파일이 누락되면 중단하며, 시드·비율이 달라도 기존 파일을 자동 덮어쓰지 않습니다.
의도적으로 분할을 바꾸는 작업은 팀과 협의하여 별도로 수행해야 합니다.

사이드바의 원본 카테고리 폴더에 로컬 bottle 폴더를 입력하면
`test/결함종류/파일명`으로 원본을 찾아 히트맵과 외곽선을 겹쳐 표시합니다.
원본이 없으면 PaDiM 히트맵과 마스크를 왼쪽에, 원본이 필요한 Sobel 안내를 오른쪽에 보여줍니다.
원본 카테고리를 추가하면 같은 고정 분할로 Sobel도 실행하여 나란히 표시합니다.
특징에서 원본 이미지를 복원하지 않습니다. 마스크는 전후 나란히·후처리 전·후 중 선택합니다.

이 모드의 시간은 **저장 특징의 거리 계산과 후처리**만 포함하며 ResNet18 추출은 제외합니다.
Sobel 결과나 두 알고리즘의 전체 CPU 시간 비교에는 사용할 수 없습니다.

## 2. 이미지 비교

사이드바에서 **이미지 비교 (PaDiM · Sobel)**를 선택합니다.

1. MVTec 카테고리 폴더 또는 정상 이미지 폴더를 입력합니다.
2. 정상 이미지 최소 세 장으로 두 방법의 통계를 준비합니다.
3. 데이터셋에서 검사 이미지를 선택하거나 제품 이미지를 업로드합니다.
4. 두 방법의 히트맵·마스크·외곽선·판정·IoU·처리 시간을 비교합니다.

카테고리 폴더는 다음 구조를 사용합니다.

```text
bottle/
  train/good/정상이미지.png
  test/good/정상이미지.png
  test/결함종류/불량이미지.png
  ground_truth/결함종류/불량이미지_mask.png
```

입력은 두 방법 모두 256×256이며 같은 고정 정상·검증·테스트 목록을 사용합니다.
PaDiM을 왼쪽, Sobel을 오른쪽에 배치하고, 동일한 마스크 전후 보기로 비교합니다.
PaDiM의 첫 이미지 실행에는 ImageNet 사전학습 ResNet18
가중치 다운로드가 필요할 수 있습니다. 두 방법의 정상 통계는 자동으로 캐시됩니다.

## 점수와 임계값

- PaDiM은 위치별 표본 공분산에 `λI`를 더하고 Cholesky 분해로 Mahalanobis 거리를 계산합니다.
- 이상 점수는 확률이 아닙니다. 이미지 점수는 보간된 점수 맵의 최댓값입니다.
- 초기 이미지 임계값은 정상 검증 이미지 최댓값들의 99백분위수입니다.
- 초기 픽셀 임계값은 정상 검증 점수 맵에 Gaussian sigma 4를 적용한 값들의 99.9백분위수입니다.
- 후처리는 임계값 이진화, opening/closing, 작은 연결 영역 제거입니다.
- 화면에서 수동 변경한 임계값은 탐색용입니다. 공식 평가에서 테스트 정답에 맞춰 조절하지 마세요.
- 정상 이미지도 오탐될 수 있으며, 두 마스크가 모두 비어 있으면 IoU는 정의되지 않습니다.

이미지 비교 모드의 시간은 **공통 리사이즈 + 특징/엣지·점수·판정 계산 + 후처리**입니다.
파일 읽기·디코딩·통계 준비·임계값 보정·화면 렌더링은 제외합니다.
표에는 전처리·점수 계산·후처리와 합계를 구분해 표시합니다.
반복 시간 측정에서도 원본으로부터 매번 리사이즈하고 워밍업을 제외한 평균·중앙값을 기록합니다.
파일 읽기와 OpenCV 그림 생성은 별도로 측정하며 브라우저 렌더링 시간은 측정하지 않습니다.
사이드바의 스레드 수는 PyTorch·OpenCV 양쪽과 같은 서버의 세션에 공통으로 적용됩니다.

## 검증

```powershell
# 단위 테스트와 통합 테스트
.\.venv\Scripts\python.exe -m unittest -v test_padim test_integration test_project_contracts

# 실제 저장 특징을 사용하는 앱 화면 테스트
.\.venv\Scripts\python.exe -m unittest -v test_streamlit

# Sobel 커널과 RGB 전처리를 OpenCV에 대조
.\.venv\Scripts\python.exe verify_sobel.py

# 실제 ResNet18·Sobel 이미지 경로도 검증: 임시 합성 데이터를 사용
$env:DEFECTVISION_IMAGE_TEST = '1'
.\.venv\Scripts\python.exe -m unittest -v test_streamlit
```

합성 데이터 검증은 실행 경로 확인용이며 MVTec 성능 측정 결과가 아닙니다.

실제 원본 데이터로 공식 평가·속도 측정을 실행할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe evaluate.py --category "D:\mvtec\bottle"
.\.venv\Scripts\python.exe benchmark.py --category "D:\mvtec\bottle" --threads 2
```

두 도구도 실제 PaDiM과 공통 분할을 사용하며 원본 데이터가 필요합니다.
평가 지표와 성능 분석은 담당 팀원이 수행합니다. 평가 도구의 지표 계산식은 변경하지 않았습니다.
벤치마크의 `--threads`는 OpenCV와 PyTorch 양쪽에 적용되며 CSV 옆의 메타데이터 JSON에
분할 ID·실행 목록·환경·반복 횟수·시간 범위를 함께 기록합니다.
`--split-file`로 팀에서 공유한 분할을 명시할 수도 있습니다.
기존 `--max-normal`로 고정 학습 목록을 줄이는 동작은 허용하지 않습니다.

## 노트북 실행

두 노트북은 [공통 데이터셋](./mvtec_dataset.py)을 불러옵니다.
Colab에서도 프로젝트의 Python 모듈과 [공통 분할 파일](./splits/bottle.json)을 함께 준비하고
프로젝트 루트로 작업 위치를 변경해야 합니다. 예전의 독립적인 10%/시드 42 분할은 제거했습니다.
특징 추출 노트북은 고정 추출기로 정상 통계·검증 특징을 함께 저장하되,
PaDiM 통계 추정 시에는 공통 목록으로 두 집합을 다시 구분합니다.
기존 특징 파일 자체를 다시 생성하지는 않았습니다. 원본 확보 후 새 공통 전처리로
특징을 다시 추출하면 원본 추론과 저장 특징 미리보기의 전처리 조건도 맞출 수 있습니다.

## 주요 코드

모델·특징·원본 데이터·압축파일·통계 캐시는 Git에서 제외합니다.
특징 파일은 [로컬 특징 파일 안내](./feature/README.md)에 따라 별도로 배치하세요.
[공통 분할 목록](./splits/bottle.json)과 소스·노트북·테스트는 함께 공유합니다.
이미 커밋한 대용량 파일은 `.gitignore`만으로 기존 커밋 이력에서 제거되지 않습니다.

| 파일 | 역할 |
| --- | --- |
| [app.py](./app.py) | 한국어 Streamlit 화면과 두 실행 모드 |
| [padim.py](./padim.py) | 정상 통계 학습·거리 맵·저장 복원 |
| [padim_preview.py](./padim_preview.py) | 특징 파일 검증·정상 분할·미리보기 캐시·임계값 보정 |
| [data_split.py](./data_split.py) | 공통 이미지 분할 파일 생성·검증·재사용 |
| [mvtec_dataset.py](./mvtec_dataset.py) | 노트북 공통 데이터셋과 전처리 |
| [verify_sobel.py](./verify_sobel.py) | 직접 구현한 Sobel과 OpenCV의 수치 대조 |
| [feature_extractor.py](./feature_extractor.py) | 기존 ResNet18 추출, 저장 인덱스 직접 전달 지원 |
| [detectors.py](./detectors.py) | 실제 PaDiM 연결과 Sobel 어댑터 |
| [sobel_detector.py](./sobel_detector.py) | 팀 Sobel 알고리즘, 계산 구현 유지 |
| [common.py](./common.py) | 공통 입출력·후처리·시각화·캐시 |
| [evaluate.py](./evaluate.py) | AUROC·IoU 평가 |
| [benchmark.py](./benchmark.py) | 단계별 CPU 처리 시간 측정 |
