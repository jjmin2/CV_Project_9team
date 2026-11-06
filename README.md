# DefectVision

사용자가 등록한 bottle 제품 이미지를 실제 ResNet18 기반 PaDiM과 Sobel로 분석하는
산업 제품 이상 탐지·결함 위치 시각화 프로젝트입니다.

## 실행

프로젝트 루트의 PowerShell에서 실행합니다. 가상환경이 이미 있으면 생성은 생략합니다.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py
```

[앱 열기](http://localhost:8501)

가상환경을 활성화했다면 `python run.py`로 실행합니다.
터미널에는 `Welcome to Streamlit!`만 표시하고 이메일 입력 없이 서버를 시작합니다.
서버가 시작된 뒤 위 링크로 접속하며, 사용하는 동안 터미널을 열어두세요.
기존 `python -m streamlit run app.py`도 이메일 입력 없이 실행되지만 기본 접속 안내는 표시됩니다.

## 사용자 흐름

1. **제품 이미지 업로드**에 PNG·JPG·JPEG·BMP 이미지 한 장을 등록합니다.
2. 선택적으로 원본과 같은 크기의 **정답 마스크**를 등록합니다. 마스크는 IoU 확인에만 사용합니다.
3. **분석 시작**을 누르면 실제 모델이 업로드 바이트를 읽어 정상·이상을 판정합니다.
4. 위쪽에는 원본과 정답 마스크, 아래쪽에는 PaDiM과 Sobel 결과가 나란히 표시됩니다.
5. 결과 보기를 히트맵·외곽선·후처리 전·후 마스크로 전환할 수 있습니다.

데이터셋의 이미지 목록을 미리 보여주거나 파일명으로 결과를 선택하지 않습니다.
새 이미지·마스크를 등록하거나 제거하면 이전 결과는 지워지며 다시 분석해야 합니다.
첫 분석은 정상 통계 준비로 시간이 걸릴 수 있습니다. 이후에는 메모리와 디스크 캐시를 재사용합니다.

## 원본 데이터

기본적으로 프로젝트 루트의 로컬 [bottle 폴더](./bottle/)를 사용합니다.
다른 위치를 쓰려면 사이드바의 정상 데이터 카테고리 경로를 변경하세요.

```text
bottle/
  train/good/정상이미지.png
  test/good/정상이미지.png
  test/결함종류/불량이미지.png
  ground_truth/결함종류/불량이미지_mask.png
```

카테고리 폴더의 정상 이미지만 통계와 임계값 준비에 사용합니다.
테스트 이미지나 업로드한 이미지로 모델을 학습하거나 임계값을 조정하지 않습니다.
업로드한 이미지의 정답을 데이터셋 파일명에서 자동으로 가져오지 않습니다.

정상 데이터 분할은 [고정 이미지 목록](./splits/bottle.json)을 따릅니다.
원본 정상 209장 중 기존 특징과 공통으로 고정한 188장을 사용하며,
**정상 통계 150장·검증 38장·테스트 83장**입니다.
추가 원본을 자동 편입하지 않으며 목록의 파일이 빠지면 중단합니다.
앱·노트북·평가·시간 측정 도구가 모두 같은 분할 파일을 재사용합니다.

이미지 추론의 첫 실행에는 사전학습 ResNet18 가중치 다운로드가 필요할 수 있습니다.
입력과 점수 맵은 256×256입니다. 특징 파일이 없어도 고정 시드로 채널을 선택하여 실행합니다.
기존 특징은 [별도 안내](./feature/README.md)에 따라 로컬에 보관할 수 있습니다.

## 결과 해석

- 두 모델 각각의 고정 임계값으로 정상·이상을 판정합니다. 두 판정이 다를 수 있습니다.
- 이상 점수는 확률이 아닙니다. 이미지 점수는 연속 점수 맵의 최댓값입니다.
- 히트맵은 원본 위 의심 영역을 표시하며 파랑은 낮은 점수, 빨강은 높은 점수입니다.
- 초기 이미지 임계값은 정상 검증 최댓값들의 99백분위수입니다.
- 초기 픽셀 임계값은 정상 검증 맵을 Gaussian sigma 4로 평활화한 값의 99.9백분위수입니다.
- IoU는 정답 마스크 등록 시에만 계산합니다. 두 마스크가 모두 비어 있으면 정의되지 않습니다.
- **Image AUROC·Pixel AUROC는 단일 업로드로 계산할 수 없어 수치를 표시하지 않습니다.**
- Sobel의 원시 엣지와 전후 마스크는 각 결과의 상세 보기에 있습니다.

CPU 시간은 공통 리사이즈부터 특징·엣지·점수·판정 계산과 후처리까지 포함합니다.
파일 읽기·디코딩·통계 준비·임계값 보정·화면 표시는 제외합니다.
단일 실행 경과 시간이며, 반복 시간 측정은 별도 도구를 사용합니다.
앱은 PyTorch와 OpenCV CPU 스레드를 각각 2개로 사용합니다.

## 검증

```powershell
# 기존 알고리즘·공통 분할·시간 범위·업로드 추론 테스트
.\.venv\Scripts\python.exe -m unittest -v test_padim test_integration test_project_contracts test_inspection

# 실제 업로드 위젯·분석 버튼·이전 결과 초기화 검증
.\.venv\Scripts\python.exe -m unittest -v test_streamlit

# 직접 구현한 Sobel과 OpenCV 수치 대조
.\.venv\Scripts\python.exe verify_sobel.py
```

원본이 있으면 업로드 추론·화면 테스트가 실제 bottle 데이터와 ResNet18을 사용합니다.
이 테스트는 프로그램 동작 확인이며 전체 테스트셋의 성능 분석을 대신하지 않습니다.
평가 지표 계산과 성능 분석은 담당 팀원이 수행합니다.

```powershell
.\.venv\Scripts\python.exe evaluate.py --category "bottle"
.\.venv\Scripts\python.exe benchmark.py --category "bottle" --threads 2
```

두 도구는 같은 공통 분할을 사용합니다. 벤치마크는 매 반복에서 원본을 다시 전처리하고
워밍업을 제외한 평균·중앙값과 단계별 시간을 기록합니다.
실행 목록·분할 ID·환경은 CSV 옆 메타데이터 JSON에 저장합니다.

## 노트북과 파일 관리

두 노트북은 [공통 데이터셋](./mvtec_dataset.py)을 불러옵니다.
Colab에서도 프로젝트 Python 모듈과 [공통 분할](./splits/bottle.json)을 함께 준비하고
프로젝트 루트로 작업 위치를 변경해야 합니다.

모델·특징·원본 데이터·압축파일·통계 캐시는 Git에서 제외합니다.
소스·노트북·분할 JSON·테스트·설정·문서는 Git으로 공유합니다.

| 파일 | 역할 |
| --- | --- |
| [app.py](./app.py) | 이미지 업로드와 2×2 결과 화면 |
| [inspection.py](./inspection.py) | 정상 원본 기반 모델 준비·업로드 바이트 추론 |
| [padim.py](./padim.py) | 정상 평균·공분산과 Mahalanobis 거리 |
| [feature_extractor.py](./feature_extractor.py) | 사전학습 ResNet18 고정 특징 추출 |
| [sobel_detector.py](./sobel_detector.py) | 팀의 직접 구현 Sobel 알고리즘 |
| [common.py](./common.py) | 공통 전처리·후처리·시간 측정·시각화 |
| [data_split.py](./data_split.py) | 고정 이미지 분할 생성·검증·재사용 |
| [padim_preview.py](./padim_preview.py) | 기존 저장 특징 처리와 정상 검증 임계값 보정 |
| [mvtec_dataset.py](./mvtec_dataset.py) | 노트북 공통 데이터셋 |
| [verify_sobel.py](./verify_sobel.py) | OpenCV 수치 대조 |
