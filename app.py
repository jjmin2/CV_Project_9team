"""DefectVision: 실제 PaDiM 히트맵 미리보기와 PaDiM·Sobel 이미지 비교 앱.

실행: python -m streamlit run app.py
"""

from pathlib import Path
import time

import cv2
import numpy as np
import streamlit as st
import torch

from common import (
    PostConfig, clean_path, draw_contours, fit_or_load, gray_to_image,
    heatmap_overlay, iou, load_bgr, load_mask,
    mask_to_image, mvtec_mask_path, postprocess, preprocess, preprocess_mask,
    read_bgr, read_mask_file, run_method, scan_mvtec, stats_signature,
)
from detectors import PaDiMDetector, SobelDetector
from padim_preview import (
    calibrate_maps, calibrate_preview, original_path, prepare_preview, test_type,
)
from data_split import SPLIT_DIR, category_split, read_split_images, split_digest


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "stats_cache"
TRAIN_FEATURES = ROOT / "feature" / "bottle_train_features.pt"
TEST_FEATURES = ROOT / "feature" / "bottle_test_features.pt"
NORMAL, DEFECTIVE = "정상", "불량"
MODE_PREVIEW = "저장 특징 미리보기"
MODE_COMPARE = "이미지 비교 (PaDiM · Sobel)"

st.set_page_config(page_title="DefectVision", layout="wide")


def file_stamp(path):
    """디스크의 파일 변경을 Streamlit 메모리 캐시에도 반영한다."""
    p = Path(path)
    info = p.stat()
    return str(p.resolve()), info.st_size, info.st_mtime_ns


@st.cache_resource(show_spinner="정상 특징 통계와 검증 임계값을 준비하고 있습니다…", max_entries=1)
def load_preview(train_path, test_path, regularization, stamps):
    bundle = prepare_preview(train_path, test_path, CACHE_DIR, regularization)
    bundle["thresholds"] = calibrate_preview(bundle)
    return bundle


@st.cache_resource(show_spinner="두 알고리즘의 정상 통계와 임계값을 준비하고 있습니다…", max_entries=1)
def load_detectors(category_root, regularization, signature):
    category = scan_mvtec(category_root)
    split = category_split(category)
    stats_images, stats_paths = read_split_images(category_root, split, "statistics_images")
    validation, _ = read_split_images(category_root, split, "validation_images")
    detectors = [PaDiMDetector(regularization=regularization), SobelDetector()]
    info, thresholds = {}, {}
    for det in detectors:
        info[det.name] = fit_or_load(det, stats_images, stats_paths, 256, str(CACHE_DIR))
        thresholds[det.name] = calibrate_maps(det.score_map(image) for image in validation)
    return detectors, thresholds, info, len(stats_images), len(validation)


@st.cache_resource(show_spinner="고정 분할로 Sobel 통계를 준비하고 있습니다…", max_entries=1)
def load_preview_sobel(category_root, signature):
    split = category_split(scan_mvtec(category_root))
    images, paths = read_split_images(category_root, split, "statistics_images")
    validation, _ = read_split_images(category_root, split, "validation_images")
    detector = SobelDetector()
    fit_or_load(detector, images, paths, 256, str(CACHE_DIR))
    return detector, calibrate_maps(detector.score_map(im) for im in validation)


def controls(name, thresholds, key):
    """정상 검증값을 초기 임계값으로 사용하고 수동 조절은 탐색용으로 제공한다."""
    with st.sidebar.expander(f"{name} 표시 설정"):
        pixel = st.number_input("픽셀 임계값", min_value=0.0, value=float(thresholds["pixel"]),
                                key=f"{key}_pixel", format="%.4f")
        image = st.number_input("이미지 임계값", min_value=0.0, value=float(thresholds["image"]),
                                key=f"{key}_image", format="%.4f")
        kernel = st.select_slider("형태학 커널 크기", options=[1, 3, 5, 7, 9], value=5, key=f"{key}_kernel")
        area = st.number_input("최소 영역 면적 (픽셀)", min_value=0, value=50, key=f"{key}_area")
    return PostConfig(blur_sigma=4.0, threshold=pixel, morph_kernel=kernel, min_area=int(area)), image


def show_metrics(score, image_threshold, milliseconds, timing_label):
    a, b, c = st.columns(3)
    a.metric("이미지 이상 점수", f"{score:.3f}")
    b.metric("판정", DEFECTIVE if score >= image_threshold else NORMAL)
    c.metric(timing_label, f"{milliseconds:.1f} ms")


def show_iou(loc, gt):
    if gt is None:
        st.caption("정답 마스크가 없어 IoU를 계산하지 않습니다.")
        return
    before, after = iou(loc.raw_mask, gt), iou(loc.mask, gt)
    a, b = st.columns(2)
    a.metric("영역 정리 전 IoU", "정의되지 않음" if before is None else f"{before:.3f}")
    b.metric("영역 정리 후 IoU", "정의되지 않음" if after is None else f"{after:.3f}")


def mask_view(key):
    return st.radio("예측 마스크 표시", ["전후 나란히", "후처리 전", "후처리 후"], horizontal=True, key=key)


def show_result(name, score_map, loc, image_score, threshold, milliseconds,
                image, gt, cfg, view, timing_label, edges=None):
    """두 알고리즘에 동일한 화면 구성을 적용한다."""
    st.subheader(name)
    vmax = max(2 * cfg.threshold, 1e-6)
    if image is not None:
        st.image(heatmap_overlay(image, score_map, vmax), channels="BGR", caption="원본 위 이상 점수 히트맵")
    else:
        color = cv2.applyColorMap((np.clip(score_map / vmax, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        st.image(color, channels="BGR", caption="이상 점수 히트맵: 파랑은 낮음, 빨강은 높음")
    if view == "전후 나란히":
        a, b = st.columns(2)
        a.image(mask_to_image(loc.raw_mask), caption="후처리 전 예측 마스크")
        b.image(mask_to_image(loc.mask), caption="후처리 후 예측 마스크")
    else:
        mask = loc.raw_mask if view == "후처리 전" else loc.mask
        st.image(mask_to_image(mask), caption=f"{view} 예측 마스크")
    if image is not None:
        st.image(draw_contours(image, loc.contours), channels="BGR", caption="후처리 후 검출 영역 외곽선")
    if edges is not None:
        st.image(gray_to_image(edges), caption="Sobel 원시 엣지 강도 · 결함 확률이 아닙니다")
    show_metrics(image_score, threshold, milliseconds, timing_label)
    show_iou(loc, gt)
    st.caption(f"픽셀 임계값 {cfg.threshold:.4f} · 이미지 임계값 {threshold:.4f} · "
               f"Gaussian sigma {cfg.blur_sigma} · 커널 {cfg.morph_kernel} · 최소 면적 {cfg.min_area}")


def run_preview(regularization):
    st.subheader("저장 특징으로 보는 실제 PaDiM 결과")
    st.caption("원본 이미지나 ResNet18 가중치 다운로드 없이 기존 bottle 특징과 정답 마스크로 실행합니다.")
    train_path = clean_path(st.sidebar.text_input("정상 학습 특징 파일", str(TRAIN_FEATURES)))
    test_path = clean_path(st.sidebar.text_input("테스트 특징 파일", str(TEST_FEATURES)))
    category_root = clean_path(st.sidebar.text_input("원본 bottle 카테고리 폴더 (선택)", ""))
    if not Path(train_path).is_file() or not Path(test_path).is_file():
        st.info("정상 학습 특징과 테스트 특징 파일의 경로를 입력하세요.")
        return
    stamps = (file_stamp(train_path), file_stamp(test_path), tuple(file_stamp(p) for p in sorted(SPLIT_DIR.glob("*.json"))))
    bundle = load_preview(train_path, test_path, regularization, stamps)
    data, det = bundle["test"], bundle["detector"]
    st.success(f"정상 통계 {bundle['n_stats']}장 · 정상 검증 {bundle['n_val']}장 · 테스트 {len(data['emb'])}장")
    st.sidebar.caption(bundle["source"])
    st.caption(f"공통 분할 ID: {split_digest(bundle['split'])}. 저장된 이미지 목록을 재사용하며 테스트 정답은 학습에 사용하지 않습니다.")
    types = sorted({test_type(p) for p in data["paths"]}, key=lambda t: (t == "good", t))
    defect = st.selectbox("결함 종류", types, key="preview_type")
    candidates = [i for i, p in enumerate(data["paths"]) if test_type(p) == defect]
    index = st.selectbox("테스트 이미지", candidates,
                         format_func=lambda i: data["paths"][i].replace("\\", "/").split("/")[-1],
                         key=f"preview_index_{defect}_{stamps[1]}")
    cfg, image_threshold = controls("PaDiM", bundle["thresholds"], f"preview_{bundle['signature']}")
    start = time.perf_counter()
    score_map = det.score_features(data["emb"][index:index + 1], output_size=(256, 256))
    loc = postprocess(score_map, cfg)
    milliseconds = (time.perf_counter() - start) * 1000
    gt = preprocess_mask((data["masks"][index, 0].numpy() > 0).astype(np.uint8), 256)
    image = None
    path = original_path(data["paths"][index], category_root)
    if path is not None:
        raw = read_bgr(str(path))
        if raw is not None:
            image = preprocess(raw, 256)
    a, b = st.columns(2)
    if image is None:
        a.info("원본 이미지가 현재 PC에 없습니다. 원본 카테고리 폴더를 지정하면 겹쳐 볼 수 있습니다.")
    else:
        a.image(image, channels="BGR", caption="원본 이미지")
    b.image(mask_to_image(gt), caption=f"정답 마스크 · 실제 라벨: {DEFECTIVE if int(data['labels'][index]) else NORMAL}")
    view = mask_view("preview_mask_view")
    left, right = st.columns(2)
    with left:
        show_result("PaDiM", score_map, loc, float(score_map.max()), image_threshold, milliseconds,
                    image, gt, cfg, view, "저장 특징 거리 계산 + 후처리")
    with right:
        category = scan_mvtec(category_root)
        if image is None or category is None or not category.train_good:
            st.subheader("Sobel")
            st.info("원본 카테고리 폴더의 train/good과 test 이미지가 필요합니다. 추가하면 같은 이미지와 고정 분할로 Sobel 결과를 나란히 표시합니다.")
        else:
            split = category_split(category)
            if split_digest(split) != split_digest(bundle["split"]):
                raise ValueError("원본과 저장 특징이 같은 공통 분할을 사용해야 합니다.")
            paths = [str(Path(category_root) / p) for p in split["statistics_images"] + split["validation_images"]]
            sobel, thresholds = load_preview_sobel(category_root, stats_signature(paths, 256, split_digest(split)))
            sobel_cfg, sobel_thr = controls("Sobel", thresholds, f"preview_sobel_{split_digest(split)}")
            result = run_method(sobel, raw, sobel_cfg, sobel_thr)
            show_result("Sobel", result.score_map, result.loc, result.image_score, sobel_thr, result.ms,
                        result.input_image, gt, sobel_cfg, view, "전처리부터 후처리까지", sobel.edges(result.input_image))
    st.info("저장 특징의 PaDiM 시간은 전처리와 ResNet18 추출을 제외합니다. 같은 범위의 전체 CPU 시간은 이미지 비교 모드에서 확인하세요.")


def run_comparison(regularization):
    st.subheader("같은 이미지에서 PaDiM과 Sobel 비교")
    st.caption("두 방법 모두 256×256 입력과 같은 후처리를 사용합니다. 최초 실행 시 ResNet18 가중치를 다운로드할 수 있습니다.")
    category_in = clean_path(st.sidebar.text_input("MVTec AD 카테고리 폴더", ""))
    category = scan_mvtec(category_in)
    if category_in and category is None:
        st.warning("카테고리 폴더에는 test/결함종류/이미지 구조가 필요합니다.")
    if category is None or not category.train_good:
        st.info("정상 이미지가 최소 세 장 있는 원본 카테고리 폴더를 입력하세요. 원본이 없으면 저장 특징 미리보기를 이용하세요.")
        for column, name in zip(st.columns(2), ("PaDiM", "Sobel")):
            column.subheader(name)
            column.info("원본 이미지가 추가되면 결과가 여기에 표시됩니다.")
        return
    split = category_split(category)
    paths = [str(Path(category.root) / p) for p in split["statistics_images"] + split["validation_images"]]
    signature = stats_signature(paths, 256, split_digest(split))
    detectors, thresholds, info, n_stats, n_val = load_detectors(category.root, regularization, signature)
    st.success(f"공통 정상 통계 {n_stats}장 · 정상 검증 {n_val}장 · 입력 256×256")
    st.caption(f"공통 분할 ID: {split_digest(split)}")
    for name, record in info.items():
        st.sidebar.caption(f"{name}: {'저장된 통계 재사용' if record['source'] == 'cache' else '통계 계산 완료'}")
    settings = {d.name: controls(d.name, thresholds[d.name], f"compare_{d.name}_{signature}_{regularization}") for d in detectors}
    options = ["데이터셋에서 선택", "이미지 업로드"] if category is not None else ["이미지 업로드"]
    source = st.radio("입력 방법", options, horizontal=True)
    if source == "데이터셋에서 선택":
        fixed_test = {}
        for relative in split["test_images"]:
            fixed_test.setdefault(relative.split("/")[1], []).append(str(Path(category.root) / relative))
        defect = st.selectbox("결함 종류", list(fixed_test))
        path = st.selectbox("테스트 이미지", fixed_test[defect], format_func=lambda p: Path(p).name, key=f"compare_image_{defect}")
        raw = read_bgr(path)
        if raw is None:
            raise ValueError("선택한 이미지를 읽을 수 없습니다.")
        image = preprocess(raw, 256)
        label = NORMAL if defect == "good" else DEFECTIVE
        mask_path = mvtec_mask_path(category, defect, path)
        gt_raw = read_mask_file(mask_path) if mask_path else None
        gt = np.zeros((256, 256), np.uint8) if defect == "good" else (preprocess_mask(gt_raw, 256) if gt_raw is not None else None)
    else:
        a, b = st.columns(2)
        upload = a.file_uploader("제품 이미지", type=["png", "jpg", "jpeg", "bmp"])
        mask_upload = b.file_uploader("정답 마스크 (선택)", type=["png", "jpg", "jpeg", "bmp"])
        label = st.radio("실제 이미지 라벨", ["알 수 없음", NORMAL, DEFECTIVE], horizontal=True)
        if upload is None:
            st.info("비교할 제품 이미지를 업로드하세요.")
            return
        raw = load_bgr(upload.getvalue())
        image = preprocess(raw, 256)
        gt = preprocess_mask(load_mask(mask_upload.getvalue()), 256) if mask_upload is not None else None
        if gt is None and label == NORMAL:
            gt = np.zeros((256, 256), np.uint8)
    a, b = st.columns(2)
    a.image(image, channels="BGR", caption=f"공통 입력 · 실제 라벨: {label}")
    if gt is not None:
        b.image(mask_to_image(gt), caption="정답 마스크")
    view = mask_view("comparison_mask_view")
    rows = []
    for column, det in zip(st.columns(2), detectors):
        with column:
            cfg, image_threshold = settings[det.name]
            result = run_method(det, raw, cfg, image_threshold)
            edges = det.edges(result.input_image) if hasattr(det, "edges") else None
            show_result(det.name, result.score_map, result.loc, result.image_score, image_threshold, result.ms,
                        result.input_image, gt, cfg, view, "전처리부터 후처리까지", edges)
            rows.append({"방법": det.name, "이미지 점수": result.image_score,
                         "판정": DEFECTIVE if result.is_defective else NORMAL, "처리 시간 (ms)": result.ms,
                         "전처리 (ms)": result.preprocess_ms, "점수 계산 (ms)": result.score_ms,
                         "후처리 (ms)": result.postprocess_ms,
                         "후처리 후 IoU": iou(result.loc.mask, gt) if gt is not None else None})
    st.dataframe(rows, hide_index=True)
    st.caption("처리 시간은 원본의 공통 리사이즈부터 점수·판정 계산과 후처리까지 포함합니다. 파일 읽기·통계 준비·화면 렌더링은 제외합니다. 단일 실행 경과 시간이므로 반복 측정은 별도 벤치마크를 사용하세요.")


def main():
    st.title("DefectVision")
    st.caption("CNN 기반 산업 이상 탐지와 결함 위치 시각화")
    mode = st.sidebar.radio("실행 모드", [MODE_PREVIEW, MODE_COMPARE])
    threads = int(st.sidebar.selectbox("CPU 스레드 수 (PyTorch · OpenCV)", [1, 2, 4], index=1))
    torch.set_num_threads(threads)
    cv2.setNumThreads(threads)
    regularization = st.sidebar.number_input("공분산 정규화 계수", min_value=0.000001, value=0.01, format="%.6f")
    st.sidebar.caption("기본 임계값은 정상 검증 점수에서 설정합니다. 수동 변경 결과는 탐색용이며 최종 성능 평가와 구분하세요.")
    if st.sidebar.button("통계 캐시 비우기"):
        if CACHE_DIR.is_dir():
            for path in CACHE_DIR.glob("*.npz"):
                path.unlink()
        st.cache_resource.clear()
        st.sidebar.success("통계 캐시를 비웠습니다.")
    try:
        if mode == MODE_PREVIEW:
            run_preview(regularization)
        else:
            run_comparison(regularization)
    except Exception as error:
        st.error(f"실행하지 못했습니다: {error}")
        st.info("데이터 경로와 패키지 설치를 확인하세요. 이미지 비교의 최초 실행에는 가중치 다운로드를 위한 인터넷 연결이 필요할 수 있습니다.")


if __name__ == "__main__":
    main()
