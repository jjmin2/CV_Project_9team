"""업로드한 제품 이미지를 실제 PaDiM·Sobel로 검사하는 Streamlit 화면.

실행: python -m streamlit run app.py
"""

import hashlib
import html
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import cv2

from common import draw_contours, gray_to_image, heatmap_overlay, iou, mask_to_image, preprocess, preprocess_mask
from inspection import DEFAULT_CATEGORY, analyze_upload, decode_upload, model_signature, prepare_models


st.set_page_config(page_title="DefectVision · 제품 검사", layout="wide", initial_sidebar_state="collapsed")

STYLES = """
<style>
.stApp {background:#191919; color:#f4f4f4;}
[data-testid="stHeader"] {background:transparent;}
.block-container {max-width:1140px; padding:1.5rem 1.6rem 2rem;}
.dv-header {display:flex; justify-content:space-between; align-items:center; margin:0 0 8px;}
.dv-brand {font-size:28px; font-weight:750; letter-spacing:-.7px;}
.dv-tag {background:#2a2a2a; border:1px solid #343434; border-radius:30px;
    padding:6px 14px; font-size:13px; color:#dedede;}
.dv-intro {color:#aaa; font-size:14px; margin:0 0 20px;}
.st-key-dv_upload, .st-key-dv_original, .st-key-dv_ground_truth,
.st-key-dv_padim, .st-key-dv_sobel {background:#111; border-radius:18px;}
.st-key-dv_original, .st-key-dv_ground_truth {min-height:315px;}
.dv-placeholder {height:215px; display:flex; flex-direction:column; align-items:center;
    justify-content:center; gap:12px; color:#aaa; text-align:center; border-radius:12px;}
.dv-placeholder svg {width:34px; height:34px; color:#d7d7d7;}
.dv-placeholder strong {color:#eee; font-size:18px; font-weight:500;}
.dv-placeholder small {max-width:300px; font-size:13px; line-height:1.5;}
[data-testid="stImage"] {margin:auto;}
[data-testid="stImage"] img {max-height:255px; object-fit:contain; border-radius:10px;}
[data-testid="stMetricValue"] {font-size:1.25rem;}
[data-testid="stFileUploader"] section {background:#111; border-radius:12px;}
.dv-result {font-size:14px; font-weight:650; padding:7px 12px; display:inline-block;
    border-radius:20px; margin-bottom:9px;}
.dv-normal {color:#b8e9c5; background:#173124;}
.dv-anomaly {color:#ffc0b6; background:#3b211e;}
.dv-footer {color:#929292; font-size:12px; margin-top:8px; line-height:1.6;}
@media(max-width:700px){.block-container{padding:1rem;} .dv-brand{font-size:24px;}}
</style>
"""


@st.cache_resource(show_spinner=False, max_entries=1)
def load_models(category_root, signature):
    """같은 정상 데이터의 모델을 재사용해 매 업로드마다 재학습하지 않는다."""
    return prepare_models(category_root)


def upload_fingerprint(image_bytes, mask_bytes, signature):
    """이미지·정답·모델이 바뀌면 이전 이미지의 결과를 표시하지 않는다."""
    digest = hashlib.sha256()
    for value in (image_bytes or b"", mask_bytes or b"", signature.encode("utf-8")):
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


def placeholder(title, description, symbol="image"):
    """사진 대신 상태를 명확히 설명하는 빈 결과 칸을 표시한다."""
    paths = {
        "image": '<rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="16" cy="8" r="2"/><path d="m3 16 5-5 6 6 3-3 4 4"/>',
        "scan": '<path d="M8 3H5a2 2 0 0 0-2 2v3m13-5h3a2 2 0 0 1 2 2v3M3 16v3a2 2 0 0 0 2 2h3m8 0h3a2 2 0 0 0 2-2v-3"/><circle cx="11" cy="11" r="4"/><path d="m14 14 3 3"/>',
    }
    st.markdown(f'<div class="dv-placeholder"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7">'
                f'{paths[symbol]}</svg><strong>{html.escape(title)}</strong><small>{html.escape(description)}</small></div>',
                unsafe_allow_html=True)


def show_result(name, inspection, models, view):
    """두 알고리즘의 동일한 결과 구성을 나란히 보여준다."""
    st.subheader(f"{name} 결과")
    if inspection is None:
        placeholder("분석 대기", "제품 이미지를 등록하고 분석 시작을 눌러주세요.", "scan")
        return
    result = inspection.results[name]
    config = models["configs"][name]
    verdict = "이상" if result.is_defective else "정상"
    color = "dv-anomaly" if result.is_defective else "dv-normal"
    st.markdown(f'<div class="dv-result {color}">{verdict}</div>', unsafe_allow_html=True)
    if view == "히트맵":
        output = heatmap_overlay(inspection.image, result.score_map, max(2 * config.threshold, 1e-6))
        st.image(output, channels="BGR", width="stretch", caption="원본 위 이상 점수 · 파랑은 낮음, 빨강은 높음")
    elif view == "외곽선":
        st.image(draw_contours(inspection.image, result.loc.contours), channels="BGR", width="stretch", caption="의심 영역 외곽선")
    else:
        mask = result.loc.raw_mask if view == "후처리 전" else result.loc.mask
        st.image(mask_to_image(mask), width="stretch", caption=f"{view} 예측 마스크")
    a, b, c = st.columns(3)
    a.metric("이상 점수", f"{result.image_score:.3f}")
    b.metric("CPU 시간", f"{result.ms:.1f} ms")
    overlap = iou(result.loc.mask, inspection.ground_truth) if inspection.ground_truth is not None else None
    c.metric("IoU", "—" if overlap is None else f"{overlap:.3f}")
    with st.expander(f"{name} 상세 결과"):
        a, b = st.columns(2)
        a.image(mask_to_image(result.loc.raw_mask), caption="후처리 전 마스크", width="stretch")
        b.image(mask_to_image(result.loc.mask), caption="후처리 후 마스크", width="stretch")
        if name == "Sobel":
            detector = next(d for d in models["detectors"] if d.name == name)
            st.image(gray_to_image(detector.edges(inspection.image)), caption="원시 Sobel 엣지 · 결함 확률이 아닙니다")
        st.caption(f"이미지 임계값 {models['thresholds'][name]['image']:.4f} · 픽셀 임계값 {config.threshold:.4f}")
        st.caption(f"전처리 {result.preprocess_ms:.1f} ms · 점수 계산 {result.score_ms:.1f} ms · 후처리 {result.postprocess_ms:.1f} ms")
        if inspection.ground_truth is None:
            st.caption("정답 마스크를 등록하면 IoU를 계산합니다.")
        elif overlap is None:
            st.caption("정답과 예측 마스크가 모두 비어 있어 IoU가 정의되지 않습니다.")


def main():
    st.markdown(STYLES, unsafe_allow_html=True)
    st.markdown('<div class="dv-header"><div class="dv-brand">DefectVision</div><div class="dv-tag">bottle 제품 검사</div></div>', unsafe_allow_html=True)
    st.markdown('<p class="dv-intro">제품 이미지를 등록하면 정상·이상 여부와 의심되는 결함 위치를 분석합니다.</p>', unsafe_allow_html=True)
    torch.set_num_threads(2)
    cv2.setNumThreads(2)

    with st.sidebar:
        st.header("분석 환경")
        category_root = st.text_input("정상 데이터 카테고리 폴더", str(DEFAULT_CATEGORY), key="category_root")
        st.caption("정상 학습·검증 데이터로 준비한 고정 임계값을 사용합니다. 업로드 이미지로 재학습하지 않습니다.")
    signature, model_error = "", None
    try:
        signature = model_signature(category_root)
    except Exception as error:
        model_error = str(error)

    with st.container(border=True, key="dv_upload"):
        st.markdown("**제품 이미지 업로드**")
        upload_col, action_col = st.columns([4, 1], vertical_alignment="center")
        with upload_col:
            uploaded = st.file_uploader("제품 이미지", type=["png", "jpg", "jpeg", "bmp"],
                                        key="product_upload", label_visibility="collapsed")
        with action_col:
            analyze = st.button("분석 시작", type="primary", disabled=uploaded is None or model_error is not None,
                                key="analyze_button", width="stretch")
        st.caption("검사할 bottle 제품 이미지 한 장을 등록하세요. PNG · JPG · JPEG · BMP")

    image_bytes = uploaded.getvalue() if uploaded is not None else None
    raw, input_error = None, None
    if image_bytes is not None:
        try:
            raw, _ = decode_upload(image_bytes)
        except ValueError as error:
            input_error = str(error)

    left, right = st.columns(2, gap="medium")
    with left, st.container(border=True, key="dv_original"):
        st.subheader("원본 이미지")
        if raw is None:
            placeholder("원본 이미지", "업로드한 이미지가 여기에 표시됩니다.")
        else:
            st.image(preprocess(raw, 256), channels="BGR", width="stretch")
            st.caption(f"{uploaded.name} · 원본 {raw.shape[1]}×{raw.shape[0]} · 분석 256×256")
    with right, st.container(border=True, key="dv_ground_truth"):
        st.subheader("정답 마스크")
        with st.popover("정답 마스크 등록 (선택)", width="stretch"):
            mask_upload = st.file_uploader("정답 마스크", type=["png", "jpg", "jpeg", "bmp"], key="mask_upload")
        mask_bytes = mask_upload.getvalue() if mask_upload is not None else None
        if mask_bytes is None:
            placeholder("정답 마스크 · 선택 사항", "마스크 없이도 정상·이상 분석이 가능합니다. 마스크는 IoU 확인에만 사용합니다.", "scan")
        elif raw is None:
            st.info("정답 마스크와 비교할 제품 이미지를 먼저 등록하세요.")
        else:
            try:
                _, mask = decode_upload(image_bytes, mask_bytes)
                st.image(mask_to_image(preprocess_mask(mask, 256)), width="stretch")
            except ValueError as error:
                input_error = str(error)

    fingerprint = upload_fingerprint(image_bytes, mask_bytes, signature)
    if st.session_state.get("inspection_key") != fingerprint:
        st.session_state.pop("inspection_result", None)
        st.session_state.pop("inspection_models", None)
    if model_error:
        st.error(f"모델 준비에 필요한 정상 데이터를 확인하세요: {model_error}")
    if input_error:
        st.error(input_error)
    if analyze and image_bytes is not None and not input_error and not model_error:
        try:
            with st.spinner("제품을 분석하고 있습니다. 최초 실행은 정상 통계 준비로 시간이 더 걸릴 수 있습니다…"):
                models = load_models(str(Path(category_root).resolve()), signature)
                inspection = analyze_upload(image_bytes, models, mask_bytes)
            st.session_state["inspection_result"] = inspection
            st.session_state["inspection_models"] = models
            st.session_state["inspection_key"] = fingerprint
        except Exception as error:
            st.error(f"분석하지 못했습니다: {error}")

    inspection = st.session_state.get("inspection_result")
    models = st.session_state.get("inspection_models")
    view = st.radio("결과 보기", ["히트맵", "외곽선", "후처리 전", "후처리 후"], horizontal=True, key="result_view")
    left, right = st.columns(2, gap="medium")
    with left, st.container(border=True, key="dv_padim"):
        show_result("PaDiM", inspection, models, view)
    with right, st.container(border=True, key="dv_sobel"):
        show_result("Sobel", inspection, models, view)

    st.markdown('<div class="dv-footer">Image AUROC · Pixel AUROC: 단일 이미지에서는 계산하지 않습니다. '
                'IoU: 정답 마스크 등록 시 표시 · CPU Time: 전처리부터 후처리까지</div>', unsafe_allow_html=True)
    if inspection is not None:
        with st.expander("분석 정보"):
            st.caption(f"고정 분할 {models['split_id']} · 정상 통계 {len(models['split']['statistics_images'])}장 · 검증 {len(models['split']['validation_images'])}장")
            st.caption("이상 점수는 확률이 아닙니다. CPU 시간은 파일 읽기·디코딩, 통계 준비, 화면 표시를 제외한 단일 실행 경과 시간입니다.")


if __name__ == "__main__":
    main()
