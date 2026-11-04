"""DefectVision: UI comparing PaDiM (ResNet18) and a from-scratch Sobel baseline.

Run:  streamlit run app.py

Layout follows the proposal: Common Input | PaDiM Results | Sobel Results | Comparison.
The app only calls detector.fit() / detector.score_map() (plus optional save()/load()) and the
helpers in common.py, so replacing the internals of detectors.py requires almost no change here.

Two ways to provide an input image:
  * Pick from the dataset: give an MVTec AD category folder in the sidebar and choose a test image;
    the label and the ground-truth mask are found automatically.
  * Upload files: upload any image (and optionally its mask).
"""
from __future__ import annotations

import os

import numpy as np
import streamlit as st

from common import (
    INPUT_SIZE,
    PostConfig,
    clean_path,
    draw_contours,
    fit_or_load,
    gray_to_image,
    heatmap_overlay,
    iou,
    load_bgr,
    load_mask,
    load_normal_images,
    mask_to_image,
    mvtec_mask_path,
    preprocess,
    preprocess_mask,
    read_bgr,
    read_mask_file,
    run_method,
    scan_mvtec,
)
from detectors import PaDiMDetector, SobelDetector

AFTER, BEFORE = "After postprocessing", "Before postprocessing"
LABEL_UNKNOWN, LABEL_NORMAL, LABEL_DEFECTIVE = "Unknown", "Normal", "Defective"
LABELS = [LABEL_UNKNOWN, LABEL_NORMAL, LABEL_DEFECTIVE]
SRC_DATASET, SRC_UPLOAD = "Pick from the dataset", "Upload files"
IMG_TYPES = ["png", "jpg", "jpeg", "bmp"]
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stats_cache")
SOURCE_TEXT = {"cache": "loaded from cache", "fit": "estimated", "fit+saved": "estimated and cached"}

st.set_page_config(page_title="DefectVision", layout="wide")


# --------------------------------------------------------------------------
# Detector setup. Cached in memory by Streamlit; statistics are also cached on disk
# (see fit_or_load), so restarting the app does not re-estimate them.
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Preparing normal statistics (loaded from cache when available)...")
def load_detectors(normal_dir: str, size: int, max_images: int):
    padim, sobel = PaDiMDetector(), SobelDetector()
    imgs, used = load_normal_images(normal_dir, size, max_images)
    info = {d.name: fit_or_load(d, imgs, used, size, CACHE_DIR) for d in (padim, sobel)}
    return padim, sobel, len(imgs), info


def method_controls(det) -> tuple[PostConfig, float]:
    """Sidebar: thresholds and postprocessing parameters for each method."""
    with st.sidebar.expander(f"{det.name} settings"):
        thr = st.slider(
            "Pixel threshold", 0.0, float(det.score_range), float(det.default_threshold),
            step=0.5, key=f"{det.name}_pix_thr",
        )
        img_thr = st.slider(
            "Image threshold", 0.0, float(det.score_range), float(det.default_threshold),
            step=0.5, key=f"{det.name}_img_thr",
        )
        sigma = st.slider("Gaussian sigma", 0.0, 10.0, 4.0, step=0.5, key=f"{det.name}_sigma")
        kernel = st.slider("Morphology kernel size", 1, 15, 5, step=2, key=f"{det.name}_kernel")
        min_area = st.slider("Minimum region area (px)", 0, 500, 50, step=10, key=f"{det.name}_area")
    return PostConfig(blur_sigma=sigma, threshold=thr, morph_kernel=kernel, min_area=min_area), img_thr


def fmt_iou(v, gt_known: bool) -> str:
    if not gt_known:
        return "no ground truth"
    return "undefined" if v is None else f"{v:.3f}"


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.header("Common settings")
size = st.sidebar.select_slider("Input resolution", options=[128, 192, 256, 320], value=INPUT_SIZE)
category_in = st.sidebar.text_input(
    "MVTec AD category folder (optional)", value="",
    help="Folder that contains train/, test/ and ground_truth/ (for example D:\\mvtec\\bottle). "
         "Lets you pick test images from the dataset instead of uploading them.",
)
normal_in = st.sidebar.text_input(
    "Normal images folder (optional)", value="",
    help="Used to estimate the normal statistics. Leave empty to use <category folder>/train/good.",
)
max_images = st.sidebar.number_input("Maximum number of normal images", min_value=1, max_value=500, value=100)
st.sidebar.caption(
    "Final thresholds must be fixed on normal validation images, "
    "not tuned on the test set."
)

if st.sidebar.button("Clear statistics cache"):
    removed = 0
    if os.path.isdir(CACHE_DIR):
        for f in os.listdir(CACHE_DIR):
            if f.endswith(".npz"):
                os.remove(os.path.join(CACHE_DIR, f))
                removed += 1
    st.cache_resource.clear()
    st.sidebar.success(f"Removed {removed} cached file(s). Statistics will be estimated again.")

category = None
if category_in.strip():
    category = scan_mvtec(clean_path(category_in))
    if category is None:
        st.sidebar.error("Not an MVTec AD category folder: expected test/<defect type>/ containing images.")
normal_dir = clean_path(normal_in)
if not normal_dir and category is not None and category.train_good:
    normal_dir = category.train_good
if normal_dir and not os.path.isdir(normal_dir):
    st.sidebar.warning("Normal images folder not found.")

padim, sobel, n_normal, fit_info = load_detectors(normal_dir, int(size), int(max_images))
detectors = [padim, sobel]
controls = {d.name: method_controls(d) for d in detectors}

if n_normal:
    st.sidebar.success(f"Using {n_normal} normal images")
    for d in detectors:
        if not d.is_placeholder:
            fi = fit_info[d.name]
            st.sidebar.caption(
                f"{d.name}: statistics {SOURCE_TEXT.get(fi['source'], fi['source'])} ({fi['seconds']:.2f} s)"
            )
else:
    st.sidebar.info("No normal images yet: Sobel uses a temporary fallback for demo purposes.")

# --------------------------------------------------------------------------
# Common input
# --------------------------------------------------------------------------
st.title("DefectVision")
st.caption("Compares PaDiM (ResNet18) with a from-scratch Sobel baseline on the same image, resolution and evaluation region.")

source = SRC_UPLOAD
if category is not None:
    source = st.radio("Input source", [SRC_DATASET, SRC_UPLOAD], horizontal=True)

if source == SRC_DATASET:
    defect = st.selectbox("Defect type (test set)", list(category.test.keys()))
    paths = category.test[defect]
    names = [os.path.basename(p) for p in paths]
    picked = st.selectbox("Test image", names)
    img_path = paths[names.index(picked)]
    raw = read_bgr(img_path)
    if raw is None:
        st.error("Could not read this image file.")
        st.stop()
    img = preprocess(raw, int(size))
    caption = f"{defect}/{picked}"
    if defect == "good":
        label = LABEL_NORMAL
        gt = np.zeros(img.shape[:2], np.uint8)  # normal image: the correct mask is empty
    else:
        label = LABEL_DEFECTIVE
        mp = mvtec_mask_path(category, defect, img_path)
        m = read_mask_file(mp) if mp else None
        gt = preprocess_mask(m, int(size)) if m is not None else None
        if gt is None:
            st.warning("No ground-truth mask found for this image.")
else:
    up_col, gt_col = st.columns(2)
    up = up_col.file_uploader("Product image", type=IMG_TYPES)
    gt_up = gt_col.file_uploader("Ground-truth mask (optional)", type=IMG_TYPES)
    label = st.radio("Image label (if known)", LABELS, horizontal=True)
    if up is None:
        st.info("Upload an image to get started.")
        st.stop()
    img = preprocess(load_bgr(up.getvalue()), int(size))
    caption = "Original image (resized)"
    gt = preprocess_mask(load_mask(gt_up.getvalue()), int(size)) if gt_up is not None else None
    if gt is None and label == LABEL_NORMAL:
        gt = np.zeros(img.shape[:2], np.uint8)  # normal image: the correct mask is empty
gt_known = gt is not None

view = st.radio("Mask display", [AFTER, BEFORE], horizontal=True)

st.header("Common input")
c1, c2, c3 = st.columns(3)
c1.image(img, channels="BGR", caption=caption)
c2.metric("Image label", label)
if gt is not None:
    c3.image(mask_to_image(gt), caption="Ground-truth mask")
else:
    c3.caption("No ground-truth mask provided")

# --------------------------------------------------------------------------
# Run both methods
# --------------------------------------------------------------------------
results = {}
for d in detectors:
    cfg, img_thr = controls[d.name]
    results[d.name] = run_method(d, img, cfg, img_thr)


def show_method(det, res) -> None:
    st.header(f"{det.name} results")
    if det.is_placeholder:
        st.warning(
            f"{det.name} is still a placeholder (it returns an all-zero score map). "
            "Replace it with the real implementation in detectors.py."
        )
    mask = res.loc.mask if view == AFTER else res.loc.raw_mask

    # Heatmap scale follows the pixel threshold: full red at 2 x threshold, so it works for any score range.
    vmax = max(2.0 * controls[det.name][0].threshold, 1e-3)
    a, b = st.columns(2)
    a.image(heatmap_overlay(img, res.score_map, vmax), channels="BGR",
            caption="Anomaly score heatmap (full red at 2 x pixel threshold)")
    b.image(mask_to_image(mask), caption=f"Predicted mask ({view.lower()})")

    c, d_ = st.columns(2)
    c.image(draw_contours(img, res.loc.contours), channels="BGR", caption="Contours (after postprocessing)")
    if hasattr(det, "edges"):
        d_.image(gray_to_image(det.edges(img)), caption="Raw Sobel edges (intermediate output)")

    m1, m2, m3 = st.columns(3)
    m1.metric("Image score", f"{res.image_score:.2f}")
    m2.metric("Decision", LABEL_DEFECTIVE if res.is_defective else LABEL_NORMAL)
    m3.metric("CPU time", f"{res.ms:.1f} ms")


show_method(padim, results[padim.name])
show_method(sobel, results[sobel.name])

# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------
st.header("Comparison")
rows = []
for d in detectors:
    res = results[d.name]
    cfg, img_thr = controls[d.name]
    row = {
        "Method": d.name,
        "Image score": round(res.image_score, 3),
        "Decision": LABEL_DEFECTIVE if res.is_defective else LABEL_NORMAL,
        "IoU before postprocessing": fmt_iou(iou(res.loc.raw_mask, gt) if gt_known else None, gt_known),
        "IoU after postprocessing": fmt_iou(iou(res.loc.mask, gt) if gt_known else None, gt_known),
        "Time (ms)": round(res.ms, 1),
        "Settings": (
            f"pixel thr={cfg.threshold}, image thr={img_thr}, sigma={cfg.blur_sigma}, "
            f"kernel={cfg.morph_kernel}, min_area={cfg.min_area}"
        ),
    }
    if label == LABEL_NORMAL:
        row["False-positive area ratio"] = f"{res.loc.mask.mean():.4f}"
    rows.append(row)
st.dataframe(rows)
st.caption(
    "IoU = intersection / union. If the ground-truth and predicted masks are both empty, IoU is undefined. "
    "Time covers scoring plus postprocessing for a single run; file reading, resizing and display rendering "
    "are excluded. For official figures (mean and median after warm-up) run benchmark.py."
)
