from __future__ import annotations

import io
import time
from datetime import datetime
from typing import Dict, List

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import streamlit as st

matplotlib.use("Agg")

from waveforge_backend import VALID_WAVELETS, compress_image_array, export_image_bytes, load_uploaded_image
from metrics import (
    actual_compression_ratio,
    compute_quality_metrics,
    generate_report_csv,
    generate_report_text,
    size_reduction_percent,
)

st.set_page_config(page_title="WaveForge", page_icon="🌊", layout="wide")
st.session_state.setdefault("recent_results", [])


def _normalize_for_display(image_array: np.ndarray) -> np.ndarray:
    normalized = np.asarray(image_array, dtype=np.float32)
    if normalized.size == 0:
        return normalized.astype(np.uint8)
    minimum = float(np.min(normalized))
    maximum = float(np.max(normalized))
    if maximum <= minimum:
        return np.zeros_like(normalized, dtype=np.uint8)
    normalized = (normalized - minimum) / (maximum - minimum)
    return (normalized * 255.0).astype(np.uint8)


def _build_report_rows(result: Dict[str, object]) -> List[Dict[str, object]]:
    return [{
        "image_dimensions": f"{result['original_dimensions'][0]}x{result['original_dimensions'][1]}",
        "output_dimensions": f"{result['output_dimensions'][0]}x{result['output_dimensions'][1]}",
        "original_file_size_bytes": int(result["original_file_size_bytes"]),
        "exported_file_size_bytes": int(result["exported_file_size_bytes"]),
        "compression_ratio": float(result["compression_ratio"]),
        "file_size_reduction_percent": float(result["file_size_reduction_percent"]),
        "psnr_db": float(result["psnr_db"]),
        "ssim": float(result["ssim"]),
        "processing_time_seconds": float(result["processing_time_seconds"]),
        "wavelet": str(result["wavelet"]),
        "mode": str(result["mode"]),
        "decomposition_level": int(result["decomposition_level"]),
        "compression_strength": int(result["compression_strength"]),
    }]


def _gray_histogram(arr: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(4.8, 2.7))
    ax.hist(np.asarray(arr).ravel(), bins=256, range=(0, 256), color="#8b5cf6", alpha=0.8)
    ax.set_title(title)
    ax.set_xlim(0, 255)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def _render_processed_download_panel(result: Dict[str, object], reconstructed: np.ndarray) -> None:
    png_bytes = result.get("png_bytes")
    jpeg_bytes = result.get("jpeg_bytes")
    if not isinstance(png_bytes, (bytes, bytearray)) or len(png_bytes) == 0:
        st.error("Processed image export failed: PNG bytes are missing or empty.")
        return
    if not isinstance(jpeg_bytes, (bytes, bytearray)) or len(jpeg_bytes) == 0:
        st.error("Processed image export failed: JPEG bytes are missing or empty.")
        return

    try:
        from PIL import Image
        Image.open(io.BytesIO(png_bytes)).verify()
        Image.open(io.BytesIO(jpeg_bytes)).verify()
    except Exception as exc:  # pragma: no cover - validation guard in UI
        st.error(f"Processed image export failed: {exc}")
        return

    st.markdown("---")
    st.subheader("Download Processed Image")
    st.success("Processed image ready for download.")
    info_col, preview_col = st.columns([1, 2])
    with info_col:
        st.write(f"Filename: {result['file_name']}")
        st.write(f"Dimensions: {result['output_dimensions'][0]}x{result['output_dimensions'][1]}")
        st.write(f"Image format: PNG / JPEG")
        st.write(f"Actual file size: {len(png_bytes)} bytes")
    with preview_col:
        st.image(reconstructed, caption=f"Final processed image — {result['file_name']}", use_container_width=True)

    download_col_1, download_col_2 = st.columns(2)
    with download_col_1:
        st.download_button(
            "Download Processed Image — PNG",
            data=png_bytes,
            file_name="waveforge_processed.png",
            mime="image/png",
        )
    with download_col_2:
        st.download_button(
            "Download Processed Image — JPEG",
            data=jpeg_bytes,
            file_name="waveforge_processed.jpg",
            mime="image/jpeg",
        )


def _wavelet_subbands(gray_image: np.ndarray, wavelet: str):
    coeffs = pywt.wavedec2(gray_image.astype(np.float32), wavelet=wavelet, level=1, mode="symmetric")
    ll, (lh, hl, hh) = coeffs[0], coeffs[1]
    return {"LL": ll, "LH": lh, "HL": hl, "HH": hh}


def _dct_preview(gray_image: np.ndarray) -> np.ndarray:
    gray = np.asarray(gray_image, dtype=np.float32)
    rows = min(gray.shape[0], 64)
    cols = min(gray.shape[1], 64)
    patch = gray[:rows, :cols]
    return np.abs(cv2.dct(patch))


st.markdown(
    """
    <style>
    :root {
        --bg: #07111f;
        --panel: rgba(15, 23, 42, 0.92);
        --panel-soft: rgba(30, 41, 59, 0.88);
        --violet: #8b5cf6;
        --indigo: #6366f1;
        --cyan: #38bdf8;
        --text: #e2e8f0;
        --muted: #a5b4cf;
        --border: rgba(148, 163, 184, 0.20);
    }
    .stApp { background: linear-gradient(180deg, #07111f 0%, #0f172a 100%); color: var(--text); }
    .block-container { padding-top: 1.8rem; padding-bottom: 2rem; }
    div[data-testid="stSidebar"] { background: rgba(8, 15, 27, 0.96); }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
    .stMetric { background: rgba(99, 102, 241, 0.10); border: 1px solid var(--border); border-radius: 14px; padding: 0.8rem; }
    .stDataFrame { background: rgba(15, 23, 42, 0.82); border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("WaveForge")
st.caption("Hybrid Image Compression Studio")

with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader(
        "Upload image",
        type=["png", "jpg", "jpeg", "webp"],
        help="Supported formats: PNG, JPG, JPEG, WEBP",
        key="uploaded_image",
    )

    if uploaded_file is not None:
        try:
            preview_image, original_format, _, _ = load_uploaded_image(uploaded_file)
            color_kind = "Grayscale" if preview_image.ndim == 2 else "Color (RGB)"
            st.info(
                f"{uploaded_file.name}\nFormat: {original_format}\nSize: {len(uploaded_file.getvalue())} bytes\nDimensions: {preview_image.shape[1]}x{preview_image.shape[0]}\nType: {color_kind}"
            )
        except Exception:
            st.warning("Unable to inspect the uploaded image metadata.")

    with st.form("compression_form"):
        compression_strength = st.slider(
            "Compression strength",
            min_value=5,
            max_value=95,
            value=55,
            step=1,
            help="Lower values preserve more detail; higher values improve compression.",
        )
        mode = st.selectbox(
            "Processing mode",
            options=["balanced", "high_quality", "high_compression"],
            index=0,
            help="Balanced is a practical all-round setting; quality favors detail preservation; compression favors smaller output.",
        )
        wavelet = st.selectbox("Wavelet selection", options=VALID_WAVELETS, index=0, help="Haar and DB1 are supported.")
        decomposition_level = st.selectbox(
            "Wavelet decomposition level",
            options=[1, 2],
            index=0,
            help="Level 1 is more reliable and lightweight for a stable CPU pipeline.",
        )
        st.markdown("---")
        submit_pressed = st.form_submit_button("Compress Image", type="primary")
        reset_button = st.form_submit_button("Reset")
        clear_image_button = st.form_submit_button("Clear Image")

    if reset_button:
        st.session_state.pop("latest_result", None)
        st.rerun()

    if clear_image_button:
        st.session_state.pop("latest_result", None)
        st.session_state["uploaded_image"] = None
        st.rerun()

    st.markdown("---")
    if st.session_state.get("recent_results"):
        st.caption("Recent results")
        for item in st.session_state["recent_results"][-4:]:
            st.write(f"- {item['filename']} | {item['compression_strength']} | {item['mode']}")
        if st.button("Clear history", key="clear_history_button"):
            st.session_state["recent_results"] = []
            st.rerun()
    st.caption("Local CPU processing. No paid service or account required.")


if uploaded_file is None:
    st.markdown("# WaveForge")
    st.write(
        "WaveForge is a hybrid DCT–DWT image compression studio for exploring trade-offs between image quality and file size on a CPU. Upload a PNG, JPG, JPEG, or WEBP image to begin."
    )
    st.info("Upload an image to begin hybrid DCT–DWT compression.")
    st.stop()

if submit_pressed:
    try:
        progress = st.progress(0, text="Preparing image...")
        with st.spinner("Applying hybrid DCT–DWT compression..."):
            start_time = time.perf_counter()
            original_array, original_format, _, file_name = load_uploaded_image(uploaded_file)
            progress.progress(25, text="Wavelet filtering...")
            reconstructed = compress_image_array(
                original_array,
                compression_strength=compression_strength,
                mode=mode,
                wavelet=wavelet,
                decomposition_level=decomposition_level,
            )
            progress.progress(83, text="Reconstructing image and computing metrics...")
            processed_time = time.perf_counter() - start_time

            original_bytes = len(uploaded_file.getvalue())
            png_bytes = export_image_bytes(reconstructed, "PNG")
            jpeg_bytes = export_image_bytes(reconstructed, "JPEG")
            metrics = compute_quality_metrics(original_array.astype(np.uint8), reconstructed.astype(np.uint8))
            ratio = actual_compression_ratio(original_bytes, len(png_bytes))
            reduction = size_reduction_percent(original_bytes, len(png_bytes))

            result = {
                "original_dimensions": original_array.shape[:2],
                "output_dimensions": reconstructed.shape[:2],
                "original_file_size_bytes": original_bytes,
                "exported_file_size_bytes": len(png_bytes),
                "compression_ratio": ratio,
                "file_size_reduction_percent": reduction,
                "psnr_db": metrics["psnr_db"],
                "ssim": metrics["ssim"],
                "processing_time_seconds": processed_time,
                "wavelet": wavelet,
                "mode": mode,
                "decomposition_level": decomposition_level,
                "compression_strength": compression_strength,
                "png_bytes": png_bytes,
                "jpeg_bytes": jpeg_bytes,
                "file_name": file_name,
                "original_format": original_format,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            st.session_state["latest_result"] = result
            history = st.session_state.setdefault("recent_results", [])
            history.append({
                "filename": file_name,
                "compression_strength": compression_strength,
                "mode": mode,
                "wavelet": wavelet,
                "timestamp": result["timestamp"],
            })
            if len(history) > 6:
                history[:] = history[-6:]
            progress.progress(100, text="Compression complete")
        st.success("Compression finished successfully.")
    except Exception as exc:
        st.error(f"Processing failed: {exc}")
        st.stop()

if "latest_result" in st.session_state:
    result = st.session_state["latest_result"]
    original_array, _, _, _ = load_uploaded_image(uploaded_file)
    reconstructed = compress_image_array(
        original_array,
        compression_strength=result["compression_strength"],
        mode=result["mode"],
        wavelet=result["wavelet"],
        decomposition_level=result["decomposition_level"],
    )

    nav = st.sidebar.radio(
        "Navigation",
        ["Compression", "Algorithm Analysis", "Comparison", "Image Analytics", "Downloads"],
        index=0,
    )

    top_cards = st.columns(6)
    with top_cards[0]:
        st.metric("Original size", f"{result['original_file_size_bytes']} B", help="Actual size of the uploaded image file in bytes.")
    with top_cards[1]:
        st.metric("Exported size", f"{result['exported_file_size_bytes']} B", help="Actual PNG file size of the reconstructed output image.")
    with top_cards[2]:
        st.metric("Compression ratio", f"{result['compression_ratio']:.2f}x", help="Original file bytes divided by exported file bytes.")
    with top_cards[3]:
        st.metric("Reduction", f"{result['file_size_reduction_percent']:.2f}%", help="Percentage reduction in file size relative to the original image file.")
    with top_cards[4]:
        st.metric("PSNR", f"{result['psnr_db']:.2f} dB", help="Higher PSNR generally indicates lower pixel-level reconstruction error.")
    with top_cards[5]:
        st.metric("SSIM", f"{result['ssim']:.4f}", help="SSIM measures structural similarity between the original and reconstructed images.")

    if nav == "Compression":
        st.markdown("### 1. Hybrid DCT–DWT Image Compression")
        st.write(
            "Image Preparation → DWT → DCT → Quantization → Inverse DCT → Inverse DWT → Reconstruction → Evaluation. The current implementation follows this sequence with local CPU processing and keeps the actual transform pipeline in the backend module."
        )
        st.markdown("### 6. Image Reconstruction")
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_array, caption=f"Original image — {result['file_name']}")
        with col2:
            st.image(reconstructed, caption="Reconstructed image")
        st.write(f"Original dimensions: {result['original_dimensions']} | Reconstructed dimensions: {result['output_dimensions']}")
        st.write(f"Original file size: {result['original_file_size_bytes']} bytes | Exported file size: {result['exported_file_size_bytes']} bytes")
        st.write(f"Compression settings: strength={result['compression_strength']}, mode={result['mode']}, wavelet={result['wavelet']}, level={result['decomposition_level']}")

        diff_map = np.abs(original_array.astype(np.int16) - reconstructed.astype(np.int16)).astype(np.uint8)
        st.image(_normalize_for_display(diff_map), caption="Absolute difference map (actual aligned pixels)", clamp=True)

        st.markdown("### 5. Compression Ratio")
        st.caption("Actual file compression ratio = original file bytes / exported compressed file bytes. File-size reduction (%) = 100 × (original file bytes − exported compressed file bytes) / original file bytes.")
        if result["exported_file_size_bytes"] < result["original_file_size_bytes"]:
            st.success("The exported output is smaller than the original file, so the image is compressed in actual file size.")
        else:
            st.warning("The exported file is not smaller than the original. This may occur for very small or highly detailed images, and it is reported honestly.")

        st.markdown("### 7. Image Quality Evaluation")
        st.write("PSNR and SSIM are computed from the actual original and reconstructed arrays. Higher PSNR generally indicates lower pixel-level reconstruction error, while SSIM evaluates structural similarity and perceptual fidelity.")
        st.metric("Processing time", f"{result['processing_time_seconds']:.3f} s", help="Time spent performing the current compression and reconstruction pipeline.")

    elif nav == "Algorithm Analysis":
        st.markdown("### 2. Discrete Cosine Transform (DCT)")
        st.write(
            "DCT transforms each image block into frequency coefficients. The large low-frequency coefficients preserve most of the image energy, while smaller high-frequency coefficients encode detail and texture. Quantization reduces the magnitude of these coefficients to save storage and maintain a practical quality–size trade-off."
        )
        gray = original_array if original_array.ndim == 2 else np.mean(original_array, axis=2)
        dct_map = _dct_preview(gray)
        st.image(_normalize_for_display(dct_map), caption="Representative 2D DCT coefficient map (magnitude)", clamp=True)

        st.markdown("### 3. Discrete Wavelet Transform (DWT)")
        st.write(
            "DWT decomposes the image into multiple levels of approximation and detail. LL preserves the low-frequency approximation, while LH, HL and HH capture horizontal, vertical and diagonal detail information. This makes DWT useful for separating strong structure from detail that can be compressed more aggressively."
        )
        gray = original_array if original_array.ndim == 2 else np.mean(original_array, axis=2).astype(np.uint8)
        subbands = _wavelet_subbands(gray, result["wavelet"])
        cols = st.columns(4)
        for index, label in enumerate(["LL", "LH", "HL", "HH"]):
            with cols[index]:
                st.image(_normalize_for_display(subbands[label]), caption=f"{label} subband", clamp=True)
        with st.expander("Wavelet explanation"):
            st.markdown(
                "- LL: low-frequency approximation.\n"
                "- LH: horizontal detail.\n"
                "- HL: vertical detail.\n"
                "- HH: diagonal detail."
            )

        st.markdown("### 4. Quantization")
        st.write(
            "Quantization controls the strength of coefficient rounding. Higher quantization values reduce the number of retained coefficients and increase compression, but they also reduce reconstruction fidelity. The selected value is applied directly in the real processing pipeline."
        )
        st.info(f"Current quantization strength: {result['compression_strength']} | Mode: {result['mode']} | Wavelet: {result['wavelet']} | Level: {result['decomposition_level']}")

    elif nav == "Comparison":
        st.markdown("## Additional Feature 1: Quality vs Compression Comparison")
        st.write(
            "This comparison runs only when requested. It evaluates three compression strengths on the same uploaded image and compares resulting output size, ratio, PSNR and SSIM. Use the results to choose between quality, size reduction or a balanced setting."
        )
        strengths = [25, 50, 75]
        comparison_rows = []
        progress = st.progress(0, text="Comparing compression settings...")
        for index, strength in enumerate(strengths, start=1):
            start = time.perf_counter()
            candidate = compress_image_array(
                original_array,
                compression_strength=strength,
                mode=result["mode"],
                wavelet=result["wavelet"],
                decomposition_level=result["decomposition_level"],
            )
            elapsed = time.perf_counter() - start
            metrics_candidate = compute_quality_metrics(original_array.astype(np.uint8), candidate.astype(np.uint8))
            export_size = len(export_image_bytes(candidate, "PNG"))
            comparison_rows.append({
                "Compression setting": strength,
                "Actual output size (bytes)": export_size,
                "Compression ratio": actual_compression_ratio(result["original_file_size_bytes"], export_size),
                "PSNR (dB)": metrics_candidate["psnr_db"],
                "SSIM": metrics_candidate["ssim"],
                "Processing time (s)": float(elapsed),
            })
            progress.progress(index / len(strengths), text=f"Evaluating compression strength {strength}...")

        comparison_df = pd.DataFrame(comparison_rows)
        st.dataframe(comparison_df, use_container_width=True)
        chart_df = comparison_df.set_index("Compression setting")[["Compression ratio", "PSNR (dB)", "SSIM"]]
        st.line_chart(chart_df, use_container_width=True)

        if comparison_rows:
            best_quality = max(comparison_rows, key=lambda row: row["PSNR (dB)"])
            smallest_file = min(comparison_rows, key=lambda row: row["Actual output size (bytes)"])
            st.caption(
                f"Best quality result: setting {best_quality['Compression setting']} | Smallest output: setting {smallest_file['Compression setting']}"
            )

        comparison_csv = generate_report_csv([
            {
                "compression_setting": int(row["Compression setting"]),
                "actual_output_size_bytes": int(row["Actual output size (bytes)"]),
                "compression_ratio": float(row["Compression ratio"]),
                "psnr_db": float(row["PSNR (dB)"]),
                "ssim": float(row["SSIM"]),
                "processing_time_seconds": float(row["Processing time (s)"]),
            }
            for row in comparison_rows
        ])
        st.session_state["comparison_csv"] = comparison_csv
        st.download_button(
            "Download comparison CSV",
            comparison_csv,
            file_name="waveforge_comparison.csv",
            mime="text/csv",
        )

    elif nav == "Image Analytics":
        st.markdown("## Additional Feature 2: Image Analytics")
        st.write("These visualizations use actual grayscale intensities, DCT coefficients and pixel differences from the processed image. They help interpret whether compression is primarily affecting detail, structure or global brightness distribution.")
        original_gray = original_array if original_array.ndim == 2 else np.mean(original_array, axis=2).astype(np.uint8)
        reconstructed_gray = reconstructed if reconstructed.ndim == 2 else np.mean(reconstructed, axis=2).astype(np.uint8)

        hist_cols = st.columns(2)
        with hist_cols[0]:
            st.pyplot(_gray_histogram(original_gray, "Original grayscale histogram"))
        with hist_cols[1]:
            st.pyplot(_gray_histogram(reconstructed_gray, "Reconstructed grayscale histogram"))

        st.markdown("DCT coefficient visualization")
        gray = original_array if original_array.ndim == 2 else np.mean(original_array, axis=2)
        st.image(_normalize_for_display(_dct_preview(gray)), caption="DCT magnitude preview", clamp=True)

        st.markdown("DWT subband visualization")
        subbands = _wavelet_subbands(gray, result["wavelet"])
        subband_cols = st.columns(4)
        for index, label in enumerate(["LL", "LH", "HL", "HH"]):
            with subband_cols[index]:
                st.image(_normalize_for_display(subbands[label]), caption=f"{label} subband", clamp=True)

        st.markdown("Absolute difference map")
        diff_map = np.abs(original_gray.astype(np.int16) - reconstructed_gray.astype(np.int16)).astype(np.uint8)
        st.image(_normalize_for_display(diff_map), caption="Absolute pixel difference map", clamp=True)

    elif nav == "Downloads":
        st.markdown("## Export Center")
        st.write("The following files are generated from the actual reconstructed image and evaluation data for the current run.")
        st.subheader("Image exports")
        download_cols = st.columns(3)
        with download_cols[0]:
            st.download_button("Download reconstructed PNG", result["png_bytes"], file_name="waveforge_reconstruction.png", mime="image/png")
        with download_cols[1]:
            st.download_button("Download reconstructed JPEG", result["jpeg_bytes"], file_name="waveforge_reconstruction.jpg", mime="image/jpeg")
        with download_cols[2]:
            st.download_button(
                "Download TXT report",
                generate_report_text(
                    _build_report_rows(result),
                    "Hybrid pipeline: RGB to YCrCb -> DWT -> block DCT -> quantization -> inverse DCT -> inverse DWT -> RGB reconstruction.",
                ),
                file_name="waveforge_report.txt",
                mime="text/plain",
            )

        st.subheader("Reports")
        col_csv, col_csv2 = st.columns(2)
        with col_csv:
            st.download_button("Download CSV evaluation report", generate_report_csv(_build_report_rows(result)), file_name="waveforge_report.csv", mime="text/csv")
        with col_csv2:
            if "comparison_csv" in st.session_state:
                st.download_button("Download comparison CSV", st.session_state["comparison_csv"], file_name="waveforge_comparison.csv", mime="text/csv")
            else:
                st.info("Run the comparison tool in the Comparison section to generate a comparison CSV.")

    _render_processed_download_panel(result, reconstructed)
    st.caption("WaveForge uses a real DWT + block DCT compression pipeline and validates all metrics from the original and reconstructed images.")
else:
    st.warning("No valid compression result yet. Upload an image and press Compress Image.")
