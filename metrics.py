from __future__ import annotations

import csv
import io
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from skimage.metrics import structural_similarity


def _safe_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    if original.shape != reconstructed.shape:
        raise ValueError("Original and reconstructed images must have matching dimensions.")

    if original.size == 0:
        raise ValueError("Image arrays cannot be empty.")

    original = np.asarray(original, dtype=np.float32)
    reconstructed = np.asarray(reconstructed, dtype=np.float32)
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float("inf")
    max_value = float(np.max(original)) if np.max(original) > 0 else 255.0
    if max_value <= 0:
        return 0.0
    return float(10.0 * np.log10((max_value ** 2) / mse))


def safe_ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
    if original.shape != reconstructed.shape:
        raise ValueError("Original and reconstructed images must have matching dimensions.")

    original = np.asarray(original)
    reconstructed = np.asarray(reconstructed)

    if original.size == 0:
        raise ValueError("Image arrays cannot be empty.")

    if np.array_equal(original, reconstructed):
        return 1.0

    if original.shape == reconstructed.shape and original.ndim == 2:
        if min(original.shape) < 7:
            return 0.0
        return float(structural_similarity(original, reconstructed, data_range=255.0))

    if original.ndim == 3 and original.shape[-1] in (3, 1):
        if min(original.shape[:2]) < 7:
            return 0.0
        channel_axis = -1
        return float(structural_similarity(original, reconstructed, channel_axis=channel_axis, data_range=255.0))

    return 0.0


def compute_quality_metrics(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
    if original.shape != reconstructed.shape:
        raise ValueError("Original and reconstructed images must have matching shapes for metric computation.")

    psnr = _safe_psnr(original, reconstructed)
    ssim = safe_ssim(original, reconstructed)

    return {"psnr_db": psnr, "ssim": ssim}


def actual_compression_ratio(original_bytes: int, exported_bytes: int) -> float:
    if original_bytes <= 0:
        raise ValueError("Original file size must be positive.")
    if exported_bytes <= 0:
        raise ValueError("Exported file size must be positive.")
    return float(original_bytes / exported_bytes)


def size_reduction_percent(original_bytes: int, exported_bytes: int) -> float:
    if original_bytes <= 0:
        raise ValueError("Original file size must be positive.")
    return float(100.0 * (original_bytes - exported_bytes) / original_bytes)


def generate_report_csv(report_rows: Iterable[Dict[str, Any]]) -> bytes:
    rows = list(report_rows)
    if not rows:
        raise ValueError("At least one report row is required.")

    fieldnames = sorted({key for row in rows for key in row.keys()})
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return output.getvalue().encode("utf-8")


def generate_report_text(report_rows: List[Dict[str, Any]], pipeline_description: str) -> bytes:
    if not report_rows:
        raise ValueError("At least one report row is required.")

    lines: List[str] = []
    lines.append("WaveForge Analysis Report")
    lines.append("=" * 24)
    lines.append(pipeline_description)
    lines.append("")
    lines.append("Metric values are calculated from the actual input and reconstructed images.")
    lines.append("")
    for index, row in enumerate(report_rows, start=1):
        lines.append(f"Run {index}:")
        for key, value in row.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    return "\n".join(lines).encode("utf-8")
