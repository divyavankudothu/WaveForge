from __future__ import annotations

import io
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import pywt
from PIL import Image

VALID_WAVELETS = ["haar", "db1"]
VALID_MODES = ["balanced", "high_quality", "high_compression"]
MAX_IMAGE_PIXELS = 12_000_000


def _ensure_valid_array(array: np.ndarray) -> np.ndarray:
    if array is None:
        raise ValueError("Image data is missing.")
    array = np.asarray(array)
    if array.size == 0:
        raise ValueError("Image cannot be empty.")
    if array.ndim not in (2, 3):
        raise ValueError("Unsupported image format. Expected a 2D grayscale or 3-channel RGB image.")
    if array.ndim == 3 and array.shape[2] not in (3, 4):
        raise ValueError("Unsupported channel count. Expected RGB or grayscale input.")
    return array


def load_uploaded_image(uploaded_file) -> Tuple[np.ndarray, str, Tuple[int, int], str]:
    if uploaded_file is None:
        raise ValueError("No image uploaded.")
    try:
        raw_data = uploaded_file.getvalue()
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError("Unable to read uploaded file data.") from exc

    if len(raw_data) == 0:
        raise ValueError("Uploaded file is empty.")

    try:
        image = Image.open(io.BytesIO(raw_data))
        image.load()
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError("The uploaded file could not be opened as an image.") from exc

    if image.format is None:
        raise ValueError("Unsupported image format.")

    file_name = getattr(uploaded_file, "name", "uploaded_image")
    lower_name = file_name.lower()
    allowed = (".png", ".jpg", ".jpeg", ".webp")
    if not any(lower_name.endswith(ext) for ext in allowed):
        raise ValueError("Unsupported image type. Please upload a PNG, JPG, JPEG, or WEBP image.")

    if image.mode in {"1", "L", "I;16", "I", "F"}:
        arr = np.asarray(image.convert("L"), dtype=np.uint8)
    else:
        rgb_image = image.convert("RGB")
        arr = np.asarray(rgb_image, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("Image conversion failed. The source image is not a valid RGB image.")

    total_pixels = arr.shape[0] * arr.shape[1]
    if total_pixels > MAX_IMAGE_PIXELS:
        raise ValueError(
            f"Image is too large ({total_pixels:,} pixels). Please use an image with fewer than {MAX_IMAGE_PIXELS:,} pixels."
        )

    return arr, image.format, (arr.shape[1], arr.shape[0]), file_name


def _safe_block_to_array(blocks: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
    row_blocks, col_blocks, _, _ = blocks.shape
    rebuilt = blocks.transpose(0, 2, 1, 3).reshape(row_blocks * 8, col_blocks * 8)
    h, w = original_shape
    return rebuilt[:h, :w]


def _process_block(block: np.ndarray, quantizer: float) -> np.ndarray:
    block = np.asarray(block, dtype=np.float32)
    dct_block = cv2.dct(block)
    q_block = np.round(dct_block / max(1.0, quantizer))
    reconstructed = cv2.idct(q_block * max(1.0, quantizer))
    return reconstructed.astype(np.float32)


def _apply_dct_quantization(subband: np.ndarray, quantizer: float) -> np.ndarray:
    subband = np.asarray(subband, dtype=np.float32)
    if subband.size == 0:
        return subband.copy()

    h, w = subband.shape
    pad_h = (8 - (h % 8)) % 8
    pad_w = (8 - (w % 8)) % 8
    padded = np.pad(subband, ((0, pad_h), (0, pad_w)), mode="edge")
    row_blocks = padded.shape[0] // 8
    col_blocks = padded.shape[1] // 8
    blocks = padded.reshape(row_blocks, 8, col_blocks, 8).transpose(0, 2, 1, 3)
    processed = np.empty_like(blocks, dtype=np.float32)

    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            processed[i, j] = _process_block(blocks[i, j], quantizer)

    return _safe_block_to_array(processed, (h, w))


def _wavelet_quantizer(strength: int, mode: str) -> float:
    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported processing mode: {mode}")

    base_quantizer = {"balanced": 18.0, "high_quality": 9.0, "high_compression": 30.0}[mode]
    strength_ratio = max(1, min(100, int(strength))) / 100.0
    quantizer = base_quantizer * (1.0 + (1.0 - strength_ratio) * 0.9) * (0.9 + strength_ratio)
    return float(max(2.0, quantizer))


def _compress_wavelet_coefficients(channel: np.ndarray, wavelet: str, level: int, strength: int, mode: str) -> np.ndarray:
    if wavelet not in VALID_WAVELETS:
        raise ValueError(f"Unsupported wavelet: {wavelet}. Supported values are: {VALID_WAVELETS}")
    if level not in (1, 2):
        raise ValueError("Wavelet decomposition level must be either 1 or 2.")

    coeffs = pywt.wavedec2(channel.astype(np.float32), wavelet=wavelet, level=level, mode="symmetric")
    quantizer = _wavelet_quantizer(strength, mode)
    processed = [coeffs[0].copy()]

    for detail in coeffs[1:]:
        if len(detail) != 3:
            continue
        lh, hl, hh = detail
        processed_detail = (
            _apply_dct_quantization(lh, quantizer),
            _apply_dct_quantization(hl, quantizer * 1.1),
            _apply_dct_quantization(hh, quantizer * 1.25),
        )
        processed.append(processed_detail)

    reconstructed = pywt.waverec2(processed, wavelet=wavelet, mode="symmetric")
    return np.asarray(reconstructed, dtype=np.float32)


def compress_image_array(
    image_array: np.ndarray,
    compression_strength: int = 55,
    mode: str = "balanced",
    wavelet: str = "haar",
    decomposition_level: int = 1,
) -> np.ndarray:
    image_array = _ensure_valid_array(image_array)
    if image_array.ndim == 2:
        channel_data = [image_array.astype(np.float32)]
    else:
        rgb = np.asarray(image_array, dtype=np.uint8)
        if rgb.shape[-1] == 4:
            rgb = rgb[:, :, :3]
        ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
        channel_data = [ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]]

    processed_channels: List[np.ndarray] = []
    for channel in channel_data:
        processed_channels.append(
            _compress_wavelet_coefficients(channel, wavelet, decomposition_level, compression_strength, mode)
        )

    if image_array.ndim == 2:
        reconstructed = np.clip(processed_channels[0], 0, 255).astype(np.uint8)
        return reconstructed

    reconstructed_ycrcb = np.stack(processed_channels, axis=-1)
    reconstructed_ycrcb = np.clip(reconstructed_ycrcb, 0, 255).astype(np.uint8)
    rgb_reconstruction = cv2.cvtColor(reconstructed_ycrcb, cv2.COLOR_YCrCb2RGB)
    return rgb_reconstruction


def export_image_bytes(image_array: np.ndarray, file_format: str = "PNG") -> bytes:
    array = np.asarray(image_array)
    if array.ndim == 2:
        pil_image = Image.fromarray(array.astype(np.uint8), mode="L")
    else:
        pil_image = Image.fromarray(array.astype(np.uint8), mode="RGB")

    buffer = io.BytesIO()
    pil_image.save(buffer, format=file_format.upper())
    return buffer.getvalue()


def compare_multiple_strengths(
    image_array: np.ndarray,
    strengths: Sequence[int],
    mode: str,
    wavelet: str,
    decomposition_level: int,
) -> List[Dict[str, float | int | str]]:
    rows: List[Dict[str, float | int | str]] = []
    original_shape = image_array.shape[:2]
    for strength in strengths:
        start_time = __import__("time").perf_counter()
        reconstructed = compress_image_array(
            image_array,
            compression_strength=int(strength),
            mode=mode,
            wavelet=wavelet,
            decomposition_level=decomposition_level,
        )
        elapsed = __import__("time").perf_counter() - start_time
        rows.append(
            {
                "strength": int(strength),
                "output_width": int(reconstructed.shape[1]),
                "output_height": int(reconstructed.shape[0]),
                "matches_original_shape": reconstructed.shape[:2] == original_shape,
                "processing_time_seconds": float(elapsed),
            }
        )
    return rows
