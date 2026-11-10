"""Shared I/O and visualization utilities for the Figure 3 ablation scripts."""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import find_boundaries


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_DIR / "raw_pic413-3-1n"
EXPECTED_IMAGE_COUNT = 396
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def natural_sort_key(path: Path) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def find_image_paths(input_dir: Path) -> list[Path]:
    paths = sorted(
        [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS],
        key=natural_sort_key,
    )
    if len(paths) != EXPECTED_IMAGE_COUNT:
        raise ValueError(f"Expected {EXPECTED_IMAGE_COUNT} images in {input_dir}, found {len(paths)}")
    return paths


def compute_std_map(image_paths: list[Path]) -> np.ndarray:
    first = cv2.imread(str(image_paths[0]), cv2.IMREAD_GRAYSCALE)
    if first is None:
        raise ValueError(f"Failed to read {image_paths[0]}")
    height, width = first.shape
    pixel_sum = np.zeros((height, width), dtype=np.float64)
    pixel_square_sum = np.zeros_like(pixel_sum)
    for index, path in enumerate(image_paths, start=1):
        image = first if index == 1 else cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (height, width):
            raise ValueError(f"Invalid image or shape mismatch: {path}")
        values = image.astype(np.float64)
        pixel_sum += values
        pixel_square_sum += values * values
        if index == 1 or index % 50 == 0 or index == len(image_paths):
            print(f"Building standard-deviation image: {index}/{len(image_paths)}", flush=True)
    count = float(len(image_paths))
    variance = np.maximum(pixel_square_sum / count - np.square(pixel_sum / count), 0.0)
    return np.sqrt(variance).astype(np.float32)


def percentile_normalize(
    image: np.ndarray,
    lower: float = 2.0,
    upper: float = 98.0,
    gamma: float = 1.0,
) -> np.ndarray:
    low, high = np.percentile(image, [lower, upper])
    if float(high - low) < 1e-8:
        return np.zeros(image.shape, dtype=np.float32)
    result = np.clip((image.astype(np.float32) - float(low)) / float(high - low), 0.0, 1.0)
    return np.power(result, gamma).astype(np.float32)


def minmax_normalize(image: np.ndarray) -> np.ndarray:
    values = image.astype(np.float32)
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-8:
        return np.zeros(values.shape, dtype=np.float32)
    return ((values - low) / (high - low)).astype(np.float32)


def to_uint8(image: np.ndarray) -> np.ndarray:
    values = image.astype(np.float32)
    if values.min() < 0.0 or values.max() > 1.0:
        values = percentile_normalize(values)
    return np.round(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)


def boundary_overlay(background: np.ndarray, labels: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(to_uint8(background), cv2.COLOR_GRAY2RGB)
    rgb[find_boundaries(labels, mode="inner")] = (255, 0, 0)
    return rgb


def save_outputs(
    output_dir: Path,
    labels: np.ndarray,
    std_display: np.ndarray,
    method_name: str,
    parameters: dict[str, object],
    extra_panels: list[tuple[str, np.ndarray, str | None]] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = labels.astype(np.int32, copy=False)
    np.save(output_dir / "slic_labels.npy", labels, allow_pickle=False)
    cv2.imwrite(str(output_dir / "std_image.png"), to_uint8(std_display))
    overlay = boundary_overlay(std_display, labels)
    cv2.imwrite(str(output_dir / "segmentation_full_resolution.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    config = {
        "method": method_name,
        "source_raw_dir": str(DEFAULT_INPUT_DIR),
        "image_count": EXPECTED_IMAGE_COUNT,
        "label_map": "slic_labels.npy",
        "std_image": "std_image.png",
        "actual_superpixels": int(len(np.unique(labels))),
        "parameters": parameters,
    }
    (output_dir / "method_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    panels: list[tuple[str, np.ndarray, str | None]] = [
        ("Standard-deviation image", std_display, "gray"),
        (method_name, overlay, None),
    ]
    if extra_panels:
        panels.extend(extra_panels)
    columns = 2
    rows = (len(panels) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(13, 5.3 * rows), squeeze=False)
    for index, axis in enumerate(axes.flat):
        if index < len(panels):
            title, image, cmap = panels[index]
            axis.imshow(image, cmap=cmap)
            axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "visualization.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {method_name}: {output_dir} ({len(np.unique(labels))} regions)")

