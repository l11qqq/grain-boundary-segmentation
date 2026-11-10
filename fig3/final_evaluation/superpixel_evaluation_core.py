"""Core data operations for the interactive superpixel evaluation tool.

This module intentionally contains no GUI code so the counting and automatic
cross-boundary rules can be unit-tested without opening a window.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image


SUPPORTED_LABEL_EXTENSIONS = {".npy", ".npz"}


@dataclass(frozen=True)
class CropRect:
    """Pixel-aligned crop rectangle using an exclusive right/bottom edge."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    def normalized(self, image_shape: tuple[int, int], minimum_size: int = 2) -> "CropRect":
        height, width = image_shape
        left, right = sorted((int(round(self.x0)), int(round(self.x1))))
        top, bottom = sorted((int(round(self.y0)), int(round(self.y1))))
        left = min(max(left, 0), max(width - minimum_size, 0))
        top = min(max(top, 0), max(height - minimum_size, 0))
        right = min(max(right, left + minimum_size), width)
        bottom = min(max(bottom, top + minimum_size), height)
        return CropRect(left, top, right, bottom)

    def as_list(self) -> list[int]:
        return [self.x0, self.y0, self.x1, self.y1]


@dataclass
class ReferenceData:
    labels: np.ndarray
    valid_mask: np.ndarray
    boundary_mask: np.ndarray
    metadata: dict[str, object]


@dataclass
class CropEvaluation:
    crop: CropRect
    crop_superpixel_ids: set[int]
    border_touching_superpixel_ids: set[int]
    auto_marked_ids: set[int]
    final_marked_ids: set[int]

    @property
    def total_superpixels(self) -> int:
        return len(self.crop_superpixel_ids)

    @property
    def border_touching_superpixels(self) -> int:
        return len(self.border_touching_superpixel_ids)

    @property
    def auto_marked_superpixels(self) -> int:
        return len(self.auto_marked_ids)

    @property
    def final_marked_superpixels(self) -> int:
        return len(self.final_marked_ids)

    @property
    def marked_ratio(self) -> float:
        return (
            self.final_marked_superpixels / self.total_superpixels
            if self.total_superpixels
            else 0.0
        )


def _ensure_2d_integer_labels(array: np.ndarray, source: Path) -> np.ndarray:
    labels = np.asarray(array)
    if labels.ndim != 2:
        raise ValueError(f"标签必须是二维数组，当前为 {labels.shape}：{source}")
    if not np.issubdtype(labels.dtype, np.integer):
        if not np.all(np.isfinite(labels)) or not np.allclose(labels, np.rint(labels)):
            raise ValueError(f"标签数组必须包含整数ID：{source}")
        labels = np.rint(labels)
    return labels.astype(np.int64, copy=False)


def load_label_array(path: Path) -> np.ndarray:
    """Load a 2-D label map from NPY or NPZ without pickle."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"标签文件不存在：{path}")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return _ensure_2d_integer_labels(np.load(path, allow_pickle=False), path)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            names = list(archive.files)
            if not names:
                raise ValueError(f"NPZ中没有数组：{path}")
            preferred = "labels" if "labels" in names else names[0]
            return _ensure_2d_integer_labels(archive[preferred], path)
    raise ValueError(f"仅支持 .npy 或 .npz 标签文件：{path}")


def compute_boundaries(labels: np.ndarray) -> np.ndarray:
    """Return a 4-connected two-sided region-boundary mask."""
    labels = np.asarray(labels)
    boundary = np.zeros(labels.shape, dtype=bool)
    horizontal = labels[:, 1:] != labels[:, :-1]
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    vertical = labels[1:, :] != labels[:-1, :]
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    return boundary


def rasterize_labelme_grains(path: Path) -> ReferenceData:
    """Rasterize LabelMe ``grain`` polygons; overlaps remain invalid/unlabeled."""
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    height = int(payload["imageHeight"])
    width = int(payload["imageWidth"])
    shapes = [
        shape
        for shape in payload.get("shapes", [])
        if shape.get("label") == "grain" and shape.get("shape_type") == "polygon"
    ]
    if not shapes:
        raise ValueError(f"LabelMe文件中没有 grain polygon：{path}")

    labels = np.zeros((height, width), dtype=np.int64)
    coverage = np.zeros((height, width), dtype=np.uint16)
    polygon_boundary = np.zeros((height, width), dtype=np.uint8)
    polygon_areas: list[int] = []

    for grain_id, shape in enumerate(shapes, start=1):
        points = np.rint(np.asarray(shape["points"], dtype=np.float64)).astype(np.int32)
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
            raise ValueError(f"第 {grain_id} 个 grain polygon 无效：{path}")
        if (
            np.any(points[:, 0] < 0)
            or np.any(points[:, 0] >= width)
            or np.any(points[:, 1] < 0)
            or np.any(points[:, 1] >= height)
        ):
            raise ValueError(f"第 {grain_id} 个 polygon 超出图像范围：{path}")
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [points], color=1)
        inside = mask.astype(bool)
        labels[inside] = grain_id
        coverage += mask
        polygon_areas.append(int(inside.sum()))
        cv2.polylines(polygon_boundary, [points], isClosed=True, color=1, thickness=1)

    valid_mask = coverage == 1
    labels[~valid_mask] = 0
    boundary_mask = polygon_boundary.astype(bool)
    metadata: dict[str, object] = {
        "source_type": "labelme_grain_polygons",
        "source_path": str(path),
        "image_shape": [height, width],
        "grain_polygon_count": len(shapes),
        "observed_grain_count": int(np.unique(labels[valid_mask]).size),
        "valid_labeled_pixels": int(valid_mask.sum()),
        "ambiguous_overlap_pixels_excluded": int(np.count_nonzero(coverage > 1)),
        "labeled_coverage_ratio": float(valid_mask.mean()),
        "polygon_area_min": int(min(polygon_areas)),
        "polygon_area_max": int(max(polygon_areas)),
    }
    return ReferenceData(labels, valid_mask, boundary_mask, metadata)


def load_reference_data(path: Path, zero_is_unlabeled: bool = False) -> ReferenceData:
    """Load LabelMe grain polygons or a 2-D reference partition label map."""
    path = Path(path).resolve()
    if path.suffix.lower() == ".json":
        return rasterize_labelme_grains(path)
    labels = load_label_array(path)
    valid_mask = labels != 0 if zero_is_unlabeled else np.ones(labels.shape, dtype=bool)
    boundary = compute_boundaries(labels)
    if zero_is_unlabeled:
        # Do not draw transitions wholly within the unlabeled background.
        boundary &= valid_mask
    metadata: dict[str, object] = {
        "source_type": "reference_partition_label_map",
        "source_path": str(path),
        "image_shape": list(labels.shape),
        "reference_region_count": int(np.unique(labels[valid_mask]).size),
        "valid_labeled_pixels": int(valid_mask.sum()),
        "zero_is_unlabeled": bool(zero_is_unlabeled),
    }
    return ReferenceData(labels, valid_mask, boundary, metadata)


def compute_auto_crossing_ids(
    candidate_labels: np.ndarray,
    reference: ReferenceData,
) -> tuple[set[int], dict[int, dict[str, int]]]:
    """Find candidate regions overlapping at least two valid reference regions.

    The calculation uses every pixel of each candidate superpixel, not only the
    part inside the crop.  This makes superpixels touching the crop border count
    as complete regions, as requested by the manual evaluation protocol.
    """
    candidate = np.asarray(candidate_labels)
    if candidate.shape != reference.labels.shape:
        raise ValueError(
            f"候选label尺寸 {candidate.shape} 与参考尺寸 {reference.labels.shape} 不一致"
        )
    valid = reference.valid_mask
    if not np.any(valid):
        return set(), {}

    candidate_values = candidate[valid].astype(np.int64, copy=False)
    reference_values = reference.labels[valid].astype(np.int64, copy=False)
    candidate_ids, candidate_inverse = np.unique(candidate_values, return_inverse=True)
    reference_ids, reference_inverse = np.unique(reference_values, return_inverse=True)
    # Aggregate only pairs that actually occur.  A dense
    # candidate_count×reference_count matrix is harmless for the default 202
    # polygons, but could exceed 1 GB if a large NPY partition is imported as
    # the reference.  The sparse pair representation remains safe in both
    # cases.
    pair_codes = candidate_inverse.astype(np.int64) * reference_ids.size + reference_inverse
    unique_pair_codes, pair_pixel_counts = np.unique(pair_codes, return_counts=True)
    pair_candidate_indices = unique_pair_codes // reference_ids.size
    pair_reference_indices = unique_pair_codes % reference_ids.size

    overlap_region_counts = np.bincount(
        pair_candidate_indices,
        minlength=candidate_ids.size,
    )
    crossing_mask = overlap_region_counts > 1
    crossing_ids = {int(value) for value in candidate_ids[crossing_mask].tolist()}

    dominant_pixels = np.zeros(candidate_ids.size, dtype=np.int64)
    np.maximum.at(dominant_pixels, pair_candidate_indices, pair_pixel_counts)
    dominant_reference_indices = np.zeros(candidate_ids.size, dtype=np.int64)
    starts = np.flatnonzero(
        np.r_[True, pair_candidate_indices[1:] != pair_candidate_indices[:-1]]
    )
    ends = np.r_[starts[1:], pair_candidate_indices.size]
    for start, end in zip(starts.tolist(), ends.tolist()):
        candidate_index = int(pair_candidate_indices[start])
        local_offset = int(np.argmax(pair_pixel_counts[start:end]))
        dominant_reference_indices[candidate_index] = int(
            pair_reference_indices[start + local_offset]
        )
    total_labeled_pixels = np.bincount(
        candidate_inverse,
        minlength=candidate_ids.size,
    )
    diagnostics: dict[int, dict[str, int]] = {}
    for index, candidate_id in enumerate(candidate_ids.tolist()):
        diagnostics[int(candidate_id)] = {
            "reference_regions_overlapped": int(overlap_region_counts[index]),
            "dominant_reference_id": int(reference_ids[dominant_reference_indices[index]]),
            "dominant_reference_pixels": int(dominant_pixels[index]),
            "other_reference_pixels": int(total_labeled_pixels[index] - dominant_pixels[index]),
            "total_reference_labeled_pixels": int(total_labeled_pixels[index]),
        }
    return crossing_ids, diagnostics


def crop_superpixel_ids(candidate_labels: np.ndarray, crop: CropRect) -> set[int]:
    crop = crop.normalized(candidate_labels.shape)
    values = candidate_labels[crop.y0 : crop.y1, crop.x0 : crop.x1]
    return {int(value) for value in np.unique(values).tolist()}


def crop_border_superpixel_ids(candidate_labels: np.ndarray, crop: CropRect) -> set[int]:
    """Return regions intersecting the one-pixel inner perimeter of a crop."""
    crop = crop.normalized(candidate_labels.shape)
    pieces = [
        candidate_labels[crop.y0, crop.x0 : crop.x1],
        candidate_labels[crop.y1 - 1, crop.x0 : crop.x1],
        candidate_labels[crop.y0 : crop.y1, crop.x0],
        candidate_labels[crop.y0 : crop.y1, crop.x1 - 1],
    ]
    return {int(value) for value in np.unique(np.concatenate(pieces)).tolist()}


def evaluate_crop(
    candidate_labels: np.ndarray,
    crop: CropRect,
    global_auto_crossing_ids: Iterable[int],
    manual_marked_ids: Iterable[int] = (),
    manual_unmarked_ids: Iterable[int] = (),
) -> CropEvaluation:
    """Evaluate a crop while treating every intersecting region as one whole SP."""
    crop = crop.normalized(candidate_labels.shape)
    crop_ids = crop_superpixel_ids(candidate_labels, crop)
    border_ids = crop_border_superpixel_ids(candidate_labels, crop)
    auto_ids = crop_ids.intersection(int(value) for value in global_auto_crossing_ids)
    manual_marked = crop_ids.intersection(int(value) for value in manual_marked_ids)
    manual_unmarked = crop_ids.intersection(int(value) for value in manual_unmarked_ids)
    final_ids = (auto_ids - manual_unmarked) | manual_marked
    return CropEvaluation(crop, crop_ids, border_ids, auto_ids, final_ids)


def normalize_id_set(values: Iterable[object]) -> set[int]:
    return {int(value) for value in values}


def extract_crop_rgb(image_rgb: np.ndarray, crop: CropRect) -> np.ndarray:
    """Copy a crop from an RGB uint8 image without resizing or resampling."""
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"待保存图像必须是 H×W×3 RGB 数组，当前为 {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"待保存图像必须是 uint8，当前为 {image.dtype}")
    normalized = crop.normalized(image.shape[:2])
    return np.ascontiguousarray(
        image[normalized.y0 : normalized.y1, normalized.x0 : normalized.x1]
    ).copy()


def save_lossless_crop_png(
    path: Path,
    image_rgb: np.ndarray,
    crop: CropRect,
) -> np.ndarray:
    """Save the exact original-resolution crop as a lossless RGB PNG.

    ``compress_level=0`` disables PNG's file-size compression.  More
    importantly, the pixel array is sliced directly: no display zoom,
    interpolation, or lossy encoder participates in the export.
    """
    crop_rgb = extract_crop_rgb(image_rgb, crop)
    Image.fromarray(crop_rgb).save(
        Path(path),
        format="PNG",
        compress_level=0,
        optimize=False,
    )
    return crop_rgb
