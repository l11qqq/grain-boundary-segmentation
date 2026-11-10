"""Figure 3, stage 2: 12-channel angular features with edge-preserving SLICO.

This reconstructs the immediate precursor to the final method: per-image mean
normalization, 3x4 angular grouping, percentile clipping, bilateral filtering,
adaptive-compactness SLICO, and connected-component size constraints.  It does
not use narrow-twin markers or watershed boundary absorption.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
from skimage.segmentation import slic

from fig3_common import (
    boundary_overlay,
    find_image_paths,
    percentile_normalize,
    save_outputs,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "crop"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "result_2"
EPS = 1e-6


def parse_position(path: Path) -> tuple[float, float]:
    match = re.fullmatch(r"pic_([-+]?\d+(?:\.\d+)?)_([-+]?\d+(?:\.\d+)?)", path.stem, re.I)
    if match is None:
        raise ValueError(f"Expected a pic_r_phi filename, got {path.name}")
    return float(match.group(1)), float(match.group(2))


def coordinate_bins(values: list[float], bin_count: int) -> dict[float, int]:
    mapping: dict[float, int] = {}
    for index, group in enumerate(np.array_split(np.asarray(sorted(set(values))), bin_count)):
        for value in group.tolist():
            mapping[float(value)] = index
    return mapping


def build_legacy_features(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    positions = [parse_position(path) for path in paths]
    r_map = coordinate_bins([position[0] for position in positions], 3)
    phi_map = coordinate_bins([position[1] for position in positions], 4)
    group_count = 12
    first = cv2.imread(str(paths[0]), cv2.IMREAD_GRAYSCALE)
    if first is None:
        raise ValueError(f"Failed to read {paths[0]}")
    features = np.zeros((*first.shape, group_count), dtype=np.float32)
    group_counts = np.zeros(group_count, dtype=np.int32)
    pixel_sum = np.zeros(first.shape, dtype=np.float64)
    pixel_square_sum = np.zeros_like(pixel_sum)
    for index, (path, position) in enumerate(zip(paths, positions), start=1):
        image = first if index == 1 else cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != first.shape:
            raise ValueError(f"Invalid image or shape mismatch: {path}")
        values = image.astype(np.float32)
        pixel_sum += values
        pixel_square_sum += values * values
        group = r_map[position[0]] * 4 + phi_map[position[1]]
        features[:, :, group] += values / (float(values.mean()) + EPS)
        group_counts[group] += 1
        if index == 1 or index % 25 == 0 or index == len(paths):
            print(f"Building legacy angular features: {index}/{len(paths)}", flush=True)
    for channel in range(group_count):
        feature = features[:, :, channel] / float(group_counts[channel])
        feature = percentile_normalize(feature, 1.0, 99.0)
        features[:, :, channel] = cv2.bilateralFilter(
            feature, d=5, sigmaColor=0.08, sigmaSpace=2.0, borderType=cv2.BORDER_REFLECT101
        )
    count = float(len(paths))
    variance = np.maximum(pixel_square_sum / count - np.square(pixel_sum / count), 0.0)
    metadata = {
        "feature_channels": group_count,
        "images_per_group": group_counts.tolist(),
        "channel_normalization": "image_divided_by_its_global_mean",
        "feature_clip_percentiles": [1.0, 99.0],
        "bilateral_diameter": 5,
        "bilateral_sigma_color": 0.08,
        "bilateral_sigma_space": 2.0,
    }
    return features, np.sqrt(variance).astype(np.float32), metadata


def angular_gradient(features: np.ndarray) -> np.ndarray:
    scale_maps: list[np.ndarray] = []
    for sigma in (0.0, 0.8):
        energy = np.zeros(features.shape[:2], dtype=np.float32)
        for channel in range(features.shape[2]):
            feature = features[:, :, channel]
            if sigma:
                feature = cv2.GaussianBlur(feature, (0, 0), sigma)
            gx = cv2.Sobel(feature, cv2.CV_32F, 1, 0, ksize=3) / 8.0
            gy = cv2.Sobel(feature, cv2.CV_32F, 0, 1, ksize=3) / 8.0
            energy += gx * gx + gy * gy
        scale_maps.append(percentile_normalize(np.sqrt(energy / features.shape[2]), 2.0, 99.5))
    return np.maximum.reduce(scale_maps)


def main(args: argparse.Namespace) -> None:
    paths = find_image_paths(args.input_dir)
    features, std_raw, metadata = build_legacy_features(paths)
    print(f"Running legacy SLICO on {features.shape[2]} channels...", flush=True)
    labels = slic(
        features,
        n_segments=args.n_segments,
        compactness=args.compactness,
        max_num_iter=30,
        sigma=0.0,
        channel_axis=-1,
        convert2lab=False,
        start_label=0,
        slic_zero=True,
        min_size_factor=0.50,
        max_size_factor=2.0,
        enforce_connectivity=True,
    ).astype(np.int32)
    std_display = percentile_normalize(std_raw, 2.0, 98.0, gamma=0.65)
    angular_strength = angular_gradient(features)
    angular_overlay = boundary_overlay(angular_strength, labels)
    save_outputs(
        args.output_dir,
        labels,
        std_display,
        "Stage 2: 12-channel edge-preserving SLICO",
        {
            "n_segments": args.n_segments,
            "compactness": args.compactness,
            "r_bins": 3,
            "phi_bins": 4,
            "slic_zero": True,
            "max_num_iter": 30,
            "min_size_factor": 0.50,
            "max_size_factor": 2.0,
            **metadata,
        },
        [("Angular-gradient overlay", angular_overlay, None)],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-segments", type=int, default=1000)
    parser.add_argument("--compactness", type=float, default=0.28)
    cli = parser.parse_args()
    cli.input_dir = cli.input_dir.resolve()
    cli.output_dir = cli.output_dir.resolve()
    main(cli)
