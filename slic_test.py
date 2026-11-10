"""Final multi-angle superpixel pipeline for grain/twin boundary labeling.

The formal pipeline combines 24-channel angular features and a lightly weighted
standard-deviation feature, compact SLICO,
narrow-twin marker supplementation, and compact marker-controlled watershed.
It writes the canonical ``slic_labels.npy``, ``std_image.png`` and
``labeling_config.json`` files consumed by ``edge_labeling_app.py`` and
``build_gnn_dataset.py``.

Default data routing is intentionally kept with the formal project paths:

* input: ``raw_pic413-3-1n``
* output: ``outputs/slic_3-1``

Use ``--output-group slic_4-1`` for another group under ``outputs`` or pass an
explicit ``--output-dir``.  The standard-deviation feature participates in the
initial SLICO input with a weight of 0.1; no mean image participates in any
segmentation stage.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import binary_closing, disk, remove_small_objects
from skimage.segmentation import find_boundaries, relabel_sequential, slic, watershed


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = PROJECT_DIR / "raw_pic413-3-1n"
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "reproduced"
DEFAULT_OUTPUT_GROUP = "slic_3-1"
DEFAULT_EXPECTED_NUM_IMAGES = 396
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
STD_CLIP_PERCENTILES = (2.0, 98.0)
STD_GAMMA = 0.65
STD_FEATURE_WEIGHT = 0
FEATURE_CLIP_PERCENTILES = (1.0, 99.0)
EPS = 1e-6


def natural_sort_key(path: Path) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def parse_light_position(path: Path) -> tuple[float, float] | None:
    match = re.fullmatch(
        r"pic_([-+]?(?:\d+(?:\.\d*)?|\.\d+))_([-+]?(?:\d+(?:\.\d*)?|\.\d+))",
        path.stem,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    return float(match.group(1)), float(match.group(2))


def find_image_paths(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    paths = [
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    ]
    if not paths:
        raise ValueError(f"No images found in: {input_dir}")
    positions = [parse_light_position(path) for path in paths]
    if all(position is not None for position in positions):
        return [
            path
            for _, path in sorted(
                zip(positions, paths),
                key=lambda item: (item[0][0], item[0][1]),  # type: ignore[index]
            )
        ]
    return sorted(paths, key=natural_sort_key)


def normalize_percentile_float(
    image: np.ndarray,
    lower_percentile: float,
    upper_percentile: float,
    gamma: float = 1.0,
) -> np.ndarray:
    if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
        raise ValueError("Percentiles must satisfy 0 <= lower < upper <= 100")
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    image_float = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(image_float, [lower_percentile, upper_percentile])
    if float(high - low) < EPS:
        return np.zeros(image_float.shape, dtype=np.float32)
    normalized = np.clip((image_float - float(low)) / float(high - low), 0.0, 1.0)
    if gamma != 1.0:
        normalized = np.power(normalized, gamma)
    return normalized.astype(np.float32, copy=False)


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    normalized = normalize_percentile_float(image, 2.0, 98.0)
    return np.round(normalized * 255.0).astype(np.uint8)


def save_gray_png(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), normalize_to_uint8(image)):
        raise OSError(f"Failed to write image: {path}")


def save_rgb_png(path: Path, rgb_image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Failed to write image: {path}")


def create_boundary_overlay(
    background_image: np.ndarray,
    segments: np.ndarray,
    boundary_color: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    background_rgb = cv2.cvtColor(normalize_to_uint8(background_image), cv2.COLOR_GRAY2RGB)
    background_rgb[find_boundaries(segments, mode="inner")] = boundary_color
    return background_rgb


def coordinate_bin_map(values: list[float], bin_count: int) -> dict[float, int]:
    unique_values = sorted(set(values))
    actual_bins = min(bin_count, len(unique_values))
    mapping: dict[float, int] = {}
    for bin_index, group in enumerate(np.array_split(np.asarray(unique_values), actual_bins)):
        for value in group.tolist():
            mapping[float(value)] = bin_index
    return mapping


def assign_angular_groups(
    image_paths: list[Path],
    r_bins: int,
    phi_bins: int,
) -> tuple[np.ndarray, int, dict[str, object]]:
    positions = [parse_light_position(path) for path in image_paths]
    if all(position is not None for position in positions):
        typed = [position for position in positions if position is not None]
        r_map = coordinate_bin_map([position[0] for position in typed], r_bins)
        phi_map = coordinate_bin_map([position[1] for position in typed], phi_bins)
        actual_r_bins = max(r_map.values()) + 1
        actual_phi_bins = max(phi_map.values()) + 1
        groups = np.asarray(
            [r_map[position[0]] * actual_phi_bins + phi_map[position[1]] for position in typed],
            dtype=np.int32,
        )
        metadata = {
            "grouping": "pic_r_phi_grid_blocks",
            "requested_r_bins": r_bins,
            "requested_phi_bins": phi_bins,
            "actual_r_bins": actual_r_bins,
            "actual_phi_bins": actual_phi_bins,
            "unique_r_values": len(r_map),
            "unique_phi_values": len(phi_map),
        }
        return groups, actual_r_bins * actual_phi_bins, metadata

    group_count = min(r_bins * phi_bins, len(image_paths))
    groups = np.empty(len(image_paths), dtype=np.int32)
    for group_index, indices in enumerate(np.array_split(np.arange(len(image_paths)), group_count)):
        groups[indices] = group_index
    return groups, group_count, {
        "grouping": "sequential_fallback",
        "requested_group_count": r_bins * phi_bins,
        "actual_group_count": group_count,
    }


def build_multi_angle_features(
    image_paths: list[Path],
    downsample_factor: float,
    r_bins: int,
    phi_bins: int,
    bilateral_diameter: int,
    bilateral_sigma_color: float,
    bilateral_sigma_space: float,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], dict[str, object]]:
    if downsample_factor < 1.0:
        raise ValueError("downsample_factor must be at least 1.0")
    if r_bins <= 0 or phi_bins <= 0:
        raise ValueError("r_bins and phi_bins must be positive")
    first = cv2.imread(str(image_paths[0]), cv2.IMREAD_GRAYSCALE)
    if first is None:
        raise ValueError(f"Failed to read image: {image_paths[0]}")
    height, width = first.shape
    work_height = max(1, int(round(height / downsample_factor)))
    work_width = max(1, int(round(width / downsample_factor)))
    group_indices, group_count, group_metadata = assign_angular_groups(image_paths, r_bins, phi_bins)
    features = np.zeros((work_height, work_width, group_count), dtype=np.float32)
    group_counts = np.zeros(group_count, dtype=np.int32)
    pixel_sum = np.zeros((work_height, work_width), dtype=np.float64)
    pixel_square_sum = np.zeros_like(pixel_sum)

    for index, (path, group_index) in enumerate(zip(image_paths, group_indices), start=1):
        image = first if index == 1 else cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to read image: {path}")
        if image.shape != (height, width):
            raise ValueError(
                f"All input images must have the same size; {path.name} is {image.shape}, "
                f"expected {(height, width)}."
            )
        if (work_height, work_width) != (height, width):
            image = cv2.resize(image, (work_width, work_height), interpolation=cv2.INTER_AREA)
        image_float = image.astype(np.float32)
        pixel_sum += image_float
        pixel_square_sum += image_float * image_float
        normalized_angle = image_float / (float(image_float.mean()) + EPS)
        features[:, :, int(group_index)] += normalized_angle
        group_counts[int(group_index)] += 1
        if index == 1 or index % 25 == 0 or index == len(image_paths):
            print(f"Building angular features: {index}/{len(image_paths)}", flush=True)

    if np.any(group_counts == 0):
        raise RuntimeError("At least one angular feature group received no images")
    for channel in range(group_count):
        feature = features[:, :, channel] / float(group_counts[channel])
        feature = normalize_percentile_float(
            feature,
            FEATURE_CLIP_PERCENTILES[0],
            FEATURE_CLIP_PERCENTILES[1],
        )
        if bilateral_diameter > 0:
            feature = cv2.bilateralFilter(
                feature,
                d=bilateral_diameter,
                sigmaColor=bilateral_sigma_color,
                sigmaSpace=bilateral_sigma_space,
                borderType=cv2.BORDER_REFLECT101,
            )
        features[:, :, channel] = feature

    count = float(len(image_paths))
    variance = np.maximum(pixel_square_sum / count - np.square(pixel_sum / count), 0.0)
    std_map = np.sqrt(variance).astype(np.float32)
    metadata = {
        **group_metadata,
        "feature_channels": group_count,
        "images_per_group": group_counts.tolist(),
        "feature_clip_percentiles": list(FEATURE_CLIP_PERCENTILES),
        "channel_normalization": "image_divided_by_its_global_mean",
        "bilateral_diameter": bilateral_diameter,
        "bilateral_sigma_color": bilateral_sigma_color,
        "bilateral_sigma_space": bilateral_sigma_space,
    }
    return features, std_map, (height, width), metadata


def save_feature_montage(path: Path, features: np.ndarray) -> None:
    channel_count = features.shape[2]
    columns = min(4, channel_count)
    rows = int(math.ceil(channel_count / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 3.2 * rows), squeeze=False)
    for index, axis in enumerate(axes.flat):
        if index < channel_count:
            axis.imshow(features[:, :, index], cmap="gray", vmin=0.0, vmax=1.0)
            axis.set_title(f"Angular feature {index + 1}")
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def normalized_multi_scale_boundary(
    features: np.ndarray,
    sigmas: tuple[float, ...],
    clip_percentiles: tuple[float, float],
) -> np.ndarray:
    if not sigmas:
        raise ValueError("At least one boundary scale is required")
    scale_maps: list[np.ndarray] = []
    for sigma in sigmas:
        if sigma < 0.0:
            raise ValueError("Boundary sigmas cannot be negative")
        energy = np.zeros(features.shape[:2], dtype=np.float32)
        for channel in range(features.shape[2]):
            feature = features[:, :, channel]
            if sigma > 0.0:
                feature = cv2.GaussianBlur(
                    feature,
                    (0, 0),
                    sigmaX=sigma,
                    sigmaY=sigma,
                    borderType=cv2.BORDER_REFLECT101,
                )
            gx = cv2.Sobel(feature, cv2.CV_32F, 1, 0, ksize=3) / 8.0
            gy = cv2.Sobel(feature, cv2.CV_32F, 0, 1, ksize=3) / 8.0
            energy += gx * gx + gy * gy
        strength = np.sqrt(energy / float(features.shape[2])).astype(np.float32)
        scale_maps.append(normalize_percentile_float(strength, *clip_percentiles))
    combined = np.maximum.reduce(scale_maps).astype(np.float32)
    return cv2.morphologyEx(
        combined,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )


def initial_slico(
    features: np.ndarray,
    n_segments: int,
    compactness: float,
    max_num_iter: int,
    min_size_factor: float,
    max_size_factor: float,
    enforce_connectivity: bool,
    slic_zero: bool = False,
) -> np.ndarray:
    print(
        f"Running initial SLICO on {features.shape[2]} angular channels "
        f"at {features.shape[1]}x{features.shape[0]}...",
        flush=True,
    )
    return slic(
        features,
        n_segments=n_segments,
        compactness=compactness,
        max_num_iter=max_num_iter,
        sigma=0.0,
        channel_axis=-1,
        convert2lab=False,
        enforce_connectivity=enforce_connectivity,
        min_size_factor=min_size_factor,
        max_size_factor=max_size_factor,
        slic_zero=slic_zero,
        start_label=0,
    ).astype(np.int32)


def core_markers_from_slic(initial_labels: np.ndarray) -> np.ndarray:
    boundary = find_boundaries(initial_labels, mode="thick")
    distance_to_boundary = ndi.distance_transform_edt(~boundary).astype(np.float32)
    label_image = initial_labels + 1
    label_count = int(initial_labels.max()) + 1
    positions = ndi.maximum_position(
        distance_to_boundary,
        labels=label_image,
        index=np.arange(1, label_count + 1),
    )
    markers = np.zeros(initial_labels.shape, dtype=np.int32)
    for marker_id, position in enumerate(positions, start=1):
        if position is not None:
            markers[int(position[0]), int(position[1])] = marker_id
    return markers


def add_narrow_region_markers(
    core_markers: np.ndarray,
    boundary_strength: np.ndarray,
    barrier_percentile: float,
    barrier_min_size: int,
    peak_min_distance: int,
    min_half_width: float,
    max_half_width: float,
    marker_exclusion_radius: int,
    max_extra_markers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    threshold = float(np.percentile(boundary_strength, barrier_percentile))
    barrier = boundary_strength >= threshold
    barrier = binary_closing(barrier, footprint=disk(1))
    barrier = remove_small_objects(barrier, min_size=barrier_min_size, connectivity=2)
    corridor_distance = ndi.distance_transform_edt(~barrier).astype(np.float32)
    candidate_limit = max(max_extra_markers * 4, max_extra_markers)
    candidates = peak_local_max(
        corridor_distance,
        min_distance=peak_min_distance,
        threshold_abs=min_half_width,
        exclude_border=False,
        num_peaks=candidate_limit,
    )
    if candidates.size:
        widths = corridor_distance[candidates[:, 0], candidates[:, 1]]
        candidates = candidates[widths <= max_half_width]

    markers = core_markers.copy()
    occupied = ndi.binary_dilation(markers > 0, iterations=marker_exclusion_radius).astype(np.uint8)
    next_marker = int(markers.max()) + 1
    added = 0
    for y, x in candidates:
        y_int, x_int = int(y), int(x)
        if occupied[y_int, x_int] != 0:
            continue
        markers[y_int, x_int] = next_marker
        next_marker += 1
        added += 1
        cv2.circle(occupied, (x_int, y_int), marker_exclusion_radius, color=1, thickness=-1)
        if added >= max_extra_markers:
            break
    return markers, barrier.astype(bool), corridor_distance, added


def absorb_small_watershed_regions(
    labels: np.ndarray,
    elevation: np.ndarray,
    min_region_size: int,
) -> tuple[np.ndarray, int]:
    labels_positive = labels.astype(np.int32, copy=False)
    counts = np.bincount(labels_positive.ravel())
    removed_count = int(np.count_nonzero((counts > 0) & (counts < min_region_size)))
    if removed_count == 0:
        relabeled, _, _ = relabel_sequential(labels_positive)
        return relabeled.astype(np.int32) - 1, 0
    retained = remove_small_objects(
        labels_positive,
        min_size=min_region_size,
        connectivity=1,
    ).astype(np.int32)
    refilled = watershed(
        elevation,
        markers=retained,
        mask=np.ones(labels.shape, dtype=bool),
        compactness=0.0,
        watershed_line=False,
    ).astype(np.int32)
    relabeled, _, _ = relabel_sequential(refilled)
    return relabeled.astype(np.int32) - 1, removed_count


def save_marker_preview(path: Path, markers: np.ndarray, barrier: np.ndarray) -> None:
    preview = np.zeros((*markers.shape, 3), dtype=np.uint8)
    preview[barrier] = (255, 255, 255)
    preview[markers > 0] = (255, 0, 0)
    save_rgb_png(path, preview)


def write_labeling_config(
    output_dir: Path,
    input_dir: Path,
    initial_parameters: dict[str, object],
    feature_metadata: dict[str, object],
    watershed_parameters: dict[str, object],
) -> Path:
    config_path = output_dir / "labeling_config.json"
    config = {
        "label_map": "slic_labels.npy",
        "std_image": "std_image.png",
        "output_dir": "edge_labels",
        "scale": 0.55,
        "source_raw_dir": str(input_dir),
        "segmentation_input": "24_channel_angular_features_with_twin_boundary_absorption",
        "initial_slic_parameters": initial_parameters,
        "feature_parameters": feature_metadata,
        "watershed_parameters": watershed_parameters,
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return config_path


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    output_group = args.output_group.strip()
    if not output_group or Path(output_group).name != output_group:
        raise ValueError("output_group must be one folder name, for example slic_3-1")
    return (args.output_root / output_group).resolve()


def main(args: argparse.Namespace) -> None:
    input_dir = args.input_dir.resolve()
    output_dir = resolve_output_dir(args)
    existing_edge_label_dir = output_dir / "edge_labels"
    existing_annotations = (
        list(existing_edge_label_dir.glob("*.csv"))
        + list(existing_edge_label_dir.glob("*.json"))
        + list(existing_edge_label_dir.glob("*.npy"))
    )
    if existing_annotations and not args.overwrite_labeled_group:
        raise FileExistsError(
            f"Refusing to replace superpixels in a group that already contains annotations: "
            f"{existing_edge_label_dir}. New superpixel IDs would invalidate the old edge labels. "
            f"Choose another --output-group, or pass --overwrite-labeled-group only after backing "
            f"up/removing the old annotations."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = find_image_paths(input_dir)
    if args.expected_num_images is not None and len(image_paths) != args.expected_num_images:
        raise ValueError(
            f"Expected {args.expected_num_images} angle images in {input_dir}, "
            f"found {len(image_paths)}."
        )
    print(f"Found {len(image_paths)} images in {input_dir}", flush=True)
    print(f"Output group: {output_dir}", flush=True)

    features, std_work, (height, width), feature_metadata = build_multi_angle_features(
        image_paths=image_paths,
        downsample_factor=args.downsample_factor,
        r_bins=args.r_bins,
        phi_bins=args.phi_bins,
        bilateral_diameter=args.bilateral_diameter,
        bilateral_sigma_color=args.bilateral_sigma_color,
        bilateral_sigma_space=args.bilateral_sigma_space,
    )
    std_enhanced_work = normalize_percentile_float(
        std_work,
        STD_CLIP_PERCENTILES[0],
        STD_CLIP_PERCENTILES[1],
        gamma=STD_GAMMA,
    )
    if args.bilateral_diameter > 0:
        std_enhanced_work = cv2.bilateralFilter(
            std_enhanced_work,
            d=args.bilateral_diameter,
            sigmaColor=args.bilateral_sigma_color,
            sigmaSpace=args.bilateral_sigma_space,
            borderType=cv2.BORDER_REFLECT101,
        )
    slic_features = np.concatenate(
        (features, (args.std_feature_weight * std_enhanced_work)[..., None]),
        axis=2,
    ).astype(np.float32, copy=False)
    feature_metadata["slic_feature_channels"] = int(slic_features.shape[2])
    feature_metadata["std_feature_weight"] = args.std_feature_weight
    boundary_strength = normalized_multi_scale_boundary(
        features,
        sigmas=tuple(args.boundary_sigmas),
        clip_percentiles=(args.boundary_clip_low, args.boundary_clip_high),
    )
    initial_labels = initial_slico(
        slic_features,
        args.n_segments,
        args.compactness,
        args.max_num_iter,
        args.min_size_factor,
        args.max_size_factor,
        args.enforce_connectivity,
        slic_zero=args.slic_zero,
    )

    print("Building SLIC core markers and narrow-twin markers...", flush=True)
    core_markers = core_markers_from_slic(initial_labels)
    markers, barrier, corridor_distance, added_markers = add_narrow_region_markers(
        core_markers,
        boundary_strength,
        args.barrier_percentile,
        args.barrier_min_size,
        args.twin_marker_min_distance,
        args.twin_min_half_width,
        args.twin_max_half_width,
        args.marker_exclusion_radius,
        args.max_extra_twin_markers,
    )
    print(
        f"Markers: {int(core_markers.max())} SLIC cores + {added_markers} narrow-region markers",
        flush=True,
    )
    print("Running compact marker-controlled watershed...", flush=True)
    watershed_labels = watershed(
        boundary_strength,
        markers=markers,
        mask=np.ones(boundary_strength.shape, dtype=bool),
        compactness=args.watershed_compactness,
        watershed_line=False,
    ).astype(np.int32)
    final_work, removed_regions = absorb_small_watershed_regions(
        watershed_labels,
        boundary_strength,
        args.min_region_size,
    )

    if final_work.shape == (height, width):
        final_full = final_work
        initial_full = initial_labels
        std_full = std_work
        std_enhanced_full = std_enhanced_work
        boundary_full = boundary_strength
        barrier_full = barrier
    else:
        final_full = cv2.resize(final_work, (width, height), interpolation=cv2.INTER_NEAREST).astype(np.int32)
        initial_full = cv2.resize(initial_labels, (width, height), interpolation=cv2.INTER_NEAREST).astype(np.int32)
        std_full = cv2.resize(std_work, (width, height), interpolation=cv2.INTER_LINEAR)
        std_enhanced_full = cv2.resize(std_enhanced_work, (width, height), interpolation=cv2.INTER_LINEAR)
        boundary_full = cv2.resize(boundary_strength, (width, height), interpolation=cv2.INTER_LINEAR)
        barrier_full = cv2.resize(barrier.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0

    np.save(output_dir / "slic_labels.npy", final_full, allow_pickle=False)
    np.save(output_dir / "initial_slic_labels.npy", initial_full, allow_pickle=False)
    save_gray_png(output_dir / "std_image.png", std_enhanced_full)
    save_gray_png(output_dir / "std_slic_input.png", std_full)
    save_gray_png(output_dir / "angular_boundary_strength.png", boundary_full)
    save_gray_png(output_dir / "boundary_barrier_mask.png", barrier_full.astype(np.float32))
    save_gray_png(output_dir / "twin_corridor_distance.png", corridor_distance)
    save_gray_png(output_dir / "segment_labels_slic_resolution.png", final_work)
    save_gray_png(output_dir / "segment_labels_full_resolution.png", final_full)
    save_feature_montage(output_dir / "angular_features_preview.png", features)
    save_marker_preview(output_dir / "watershed_markers_and_barriers.png", markers, barrier)

    initial_overlay = create_boundary_overlay(std_enhanced_full, initial_full)
    final_overlay = create_boundary_overlay(std_enhanced_full, final_full)
    angular_overlay = create_boundary_overlay(boundary_full, final_full)
    save_rgb_png(output_dir / "segmentation_initial_slic.png", initial_overlay)
    save_rgb_png(output_dir / "segmentation_full_resolution.png", final_overlay)
    save_rgb_png(output_dir / "segmentation_slic_resolution.png", create_boundary_overlay(std_enhanced_work, final_work))
    save_rgb_png(output_dir / "segmentation_on_angular_boundary.png", angular_overlay)

    initial_parameters: dict[str, object] = {
        "n_segments": args.n_segments,
        "compactness": args.compactness,
        "downsample_factor": args.downsample_factor,
        "max_num_iter": args.max_num_iter,
        "min_size_factor": args.min_size_factor,
        "max_size_factor": args.max_size_factor,
        "slic_zero": args.slic_zero,
        "enforce_connectivity": args.enforce_connectivity,
        "r_bins": args.r_bins,
        "phi_bins": args.phi_bins,
        "std_feature_weight": args.std_feature_weight,
        "bilateral_diameter": args.bilateral_diameter,
        "bilateral_sigma_color": args.bilateral_sigma_color,
        "bilateral_sigma_space": args.bilateral_sigma_space,
    }
    watershed_parameters: dict[str, object] = {
        "boundary_sigmas": args.boundary_sigmas,
        "boundary_clip_percentiles": [args.boundary_clip_low, args.boundary_clip_high],
        "barrier_percentile": args.barrier_percentile,
        "barrier_min_size": args.barrier_min_size,
        "twin_marker_min_distance": args.twin_marker_min_distance,
        "twin_half_width_range": [args.twin_min_half_width, args.twin_max_half_width],
        "marker_exclusion_radius": args.marker_exclusion_radius,
        "max_extra_twin_markers": args.max_extra_twin_markers,
        "actual_extra_twin_markers": added_markers,
        "watershed_compactness": args.watershed_compactness,
        "min_region_size": args.min_region_size,
        "removed_small_regions": removed_regions,
    }
    config_path = write_labeling_config(
        output_dir,
        input_dir,
        initial_parameters,
        feature_metadata,
        watershed_parameters,
    )

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    panels = (
        (axes[0, 0], std_enhanced_full, "Enhanced standard deviation", "gray"),
        (axes[0, 1], boundary_full, "Multi-scale angular boundary", "gray"),
        (axes[0, 2], barrier_full, "Boundary barrier mask", "gray"),
        (axes[1, 0], initial_overlay, "Initial 24-channel SLICO", None),
        (axes[1, 1], final_overlay, "Twin-boundary-absorbing result", None),
        (axes[1, 2], angular_overlay, "Result on angular boundary", None),
    )
    for axis, image, title, cmap in panels:
        axis.imshow(image, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "visualization.png", dpi=150, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close(fig)

    print(f"Initial SLIC regions: {len(np.unique(initial_full))}")
    print(f"Final watershed superpixels: {len(np.unique(final_full))}")
    print(f"Added narrow-region markers: {added_markers}")
    print(f"Removed small watershed regions: {removed_regions}")
    print(f"Labels: {output_dir / 'slic_labels.npy'}")
    print(f"Labeling config: {config_path}")
    print("Next step: python edge_labeling_app.py")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final twin-boundary-absorbing superpixel pipeline.")
    parser.add_argument("--input-dir", "--input_dir", dest="input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--expected-num-images",
        type=int,
        default=DEFAULT_EXPECTED_NUM_IMAGES,
        help="Required number of angle images in the input directory (default: 396).",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-group", type=str, default=DEFAULT_OUTPUT_GROUP)
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=Path,
        default=None,
        help="Explicit output directory; overrides --output-root/--output-group.",
    )
    parser.add_argument(
        "--overwrite-labeled-group",
        action="store_true",
        help="Allow replacing superpixels even when edge_labels already exist (old labels become invalid).",
    )
    parser.add_argument("--n-segments", type=int, default=10000)
    parser.add_argument("--compactness", type=float, default=1)
    parser.add_argument("--std-feature-weight", type=float, default=STD_FEATURE_WEIGHT)
    parser.add_argument("--slic-zero", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--downsample-factor", type=float, default=1.0)
    parser.add_argument("--max-num-iter", type=int, default=30)
    parser.add_argument("--min-size-factor", type=float, default=0.35)
    parser.add_argument("--max-size-factor", type=float, default=2.0)
    parser.add_argument("--r-bins", type=int, default=3)
    parser.add_argument("--phi-bins", type=int, default=8)
    parser.add_argument("--bilateral-diameter", type=int, default=3)
    parser.add_argument("--bilateral-sigma-color", type=float, default=0.045)
    parser.add_argument("--bilateral-sigma-space", type=float, default=1.5)
    parser.add_argument("--boundary-sigmas", type=float, nargs="+", default=[0.0, 0.8, 1.6])
    parser.add_argument("--boundary-clip-low", type=float, default=2.0)
    parser.add_argument("--boundary-clip-high", type=float, default=99.5)
    parser.add_argument("--barrier-percentile", type=float, default=80.0)
    parser.add_argument("--barrier-min-size", type=int, default=8)
    parser.add_argument("--twin-marker-min-distance", type=int, default=4)
    parser.add_argument("--twin-min-half-width", type=float, default=1.5)
    parser.add_argument("--twin-max-half-width", type=float, default=8.0)
    parser.add_argument("--marker-exclusion-radius", type=int, default=4)
    parser.add_argument("--max-extra-twin-markers", type=int, default=7000)
    parser.add_argument("--watershed-compactness", type=float, default=0.01)
    parser.add_argument("--min-region-size", type=int, default=35)
    parser.add_argument("--no-enforce-connectivity", dest="enforce_connectivity", action="store_false")
    parser.set_defaults(enforce_connectivity=True)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args([
        "--compactness", "0.28",
        "--barrier-percentile", "84",
        "--std-feature-weight", "0.1",
        "--slic-zero",
        *sys.argv[1:],
    ]))
