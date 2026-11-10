"""Stage-3 ablation with std-channel and std-gradient fusion.

Feature fusion:
    [24 angular channels, alpha * enhanced_std], alpha = 0.30

Boundary-energy fusion:
    (1 - beta) * angular_energy + beta * std_gradient, beta = 0.20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
import slic_test  # noqa: E402


OUTPUT_DIR = PROJECT_DIR / "fig3" / "std" / "stage3_with_std"
STD_FEATURE_WEIGHT = 0.30
STD_BOUNDARY_WEIGHT = 0.20


def normalized_std_gradient(std_feature: np.ndarray, sigmas: tuple[float, ...]) -> np.ndarray:
    scale_maps: list[np.ndarray] = []
    for sigma in sigmas:
        feature = std_feature
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
        gradient = np.sqrt(gx * gx + gy * gy).astype(np.float32)
        scale_maps.append(slic_test.normalize_percentile_float(gradient, 2.0, 99.5))
    return np.maximum.reduce(scale_maps).astype(np.float32)


def save_main_outputs(
    output_dir: Path,
    labels: np.ndarray,
    initial_labels: np.ndarray,
    std_raw: np.ndarray,
    std_feature: np.ndarray,
    angular_energy: np.ndarray,
    std_energy: np.ndarray,
    fused_energy: np.ndarray,
    barrier: np.ndarray,
    corridor_distance: np.ndarray,
    markers: np.ndarray,
    parameters: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "slic_labels.npy", labels.astype(np.int32), allow_pickle=False)
    np.save(output_dir / "initial_slic_labels.npy", initial_labels.astype(np.int32), allow_pickle=False)
    slic_test.save_gray_png(output_dir / "std_image.png", std_feature)
    slic_test.save_gray_png(output_dir / "std_slic_input.png", std_raw)
    slic_test.save_gray_png(output_dir / "angular_boundary_strength.png", angular_energy)
    slic_test.save_gray_png(output_dir / "std_boundary_strength.png", std_energy)
    slic_test.save_gray_png(output_dir / "fused_boundary_strength.png", fused_energy)
    slic_test.save_gray_png(output_dir / "boundary_barrier_mask.png", barrier.astype(np.float32))
    slic_test.save_gray_png(output_dir / "twin_corridor_distance.png", corridor_distance)
    slic_test.save_marker_preview(output_dir / "watershed_markers_and_barriers.png", markers, barrier)

    initial_overlay = slic_test.create_boundary_overlay(std_feature, initial_labels)
    final_overlay = slic_test.create_boundary_overlay(std_feature, labels)
    fused_overlay = slic_test.create_boundary_overlay(fused_energy, labels)
    slic_test.save_rgb_png(output_dir / "segmentation_initial_slic.png", initial_overlay)
    slic_test.save_rgb_png(output_dir / "segmentation_full_resolution.png", final_overlay)
    slic_test.save_rgb_png(output_dir / "segmentation_on_fused_boundary.png", fused_overlay)

    config = {
        "method": "Stage 3 with std feature and std-gradient fusion",
        "source_raw_dir": str(slic_test.DEFAULT_INPUT_DIR),
        "image_count": 396,
        "label_map": "slic_labels.npy",
        "std_image": "std_image.png",
        "actual_superpixels": int(len(np.unique(labels))),
        "parameters": parameters,
    }
    import json
    (output_dir / "method_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    panels = (
        (std_feature, "Enhanced standard deviation", "gray"),
        (angular_energy, "24-channel angular boundary", "gray"),
        (std_energy, "Standard-deviation boundary", "gray"),
        (initial_overlay, "25-channel initial SLICO", None),
        (final_overlay, "Std-assisted twin-aware result", None),
        (fused_overlay, "Result on fused boundary energy", None),
    )
    for axis, (image, title, cmap) in zip(axes.flat, panels):
        axis.imshow(image, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "visualization.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = slic_test.parse_args(
        [
            "--input-dir", str(slic_test.DEFAULT_INPUT_DIR),
            "--output-dir", str(OUTPUT_DIR),
            "--n-segments", "10000",
            "--compactness", "0.28",
        ]
    )
    paths = slic_test.find_image_paths(args.input_dir.resolve())
    if len(paths) != 396:
        raise ValueError(f"Expected 396 input images, found {len(paths)}")

    angular_features, std_raw, _, feature_metadata = slic_test.build_multi_angle_features(
        paths,
        args.downsample_factor,
        args.r_bins,
        args.phi_bins,
        args.bilateral_diameter,
        args.bilateral_sigma_color,
        args.bilateral_sigma_space,
    )
    std_feature = slic_test.normalize_percentile_float(std_raw, 2.0, 98.0, gamma=0.65)
    std_feature = cv2.bilateralFilter(
        std_feature,
        d=args.bilateral_diameter,
        sigmaColor=args.bilateral_sigma_color,
        sigmaSpace=args.bilateral_sigma_space,
        borderType=cv2.BORDER_REFLECT101,
    )
    fused_features = np.concatenate(
        (angular_features, (STD_FEATURE_WEIGHT * std_feature)[..., None]),
        axis=2,
    ).astype(np.float32, copy=False)

    boundary_sigmas = tuple(args.boundary_sigmas)
    angular_energy = slic_test.normalized_multi_scale_boundary(
        angular_features,
        boundary_sigmas,
        (args.boundary_clip_low, args.boundary_clip_high),
    )
    std_energy = normalized_std_gradient(std_feature, boundary_sigmas)
    fused_energy = (
        (1.0 - STD_BOUNDARY_WEIGHT) * angular_energy
        + STD_BOUNDARY_WEIGHT * std_energy
    ).astype(np.float32)
    fused_energy = slic_test.normalize_percentile_float(fused_energy, 0.0, 100.0)

    initial_labels = slic_test.initial_slico(
        fused_features,
        args.n_segments,
        args.compactness,
        args.max_num_iter,
        args.min_size_factor,
        args.max_size_factor,
        args.enforce_connectivity,
    )
    core_markers = slic_test.core_markers_from_slic(initial_labels)
    markers, barrier, corridor_distance, added_markers = slic_test.add_narrow_region_markers(
        core_markers,
        fused_energy,
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
    watershed_labels = slic_test.watershed(
        fused_energy,
        markers=markers,
        mask=np.ones(fused_energy.shape, dtype=bool),
        compactness=args.watershed_compactness,
        watershed_line=False,
    ).astype(np.int32)
    final_labels, removed_regions = slic_test.absorb_small_watershed_regions(
        watershed_labels,
        fused_energy,
        args.min_region_size,
    )

    save_main_outputs(
        OUTPUT_DIR,
        final_labels,
        initial_labels,
        std_raw,
        std_feature,
        angular_energy,
        std_energy,
        fused_energy,
        barrier,
        corridor_distance,
        markers,
        {
            "n_segments": args.n_segments,
            "compactness": args.compactness,
            "angular_feature_channels": 24,
            "std_feature_channels": 1,
            "std_feature_weight_alpha": STD_FEATURE_WEIGHT,
            "angular_boundary_weight": 1.0 - STD_BOUNDARY_WEIGHT,
            "std_boundary_weight_beta": STD_BOUNDARY_WEIGHT,
            "boundary_sigmas": list(boundary_sigmas),
            "actual_extra_twin_markers": added_markers,
            "removed_small_regions": removed_regions,
            "feature_metadata": feature_metadata,
        },
    )
    print(f"Initial regions: {len(np.unique(initial_labels))}")
    print(f"Final regions: {len(np.unique(final_labels))}")


if __name__ == "__main__":
    main()

