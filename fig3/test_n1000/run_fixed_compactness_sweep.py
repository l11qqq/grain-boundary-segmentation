"""Isolated fixed-compactness sweep for all three Figure 3 methods.

The original scripts remain unchanged. Stage 2 and stage 3 normally enable
SLICO (adaptive compactness); this runner replaces only their SLIC call with
standard fixed-compactness SLIC for the controlled m sweep.
"""

from __future__ import annotations

import copy
import json
import sys
from argparse import Namespace
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import find_boundaries, slic as skimage_slic


TEST_DIR = Path(__file__).resolve().parent
FIG3_DIR = TEST_DIR.parent
PROJECT_DIR = FIG3_DIR.parent
INPUT_DIR = FIG3_DIR / "crop"
COMMON_BACKGROUND_PATH = FIG3_DIR / "result_3" / "std_image.png"
N_SEGMENTS = 10_000
COMPACTNESS_VALUES = (
    ("0.15", 0.15),
    ("1", 1.0),
    ("2", 2.0),
    ("5", 5.0),
    ("10", 10.0),
)
METHODS = (
    ("method_1", "Basic single-channel SLIC"),
    ("method_2", "Multichannel edge-preserving pipeline"),
    ("method_3", "Final twin-boundary pipeline"),
)

sys.path.insert(0, str(FIG3_DIR))
sys.path.insert(0, str(PROJECT_DIR))

import fig3_common  # noqa: E402
import slic_test  # noqa: E402
import slic_version_1_basic as stage1  # noqa: E402
import slic_version_2_multichannel_slico as stage2  # noqa: E402


def fixed_compactness_slic(*args, **kwargs):
    """Call scikit-image SLIC with spatial/color weight fixed throughout."""
    kwargs["slic_zero"] = False
    return skimage_slic(*args, **kwargs)


def common_background_overlay(background: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if labels.shape != background.shape:
        raise ValueError(
            f"Label/background shape mismatch: {labels.shape} versus {background.shape}"
        )
    rgb = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
    rgb[find_boundaries(labels, mode="inner")] = (255, 0, 0)
    return rgb


def update_json(path: Path, updates: dict[str, object]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(updates)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_comparison(path: Path, overlays: list[tuple[str, np.ndarray]]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for axis, (title, overlay) in zip(axes, overlays):
        axis.imshow(overlay)
        axis.set_title(title)
        axis.axis("off")
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def prepare_cached_inputs() -> tuple[
    list[Path],
    np.ndarray,
    tuple[np.ndarray, np.ndarray, dict[str, object], np.ndarray],
    tuple[np.ndarray, np.ndarray, tuple[int, int], dict[str, object], np.ndarray],
]:
    paths = fig3_common.find_image_paths(INPUT_DIR)

    print("Preparing stage-1 standard-deviation input once...", flush=True)
    stage1_std_raw = fig3_common.compute_std_map(paths)

    print("Preparing stage-2 angular features once...", flush=True)
    stage2_features, stage2_std_raw, stage2_metadata = stage2.build_legacy_features(paths)
    stage2_angular_strength = stage2.angular_gradient(stage2_features)

    base_stage3_args = slic_test.parse_args(
        [
            "--input-dir",
            str(INPUT_DIR),
            "--output-dir",
            str(TEST_DIR / "_stage3_cache_probe"),
            "--n-segments",
            str(N_SEGMENTS),
        ]
    )
    print("Preparing stage-3 angular features once...", flush=True)
    stage3_features, stage3_std_raw, stage3_shape, stage3_metadata = (
        slic_test.build_multi_angle_features(
            image_paths=paths,
            downsample_factor=base_stage3_args.downsample_factor,
            r_bins=base_stage3_args.r_bins,
            phi_bins=base_stage3_args.phi_bins,
            bilateral_diameter=base_stage3_args.bilateral_diameter,
            bilateral_sigma_color=base_stage3_args.bilateral_sigma_color,
            bilateral_sigma_space=base_stage3_args.bilateral_sigma_space,
        )
    )
    stage3_boundary_strength = slic_test.normalized_multi_scale_boundary(
        stage3_features,
        sigmas=tuple(base_stage3_args.boundary_sigmas),
        clip_percentiles=(
            base_stage3_args.boundary_clip_low,
            base_stage3_args.boundary_clip_high,
        ),
    )
    return (
        paths,
        stage1_std_raw,
        (stage2_features, stage2_std_raw, stage2_metadata, stage2_angular_strength),
        (
            stage3_features,
            stage3_std_raw,
            stage3_shape,
            stage3_metadata,
            stage3_boundary_strength,
        ),
    )


def main() -> None:
    background = cv2.imread(str(COMMON_BACKGROUND_PATH), cv2.IMREAD_GRAYSCALE)
    if background is None:
        raise FileNotFoundError(f"Cannot read common background: {COMMON_BACKGROUND_PATH}")

    (
        paths,
        stage1_std_raw,
        stage2_cache,
        stage3_cache,
    ) = prepare_cached_inputs()
    stage2_features, stage2_std_raw, stage2_metadata, stage2_angular_strength = stage2_cache
    (
        stage3_features,
        stage3_std_raw,
        stage3_shape,
        stage3_metadata,
        stage3_boundary_strength,
    ) = stage3_cache

    # Reuse the expensive image-derived inputs while retaining each original
    # method's segmentation and postprocessing implementation.
    stage1.compute_std_map = lambda _paths: stage1_std_raw
    stage2.build_legacy_features = lambda _paths: (
        stage2_features,
        stage2_std_raw,
        copy.deepcopy(stage2_metadata),
    )
    stage2.angular_gradient = lambda _features: stage2_angular_strength
    stage2.slic = fixed_compactness_slic
    slic_test.build_multi_angle_features = lambda **_kwargs: (
        stage3_features,
        stage3_std_raw,
        stage3_shape,
        copy.deepcopy(stage3_metadata),
    )
    slic_test.normalized_multi_scale_boundary = lambda *_args, **_kwargs: stage3_boundary_strength
    slic_test.slic = fixed_compactness_slic

    manifest: dict[str, object] = {
        "input_dir": str(INPUT_DIR.resolve()),
        "image_count": len(paths),
        "n_segments": N_SEGMENTS,
        "compactness_mode": "fixed (slic_zero=False)",
        "compactness_values": [value for _, value in COMPACTNESS_VALUES],
        "common_visualization_background": str(COMMON_BACKGROUND_PATH.resolve()),
        "methods": [description for _, description in METHODS],
        "runs": [],
    }

    for folder_name, compactness in COMPACTNESS_VALUES:
        m_dir = TEST_DIR / folder_name
        m_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Fixed compactness m={compactness:g} ===", flush=True)

        method1_dir = m_dir / "method_1_data"
        stage1.main(INPUT_DIR, method1_dir, N_SEGMENTS, compactness)
        update_json(
            method1_dir / "method_config.json",
            {
                "controlled_sweep": {
                    "compactness_mode": "fixed",
                    "slic_zero": False,
                }
            },
        )

        method2_dir = m_dir / "method_2_data"
        stage2.main(
            Namespace(
                input_dir=INPUT_DIR,
                output_dir=method2_dir,
                n_segments=N_SEGMENTS,
                compactness=compactness,
            )
        )
        method2_config_path = method2_dir / "method_config.json"
        method2_config = json.loads(method2_config_path.read_text(encoding="utf-8"))
        method2_config["parameters"]["slic_zero"] = False
        method2_config["controlled_sweep"] = {"compactness_mode": "fixed"}
        method2_config_path.write_text(
            json.dumps(method2_config, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        method3_dir = m_dir / "method_3_data"
        stage3_args = slic_test.parse_args(
            [
                "--input-dir",
                str(INPUT_DIR),
                "--output-dir",
                str(method3_dir),
                "--n-segments",
                str(N_SEGMENTS),
                "--compactness",
                str(compactness),
            ]
        )
        slic_test.main(stage3_args)
        method3_config_path = method3_dir / "labeling_config.json"
        method3_config = json.loads(method3_config_path.read_text(encoding="utf-8"))
        method3_config["initial_slic_parameters"]["slic_zero"] = False
        method3_config["controlled_sweep"] = {"compactness_mode": "fixed"}
        method3_config_path.write_text(
            json.dumps(method3_config, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        method_dirs = (method1_dir, method2_dir, method3_dir)
        overlays: list[tuple[str, np.ndarray]] = []
        counts: dict[str, int] = {}
        for (method_name, method_title), method_dir in zip(METHODS, method_dirs):
            labels = np.load(method_dir / "slic_labels.npy", allow_pickle=False)
            overlay = common_background_overlay(background, labels)
            visualization_path = m_dir / f"{method_name}_visualization.png"
            cv2.imwrite(
                str(visualization_path),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            )
            overlays.append((method_title, overlay))
            counts[method_name] = int(np.unique(labels).size)

        comparison_path = m_dir / "comparison.png"
        save_comparison(comparison_path, overlays)
        run_summary = {
            "compactness": compactness,
            "compactness_mode": "fixed",
            "slic_zero": False,
            "n_segments": N_SEGMENTS,
            "actual_superpixel_counts": counts,
            "visualizations": [f"{name}_visualization.png" for name, _ in METHODS],
            "comparison": comparison_path.name,
        }
        (m_dir / "run_summary.json").write_text(
            json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        manifest["runs"].append(run_summary)
        print(f"Completed m={compactness:g}: {counts}", flush=True)

    (TEST_DIR / "fixed_compactness_sweep_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nFixed-compactness sweep completed.", flush=True)


if __name__ == "__main__":
    main()
