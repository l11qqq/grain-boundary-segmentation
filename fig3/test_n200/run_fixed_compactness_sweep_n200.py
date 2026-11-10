"""Isolated fixed-compactness comparison of the three Figure 3 methods at n=200."""

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


OUTPUT_DIR = Path(__file__).resolve().parent
FIG3_DIR = OUTPUT_DIR.parent
PROJECT_DIR = FIG3_DIR.parent
INPUT_DIR = FIG3_DIR / "crop"
COMMON_BACKGROUND_PATH = FIG3_DIR / "result_3" / "std_image.png"
N_SEGMENTS = 200
COMPACTNESS_VALUES = (("0.15", 0.15), ("1", 1.0), ("2", 2.0), ("5", 5.0))
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


def fixed_slic(*args, **kwargs):
    kwargs["slic_zero"] = False
    return skimage_slic(*args, **kwargs)


def overlay_on_common_background(background: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if labels.shape != background.shape:
        raise ValueError(f"Shape mismatch: labels={labels.shape}, background={background.shape}")
    overlay = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
    overlay[find_boundaries(labels, mode="inner")] = (255, 0, 0)
    return overlay


def save_comparison(path: Path, overlays: list[tuple[str, np.ndarray]]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for axis, (title, image) in zip(axes, overlays):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    background = cv2.imread(str(COMMON_BACKGROUND_PATH), cv2.IMREAD_GRAYSCALE)
    if background is None:
        raise FileNotFoundError(f"Cannot read {COMMON_BACKGROUND_PATH}")
    paths = fig3_common.find_image_paths(INPUT_DIR)

    # Build each method's image-derived inputs once, then reuse them across m.
    print("Preparing stage-1 input...", flush=True)
    stage1_std = fig3_common.compute_std_map(paths)
    print("Preparing stage-2 inputs...", flush=True)
    stage2_features, stage2_std, stage2_metadata = stage2.build_legacy_features(paths)
    stage2_angular = stage2.angular_gradient(stage2_features)
    base3 = slic_test.parse_args(
        ["--input-dir", str(INPUT_DIR), "--output-dir", str(OUTPUT_DIR / "_probe")]
    )
    print("Preparing stage-3 inputs...", flush=True)
    stage3_features, stage3_std, stage3_shape, stage3_metadata = (
        slic_test.build_multi_angle_features(
            image_paths=paths,
            downsample_factor=base3.downsample_factor,
            r_bins=base3.r_bins,
            phi_bins=base3.phi_bins,
            bilateral_diameter=base3.bilateral_diameter,
            bilateral_sigma_color=base3.bilateral_sigma_color,
            bilateral_sigma_space=base3.bilateral_sigma_space,
        )
    )
    stage3_boundary = slic_test.normalized_multi_scale_boundary(
        stage3_features,
        sigmas=tuple(base3.boundary_sigmas),
        clip_percentiles=(base3.boundary_clip_low, base3.boundary_clip_high),
    )

    stage1.compute_std_map = lambda _paths: stage1_std
    stage2.build_legacy_features = lambda _paths: (
        stage2_features,
        stage2_std,
        copy.deepcopy(stage2_metadata),
    )
    stage2.angular_gradient = lambda _features: stage2_angular
    stage2.slic = fixed_slic
    slic_test.build_multi_angle_features = lambda **_kwargs: (
        stage3_features,
        stage3_std,
        stage3_shape,
        copy.deepcopy(stage3_metadata),
    )
    slic_test.normalized_multi_scale_boundary = lambda *_args, **_kwargs: stage3_boundary
    slic_test.slic = fixed_slic

    manifest: dict[str, object] = {
        "input_dir": str(INPUT_DIR.resolve()),
        "image_count": len(paths),
        "n_segments": N_SEGMENTS,
        "compactness_mode": "fixed (slic_zero=False)",
        "compactness_values": [value for _, value in COMPACTNESS_VALUES],
        "common_visualization_background": str(COMMON_BACKGROUND_PATH.resolve()),
        "runs": [],
    }

    for folder_name, compactness in COMPACTNESS_VALUES:
        run_dir = OUTPUT_DIR / folder_name
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== n=200, fixed m={compactness:g} ===", flush=True)

        method1_dir = run_dir / "method_1_data"
        stage1.main(INPUT_DIR, method1_dir, N_SEGMENTS, compactness)
        config1_path = method1_dir / "method_config.json"
        config1 = json.loads(config1_path.read_text(encoding="utf-8"))
        config1["controlled_sweep"] = {"compactness_mode": "fixed", "slic_zero": False}
        config1_path.write_text(json.dumps(config1, ensure_ascii=False, indent=2), encoding="utf-8")

        method2_dir = run_dir / "method_2_data"
        stage2.main(
            Namespace(
                input_dir=INPUT_DIR,
                output_dir=method2_dir,
                n_segments=N_SEGMENTS,
                compactness=compactness,
            )
        )
        config2_path = method2_dir / "method_config.json"
        config2 = json.loads(config2_path.read_text(encoding="utf-8"))
        config2["parameters"]["slic_zero"] = False
        config2["controlled_sweep"] = {"compactness_mode": "fixed"}
        config2_path.write_text(json.dumps(config2, ensure_ascii=False, indent=2), encoding="utf-8")

        method3_dir = run_dir / "method_3_data"
        args3 = slic_test.parse_args(
            [
                "--input-dir", str(INPUT_DIR),
                "--output-dir", str(method3_dir),
                "--n-segments", str(N_SEGMENTS),
                "--compactness", str(compactness),
            ]
        )
        slic_test.main(args3)
        config3_path = method3_dir / "labeling_config.json"
        config3 = json.loads(config3_path.read_text(encoding="utf-8"))
        config3["initial_slic_parameters"]["slic_zero"] = False
        config3["controlled_sweep"] = {"compactness_mode": "fixed"}
        config3_path.write_text(json.dumps(config3, ensure_ascii=False, indent=2), encoding="utf-8")

        overlays: list[tuple[str, np.ndarray]] = []
        counts: dict[str, int] = {}
        for (method_name, title), data_dir in zip(
            METHODS, (method1_dir, method2_dir, method3_dir)
        ):
            labels = np.load(data_dir / "slic_labels.npy", allow_pickle=False)
            image = overlay_on_common_background(background, labels)
            output_path = run_dir / f"{method_name}_visualization.png"
            cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            overlays.append((title, image))
            counts[method_name] = int(np.unique(labels).size)

        save_comparison(run_dir / "comparison.png", overlays)
        summary = {
            "n_segments": N_SEGMENTS,
            "compactness": compactness,
            "compactness_mode": "fixed",
            "slic_zero": False,
            "actual_superpixel_counts": counts,
        }
        (run_dir / "run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        manifest["runs"].append(summary)
        print(f"Completed m={compactness:g}: {counts}", flush=True)

    (OUTPUT_DIR / "fixed_compactness_n200_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nn=200 fixed-compactness sweep completed.", flush=True)


if __name__ == "__main__":
    main()
