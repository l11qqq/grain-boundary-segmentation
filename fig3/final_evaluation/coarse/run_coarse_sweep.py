"""Run an isolated 4 x 3 coarse parameter sweep with ``slic_test.py``.

The expensive raw-image features are computed once, then the original final
pipeline is run for all combinations of n={4000,6000,8000,10000} and
m={0.05,0.5,5}.  SLIC is switched from SLICO to fixed-compactness SLIC so that
``m`` has the same controlled meaning in every run.  All generated artifacts
remain below this script's directory.

The requested visualization background is the first angular-group mean image
(channel 1 of the 24-channel feature tensor built by ``slic_test.py``).
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import find_boundaries, slic as skimage_slic


COARSE_DIR = Path(__file__).resolve().parent
FIG3_DIR = COARSE_DIR.parents[1]
PROJECT_DIR = FIG3_DIR.parent
INPUT_DIR = PROJECT_DIR / "raw_pic413-3-1n"
FIRST_MEAN_PATH = COARSE_DIR / "first_angular_mean_background.png"
MANIFEST_PATH = COARSE_DIR / "coarse_parameter_sweep_manifest.json"
CONTACT_SHEET_PATH = COARSE_DIR / "coarse_12_group_comparison.png"

N_VALUES = (4_000, 6_000, 8_000, 10_000)
M_VALUES = (0.05, 0.5, 5.0)
PARAMETER_SETS = tuple(
    (f"n{n}_m{m:g}", n, m)
    for n in N_VALUES
    for m in M_VALUES
)

sys.path.insert(0, str(PROJECT_DIR))
import slic_test  # noqa: E402


def fixed_compactness_slic(*args, **kwargs):
    """Use standard fixed-compactness SLIC instead of adaptive SLICO."""
    kwargs["slic_zero"] = False
    return skimage_slic(*args, **kwargs)


def first_mean_overlay(background: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if labels.shape != background.shape:
        raise ValueError(f"Shape mismatch: labels={labels.shape}, background={background.shape}")
    gray = slic_test.normalize_to_uint8(background)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    rgb[find_boundaries(labels, mode="inner")] = (255, 0, 0)
    return rgb


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Failed to write image: {path}")


def save_contact_sheet(overlays: list[tuple[int, float, int, np.ndarray]]) -> None:
    figure, axes = plt.subplots(
        len(N_VALUES),
        len(M_VALUES),
        figsize=(15, 16),
        constrained_layout=True,
    )
    for axis, (n_segments, compactness, actual_count, overlay) in zip(axes.flat, overlays):
        axis.imshow(overlay)
        axis.set_title(f"n={n_segments}, m={compactness:g}\nactual={actual_count}")
        axis.axis("off")
    figure.suptitle(
        "slic_test.py coarse sweep — boundaries on first angular mean image",
        fontsize=16,
    )
    figure.savefig(CONTACT_SHEET_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    COARSE_DIR.mkdir(parents=True, exist_ok=True)
    image_paths = slic_test.find_image_paths(INPUT_DIR)
    base_args = slic_test.parse_args(
        [
            "--input-dir",
            str(INPUT_DIR),
            "--output-dir",
            str(COARSE_DIR / "_cache_probe"),
        ]
    )

    print("Building the 24 angular-group mean features once...", flush=True)
    features, std_raw, image_shape, feature_metadata = slic_test.build_multi_angle_features(
        image_paths=image_paths,
        downsample_factor=base_args.downsample_factor,
        r_bins=base_args.r_bins,
        phi_bins=base_args.phi_bins,
        bilateral_diameter=base_args.bilateral_diameter,
        bilateral_sigma_color=base_args.bilateral_sigma_color,
        bilateral_sigma_space=base_args.bilateral_sigma_space,
    )
    if features.shape[2] != 24:
        raise ValueError(f"Expected 24 angular mean channels, found {features.shape[2]}")
    first_mean = features[:, :, 0].astype(np.float32, copy=True)
    boundary_strength = slic_test.normalized_multi_scale_boundary(
        features,
        sigmas=tuple(base_args.boundary_sigmas),
        clip_percentiles=(base_args.boundary_clip_low, base_args.boundary_clip_high),
    )

    if first_mean.shape != image_shape:
        first_mean_full = cv2.resize(
            first_mean,
            (image_shape[1], image_shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        first_mean_full = first_mean
    if not cv2.imwrite(str(FIRST_MEAN_PATH), slic_test.normalize_to_uint8(first_mean_full)):
        raise OSError(f"Failed to write image: {FIRST_MEAN_PATH}")

    # Keep the original slic_test.py pipeline, while reusing immutable cached
    # inputs and making compactness a fixed controlled parameter.
    slic_test.build_multi_angle_features = lambda **_kwargs: (
        features,
        std_raw,
        image_shape,
        copy.deepcopy(feature_metadata),
    )
    slic_test.normalized_multi_scale_boundary = lambda *_args, **_kwargs: boundary_strength
    slic_test.slic = fixed_compactness_slic

    manifest: dict[str, object] = {
        "runner": str(Path(__file__).resolve()),
        "pipeline_source": str((PROJECT_DIR / "slic_test.py").resolve()),
        "input_dir": str(INPUT_DIR.resolve()),
        "input_image_count": len(image_paths),
        "image_shape": list(image_shape),
        "parameter_grid": {
            "n_segments": list(N_VALUES),
            "compactness_m": list(M_VALUES),
            "combination_count": len(PARAMETER_SETS),
        },
        "compactness_mode": "fixed standard SLIC (slic_zero=False)",
        "postprocessing": "original slic_test.py marker-controlled watershed pipeline",
        "visualization_background": {
            "definition": "first angular-group mean feature (channel 1 of 24)",
            "path": str(FIRST_MEAN_PATH.resolve()),
        },
        "runs": [],
    }
    contact_overlays: list[tuple[int, float, int, np.ndarray]] = []

    for run_index, (folder, n_segments, compactness) in enumerate(PARAMETER_SETS, start=1):
        output_dir = COARSE_DIR / folder
        print(
            f"\n[{run_index}/{len(PARAMETER_SETS)}] n={n_segments}, m={compactness:g}",
            flush=True,
        )
        args = slic_test.parse_args(
            [
                "--input-dir",
                str(INPUT_DIR),
                "--output-dir",
                str(output_dir),
                "--n-segments",
                str(n_segments),
                "--compactness",
                str(compactness),
            ]
        )
        slic_test.main(args)

        label_path = output_dir / "slic_labels.npy"
        initial_label_path = output_dir / "initial_slic_labels.npy"
        labels = np.load(label_path, allow_pickle=False)
        initial_labels = np.load(initial_label_path, allow_pickle=False)
        actual_count = int(np.unique(labels).size)
        initial_count = int(np.unique(initial_labels).size)
        overlay = first_mean_overlay(first_mean_full, labels)
        overlay_path = output_dir / "segmentation_on_first_mean.png"
        save_rgb(overlay_path, overlay)
        contact_overlays.append((n_segments, compactness, actual_count, overlay))

        config_path = output_dir / "labeling_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["initial_slic_parameters"]["slic_zero"] = False
        config["controlled_coarse_evaluation"] = {
            "compactness_mode": "fixed",
            "visualization_background": "../first_angular_mean_background.png",
            "isolated_output_root": str(COARSE_DIR.resolve()),
        }
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        run_summary = {
            "folder": folder,
            "n_segments_requested": n_segments,
            "compactness_m": compactness,
            "initial_slic_superpixels": initial_count,
            "final_superpixels": actual_count,
            "label_map": str(label_path.resolve()),
            "visualization": str(overlay_path.resolve()),
        }
        (output_dir / "run_summary.json").write_text(
            json.dumps(run_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        manifest["runs"].append(run_summary)
        print(
            f"Completed {folder}: initial={initial_count}, final={actual_count}",
            flush=True,
        )

    save_contact_sheet(contact_overlays)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nCompleted all {len(PARAMETER_SETS)} isolated runs.", flush=True)
    print(f"Manifest: {MANIFEST_PATH}", flush=True)
    print(f"Comparison: {CONTACT_SHEET_PATH}", flush=True)


if __name__ == "__main__":
    main()
