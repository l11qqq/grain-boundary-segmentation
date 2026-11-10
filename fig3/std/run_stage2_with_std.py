"""Stage-2 ablation: 12 angular channels plus a weighted std channel."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.segmentation import slic


PROJECT_DIR = Path(__file__).resolve().parents[2]
FIG3_DIR = PROJECT_DIR / "fig3"
sys.path.insert(0, str(FIG3_DIR))

from fig3_common import DEFAULT_INPUT_DIR, boundary_overlay, find_image_paths, percentile_normalize, save_outputs  # noqa: E402
from slic_version_2_multichannel_slico import angular_gradient, build_legacy_features  # noqa: E402


OUTPUT_DIR = FIG3_DIR / "std" / "stage2_with_std"
STD_FEATURE_WEIGHT = 0.30
N_SEGMENTS = 10000
COMPACTNESS = 0.28


def main() -> None:
    paths = find_image_paths(DEFAULT_INPUT_DIR)
    angular_features, std_raw, metadata = build_legacy_features(paths)
    std_feature = percentile_normalize(std_raw, 2.0, 98.0, gamma=0.65)
    std_feature = cv2.bilateralFilter(
        std_feature,
        d=5,
        sigmaColor=0.08,
        sigmaSpace=2.0,
        borderType=cv2.BORDER_REFLECT101,
    )
    fused_features = np.concatenate(
        (angular_features, (STD_FEATURE_WEIGHT * std_feature)[..., None]),
        axis=2,
    ).astype(np.float32, copy=False)

    print(f"Running SLICO on {fused_features.shape[2]} channels (12 angular + 1 std)...", flush=True)
    labels = slic(
        fused_features,
        n_segments=N_SEGMENTS,
        compactness=COMPACTNESS,
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

    angular_strength = angular_gradient(angular_features)
    save_outputs(
        OUTPUT_DIR,
        labels,
        std_feature,
        "Stage 2 + std: 12 angular channels and weighted std",
        {
            "n_segments": N_SEGMENTS,
            "compactness": COMPACTNESS,
            "angular_feature_channels": 12,
            "std_feature_channels": 1,
            "std_feature_weight_alpha": STD_FEATURE_WEIGHT,
            "std_preprocessing": "2-98 percentile, gamma=0.65, bilateral d=5",
            **metadata,
        },
        [("Angular-gradient overlay", boundary_overlay(angular_strength, labels), None)],
    )
    print(json.dumps({"actual_superpixels": int(len(np.unique(labels)))}, indent=2))


if __name__ == "__main__":
    main()

