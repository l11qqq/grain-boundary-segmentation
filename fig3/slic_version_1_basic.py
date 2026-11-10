"""Figure 3, stage 1: basic SLIC on a single standard-deviation image.

Only two segmentation controls are exposed: the requested number of
superpixels and the SLIC compactness (spatial/gray-distance weight).  Other
behavior uses the library defaults.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from skimage.segmentation import slic

from fig3_common import compute_std_map, find_image_paths, minmax_normalize, save_outputs


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "crop"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "result_1"


def main(input_dir: Path, output_dir: Path, n_segments: int, compactness: float) -> None:
    paths = find_image_paths(input_dir)
    std_raw = compute_std_map(paths)
    std_input = minmax_normalize(std_raw)
    labels = slic(
        std_input,
        n_segments=n_segments,
        compactness=compactness,
        channel_axis=None,
        start_label=0,
    ).astype(np.int32)
    save_outputs(
        output_dir,
        labels,
        std_input,
        "Stage 1: basic single-channel SLIC",
        {"n_segments": n_segments, "compactness": compactness},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-segments", type=int, default=200)
    parser.add_argument("--compactness", type=float, default=0.28)
    args = parser.parse_args()
    main(args.input_dir.resolve(), args.output_dir.resolve(), args.n_segments, args.compactness)
