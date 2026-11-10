"""Figure 3, stage 3: run the current formal pipeline into fig3/result_3."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
import slic_test  # noqa: E402


if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    input_dir = SCRIPT_DIR / "crop"
    output_dir = SCRIPT_DIR / "result_3"
    forwarded = sys.argv[1:]
    args = slic_test.parse_args(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--n-segments",
            "200",
            "--compactness",
            "0.28",
            "--barrier-percentile", "84",
            "--std-feature-weight", "0.1",
            "--slic-zero",
            *forwarded,
        ]
    )
    slic_test.main(args)
