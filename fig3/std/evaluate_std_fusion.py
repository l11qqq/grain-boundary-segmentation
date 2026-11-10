"""Compare stage 2/3 results before and after standard-deviation fusion."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries


PROJECT_DIR = Path(__file__).resolve().parents[2]
FIG3_DIR = PROJECT_DIR / "fig3"
sys.path.insert(0, str(FIG3_DIR))
from evaluate_superpixels import boundary_scores, compactness_score  # noqa: E402


METHODS = [
    ("Stage 2", FIG3_DIR / "result_2"),
    ("Stage 2 + std", FIG3_DIR / "std" / "stage2_with_std"),
    ("Stage 3", FIG3_DIR / "result_3"),
    ("Stage 3 + std", FIG3_DIR / "std" / "stage3_with_std"),
]


def main() -> None:
    reference = cv2.imread(str(FIG3_DIR / "angular_gradient_reference.png"), cv2.IMREAD_GRAYSCALE)
    if reference is None:
        raise FileNotFoundError("Run fig3/evaluate_superpixels.py first")
    strength = reference.astype(np.float32) / 255.0
    reference_boundary = strength >= np.percentile(strength, 90.0)
    rows: list[dict[str, object]] = []
    for name, directory in METHODS:
        labels = np.load(directory / "slic_labels.npy", allow_pickle=False)
        predicted = find_boundaries(labels, mode="thick")
        precision, recall, f1 = boundary_scores(predicted, reference_boundary, tolerance=2)
        rows.append(
            {
                "method": name,
                "superpixels": int(len(np.unique(labels))),
                "boundary_precision": precision,
                "boundary_recall": recall,
                "boundary_f1": f1,
                "area_weighted_compactness": compactness_score(labels),
            }
        )

    output_dir = FIG3_DIR / "std"
    with (output_dir / "std_fusion_metrics.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "std_fusion_metrics.json").write_text(
        json.dumps(
            {
                "reference": "cached 396-D angular-gradient proxy, top 10%, tolerance=2 px",
                "results": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(2)
    width = 0.36
    for axis, base_index, title in ((axes[0], 0, "Stage 2 std-channel ablation"), (axes[1], 2, "Stage 3 std-fusion ablation")):
        pair = rows[base_index : base_index + 2]
        for metric_index, (key, label) in enumerate(
            (("boundary_precision", "Precision"), ("boundary_recall", "Recall"), ("boundary_f1", "F1"))
        ):
            offsets = x + (metric_index - 1) * width / 1.7
            axis.bar(offsets, [float(item[key]) for item in pair], width / 1.8, label=label)
        axis.set_xticks(x, [pair[0]["method"], pair[1]["method"]])
        axis.set_ylim(0.0, 1.0)
        axis.set_title(title)
        axis.set_ylabel("Proxy boundary score")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "std_fusion_metric_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for axis, (name, directory) in zip(axes.flat, METHODS):
        overlay = cv2.imread(str(directory / "segmentation_full_resolution.png"), cv2.IMREAD_COLOR)
        axis.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axis.set_title(name)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "std_fusion_visual_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

