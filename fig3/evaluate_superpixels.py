"""Evaluate the three Figure 3 superpixel stages and draw comparison curves.

Without pixel-level manual ground truth, the script builds an independent
reference from the local RMSE gradient of all 396 normalized angle images.
Metrics obtained this way are explicitly reported as proxy metrics.  Supplying
``--ground-truth grain_labels.npy`` switches boundary metrics to manual truth
and additionally reports achievable segmentation accuracy (ASA) and
undersegmentation error (UE = 1 - ASA).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries

from fig3_common import DEFAULT_INPUT_DIR, PROJECT_DIR, find_image_paths, percentile_normalize


RESULT_DIRS = [PROJECT_DIR / "fig3" / f"result_{index}" for index in (1, 2, 3)]
STAGE_NAMES = ["Basic SLIC", "Angular SLICO", "Twin-aware watershed"]


def full_stack_angular_gradient(image_paths: list[Path]) -> np.ndarray:
    first = cv2.imread(str(image_paths[0]), cv2.IMREAD_GRAYSCALE)
    if first is None:
        raise ValueError(f"Failed to read {image_paths[0]}")
    sum_squared_gradient = np.zeros(first.shape, dtype=np.float64)
    for index, path in enumerate(image_paths, start=1):
        image = first if index == 1 else cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != first.shape:
            raise ValueError(f"Invalid image or shape mismatch: {path}")
        feature = image.astype(np.float32) / (float(image.mean()) + 1e-6)
        gx = cv2.Sobel(feature, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(feature, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        sum_squared_gradient += gx * gx + gy * gy
        if index == 1 or index % 50 == 0 or index == len(image_paths):
            print(f"Building independent 396-D gradient reference: {index}/{len(image_paths)}", flush=True)
    gradient = np.sqrt(sum_squared_gradient / float(len(image_paths))).astype(np.float32)
    return percentile_normalize(gradient, 2.0, 99.5)


def boundary_scores(prediction: np.ndarray, reference: np.ndarray, tolerance: int) -> tuple[float, float, float]:
    pred_dilated = ndi.binary_dilation(prediction, iterations=tolerance)
    ref_dilated = ndi.binary_dilation(reference, iterations=tolerance)
    precision = float(np.count_nonzero(prediction & ref_dilated) / max(np.count_nonzero(prediction), 1))
    recall = float(np.count_nonzero(reference & pred_dilated) / max(np.count_nonzero(reference), 1))
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1


def compactness_score(labels: np.ndarray) -> float:
    labels = labels.astype(np.int64, copy=False)
    region_count = int(labels.max()) + 1
    areas = np.bincount(labels.ravel(), minlength=region_count).astype(np.float64)
    perimeters = np.zeros(region_count, dtype=np.float64)
    vertical = labels[:-1, :] != labels[1:, :]
    horizontal = labels[:, :-1] != labels[:, 1:]
    for values in (labels[:-1, :][vertical], labels[1:, :][vertical]):
        perimeters += np.bincount(values, minlength=region_count)
    for values in (labels[:, :-1][horizontal], labels[:, 1:][horizontal]):
        perimeters += np.bincount(values, minlength=region_count)
    perimeters += np.bincount(labels[0, :], minlength=region_count)
    perimeters += np.bincount(labels[-1, :], minlength=region_count)
    perimeters += np.bincount(labels[:, 0], minlength=region_count)
    perimeters += np.bincount(labels[:, -1], minlength=region_count)
    valid = (areas > 0) & (perimeters > 0)
    circularity = np.clip(4.0 * np.pi * areas[valid] / np.square(perimeters[valid]), 0.0, 1.0)
    return float(np.average(circularity, weights=areas[valid]))


def asa_and_ue(superpixels: np.ndarray, ground_truth: np.ndarray) -> tuple[float, float]:
    sp = superpixels.astype(np.int64, copy=False).ravel()
    gt_values, gt = np.unique(ground_truth, return_inverse=True)
    del gt_values
    pair_code = sp * (int(gt.max()) + 1) + gt
    pairs, overlaps = np.unique(pair_code, return_counts=True)
    sp_ids = pairs // (int(gt.max()) + 1)
    maximum_overlap = np.zeros(int(sp.max()) + 1, dtype=np.int64)
    np.maximum.at(maximum_overlap, sp_ids, overlaps)
    asa = float(maximum_overlap.sum() / sp.size)
    return asa, 1.0 - asa


def load_ground_truth(path: Path, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray | None]:
    if path.suffix.lower() == ".npy":
        values = np.load(path, allow_pickle=False)
    else:
        values = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if values is None:
            raise ValueError(f"Failed to read ground truth: {path}")
        if values.ndim == 3:
            values = cv2.cvtColor(values, cv2.COLOR_BGR2GRAY)
    if values.shape != shape:
        raise ValueError(f"Ground truth shape {values.shape} does not match {shape}")
    unique = np.unique(values)
    if values.dtype == bool or (len(unique) <= 2 and np.min(unique) == 0):
        return values > 0, None
    labels = values.astype(np.int64)
    return find_boundaries(labels, mode="thick"), labels


def main(args: argparse.Namespace) -> None:
    labels_list = [np.load(path / "slic_labels.npy", allow_pickle=False) for path in RESULT_DIRS]
    shape = labels_list[0].shape
    if any(labels.shape != shape for labels in labels_list):
        raise ValueError("All Figure 3 label maps must have the same shape")

    if args.ground_truth is None:
        reference_path = PROJECT_DIR / "fig3" / "angular_gradient_reference.png"
        if reference_path.is_file() and not args.rebuild_proxy:
            reference_u8 = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
            if reference_u8 is None or reference_u8.shape != shape:
                raise ValueError(f"Invalid cached angular-gradient reference: {reference_path}")
            reference_strength = reference_u8.astype(np.float32) / 255.0
            print(f"Using cached angular-gradient reference: {reference_path}")
        else:
            reference_strength = full_stack_angular_gradient(find_image_paths(args.input_dir))
            cv2.imwrite(
                str(reference_path),
                np.round(reference_strength * 255.0).astype(np.uint8),
            )
        threshold = float(np.percentile(reference_strength, args.proxy_percentile))
        reference_boundary = reference_strength >= threshold
        reference_labels = None
        reference_type = f"396-D angular-gradient proxy (top {100.0 - args.proxy_percentile:g}%)"
    else:
        reference_boundary, reference_labels = load_ground_truth(args.ground_truth, shape)
        reference_type = f"manual ground truth: {args.ground_truth}"

    rows: list[dict[str, object]] = []
    for stage, labels in zip(STAGE_NAMES, labels_list):
        predicted_boundary = find_boundaries(labels, mode="thick")
        precision, recall, f1 = boundary_scores(predicted_boundary, reference_boundary, args.tolerance)
        row: dict[str, object] = {
            "stage": stage,
            "superpixels": int(len(np.unique(labels))),
            "boundary_precision": precision,
            "boundary_recall": recall,
            "boundary_f1": f1,
            "area_weighted_compactness": compactness_score(labels),
        }
        if reference_labels is not None:
            row["asa"], row["undersegmentation_error"] = asa_and_ue(labels, reference_labels)
        rows.append(row)

    output_csv = PROJECT_DIR / "fig3" / "metrics.csv"
    with output_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (PROJECT_DIR / "fig3" / "metrics.json").write_text(
        json.dumps(
            {
                "reference": reference_type,
                "boundary_tolerance_pixels": args.tolerance,
                "important_note": (
                    "Proxy boundary scores are not substitutes for manual ground-truth metrics."
                    if args.ground_truth is None
                    else "Boundary and region metrics use manual ground truth."
                ),
                "results": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    x = np.arange(1, 4)
    fig, axis = plt.subplots(figsize=(9.5, 6.2))
    for key, label, marker in (
        ("boundary_precision", "Boundary precision", "o"),
        ("boundary_recall", "Boundary recall", "s"),
        ("boundary_f1", "Boundary F1", "^"),
        ("area_weighted_compactness", "Compactness", "D"),
    ):
        axis.plot(x, [float(row[key]) for row in rows], marker=marker, linewidth=2.0, label=label)
    if reference_labels is not None:
        axis.plot(x, [float(row["asa"]) for row in rows], marker="P", linewidth=2.0, label="ASA")
    axis.set_xticks(x, ["Stage 1\nBasic SLIC", "Stage 2\nAngular SLICO", "Stage 3\nTwin-aware watershed"])
    axis.set_ylim(0.0, 1.02)
    axis.set_ylabel("Score")
    axis.set_title("Superpixel optimization across three stages")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(PROJECT_DIR / "fig3" / "metrics_line_plot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # A compact publication-oriented visual comparison using identical crops,
    # background rendering, and boundary color for all stages.
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for axis, stage, result_dir in zip(axes, STAGE_NAMES, RESULT_DIRS):
        overlay = cv2.imread(str(result_dir / "segmentation_full_resolution.png"), cv2.IMREAD_COLOR)
        if overlay is None:
            raise ValueError(f"Missing visualization in {result_dir}")
        axis.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axis.set_title(stage)
        axis.axis("off")
    fig.tight_layout(pad=0.5)
    fig.savefig(PROJECT_DIR / "fig3" / "three_stage_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--ground-truth", type=Path, default=None)
    parser.add_argument("--tolerance", type=int, default=2)
    parser.add_argument("--proxy-percentile", type=float, default=90.0)
    parser.add_argument("--rebuild-proxy", action="store_true")
    arguments = parser.parse_args()
    arguments.input_dir = arguments.input_dir.resolve()
    if arguments.ground_truth is not None:
        arguments.ground_truth = arguments.ground_truth.resolve()
    main(arguments)
