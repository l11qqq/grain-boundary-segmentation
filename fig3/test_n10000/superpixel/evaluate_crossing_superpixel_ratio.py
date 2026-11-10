"""Evaluate the ratio of superpixels that cross reference grain boundaries.

Only pixels covered by exactly one LabelMe ``grain`` polygon participate in
the evaluation.  For each candidate superpixel, the reference grain with the
largest overlap is its dominant grain.  If any evaluated pixels of that
superpixel belong to another grain, the whole superpixel is classified as a
boundary-crossing superpixel.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


OUTPUT_DIR = Path(__file__).resolve().parent
TEST_DIR = OUTPUT_DIR.parent
PROJECT_DIR = TEST_DIR.parents[1]
REFERENCE_JSON = PROJECT_DIR / "outputs" / "slic_3-1" / "angular_boundary_strength.json"
BACKGROUND_PATH = TEST_DIR / "common_std_background.png"

PARAMETER_FOLDERS = (
    ("n2000_m0.28", 2_000, 0.28),
    ("n10000_m0.28", 10_000, 0.28),
    ("n2000_m1", 2_000, 1.0),
    ("n10000_m1", 10_000, 1.0),
)


def rasterize_reference() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    payload = json.loads(REFERENCE_JSON.read_text(encoding="utf-8"))
    height = int(payload["imageHeight"])
    width = int(payload["imageWidth"])
    shapes = [
        shape
        for shape in payload.get("shapes", [])
        if shape.get("label") == "grain" and shape.get("shape_type") == "polygon"
    ]
    if not shapes:
        raise ValueError(f"No grain polygons found in {REFERENCE_JSON}")

    grain_labels = np.zeros((height, width), dtype=np.int32)
    coverage_count = np.zeros((height, width), dtype=np.uint16)
    boundary = np.zeros((height, width), dtype=np.uint8)

    for grain_id, shape in enumerate(shapes, start=1):
        points = np.rint(np.asarray(shape["points"], dtype=np.float64)).astype(np.int32)
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
            raise ValueError(f"Invalid polygon for grain {grain_id}")
        polygon = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(polygon, [points], color=1)
        inside = polygon.astype(bool)
        coverage_count += polygon
        grain_labels[inside] = grain_id
        cv2.polylines(boundary, [points], isClosed=True, color=1, thickness=1)

    evaluation_mask = coverage_count == 1
    ambiguous_mask = coverage_count > 1
    grain_labels[~evaluation_mask] = 0
    qc = {
        "source_json": str(REFERENCE_JSON.resolve()),
        "image_shape": [height, width],
        "grain_polygon_count": len(shapes),
        "observed_reference_grains": int(np.unique(grain_labels[evaluation_mask]).size),
        "evaluated_label_pixels": int(evaluation_mask.sum()),
        "ambiguous_overlap_pixels_excluded": int(ambiguous_mask.sum()),
        "unlabeled_policy": "excluded",
        "ambiguous_polygon_overlap_policy": "excluded",
    }
    return grain_labels, evaluation_mask, boundary.astype(bool), qc


def evaluate_one(
    folder: str,
    n_segments: int,
    compactness: float,
    reference: np.ndarray,
    evaluation_mask: np.ndarray,
) -> tuple[dict[str, object], list[dict[str, object]], np.ndarray, np.ndarray]:
    label_path = TEST_DIR / folder / "method_3_data" / "slic_labels.npy"
    candidate = np.load(label_path)
    if candidate.shape != reference.shape:
        raise ValueError(f"Shape mismatch for {folder}: {candidate.shape} != {reference.shape}")

    candidate_values = candidate[evaluation_mask].astype(np.int64, copy=False)
    grain_values = reference[evaluation_mask].astype(np.int64, copy=False)
    candidate_ids, candidate_inverse = np.unique(candidate_values, return_inverse=True)
    grain_count = int(reference.max())

    pair_codes = candidate_inverse * (grain_count + 1) + grain_values
    overlaps = np.bincount(
        pair_codes,
        minlength=candidate_ids.size * (grain_count + 1),
    ).reshape(candidate_ids.size, grain_count + 1)[:, 1:]

    evaluated_pixels = overlaps.sum(axis=1)
    dominant_index = overlaps.argmax(axis=1)
    dominant_pixels = overlaps[np.arange(candidate_ids.size), dominant_index]
    minority_pixels = evaluated_pixels - dominant_pixels
    grain_counts = np.count_nonzero(overlaps, axis=1)
    crossing = minority_pixels > 0
    ties = np.count_nonzero(overlaps == dominant_pixels[:, None], axis=1) > 1

    evaluated_superpixels = int(candidate_ids.size)
    crossing_superpixels = int(crossing.sum())
    summary = {
        "parameter_group": folder,
        "n_segments_requested": n_segments,
        "compactness_m": compactness,
        "method": "method_3",
        "label_map": str(label_path.resolve()),
        "actual_superpixels_full_image": int(np.unique(candidate).size),
        "evaluated_superpixels": evaluated_superpixels,
        "crossing_superpixels": crossing_superpixels,
        "non_crossing_superpixels": evaluated_superpixels - crossing_superpixels,
        "crossing_superpixel_ratio": float(crossing_superpixels / evaluated_superpixels),
        "crossing_superpixel_ratio_pct": float(100 * crossing_superpixels / evaluated_superpixels),
        "evaluated_label_pixels": int(evaluated_pixels.sum()),
        "dominant_grain_pixels": int(dominant_pixels.sum()),
        "minority_crossing_pixels": int(minority_pixels.sum()),
        "minority_crossing_pixel_ratio": float(minority_pixels.sum() / evaluated_pixels.sum()),
        "dominant_grain_tie_superpixels": int(ties.sum()),
    }

    details: list[dict[str, object]] = []
    for index, superpixel_id in enumerate(candidate_ids.tolist()):
        details.append(
            {
                "parameter_group": folder,
                "n_segments_requested": n_segments,
                "compactness_m": compactness,
                "method": "method_3",
                "superpixel_id": int(superpixel_id),
                "evaluated_label_pixels": int(evaluated_pixels[index]),
                "dominant_reference_grain_id": int(dominant_index[index] + 1),
                "dominant_grain_pixels": int(dominant_pixels[index]),
                "minority_crossing_pixels": int(minority_pixels[index]),
                "minority_crossing_pixel_ratio": float(minority_pixels[index] / evaluated_pixels[index]),
                "reference_grains_overlapped": int(grain_counts[index]),
                "crosses_grain_boundary": bool(crossing[index]),
                "dominant_grain_tie": bool(ties[index]),
            }
        )

    return summary, details, candidate, candidate_ids[crossing]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_overlay(
    folder: str,
    candidate: np.ndarray,
    crossing_ids: np.ndarray,
    evaluation_mask: np.ndarray,
    reference_boundary: np.ndarray,
    ratio_pct: float,
) -> None:
    background_bgr = cv2.imread(str(BACKGROUND_PATH), cv2.IMREAD_COLOR)
    if background_bgr is None:
        background = np.zeros((*candidate.shape, 3), dtype=np.uint8)
    else:
        background = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2RGB)
    if background.shape[:2] != candidate.shape:
        raise ValueError(f"Background shape mismatch: {background.shape[:2]} != {candidate.shape}")

    crossing_mask = np.isin(candidate, crossing_ids) & evaluation_mask
    non_crossing_mask = evaluation_mask & ~crossing_mask
    overlay = background.astype(np.float32)
    overlay[non_crossing_mask] = 0.65 * overlay[non_crossing_mask] + 0.35 * np.array([34, 197, 94])
    overlay[crossing_mask] = 0.45 * overlay[crossing_mask] + 0.55 * np.array([239, 68, 68])
    overlay[reference_boundary] = np.array([250, 204, 21])
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(12, 9), dpi=160)
    ax.imshow(overlay)
    ax.set_title(f"{folder}: crossing-superpixel ratio = {ratio_pct:.4f}%")
    ax.axis("off")
    ax.legend(
        handles=[
            Patch(color="#ef4444", label="Crossing superpixel (labeled pixels)"),
            Patch(color="#22c55e", label="Non-crossing superpixel (labeled pixels)"),
            Patch(color="#facc15", label="Reference grain boundary"),
        ],
        loc="lower center",
        ncol=3,
        framealpha=0.9,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{folder}_crossing_superpixels_overlay.png", bbox_inches="tight")
    plt.close(fig)


def save_comparison_chart(summary_rows: list[dict[str, object]]) -> None:
    ordered = sorted(summary_rows, key=lambda row: float(row["crossing_superpixel_ratio_pct"]))
    labels = [str(row["parameter_group"]) for row in ordered]
    values = [float(row["crossing_superpixel_ratio_pct"]) for row in ordered]
    colors = ["#2563eb", "#0891b2", "#7c3aed", "#9333ea"]

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=180)
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Crossing-superpixel ratio (%)")
    ax.set_xlabel("Parameter group (method_3)")
    ax.set_title("Superpixels crossing labeled grain boundaries (lower is better)")
    ax.set_ylim(0, max(values) * 1.22)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.025,
            f"{value:.4f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "crossing_superpixel_ratio_comparison.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reference, evaluation_mask, reference_boundary, reference_qc = rasterize_reference()
    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []

    for folder, n_segments, compactness in PARAMETER_FOLDERS:
        summary, details, candidate, crossing_ids = evaluate_one(
            folder,
            n_segments,
            compactness,
            reference,
            evaluation_mask,
        )
        summary_rows.append(summary)
        detail_rows.extend(details)
        save_overlay(
            folder,
            candidate,
            crossing_ids,
            evaluation_mask,
            reference_boundary,
            float(summary["crossing_superpixel_ratio_pct"]),
        )

    ranked = sorted(summary_rows, key=lambda row: float(row["crossing_superpixel_ratio"]))
    for rank, row in enumerate(ranked, start=1):
        row["rank_lower_is_better"] = rank

    write_csv(OUTPUT_DIR / "crossing_superpixel_ratio_summary.csv", ranked)
    write_csv(OUTPUT_DIR / "crossing_superpixel_details.csv", detail_rows)
    results = {
        "metric_definition": {
            "name": "crossing-superpixel ratio",
            "formula": "number of evaluated superpixels containing labeled pixels from more than one reference grain / number of superpixels containing at least one unambiguous labeled pixel",
            "dominant_grain_rule": "the reference grain contributing the most evaluated pixels is the superpixel's dominant grain; all other evaluated pixels are minority crossing pixels",
            "crossing_rule": "minority crossing pixels > 0",
            "interpretation": "lower is better",
            "boundary_overlap_credit": "not applied",
        },
        "reference_qc": reference_qc,
        "summary": ranked,
        "detail_row_count": len(detail_rows),
    }
    (OUTPUT_DIR / "crossing_superpixel_ratio_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_comparison_chart(summary_rows)

    print("parameter_group,evaluated_superpixels,crossing_superpixels,ratio_pct")
    for row in ranked:
        print(
            f"{row['parameter_group']},{row['evaluated_superpixels']},"
            f"{row['crossing_superpixels']},{row['crossing_superpixel_ratio_pct']:.6f}"
        )


if __name__ == "__main__":
    main()
