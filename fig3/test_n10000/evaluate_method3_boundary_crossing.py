"""Evaluate method-3 superpixels against manually labeled grain polygons.

For each candidate superpixel, only pixels covered by exactly one LabelMe
``grain`` polygon are evaluated.  The reference grain with the largest pixel
overlap is treated as the superpixel's dominant grain; all remaining labeled
pixels are counted as boundary-crossing pixels.  Unlabeled and ambiguously
overlapped reference pixels are excluded.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


TEST_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TEST_DIR.parents[1]
REFERENCE_DIR = PROJECT_DIR / "outputs" / "slic_3-1"
REFERENCE_JSON = REFERENCE_DIR / "angular_boundary_strength.json"
BACKGROUND_PATH = TEST_DIR / "common_std_background.png"
EVALUATION_DIR = TEST_DIR / "method_3_boundary_evaluation"
RESULTS_JSON = EVALUATION_DIR / "evaluation_results.json"

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
    shapes = payload.get("shapes", [])
    if not shapes:
        raise ValueError(f"No polygon labels found in {REFERENCE_JSON}")

    reference_grains = np.zeros((height, width), dtype=np.int32)
    coverage_count = np.zeros((height, width), dtype=np.uint16)
    boundary_mask = np.zeros((height, width), dtype=np.uint8)
    polygon_areas: list[int] = []

    for grain_id, shape in enumerate(shapes, start=1):
        if shape.get("label") != "grain" or shape.get("shape_type") != "polygon":
            raise ValueError(
                f"Unsupported reference shape {grain_id}: "
                f"label={shape.get('label')!r}, type={shape.get('shape_type')!r}"
            )
        points = np.rint(np.asarray(shape["points"], dtype=np.float64)).astype(np.int32)
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
            raise ValueError(f"Invalid polygon {grain_id} in {REFERENCE_JSON}")
        if (
            np.any(points[:, 0] < 0)
            or np.any(points[:, 0] >= width)
            or np.any(points[:, 1] < 0)
            or np.any(points[:, 1] >= height)
        ):
            raise ValueError(f"Polygon {grain_id} has coordinates outside the reference image")

        polygon_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(polygon_mask, [points], color=1)
        polygon_pixels = polygon_mask.astype(bool)
        polygon_areas.append(int(polygon_pixels.sum()))
        coverage_count += polygon_mask
        reference_grains[polygon_pixels] = grain_id
        cv2.polylines(boundary_mask, [points], isClosed=True, color=255, thickness=1)

    ambiguous_mask = coverage_count > 1
    evaluation_mask = coverage_count == 1
    reference_grains[~evaluation_mask] = 0

    observed_grains = np.unique(reference_grains[evaluation_mask])
    reference_qc = {
        "source_json": str(REFERENCE_JSON.resolve()),
        "source_image": str((REFERENCE_DIR / payload["imagePath"]).resolve()),
        "image_shape": [height, width],
        "polygon_count": len(shapes),
        "observed_reference_grains": int(observed_grains.size),
        "sum_polygon_pixels_before_overlap_exclusion": int(sum(polygon_areas)),
        "union_labeled_pixels": int(np.count_nonzero(coverage_count)),
        "ambiguous_overlap_pixels_excluded": int(ambiguous_mask.sum()),
        "evaluated_pixels": int(evaluation_mask.sum()),
        "total_image_pixels": int(height * width),
        "labeled_coverage_ratio": float(evaluation_mask.mean()),
        "polygon_area_min_pixels": int(min(polygon_areas)),
        "polygon_area_median_pixels": float(np.median(polygon_areas)),
        "polygon_area_max_pixels": int(max(polygon_areas)),
        "coordinate_rasterization": "round-to-nearest integer pixel; cv2.fillPoly",
    }
    return reference_grains, evaluation_mask, boundary_mask, reference_qc


def compute_boundaries(labels: np.ndarray) -> np.ndarray:
    """Return a 4-connected, two-sided candidate-superpixel boundary mask."""
    boundaries = np.zeros(labels.shape, dtype=bool)
    horizontal = labels[:, 1:] != labels[:, :-1]
    boundaries[:, 1:] |= horizontal
    boundaries[:, :-1] |= horizontal
    vertical = labels[1:, :] != labels[:-1, :]
    boundaries[1:, :] |= vertical
    boundaries[:-1, :] |= vertical
    return boundaries


def compute_region_metrics(
    candidate: np.ndarray,
    candidate_boundary: np.ndarray,
    reference_grains: np.ndarray,
    region_mask: np.ndarray,
    reference_boundary: np.ndarray,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Compute original and exact-boundary-overlap-adjusted metrics in a mask."""
    valid_grain_values = reference_grains[region_mask].astype(np.int64, copy=False)
    if valid_grain_values.size == 0:
        raise ValueError("Evaluation region contains no labeled pixels")
    valid_candidate_values = candidate[region_mask].astype(np.int64, copy=False)
    candidate_ids, candidate_inverse = np.unique(valid_candidate_values, return_inverse=True)
    grain_count = int(reference_grains.max())
    pair_codes = candidate_inverse * (grain_count + 1) + valid_grain_values
    overlap_counts = np.bincount(
        pair_codes,
        minlength=candidate_ids.size * (grain_count + 1),
    ).reshape(candidate_ids.size, grain_count + 1)
    overlap_counts = overlap_counts[:, 1:]

    evaluated_pixels_per_superpixel = overlap_counts.sum(axis=1)
    dominant_grain_index = overlap_counts.argmax(axis=1)
    dominant_pixels = overlap_counts[
        np.arange(candidate_ids.size), dominant_grain_index
    ]
    crossing_pixels = evaluated_pixels_per_superpixel - dominant_pixels
    overlapped_grain_counts = np.count_nonzero(overlap_counts, axis=1)
    crossing_rates = crossing_pixels / evaluated_pixels_per_superpixel

    dominant_grain_per_pixel = dominant_grain_index[candidate_inverse] + 1
    crossing_valid = valid_grain_values != dominant_grain_per_pixel
    boundary_overlap_valid = (
        candidate_boundary[region_mask]
        & reference_boundary[region_mask].astype(bool)
    )
    credited_crossing_valid = crossing_valid & boundary_overlap_valid
    boundary_overlap_per_superpixel = np.bincount(
        candidate_inverse[boundary_overlap_valid], minlength=candidate_ids.size
    )
    credited_crossing_per_superpixel = np.bincount(
        candidate_inverse[credited_crossing_valid], minlength=candidate_ids.size
    )
    adjusted_crossing_pixels = crossing_pixels - credited_crossing_per_superpixel
    adjusted_crossing_rates = adjusted_crossing_pixels / evaluated_pixels_per_superpixel

    evaluated_pixels = int(evaluated_pixels_per_superpixel.sum())
    dominant_pixel_sum = int(dominant_pixels.sum())
    crossing_pixel_sum = int(crossing_pixels.sum())
    credited_crossing_pixel_sum = int(credited_crossing_per_superpixel.sum())
    adjusted_crossing_pixel_sum = int(adjusted_crossing_pixels.sum())
    if evaluated_pixels != int(region_mask.sum()):
        raise AssertionError("Evaluated-pixel mismatch")
    if dominant_pixel_sum + crossing_pixel_sum != evaluated_pixels:
        raise AssertionError("Original pixel reconciliation failed")
    if adjusted_crossing_pixel_sum + dominant_pixel_sum + credited_crossing_pixel_sum != evaluated_pixels:
        raise AssertionError("Boundary-credit pixel reconciliation failed")

    crossing_superpixel_mask = crossing_pixels > 0
    adjusted_crossing_superpixel_mask = adjusted_crossing_pixels > 0
    reference_boundary_pixels = int(
        np.count_nonzero(reference_boundary[region_mask])
    )
    candidate_boundary_pixels = int(
        np.count_nonzero(candidate_boundary[region_mask])
    )
    boundary_overlap_pixels = int(boundary_overlap_valid.sum())
    metrics: dict[str, object] = {
        "evaluated_superpixels": int(candidate_ids.size),
        "crossing_superpixels": int(crossing_superpixel_mask.sum()),
        "crossing_superpixel_ratio": float(crossing_superpixel_mask.mean()),
        "adjusted_crossing_superpixels": int(adjusted_crossing_superpixel_mask.sum()),
        "adjusted_crossing_superpixel_ratio": float(adjusted_crossing_superpixel_mask.mean()),
        "evaluated_pixels": evaluated_pixels,
        "dominant_grain_pixels": dominant_pixel_sum,
        "boundary_crossing_pixels": crossing_pixel_sum,
        "boundary_crossing_pixel_ratio": float(crossing_pixel_sum / evaluated_pixels),
        "reference_boundary_pixels": reference_boundary_pixels,
        "candidate_boundary_pixels": candidate_boundary_pixels,
        "overlapping_boundary_pixels": boundary_overlap_pixels,
        "boundary_overlap_recall": float(
            boundary_overlap_pixels / reference_boundary_pixels
            if reference_boundary_pixels
            else 0.0
        ),
        "boundary_overlap_precision": float(
            boundary_overlap_pixels / candidate_boundary_pixels
            if candidate_boundary_pixels
            else 0.0
        ),
        "credited_crossing_pixels": credited_crossing_pixel_sum,
        "adjusted_boundary_crossing_pixels": adjusted_crossing_pixel_sum,
        "adjusted_boundary_crossing_pixel_ratio": float(
            adjusted_crossing_pixel_sum / evaluated_pixels
        ),
        "adjusted_correct_pixel_ratio": float(
            (dominant_pixel_sum + credited_crossing_pixel_sum) / evaluated_pixels
        ),
        "dominant_pixel_ratio": float(dominant_pixel_sum / evaluated_pixels),
        "mean_per_superpixel_crossing_ratio": float(crossing_rates.mean()),
        "mean_per_superpixel_adjusted_crossing_ratio": float(
            adjusted_crossing_rates.mean()
        ),
        "median_per_superpixel_crossing_ratio": float(np.median(crossing_rates)),
        "median_per_superpixel_adjusted_crossing_ratio": float(
            np.median(adjusted_crossing_rates)
        ),
        "max_reference_grains_overlapped": int(overlapped_grain_counts.max()),
    }
    arrays = {
        "candidate_ids": candidate_ids,
        "evaluated_pixels_per_superpixel": evaluated_pixels_per_superpixel,
        "dominant_grain_index": dominant_grain_index,
        "dominant_pixels": dominant_pixels,
        "crossing_pixels": crossing_pixels,
        "crossing_rates": crossing_rates,
        "overlapped_grain_counts": overlapped_grain_counts,
        "crossing_superpixel_mask": crossing_superpixel_mask,
        "boundary_overlap_per_superpixel": boundary_overlap_per_superpixel,
        "credited_crossing_per_superpixel": credited_crossing_per_superpixel,
        "adjusted_crossing_pixels": adjusted_crossing_pixels,
        "adjusted_crossing_rates": adjusted_crossing_rates,
        "adjusted_crossing_superpixel_mask": adjusted_crossing_superpixel_mask,
    }
    return metrics, arrays


def evaluate_candidate(
    folder_name: str,
    n_segments: int,
    compactness: float,
    candidate: np.ndarray,
    label_path: Path,
    reference_grains: np.ndarray,
    evaluation_mask: np.ndarray,
    reference_boundary: np.ndarray,
) -> tuple[dict[str, object], list[dict[str, object]], np.ndarray]:
    if candidate.shape != reference_grains.shape:
        raise ValueError(
            f"Shape mismatch for {folder_name}: {candidate.shape} vs {reference_grains.shape}"
        )
    candidate_boundary = compute_boundaries(candidate)
    metrics, arrays = compute_region_metrics(
        candidate,
        candidate_boundary,
        reference_grains,
        evaluation_mask,
        reference_boundary,
    )
    summary = {
        "folder": folder_name,
        "n_segments": n_segments,
        "compactness": compactness,
        "method": "method_3",
        "label_map": str(label_path.resolve()),
        "actual_superpixels_full_image": int(np.unique(candidate).size),
        **metrics,
    }

    details: list[dict[str, object]] = []
    for index, superpixel_id in enumerate(arrays["candidate_ids"].tolist()):
        details.append(
            {
                "folder": folder_name,
                "n_segments": n_segments,
                "compactness": compactness,
                "method": "method_3",
                "superpixel_id": int(superpixel_id),
                "evaluated_pixels": int(arrays["evaluated_pixels_per_superpixel"][index]),
                "dominant_reference_grain_id": int(arrays["dominant_grain_index"][index] + 1),
                "dominant_grain_pixels": int(arrays["dominant_pixels"][index]),
                "boundary_crossing_pixels": int(arrays["crossing_pixels"][index]),
                "boundary_crossing_pixel_ratio": float(arrays["crossing_rates"][index]),
                "overlapping_boundary_pixels": int(arrays["boundary_overlap_per_superpixel"][index]),
                "credited_crossing_pixels": int(arrays["credited_crossing_per_superpixel"][index]),
                "adjusted_boundary_crossing_pixels": int(arrays["adjusted_crossing_pixels"][index]),
                "adjusted_boundary_crossing_pixel_ratio": float(arrays["adjusted_crossing_rates"][index]),
                "reference_grains_overlapped": int(arrays["overlapped_grain_counts"][index]),
                "crosses_boundary": bool(arrays["crossing_superpixel_mask"][index]),
                "crosses_boundary_after_credit": bool(
                    arrays["adjusted_crossing_superpixel_mask"][index]
                ),
            }
        )
    return summary, details, candidate_boundary


def build_local_region_analysis(
    candidates: dict[str, np.ndarray],
    candidate_boundaries: dict[str, np.ndarray],
    reference_grains: np.ndarray,
    evaluation_mask: np.ndarray,
    reference_boundary: np.ndarray,
) -> dict[str, object]:
    """Evaluate every cell in fixed 2x2 and 3x3 grids over the labeled bbox."""
    y_values, x_values = np.nonzero(evaluation_mask)
    x_min, x_max_exclusive = int(x_values.min()), int(x_values.max()) + 1
    y_min, y_max_exclusive = int(y_values.min()), int(y_values.max()) + 1
    rows: list[dict[str, object]] = []

    for grid_size in (2, 3):
        x_edges = np.rint(
            np.linspace(x_min, x_max_exclusive, grid_size + 1)
        ).astype(int)
        y_edges = np.rint(
            np.linspace(y_min, y_max_exclusive, grid_size + 1)
        ).astype(int)
        for grid_row in range(grid_size):
            for grid_col in range(grid_size):
                x0, x1 = int(x_edges[grid_col]), int(x_edges[grid_col + 1])
                y0, y1 = int(y_edges[grid_row]), int(y_edges[grid_row + 1])
                region_mask = np.zeros(evaluation_mask.shape, dtype=bool)
                region_mask[y0:y1, x0:x1] = evaluation_mask[y0:y1, x0:x1]
                evaluated_pixels = int(region_mask.sum())
                if evaluated_pixels == 0:
                    continue

                result_by_folder: dict[str, dict[str, object]] = {}
                for folder_name, _n_segments, _compactness in PARAMETER_FOLDERS:
                    metrics, _arrays = compute_region_metrics(
                        candidates[folder_name],
                        candidate_boundaries[folder_name],
                        reference_grains,
                        region_mask,
                        reference_boundary,
                    )
                    result_by_folder[folder_name] = metrics

                ranking = sorted(
                    result_by_folder,
                    key=lambda folder: (
                        result_by_folder[folder]["adjusted_boundary_crossing_pixel_ratio"],
                        folder,
                    ),
                )
                winner = ranking[0]
                winner_ratio = float(
                    result_by_folder[winner]["adjusted_boundary_crossing_pixel_ratio"]
                )
                second_ratio = float(
                    result_by_folder[ranking[1]]["adjusted_boundary_crossing_pixel_ratio"]
                )
                rows.append(
                    {
                        "grid": f"{grid_size}x{grid_size}",
                        "region_id": f"{grid_size}x{grid_size}_r{grid_row + 1}_c{grid_col + 1}",
                        "grid_row": grid_row + 1,
                        "grid_col": grid_col + 1,
                        "x_min_inclusive": x0,
                        "x_max_exclusive": x1,
                        "y_min_inclusive": y0,
                        "y_max_exclusive": y1,
                        "evaluated_pixels": evaluated_pixels,
                        "reference_grains_present": int(
                            np.unique(reference_grains[region_mask]).size
                        ),
                        "winner": winner,
                        "winner_adjusted_ratio": winner_ratio,
                        "winner_margin_to_second": second_ratio - winner_ratio,
                        "n10000_m1_is_best": winner == "n10000_m1",
                        "n10000_m1_advantage_vs_n10000_m0.28": float(
                            result_by_folder["n10000_m0.28"][
                                "adjusted_boundary_crossing_pixel_ratio"
                            ]
                            - result_by_folder["n10000_m1"][
                                "adjusted_boundary_crossing_pixel_ratio"
                            ]
                        ),
                        "results": result_by_folder,
                    }
                )

    winning_rows = [row for row in rows if row["n10000_m1_is_best"]]
    best_row = (
        max(
            winning_rows,
            key=lambda row: (
                row["winner_margin_to_second"],
                row["evaluated_pixels"],
            ),
        )
        if winning_rows
        else None
    )
    return {
        "selection_protocol": (
            "Exploratory sensitivity analysis over every cell of fixed 2x2 and "
            "3x3 grids covering the labeled-polygon bounding box; no cell omitted."
        ),
        "labeled_bbox": {
            "x_min_inclusive": x_min,
            "x_max_exclusive": x_max_exclusive,
            "y_min_inclusive": y_min,
            "y_max_exclusive": y_max_exclusive,
        },
        "region_count": len(rows),
        "n10000_m1_winning_region_count": len(winning_rows),
        "best_exploratory_region_for_n10000_m1": best_row,
        "regions": rows,
    }


def save_reference_outputs(
    reference_grains: np.ndarray,
    evaluation_mask: np.ndarray,
    boundary_mask: np.ndarray,
) -> None:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EVALUATION_DIR / "reference_grain_labels.npy", reference_grains, allow_pickle=False)
    cv2.imwrite(
        str(EVALUATION_DIR / "reference_evaluation_mask.png"),
        evaluation_mask.astype(np.uint8) * 255,
    )
    cv2.imwrite(str(EVALUATION_DIR / "reference_grain_boundary.png"), boundary_mask)

    background = cv2.imread(str(BACKGROUND_PATH), cv2.IMREAD_GRAYSCALE)
    if background is None or background.shape != reference_grains.shape:
        background = cv2.imread(
            str(REFERENCE_DIR / "angular_boundary_strength.png"), cv2.IMREAD_GRAYSCALE
        )
    if background is None or background.shape != reference_grains.shape:
        raise ValueError("Could not load a shape-compatible reference background")
    overlay = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    overlay[boundary_mask > 0] = (0, 255, 0)
    overlay[~evaluation_mask] = (overlay[~evaluation_mask] * 0.30).astype(np.uint8)
    cv2.imwrite(str(EVALUATION_DIR / "reference_grain_overlay.png"), overlay)


def save_exploratory_region_overlay(local_analysis: dict[str, object]) -> None:
    best = local_analysis.get("best_exploratory_region_for_n10000_m1")
    if not isinstance(best, dict):
        return
    source_path = EVALUATION_DIR / "reference_grain_overlay.png"
    overlay = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if overlay is None:
        raise ValueError(f"Cannot read local-region overlay background: {source_path}")
    x0 = int(best["x_min_inclusive"])
    x1 = int(best["x_max_exclusive"])
    y0 = int(best["y_min_inclusive"])
    y1 = int(best["y_max_exclusive"])
    cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 255), thickness=4)
    label = (
        f"Exploratory {best['region_id']}: n10000_m1 best, "
        f"{float(best['winner_adjusted_ratio']):.3%}"
    )
    text_y = max(28, y0 - 12)
    cv2.putText(
        overlay,
        label,
        (x0, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(EVALUATION_DIR / "exploratory_n10000_m1_best_region.png"), overlay)


def main() -> None:
    reference_grains, evaluation_mask, boundary_mask, reference_qc = rasterize_reference()
    save_reference_outputs(reference_grains, evaluation_mask, boundary_mask)
    reference_qc["evaluated_reference_boundary_pixels"] = int(
        np.count_nonzero(boundary_mask[evaluation_mask])
    )

    summaries: list[dict[str, object]] = []
    all_details: list[dict[str, object]] = []
    candidates: dict[str, np.ndarray] = {}
    candidate_boundaries: dict[str, np.ndarray] = {}
    for folder_name, n_segments, compactness in PARAMETER_FOLDERS:
        label_path = TEST_DIR / folder_name / "method_3_data" / "slic_labels.npy"
        candidate = np.load(label_path, allow_pickle=False)
        candidates[folder_name] = candidate
        summary, details, candidate_boundary = evaluate_candidate(
            folder_name,
            n_segments,
            compactness,
            candidate,
            label_path,
            reference_grains,
            evaluation_mask,
            boundary_mask,
        )
        candidate_boundaries[folder_name] = candidate_boundary
        summaries.append(summary)
        all_details.extend(details)
        print(
            f"{folder_name}: original={summary['boundary_crossing_pixels']:,}/"
            f"{summary['evaluated_pixels']:,} "
            f"({summary['boundary_crossing_pixel_ratio']:.6%}), "
            f"credited={summary['credited_crossing_pixels']:,}, "
            f"adjusted={summary['adjusted_boundary_crossing_pixel_ratio']:.6%}",
            flush=True,
        )

    original_ranking = sorted(
        range(len(summaries)),
        key=lambda index: (
            summaries[index]["boundary_crossing_pixel_ratio"],
            summaries[index]["folder"],
        ),
    )
    adjusted_ranking = sorted(
        range(len(summaries)),
        key=lambda index: (
            summaries[index]["adjusted_boundary_crossing_pixel_ratio"],
            summaries[index]["folder"],
        ),
    )
    for rank, index in enumerate(original_ranking, start=1):
        summaries[index]["rank_original_lower_is_better"] = rank
    for rank, index in enumerate(adjusted_ranking, start=1):
        summaries[index]["rank_adjusted_lower_is_better"] = rank

    local_analysis = build_local_region_analysis(
        candidates,
        candidate_boundaries,
        reference_grains,
        evaluation_mask,
        boundary_mask,
    )
    save_exploratory_region_overlay(local_analysis)

    result = {
        "metric_definition": {
            "original_name": "boundary-crossing pixel ratio",
            "original_formula": (
                "sum over candidate superpixels of "
                "(labeled pixels - largest single-reference-grain overlap) "
                "/ total evaluated labeled pixels"
            ),
            "adjusted_name": "boundary-overlap-credited crossing pixel ratio",
            "adjusted_formula": (
                "(original crossing pixels - crossing pixels that lie on both "
                "the exact reference-polygon boundary and the candidate-superpixel "
                "boundary) / total evaluated labeled pixels"
            ),
            "candidate_boundary_definition": (
                "4-connected label transitions, marking pixels on both sides"
            ),
            "reference_boundary_definition": "one-pixel LabelMe polygon outline",
            "boundary_tolerance_pixels": 0,
            "interpretation": "lower is better",
            "unlabeled_policy": "excluded",
            "ambiguous_polygon_overlap_policy": "excluded",
            "reference_unit": "each LabelMe grain polygon is one reference grain",
            "local_analysis_status": (
                "exploratory; every fixed-grid cell is reported to avoid selective cropping"
            ),
        },
        "reference_qc": reference_qc,
        "summary": summaries,
        "local_region_analysis": local_analysis,
        "per_superpixel_detail": all_details,
    }
    RESULTS_JSON.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved evaluation data: {RESULTS_JSON}", flush=True)


if __name__ == "__main__":
    main()
