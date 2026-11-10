"""Figure-3 ablation for improving discontinuous narrow-twin segmentation.

The experiment reuses the Stage-3 angular boundary energy and initial SLICO
labels.  It adds two conservative operations before marker-controlled
watershed:

1. dual-threshold boundary tracking retains weak boundary pixels only when
   connected to a strong response;
2. multi-directional line closing bridges short interruptions along elongated
   twin boundaries.

Three profiles are generated so that increased twin recall can be compared
against boundary precision and over-segmentation.  The formal ``slic_test.py``
is not modified by this experiment.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects
from skimage.segmentation import find_boundaries, watershed


PROJECT_DIR = Path(__file__).resolve().parents[2]
FIG3_DIR = PROJECT_DIR / "fig3"
SOURCE_DIR = FIG3_DIR / "result_3"
OUTPUT_ROOT = FIG3_DIR / "twin_continuity"
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(FIG3_DIR))

import slic_test  # noqa: E402
from evaluate_superpixels import boundary_scores, compactness_score  # noqa: E402


@dataclass(frozen=True)
class Profile:
    name: str
    strong_percentile: float
    weak_percentile: float
    bridge_length: int
    barrier_min_size: int
    peak_min_distance: int
    min_half_width: float
    max_half_width: float
    marker_exclusion_radius: int
    max_extra_markers: int
    watershed_compactness: float
    min_region_size: int


PROFILES = (
    Profile("gentle", 82.0, 70.0, 5, 6, 4, 1.3, 9.0, 4, 7500, 0.008, 28),
    Profile("balanced", 80.0, 65.0, 7, 5, 3, 1.2, 10.0, 3, 9000, 0.005, 20),
    Profile("strong", 78.0, 58.0, 9, 4, 2, 1.0, 11.0, 2, 11000, 0.003, 14),
)


def line_kernels(length: int) -> list[np.ndarray]:
    if length < 3:
        return []
    if length % 2 == 0:
        length += 1
    center = length // 2
    radius = center
    kernels: list[np.ndarray] = []
    for angle in (0.0, 30.0, 60.0, 90.0, 120.0, 150.0):
        radians = np.deg2rad(angle)
        dx = int(round(radius * np.cos(radians)))
        dy = int(round(radius * np.sin(radians)))
        kernel = np.zeros((length, length), dtype=np.uint8)
        cv2.line(
            kernel,
            (center - dx, center - dy),
            (center + dx, center + dy),
            color=1,
            thickness=1,
        )
        kernels.append(kernel)
    return kernels


def continuity_barrier(
    strength: np.ndarray,
    profile: Profile,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    strong_threshold = float(np.percentile(strength, profile.strong_percentile))
    weak_threshold = float(np.percentile(strength, profile.weak_percentile))
    strong = strength >= strong_threshold
    weak = strength >= weak_threshold

    # Geodesic tracking: isolated weak texture cannot become a barrier.
    tracked = ndi.binary_propagation(strong, mask=weak)
    bridged = tracked.copy()
    for kernel in line_kernels(profile.bridge_length):
        closed = cv2.morphologyEx(
            tracked.astype(np.uint8),
            cv2.MORPH_CLOSE,
            kernel,
            borderType=cv2.BORDER_REFLECT101,
        ) > 0
        bridged |= closed

    # Restrict new bridge pixels to the vicinity of measured weak responses.
    weak_support = ndi.binary_dilation(weak, iterations=1)
    barrier = tracked | (bridged & weak_support)
    barrier = remove_small_objects(
        barrier,
        min_size=profile.barrier_min_size,
        connectivity=2,
    )
    stats: dict[str, float | int] = {
        "strong_threshold": strong_threshold,
        "weak_threshold": weak_threshold,
        "strong_pixels": int(np.count_nonzero(strong)),
        "tracked_pixels": int(np.count_nonzero(tracked)),
        "bridge_pixels_added": int(np.count_nonzero(barrier & ~tracked)),
        "final_barrier_pixels": int(np.count_nonzero(barrier)),
    }
    return barrier.astype(bool), tracked.astype(bool), stats


def add_markers(
    core_markers: np.ndarray,
    barrier: np.ndarray,
    profile: Profile,
) -> tuple[np.ndarray, np.ndarray, int]:
    distance = ndi.distance_transform_edt(~barrier).astype(np.float32)
    candidates = peak_local_max(
        distance,
        min_distance=profile.peak_min_distance,
        threshold_abs=profile.min_half_width,
        exclude_border=False,
        num_peaks=profile.max_extra_markers * 5,
    )
    if candidates.size:
        widths = distance[candidates[:, 0], candidates[:, 1]]
        candidates = candidates[widths <= profile.max_half_width]

    markers = core_markers.copy()
    occupied = ndi.binary_dilation(
        markers > 0,
        iterations=profile.marker_exclusion_radius,
    ).astype(np.uint8)
    next_marker = int(markers.max()) + 1
    added = 0
    for y, x in candidates:
        y_int, x_int = int(y), int(x)
        if occupied[y_int, x_int]:
            continue
        markers[y_int, x_int] = next_marker
        next_marker += 1
        added += 1
        cv2.circle(
            occupied,
            (x_int, y_int),
            profile.marker_exclusion_radius,
            color=1,
            thickness=-1,
        )
        if added >= profile.max_extra_markers:
            break
    return markers, distance, added


def save_profile_outputs(
    output_dir: Path,
    labels: np.ndarray,
    strength: np.ndarray,
    background: np.ndarray,
    tracked: np.ndarray,
    barrier: np.ndarray,
    markers: np.ndarray,
    distance: np.ndarray,
    config: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "slic_labels.npy", labels.astype(np.int32), allow_pickle=False)
    slic_test.save_gray_png(output_dir / "tracked_weak_boundary.png", tracked.astype(np.float32))
    slic_test.save_gray_png(output_dir / "continuity_barrier_mask.png", barrier.astype(np.float32))
    slic_test.save_gray_png(output_dir / "twin_corridor_distance.png", distance)
    slic_test.save_marker_preview(output_dir / "markers_and_continuity_barrier.png", markers, barrier)
    overlay = slic_test.create_boundary_overlay(background, labels)
    energy_overlay = slic_test.create_boundary_overlay(strength, labels)
    slic_test.save_rgb_png(output_dir / "segmentation_full_resolution.png", overlay)
    slic_test.save_rgb_png(output_dir / "segmentation_on_angular_boundary.png", energy_overlay)
    (output_dir / "method_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    panels = (
        (background, "Enhanced standard deviation", "gray"),
        (strength, "Angular boundary energy", "gray"),
        (tracked, "Dual-threshold tracked boundary", "gray"),
        (barrier, "Directionally bridged barrier", "gray"),
        (overlay, "Continuity-enhanced segmentation", None),
        (energy_overlay, "Segmentation on angular boundary", None),
    )
    for axis, (image, title, cmap) in zip(axes.flat, panels):
        axis.imshow(image, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "visualization.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    initial_labels = np.load(SOURCE_DIR / "initial_slic_labels.npy", allow_pickle=False)
    source_labels = np.load(SOURCE_DIR / "slic_labels.npy", allow_pickle=False)
    strength_u8 = cv2.imread(str(SOURCE_DIR / "angular_boundary_strength.png"), cv2.IMREAD_GRAYSCALE)
    background_u8 = cv2.imread(str(SOURCE_DIR / "std_image.png"), cv2.IMREAD_GRAYSCALE)
    reference_u8 = cv2.imread(str(FIG3_DIR / "angular_gradient_reference.png"), cv2.IMREAD_GRAYSCALE)
    if strength_u8 is None or background_u8 is None or reference_u8 is None:
        raise FileNotFoundError("Stage-3 energy, background, or cached proxy reference is missing")
    strength = strength_u8.astype(np.float32) / 255.0
    background = background_u8.astype(np.float32) / 255.0
    reference_strength = reference_u8.astype(np.float32) / 255.0
    reference_boundary = reference_strength >= np.percentile(reference_strength, 90.0)
    core_markers = slic_test.core_markers_from_slic(initial_labels)

    rows: list[dict[str, object]] = []
    baseline_boundary = find_boundaries(source_labels, mode="thick")
    precision, recall, f1 = boundary_scores(baseline_boundary, reference_boundary, tolerance=2)
    rows.append(
        {
            "profile": "stage3_baseline",
            "superpixels": int(len(np.unique(source_labels))),
            "extra_markers": 1680,
            "boundary_precision": precision,
            "boundary_recall": recall,
            "boundary_f1": f1,
            "compactness": compactness_score(source_labels),
        }
    )

    for profile in PROFILES:
        print(f"Running profile: {profile.name}", flush=True)
        barrier, tracked, barrier_stats = continuity_barrier(strength, profile)
        markers, distance, added = add_markers(core_markers, barrier, profile)
        raw_labels = watershed(
            strength,
            markers=markers,
            mask=np.ones(strength.shape, dtype=bool),
            compactness=profile.watershed_compactness,
            watershed_line=False,
        ).astype(np.int32)
        labels, removed = slic_test.absorb_small_watershed_regions(
            raw_labels,
            strength,
            profile.min_region_size,
        )
        predicted_boundary = find_boundaries(labels, mode="thick")
        precision, recall, f1 = boundary_scores(predicted_boundary, reference_boundary, tolerance=2)
        row = {
            "profile": profile.name,
            "superpixels": int(len(np.unique(labels))),
            "extra_markers": added,
            "boundary_precision": precision,
            "boundary_recall": recall,
            "boundary_f1": f1,
            "compactness": compactness_score(labels),
        }
        rows.append(row)
        config: dict[str, object] = {
            "method": "Twin-continuity-enhanced marker-controlled watershed",
            "source": str(SOURCE_DIR),
            "profile": asdict(profile),
            "barrier_statistics": barrier_stats,
            "actual_extra_markers": added,
            "removed_small_regions": removed,
            "actual_superpixels": row["superpixels"],
            "proxy_metrics": row,
        }
        save_profile_outputs(
            OUTPUT_ROOT / profile.name,
            labels,
            strength,
            background,
            tracked,
            barrier,
            markers,
            distance,
            config,
        )
        print(
            f"{profile.name}: regions={row['superpixels']}, markers={added}, "
            f"recall={recall:.4f}, F1={f1:.4f}",
            flush=True,
        )

    csv_path = OUTPUT_ROOT / "continuity_metrics.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT_ROOT / "continuity_metrics.json").write_text(
        json.dumps(
            {
                "reference": "396-D angular-gradient proxy, top 10%, tolerance 2 px",
                "note": "Proxy metrics must not replace evaluation using manual twin-boundary ground truth.",
                "results": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    names = [str(row["profile"]) for row in rows]
    x = np.arange(len(rows))
    fig, axis = plt.subplots(figsize=(10, 6))
    for key, label, marker in (
        ("boundary_precision", "Precision", "o"),
        ("boundary_recall", "Recall", "s"),
        ("boundary_f1", "F1", "^"),
        ("compactness", "Compactness", "D"),
    ):
        axis.plot(x, [float(row[key]) for row in rows], marker=marker, linewidth=2, label=label)
    axis.set_xticks(x, names)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Score")
    axis.set_title("Twin-continuity parameter sweep (proxy evaluation)")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "continuity_metric_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
