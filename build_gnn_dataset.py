"""Build node features, graph topology, and edge supervision for GNN training.

Each graph node is one SLIC superpixel.  Column j of ``node_features.npy`` is
the mean grayscale intensity of *all* pixels in that superpixel in raw image
j.  The RGB/thresholded-pixel experiments in ``build_gnn_dataset_test.py`` are
not used by this formal data-building pipeline.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np
from scipy import sparse


PROJECT_DIR = Path(__file__).resolve().parent
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def natural_sort_key(path: Path) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def find_image_paths(raw_dir: Path) -> list[Path]:
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw-image directory does not exist: {raw_dir}")
    paths = sorted(
        (
            path
            for path in raw_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
        ),
        key=natural_sort_key,
    )
    if not paths:
        raise ValueError(f"No raw images found in: {raw_dir}")
    return paths


def load_node_index_map(label_map_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.load(label_map_path, allow_pickle=False)
    if labels.ndim != 2:
        raise ValueError(f"SLIC label map must be 2D, got {labels.shape}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"SLIC labels must be integers, got {labels.dtype}")

    labels = labels.astype(np.int32, copy=False)
    node_ids = np.unique(labels).astype(np.int32)
    if np.array_equal(node_ids, np.arange(len(node_ids), dtype=np.int32)):
        node_index_map = labels
    else:
        # Convert arbitrary SLIC IDs to contiguous matrix row indices.
        node_index_map = np.searchsorted(node_ids, labels).astype(np.int32)

    pixel_counts = np.bincount(node_index_map.ravel(), minlength=len(node_ids)).astype(np.int64)
    if np.any(pixel_counts == 0):
        raise ValueError("At least one graph node contains no pixels.")
    return labels, node_ids, pixel_counts


def extract_node_features(
    image_paths: list[Path],
    node_index_map: np.ndarray,
    pixel_counts: np.ndarray,
) -> np.ndarray:
    """Return [num_nodes, num_images] all-pixel mean-grayscale features.

    No pixel selection, clustering, threshold rejection, or RGB expansion is
    performed: every pixel assigned to a superpixel contributes once.
    """
    height, width = node_index_map.shape
    num_nodes = len(pixel_counts)
    features = np.empty((num_nodes, len(image_paths)), dtype=np.float32)
    flat_node_indices = node_index_map.ravel()

    for column, path in enumerate(image_paths):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to read raw image: {path}")
        if image.shape != (height, width):
            raise ValueError(
                f"Image {path.name} has shape {image.shape}; expected {(height, width)} "
                "to match slic_labels.npy."
            )

        grayscale_sums = np.bincount(
            flat_node_indices,
            weights=image.ravel(),
            minlength=num_nodes,
        )
        features[:, column] = (grayscale_sums / pixel_counts).astype(np.float32)

        completed = column + 1
        if completed == 1 or completed % 25 == 0 or completed == len(image_paths):
            print(f"Extracting node features: {completed}/{len(image_paths)}", flush=True)

    return features


def build_undirected_edges(node_index_map: np.ndarray) -> np.ndarray:
    """Return unique canonical edges as an [E, 2] int64 array."""
    pair_blocks: list[np.ndarray] = []
    for side_a, side_b in (
        (node_index_map[:, :-1], node_index_map[:, 1:]),
        (node_index_map[:-1, :], node_index_map[1:, :]),
    ):
        different = side_a != side_b
        values_a = side_a[different].astype(np.int64)
        values_b = side_b[different].astype(np.int64)
        pair_blocks.append(
            np.column_stack((np.minimum(values_a, values_b), np.maximum(values_a, values_b)))
        )
    return np.unique(np.vstack(pair_blocks), axis=0)


def compute_node_centroids(node_index_map: np.ndarray, num_nodes: int) -> np.ndarray:
    """Return normalized ``[x, y]`` centroid coordinates for every node."""
    height, width = node_index_map.shape
    yy, xx = np.indices((height, width), dtype=np.float64)
    flat_nodes = node_index_map.ravel()
    counts = np.bincount(flat_nodes, minlength=num_nodes).astype(np.float64)
    centroid_x = np.bincount(flat_nodes, weights=xx.ravel(), minlength=num_nodes) / counts
    centroid_y = np.bincount(flat_nodes, weights=yy.ravel(), minlength=num_nodes) / counts
    return np.column_stack((centroid_x / max(width, 1), centroid_y / max(height, 1))).astype(np.float32)


def build_adjacency_outputs(
    undirected_edges: np.ndarray,
    num_nodes: int,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    # PyTorch Geometric expects both directions in edge_index.
    forward = undirected_edges.T
    backward = undirected_edges[:, ::-1].T
    edge_index = np.concatenate((forward, backward), axis=1).astype(np.int64)
    adjacency = sparse.csr_matrix(
        (
            np.ones(edge_index.shape[1], dtype=np.uint8),
            (edge_index[0], edge_index[1]),
        ),
        shape=(num_nodes, num_nodes),
        dtype=np.uint8,
    )
    return adjacency, edge_index


def load_edge_supervision(
    csv_path: Path,
    node_ids: np.ndarray,
    undirected_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map CSV node IDs to matrix rows and mark unknown edges with -1."""
    node_id_to_index = {int(node_id): index for index, node_id in enumerate(node_ids)}
    edge_to_index = {tuple(map(int, edge)): index for index, edge in enumerate(undirected_edges)}
    labeled_pairs: list[tuple[int, int]] = []
    labels: list[int] = []
    seen: dict[tuple[int, int], int] = {}

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {"node_a", "node_b", "edge_label"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"Edge-label CSV must contain {sorted(required_columns)}: {csv_path}")

        for row_number, row in enumerate(reader, start=2):
            source_id = int(row["node_a"])
            target_id = int(row["node_b"])
            label = int(row["edge_label"])
            if label not in (0, 1):
                raise ValueError(f"Invalid edge label {label} at CSV row {row_number}")
            if source_id not in node_id_to_index or target_id not in node_id_to_index:
                raise ValueError(f"Unknown node ID at CSV row {row_number}: {source_id}-{target_id}")

            source = node_id_to_index[source_id]
            target = node_id_to_index[target_id]
            pair = (min(source, target), max(source, target))
            if pair not in edge_to_index:
                raise ValueError(f"Labeled pair is not a SLIC adjacency edge: {source_id}-{target_id}")
            if pair in seen and seen[pair] != label:
                raise ValueError(f"Conflicting labels for edge {source_id}-{target_id}")
            if pair in seen:
                continue
            seen[pair] = label
            labeled_pairs.append(pair)
            labels.append(label)

    edge_label_index = np.asarray(labeled_pairs, dtype=np.int64).T
    edge_labels = np.asarray(labels, dtype=np.int64)
    all_edge_labels = np.full(len(undirected_edges), -1, dtype=np.int8)
    for pair, label in zip(labeled_pairs, labels):
        all_edge_labels[edge_to_index[pair]] = label
    edge_label_mask = all_edge_labels >= 0
    return edge_label_index, edge_labels, all_edge_labels, edge_label_mask


def build_dataset(
    raw_dir: Path,
    label_map_path: Path,
    edge_labels_path: Path,
    output_dir: Path,
    expected_num_images: int | None = None,
) -> None:
    image_paths = find_image_paths(raw_dir)
    if expected_num_images is not None and len(image_paths) != expected_num_images:
        raise ValueError(
            f"Expected {expected_num_images} raw images in {raw_dir}, found {len(image_paths)}."
        )
    labels, node_ids, pixel_counts = load_node_index_map(label_map_path)
    node_index_map = (
        labels
        if np.array_equal(node_ids, np.arange(len(node_ids), dtype=np.int32))
        else np.searchsorted(node_ids, labels).astype(np.int32)
    )

    print(f"Raw images: {len(image_paths)}")
    print(f"Graph nodes: {len(node_ids)}")
    node_features = extract_node_features(image_paths, node_index_map, pixel_counts)
    undirected_edges = build_undirected_edges(node_index_map)
    node_centroids = compute_node_centroids(node_index_map, len(node_ids))
    adjacency, edge_index = build_adjacency_outputs(undirected_edges, len(node_ids))

    if not edge_labels_path.is_file():
        raise FileNotFoundError(f"Edge-label CSV does not exist: {edge_labels_path}")
    edge_label_index, edge_labels, all_edge_labels, edge_label_mask = load_edge_supervision(
        edge_labels_path,
        node_ids,
        undirected_edges,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "node_features.npy", node_features, allow_pickle=False)
    np.save(output_dir / "node_ids.npy", node_ids, allow_pickle=False)
    np.save(output_dir / "node_pixel_counts.npy", pixel_counts, allow_pickle=False)
    np.save(output_dir / "node_centroids.npy", node_centroids, allow_pickle=False)
    np.save(output_dir / "undirected_edges.npy", undirected_edges, allow_pickle=False)
    np.save(output_dir / "edge_index.npy", edge_index, allow_pickle=False)
    sparse.save_npz(output_dir / "adjacency_matrix.npz", adjacency, compressed=True)
    np.save(output_dir / "edge_label_index.npy", edge_label_index, allow_pickle=False)
    np.save(output_dir / "edge_labels.npy", edge_labels, allow_pickle=False)
    np.save(output_dir / "all_edge_labels.npy", all_edge_labels, allow_pickle=False)
    np.save(output_dir / "edge_label_mask.npy", edge_label_mask, allow_pickle=False)

    metadata = {
        "num_nodes": int(len(node_ids)),
        "num_raw_images": int(len(image_paths)),
        "num_node_features": int(node_features.shape[1]),
        "node_centroid_definition": "normalized [x, y] centroid of each superpixel",
        "num_undirected_edges": int(len(undirected_edges)),
        "num_directed_message_passing_edges": int(edge_index.shape[1]),
        "num_labeled_edges": int(len(edge_labels)),
        "num_class_0_same_grain": int(np.sum(edge_labels == 0)),
        "num_class_1_boundary": int(np.sum(edge_labels == 1)),
        "feature_definition": (
            "node_features[i, j] is the mean grayscale value of all pixels belonging "
            "to graph node i in raw_images[j]"
        ),
        "edge_label_definition": {"0": "same grain", "1": "different grains / boundary", "-1": "unknown"},
        "raw_images": [path.name for path in image_paths],
        "source_label_map": str(label_map_path.resolve()),
        "source_edge_labels": str(edge_labels_path.resolve()),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Node feature matrix: {node_features.shape} ({node_features.dtype})")
    print(f"Node centroids: {node_centroids.shape} ({node_centroids.dtype})")
    print(f"Sparse adjacency matrix: {adjacency.shape}, nnz={adjacency.nnz}")
    print(f"Labeled edges: {len(edge_labels)}")
    print(f"Dataset written to: {output_dir}")


def main() -> None:
    """Build one independent GNN dataset per raw/SLIC data group.

    Add or edit entries in ``data_groups`` when new regions are available.
    ``edge_labels_dir`` points to the folder produced by
    ``edge_labeling_app.py``; ``label_map_path`` points to the matching SLIC
    label map in the same data group.
    """
    data_groups = [
        {
            "raw_dir": PROJECT_DIR / "raw_pic413-3-1n",
            "label_map_path": PROJECT_DIR / "outputs" / "slic_3-1" / "slic_labels.npy",
            "edge_labels_dir": PROJECT_DIR / "outputs" / "slic_3-1" / "edge_labels",
            "output_dir": PROJECT_DIR / "reproduced" / "gnn_data_3-1",
            # The completed acquisition grid contains 11 x 36 angle images.
            # Keep this check enabled so an incomplete raw folder cannot
            # silently produce a feature matrix with the wrong width.
            "expected_num_images": 396,
        },
    ]

    for group_number, group in enumerate(data_groups, start=1):
        raw_dir = Path(group["raw_dir"]).resolve()
        label_map_path = Path(group["label_map_path"]).resolve()
        edge_labels_dir = Path(group["edge_labels_dir"]).resolve()
        output_dir = Path(group.get("output_dir", PROJECT_DIR / "outputs" / f"gnn_data_{group_number}")).resolve()
        edge_labels_path = edge_labels_dir / "edge_labels.csv"
        expected_num_images = group.get("expected_num_images")
        if expected_num_images is not None:
            expected_num_images = int(expected_num_images)
        print(f"\n=== Building GNN dataset group {group_number} ===")
        print(f"Raw images: {raw_dir}")
        print(f"SLIC label map: {label_map_path}")
        print(f"Edge labels: {edge_labels_dir}")
        build_dataset(
            raw_dir=raw_dir,
            label_map_path=label_map_path,
            edge_labels_path=edge_labels_path,
            output_dir=output_dir,
            expected_num_images=expected_num_images,
        )


if __name__ == "__main__":
    main()
