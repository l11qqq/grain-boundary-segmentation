"""Run a trained edge-GNN on raw images and visualize predicted grains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from build_gnn_dataset import (
    build_adjacency_outputs,
    build_undirected_edges,
    find_image_paths,
    load_node_index_map,
)
from train_gnn import EdgeGraphSAGE


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_DIR = PROJECT_DIR / "raw_pic413-3-1n"
DEFAULT_LABEL_MAP = PROJECT_DIR / "outputs" / "slic_3-1" / "slic_labels.npy"
DEFAULT_DATA_DIR = PROJECT_DIR / "outputs" / "gnn_data_3-1"
DEFAULT_MODEL = PROJECT_DIR / "outputs" / "gnn_results_3-1" / "best_model.pt"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "reproduced" / "prediction_visualization_3-1"


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.int8)

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[item] != item:
            next_item = int(self.parent[item])
            self.parent[item] = root
            item = next_item
        return root

    def union(self, item_a: int, item_b: int) -> None:
        root_a = self.find(item_a)
        root_b = self.find(item_b)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    minimum = float(image.min())
    maximum = float(image.max())
    if maximum - minimum < 1e-8:
        return np.zeros(image.shape, dtype=np.uint8)
    return np.round((image - minimum) / (maximum - minimum) * 255).astype(np.uint8)


def extract_features_and_std_image(
    image_paths: list[Path],
    node_index_map: np.ndarray,
    pixel_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Read the raw images once and compute both GNN features and the display background."""
    height, width = node_index_map.shape
    num_nodes = len(pixel_counts)
    features = np.empty((num_nodes, len(image_paths)), dtype=np.float32)
    pixel_sum = np.zeros((height, width), dtype=np.float64)
    pixel_square_sum = np.zeros((height, width), dtype=np.float64)
    flat_node_indices = node_index_map.ravel()

    for column, path in enumerate(image_paths):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to read raw image: {path}")
        if image.shape != (height, width):
            raise ValueError(
                f"Image {path.name} has shape {image.shape}; expected {(height, width)} "
                "to match the SLIC label map."
            )

        image_64 = image.astype(np.float64)
        pixel_sum += image_64
        pixel_square_sum += image_64 * image_64
        node_sums = np.bincount(
            flat_node_indices,
            weights=image.ravel(),
            minlength=num_nodes,
        )
        features[:, column] = (node_sums / pixel_counts).astype(np.float32)

        completed = column + 1
        if completed == 1 or completed % 25 == 0 or completed == len(image_paths):
            print(f"Reading raw images: {completed}/{len(image_paths)}", flush=True)

    count = float(len(image_paths))
    variance = np.maximum(pixel_square_sum / count - np.square(pixel_sum / count), 0.0)
    std_image = normalize_to_uint8(np.sqrt(variance))
    return features, std_image


def validate_image_order(image_paths: list[Path], data_dir: Path, allow_mismatch: bool) -> None:
    metadata_path = data_dir / "metadata.json"
    if not metadata_path.is_file():
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_names = metadata.get("raw_images")
    actual_names = [path.name for path in image_paths]
    if isinstance(expected_names, list) and actual_names != expected_names and not allow_mismatch:
        raise ValueError(
            "Raw-image names/order do not match the images used to train the model. "
            "Use --allow-image-order-mismatch only when the feature columns have the same semantics."
        )


def resolve_label_map(label_map: Path | None, data_dir: Path) -> Path:
    """Resolve the 2-D SLIC map, preferably from the graph-dataset metadata."""
    if label_map is not None:
        resolved = label_map.resolve()
    else:
        resolved = DEFAULT_LABEL_MAP.resolve()
        metadata_path = data_dir / "metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata_label_map = metadata.get("source_label_map")
            if isinstance(metadata_label_map, str) and metadata_label_map.strip():
                candidate = Path(metadata_label_map)
                if not candidate.is_absolute():
                    candidate = data_dir / candidate
                resolved = candidate.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"SLIC label map does not exist: {resolved}")
    return resolved


def validate_dataset_layout(
    data_dir: Path,
    node_ids: np.ndarray,
    num_raw_images: int,
    allow_new_graph: bool = False,
) -> None:
    """Validate feature semantics, and optionally require the original graph layout."""
    feature_path = data_dir / "node_features.npy"
    node_ids_path = data_dir / "node_ids.npy"
    if not feature_path.is_file() or not node_ids_path.is_file():
        raise FileNotFoundError(
            f"GNN data must contain node_features.npy and node_ids.npy: {data_dir}"
        )
    saved_features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    saved_node_ids = np.load(node_ids_path, allow_pickle=False)
    if saved_features.ndim != 2 or saved_features.shape[1] != num_raw_images:
        raise ValueError(
            f"GNN feature matrix has shape {saved_features.shape}; expected "
            f"{num_raw_images} angular feature columns."
        )
    if allow_new_graph:
        return
    expected_shape = (len(node_ids), num_raw_images)
    if saved_features.shape != expected_shape:
        raise ValueError(
            f"GNN feature matrix has shape {saved_features.shape}, expected {expected_shape}."
        )
    if not np.array_equal(saved_node_ids, node_ids):
        raise ValueError(
            "The SLIC label map node IDs do not match outputs/gnn_data/node_ids.npy."
        )


def load_model(checkpoint_path: Path, input_dim: int) -> tuple[EdgeGraphSAGE, dict[str, object]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Trained model does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if int(checkpoint["input_dim"]) != input_dim:
        raise ValueError(
            f"Model expects {checkpoint['input_dim']} input images/features, but received {input_dim}."
        )
    model = EdgeGraphSAGE(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_layers=int(checkpoint["num_layers"]),
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def merge_same_grain_nodes(
    num_nodes: int,
    undirected_edges: np.ndarray,
    predicted_edge_labels: np.ndarray,
) -> tuple[np.ndarray, int]:
    union_find = UnionFind(num_nodes)
    for (node_a, node_b), edge_label in zip(undirected_edges, predicted_edge_labels):
        if int(edge_label) == 0:
            union_find.union(int(node_a), int(node_b))

    roots = np.fromiter((union_find.find(node) for node in range(num_nodes)), dtype=np.int64)
    _, node_grain_ids = np.unique(roots, return_inverse=True)

    inconsistent_boundary_edges = 0
    for (node_a, node_b), edge_label in zip(undirected_edges, predicted_edge_labels):
        if int(edge_label) == 1 and node_grain_ids[node_a] == node_grain_ids[node_b]:
            inconsistent_boundary_edges += 1
    return node_grain_ids.astype(np.int32), inconsistent_boundary_edges


def compute_boundaries(grain_map: np.ndarray) -> np.ndarray:
    boundaries = np.zeros(grain_map.shape, dtype=bool)
    horizontal = grain_map[:, 1:] != grain_map[:, :-1]
    boundaries[:, 1:] |= horizontal
    boundaries[:, :-1] |= horizontal
    vertical = grain_map[1:, :] != grain_map[:-1, :]
    boundaries[1:, :] |= vertical
    boundaries[:-1, :] |= vertical
    return boundaries


def create_visualizations(
    grain_map: np.ndarray,
    std_image: np.ndarray,
    color_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_grains = int(grain_map.max()) + 1
    rng = np.random.default_rng(color_seed)
    colors = rng.integers(35, 256, size=(num_grains, 3), dtype=np.uint8)
    random_color_image = colors[grain_map]
    boundaries = compute_boundaries(grain_map)

    color_with_boundaries = random_color_image.copy()
    color_with_boundaries[boundaries] = np.array([0, 0, 0], dtype=np.uint8)
    std_rgb = np.repeat(std_image[:, :, None], 3, axis=2)
    std_overlay = (
        random_color_image.astype(np.float32) * 0.58
        + std_rgb.astype(np.float32) * 0.42
    ).astype(np.uint8)
    std_overlay[boundaries] = np.array([0, 0, 0], dtype=np.uint8)

    # Paper-friendly binary boundary map: white background, black grain boundaries.
    boundary_only_image = np.full(grain_map.shape, 255, dtype=np.uint8)
    boundary_only_image[boundaries] = 0
    return random_color_image, color_with_boundaries, std_overlay, boundary_only_image


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    Image.fromarray(rgb, mode="RGB").save(path)


def run_inference(args: argparse.Namespace) -> dict[str, object]:
    if args.threshold is not None and not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")
    args.label_map = resolve_label_map(args.label_map, args.data_dir)
    image_paths = find_image_paths(args.raw_dir)
    validate_image_order(image_paths, args.data_dir, args.allow_image_order_mismatch)

    labels, node_ids, pixel_counts = load_node_index_map(args.label_map)
    validate_dataset_layout(
        args.data_dir,
        node_ids,
        len(image_paths),
        allow_new_graph=getattr(args, "allow_new_graph", False),
    )
    model, checkpoint = load_model(args.model, len(image_paths))
    node_index_map = (
        labels
        if np.array_equal(node_ids, np.arange(len(node_ids), dtype=np.int32))
        else np.searchsorted(node_ids, labels).astype(np.int32)
    )
    print(f"Raw images: {len(image_paths)}")
    print(f"Superpixel nodes: {len(node_ids)}")
    node_features, std_image = extract_features_and_std_image(
        image_paths,
        node_index_map,
        pixel_counts,
    )

    undirected_edges = build_undirected_edges(node_index_map)
    _, graph_edge_index_np = build_adjacency_outputs(undirected_edges, len(node_ids))
    decision_threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(checkpoint.get("decision_threshold", 0.5))
    )

    features = torch.from_numpy(node_features)
    feature_mean = checkpoint["feature_mean"]
    feature_std = checkpoint["feature_std"]
    normalized_features = (features - feature_mean) / feature_std
    graph_edge_index = torch.from_numpy(graph_edge_index_np)
    degree = torch.bincount(graph_edge_index[1], minlength=len(node_ids)).to(torch.float32)
    query_edges = torch.from_numpy(undirected_edges.T.astype(np.int64))

    with torch.no_grad():
        embeddings = model.encode(normalized_features, graph_edge_index, degree)
        probabilities = torch.sigmoid(model.decode(embeddings, query_edges)).cpu().numpy()
    predicted_edge_labels = (probabilities >= decision_threshold).astype(np.int8)
    node_grain_ids, inconsistent_edges = merge_same_grain_nodes(
        len(node_ids),
        undirected_edges,
        predicted_edge_labels,
    )
    grain_map = node_grain_ids[node_index_map]
    random_colors, colors_with_boundaries, std_overlay, boundary_only_image = create_visualizations(
        grain_map,
        std_image,
        args.color_seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_rgb(args.output_dir / "predicted_grains_random_color.png", random_colors)
    save_rgb(args.output_dir / "predicted_grains_with_boundaries.png", colors_with_boundaries)
    save_rgb(args.output_dir / "predicted_grains_std_overlay.png", std_overlay)
    Image.fromarray(boundary_only_image, mode="L").save(
        args.output_dir / "predicted_grain_boundaries_only.png"
    )
    Image.fromarray(std_image, mode="L").save(args.output_dir / "std_image_from_raw.png")
    np.save(args.output_dir / "predicted_edge_probabilities.npy", probabilities, allow_pickle=False)
    np.save(args.output_dir / "predicted_edge_labels.npy", predicted_edge_labels, allow_pickle=False)
    np.save(args.output_dir / "predicted_node_grain_ids.npy", node_grain_ids, allow_pickle=False)
    np.save(args.output_dir / "predicted_grain_map.npy", grain_map.astype(np.int32), allow_pickle=False)
    np.save(args.output_dir / "undirected_edges.npy", undirected_edges, allow_pickle=False)

    grain_pixel_counts = np.bincount(grain_map.ravel())
    summary: dict[str, object] = {
        "num_raw_images": len(image_paths),
        "num_superpixel_nodes": len(node_ids),
        "num_adjacency_edges": len(undirected_edges),
        "threshold": decision_threshold,
        "threshold_source": "command_line" if args.threshold is not None else "model_checkpoint",
        "predicted_same_grain_edges": int(np.sum(predicted_edge_labels == 0)),
        "predicted_boundary_edges": int(np.sum(predicted_edge_labels == 1)),
        "predicted_grains": int(len(grain_pixel_counts)),
        "inconsistent_boundary_edges_after_connectivity_merge": int(inconsistent_edges),
        "grain_pixel_size": {
            "minimum": int(grain_pixel_counts.min()),
            "median": float(np.median(grain_pixel_counts)),
            "maximum": int(grain_pixel_counts.max()),
        },
        "edge_probability": {
            "minimum": float(probabilities.min()),
            "mean": float(probabilities.mean()),
            "maximum": float(probabilities.max()),
        },
        "model": str(args.model),
        "label_map": str(args.label_map),
        "raw_dir": str(args.raw_dir),
        "allow_new_graph": bool(getattr(args, "allow_new_graph", False)),
    }
    (args.output_dir / "prediction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Visualizations written to: {args.output_dir}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize trained GNN grain predictions.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--label-map",
        type=Path,
        default=None,
        help=(
            "2-D SLIC label map. By default it is read from "
            "outputs/gnn_data_3-1/metadata.json (source_label_map)."
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Boundary threshold; default reads the validation-selected value from the model.",
    )
    parser.add_argument("--color-seed", type=int, default=42)
    parser.add_argument("--allow-image-order-mismatch", action="store_true")
    parser.add_argument(
        "--allow-new-graph",
        action="store_true",
        help=(
            "Permit inference on a new SLIC graph while retaining the training "
            "feature definition and angular-image order checks."
        ),
    )
    args = parser.parse_args()
    for name in ("raw_dir", "data_dir", "model", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.label_map is not None:
        args.label_map = args.label_map.resolve()
    return args


if __name__ == "__main__":
    run_inference(parse_args())
