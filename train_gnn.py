"""Train one full-batch GraphSAGE model sequentially on multiple graph groups."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "reproduced" / "gnn_results_3-1"
DEFAULT_DATA_DIR = PROJECT_DIR / "outputs" / "gnn_data_3-1"
DEFAULT_EXPECTED_FEATURE_DIM = 396
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
DEFAULT_MAX_EPOCHS = 300  # Safety ceiling used for the archived 3-1 run.
DEFAULT_PATIENCE = 40


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class GraphSAGELayer(nn.Module):
    """Mean-neighbor GraphSAGE layer implemented with index_add_."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.self_linear = nn.Linear(input_dim, output_dim)
        self.neighbor_linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        degree: torch.Tensor,
    ) -> torch.Tensor:
        source, target = edge_index
        neighbor_sum = node_features.new_zeros(node_features.shape)
        neighbor_sum.index_add_(0, target, node_features[source])
        neighbor_mean = neighbor_sum / degree.clamp_min(1.0).unsqueeze(1)
        return self.self_linear(node_features) + self.neighbor_linear(neighbor_mean)


class GraphSAGEEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.layers = nn.ModuleList(
            GraphSAGELayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_layers))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        degree: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.input_projection(node_features)
        for layer, norm in zip(self.layers, self.norms):
            update = layer(hidden, edge_index, degree)
            update = self.dropout(torch.relu(norm(update)))
            hidden = hidden + update
        return hidden


class SymmetricEdgeDecoder(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_embeddings: torch.Tensor, labeled_edges: torch.Tensor) -> torch.Tensor:
        source_embedding = node_embeddings[labeled_edges[0]]
        target_embedding = node_embeddings[labeled_edges[1]]
        symmetric_features = torch.cat(
            (
                torch.abs(source_embedding - target_embedding),
                source_embedding * target_embedding,
                source_embedding + target_embedding,
            ),
            dim=1,
        )
        return self.mlp(symmetric_features).squeeze(1)


class EdgeGraphSAGE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = GraphSAGEEncoder(input_dim, hidden_dim, num_layers, dropout)
        self.decoder = SymmetricEdgeDecoder(hidden_dim, dropout)

    def encode(
        self,
        node_features: torch.Tensor,
        graph_edge_index: torch.Tensor,
        degree: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(node_features, graph_edge_index, degree)

    def decode(self, node_embeddings: torch.Tensor, query_edges: torch.Tensor) -> torch.Tensor:
        return self.decoder(node_embeddings, query_edges)

    def forward(
        self,
        node_features: torch.Tensor,
        graph_edge_index: torch.Tensor,
        degree: torch.Tensor,
        query_edges: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = self.encode(node_features, graph_edge_index, degree)
        return self.decode(embeddings, query_edges)


def random_stratified_split(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    train_indices, temporary_indices = train_test_split(
        indices,
        test_size=VALIDATION_RATIO + TEST_RATIO,
        random_state=seed,
        stratify=labels,
    )
    validation_indices, test_indices = train_test_split(
        temporary_indices,
        test_size=TEST_RATIO / (VALIDATION_RATIO + TEST_RATIO),
        random_state=seed,
        stratify=labels[temporary_indices],
    )
    return (
        np.sort(train_indices),
        np.sort(validation_indices),
        np.sort(test_indices),
    )


def farthest_point_sample(
    candidate_indices: np.ndarray,
    coordinates: np.ndarray,
    sample_count: int,
    rng: np.random.Generator,
    anchor_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Select spatially dispersed points from one edge-label class."""
    if sample_count <= 0:
        return np.empty(0, dtype=np.int64)
    if sample_count > len(candidate_indices):
        raise ValueError("Cannot sample more spatial points than available candidates.")

    candidate_coordinates = coordinates[candidate_indices]
    if anchor_indices is not None and len(anchor_indices) > 0:
        anchor_coordinates = coordinates[anchor_indices]
        distances = np.min(
            np.sum((candidate_coordinates[:, None, :] - anchor_coordinates[None, :, :]) ** 2, axis=2),
            axis=1,
        )
    else:
        first_position = int(rng.integers(len(candidate_indices)))
        first_coordinate = candidate_coordinates[first_position]
        distances = np.sum((candidate_coordinates - first_coordinate) ** 2, axis=1)

    available = np.ones(len(candidate_indices), dtype=bool)
    selected_positions: list[int] = []
    for _ in range(sample_count):
        masked_distances = np.where(available, distances, -1.0)
        position = int(np.argmax(masked_distances))
        selected_positions.append(position)
        available[position] = False
        new_distances = np.sum((candidate_coordinates - candidate_coordinates[position]) ** 2, axis=1)
        distances = np.minimum(distances, new_distances)
    return candidate_indices[np.asarray(selected_positions, dtype=np.int64)]


def spatially_distributed_split(
    labels: np.ndarray,
    edge_midpoints: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create class-balanced validation/test sets spread over the labeled region."""
    rng = np.random.default_rng(seed)
    validation_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    train_parts: list[np.ndarray] = []

    for label in np.unique(labels):
        class_indices = np.flatnonzero(labels == label)
        test_count = int(round(len(class_indices) * TEST_RATIO))
        validation_count = int(round(len(class_indices) * VALIDATION_RATIO))
        test_indices = farthest_point_sample(
            class_indices,
            edge_midpoints,
            test_count,
            rng,
        )
        remaining = np.setdiff1d(class_indices, test_indices, assume_unique=True)
        validation_indices = farthest_point_sample(
            remaining,
            edge_midpoints,
            validation_count,
            rng,
            anchor_indices=test_indices,
        )
        train_indices = np.setdiff1d(remaining, validation_indices, assume_unique=True)
        test_parts.append(test_indices)
        validation_parts.append(validation_indices)
        train_parts.append(train_indices)

    return (
        np.sort(np.concatenate(train_parts)),
        np.sort(np.concatenate(validation_parts)),
        np.sort(np.concatenate(test_parts)),
    )


def compute_labeled_edge_midpoints(
    label_map_path: Path,
    node_ids: np.ndarray,
    labeled_edge_index: np.ndarray,
) -> np.ndarray:
    label_map = np.load(label_map_path, allow_pickle=False)
    if label_map.ndim != 2:
        raise ValueError(f"SLIC label map must be 2D, got {label_map.shape}")
    node_index_map = (
        label_map.astype(np.int64, copy=False)
        if np.array_equal(node_ids, np.arange(len(node_ids)))
        else np.searchsorted(node_ids, label_map).astype(np.int64)
    )
    yy, xx = np.indices(label_map.shape)
    flat_nodes = node_index_map.ravel()
    counts = np.bincount(flat_nodes, minlength=len(node_ids))
    centroid_x = np.bincount(flat_nodes, weights=xx.ravel(), minlength=len(node_ids)) / counts
    centroid_y = np.bincount(flat_nodes, weights=yy.ravel(), minlength=len(node_ids)) / counts
    source, target = labeled_edge_index
    midpoint_x = (centroid_x[source] + centroid_x[target]) * 0.5 / label_map.shape[1]
    midpoint_y = (centroid_y[source] + centroid_y[target]) * 0.5 / label_map.shape[0]
    return np.column_stack((midpoint_x, midpoint_y)).astype(np.float32)


def spatial_distribution_metrics(
    edge_midpoints: np.ndarray,
    indices: np.ndarray,
    grid_size: int = 8,
) -> dict[str, float | int]:
    points = edge_midpoints[indices]
    cells = np.clip((points * grid_size).astype(np.int64), 0, grid_size - 1)
    occupied_cells = len(np.unique(cells, axis=0))
    if len(points) <= 1:
        mean_nearest_distance = 0.0
    else:
        distances = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2)
        np.fill_diagonal(distances, np.inf)
        mean_nearest_distance = float(np.mean(np.sqrt(np.min(distances, axis=1))))
    return {
        "occupied_8x8_cells": occupied_cells,
        "mean_normalized_nearest_neighbor_distance": mean_nearest_distance,
    }


def select_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if len(thresholds) == 0:
        return 0.5, 0.0
    f1_values = 2.0 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1],
        1e-12,
    )
    best_index = int(np.argmax(f1_values))
    return float(thresholds[best_index]), float(f1_values[best_index])


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, object]:
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    has_both_classes = len(np.unique(labels)) > 1
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if has_both_classes else None,
        "pr_auc": float(average_precision_score(labels, probabilities)) if has_both_classes else None,
        "decision_threshold": float(threshold),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


@torch.no_grad()
def evaluate_split(
    model: EdgeGraphSAGE,
    node_features: torch.Tensor,
    graph_edge_index: torch.Tensor,
    degree: torch.Tensor,
    labeled_edge_index: torch.Tensor,
    labels: torch.Tensor,
    split_indices: torch.Tensor,
    loss_function: nn.Module,
) -> tuple[float, dict[str, object], np.ndarray]:
    model.eval()
    embeddings = model.encode(node_features, graph_edge_index, degree)
    logits = model.decode(embeddings, labeled_edge_index[:, split_indices])
    split_labels = labels[split_indices]
    loss = float(loss_function(logits, split_labels).item())
    probabilities = torch.sigmoid(logits).cpu().numpy()
    metrics = classification_metrics(split_labels.cpu().numpy().astype(np.int64), probabilities)
    return loss, metrics, probabilities


def load_training_data(data_dir: Path) -> dict[str, np.ndarray]:
    required_files = {
        "node_features": "node_features.npy",
        "edge_index": "edge_index.npy",
        "edge_label_index": "edge_label_index.npy",
        "edge_labels": "edge_labels.npy",
        "undirected_edges": "undirected_edges.npy",
        "node_ids": "node_ids.npy",
    }
    arrays: dict[str, np.ndarray] = {}
    for key, filename in required_files.items():
        path = data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing GNN data file: {path}. Run build_gnn_dataset.py first.")
        arrays[key] = np.load(path, allow_pickle=False)
    return arrays


def load_node_centroids(data_dir: Path, arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Load saved centroids, with a metadata-based fallback for old datasets."""
    centroid_path = data_dir / "node_centroids.npy"
    if centroid_path.is_file():
        centroids = np.asarray(np.load(centroid_path, allow_pickle=False), dtype=np.float32)
        if centroids.shape != (len(arrays["node_ids"]), 2):
            raise ValueError(f"Invalid node_centroids shape in {centroid_path}: {centroids.shape}")
        return centroids

    metadata_path = data_dir / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        source_label_map = metadata.get("source_label_map")
        if isinstance(source_label_map, str):
            label_map_path = Path(source_label_map)
            if label_map_path.is_file():
                label_map = np.load(label_map_path, allow_pickle=False)
                node_ids = arrays["node_ids"]
                node_index_map = (
                    label_map.astype(np.int64, copy=False)
                    if np.array_equal(node_ids, np.arange(len(node_ids)))
                    else np.searchsorted(node_ids, label_map).astype(np.int64)
                )
                height, width = node_index_map.shape
                yy, xx = np.indices((height, width), dtype=np.float64)
                flat_nodes = node_index_map.ravel()
                counts = np.bincount(flat_nodes, minlength=len(node_ids)).astype(np.float64)
                centroid_x = np.bincount(flat_nodes, weights=xx.ravel(), minlength=len(node_ids)) / counts
                centroid_y = np.bincount(flat_nodes, weights=yy.ravel(), minlength=len(node_ids)) / counts
                return np.column_stack(
                    (centroid_x / max(width, 1), centroid_y / max(height, 1))
                ).astype(np.float32)
    raise FileNotFoundError(
        f"Missing node_centroids.npy in {data_dir}; rebuild the dataset with build_gnn_dataset.py."
    )


def prepare_group(
    data_dir: Path,
    arrays: dict[str, np.ndarray],
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    args: argparse.Namespace,
    group_seed: int,
) -> dict[str, object]:
    raw_features = torch.from_numpy(arrays["node_features"].astype(np.float32))
    if raw_features.shape[1] != feature_mean.shape[0]:
        raise ValueError(
            f"Feature dimension mismatch in {data_dir}: {raw_features.shape[1]} vs {feature_mean.shape[0]}"
        )
    node_features = (raw_features - feature_mean) / feature_std
    graph_edge_index = torch.from_numpy(arrays["edge_index"].astype(np.int64))
    labeled_edge_index = torch.from_numpy(arrays["edge_label_index"].astype(np.int64))
    labels = torch.from_numpy(arrays["edge_labels"].astype(np.float32))
    num_nodes = node_features.shape[0]
    degree = torch.bincount(graph_edge_index[1], minlength=num_nodes).to(torch.float32)
    centroids = load_node_centroids(data_dir, arrays)
    source, target = arrays["edge_label_index"]
    edge_midpoints = ((centroids[source] + centroids[target]) * 0.5).astype(np.float32)

    if args.split_strategy == "spatial":
        train_np, validation_np, test_np = spatially_distributed_split(
            arrays["edge_labels"], edge_midpoints, group_seed
        )
    else:
        train_np, validation_np, test_np = random_stratified_split(arrays["edge_labels"], group_seed)

    return {
        "data_dir": data_dir,
        "arrays": arrays,
        "node_features": node_features,
        "graph_edge_index": graph_edge_index,
        "labeled_edge_index": labeled_edge_index,
        "labels": labels,
        "degree": degree,
        "edge_midpoints": edge_midpoints,
        "train_np": train_np,
        "validation_np": validation_np,
        "test_np": test_np,
    }


def compute_global_feature_statistics(
    arrays_list: list[dict[str, np.ndarray]],
) -> tuple[torch.Tensor, torch.Tensor]:
    total_count = 0
    total_sum: np.ndarray | None = None
    total_square_sum: np.ndarray | None = None
    for arrays in arrays_list:
        features = arrays["node_features"].astype(np.float64)
        current_sum = features.sum(axis=0)
        current_square_sum = np.square(features).sum(axis=0)
        total_sum = current_sum if total_sum is None else total_sum + current_sum
        total_square_sum = current_square_sum if total_square_sum is None else total_square_sum + current_square_sum
        total_count += features.shape[0]
    if total_count == 0 or total_sum is None or total_square_sum is None:
        raise ValueError("No node features available for training.")
    mean = total_sum / total_count
    variance = np.maximum(total_square_sum / total_count - np.square(mean), 0.0)
    return (
        torch.from_numpy(mean.astype(np.float32)),
        torch.from_numpy(np.sqrt(variance).clip(min=1e-6).astype(np.float32)),
    )


def _aggregate_split_metrics(
    prepared_groups: list[dict[str, object]],
    model: EdgeGraphSAGE,
    loss_function: nn.Module,
    threshold: float,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object], dict[str, object]]:
    group_results: list[dict[str, object]] = []
    split_probabilities: dict[str, list[np.ndarray]] = {"train": [], "validation": [], "test": []}
    split_labels: dict[str, list[np.ndarray]] = {"train": [], "validation": [], "test": []}
    for group_index, group in enumerate(prepared_groups, start=1):
        arrays = group["arrays"]
        train_np = group["train_np"]
        validation_np = group["validation_np"]
        test_np = group["test_np"]
        for split_name, split_np in (("train", train_np), ("validation", validation_np), ("test", test_np)):
            split_labels[split_name].append(arrays["edge_labels"][split_np])
        train_loss, _, train_prob = evaluate_split(
            model, group["node_features"], group["graph_edge_index"], group["degree"],
            group["labeled_edge_index"], group["labels"], torch.from_numpy(train_np), loss_function
        )
        validation_loss, _, validation_prob = evaluate_split(
            model, group["node_features"], group["graph_edge_index"], group["degree"],
            group["labeled_edge_index"], group["labels"], torch.from_numpy(validation_np), loss_function
        )
        test_loss, _, test_prob = evaluate_split(
            model, group["node_features"], group["graph_edge_index"], group["degree"],
            group["labeled_edge_index"], group["labels"], torch.from_numpy(test_np), loss_function
        )
        split_probabilities["train"].append(train_prob)
        split_probabilities["validation"].append(validation_prob)
        split_probabilities["test"].append(test_prob)
        group_results.append(
            {
                "group": group_index,
                "data_dir": str(group["data_dir"]),
                "split_sizes": {"train": len(train_np), "validation": len(validation_np), "test": len(test_np)},
                "split_spatial_distribution": {
                    name: spatial_distribution_metrics(group["edge_midpoints"], indices)
                    for name, indices in (("train", train_np), ("validation", validation_np), ("test", test_np))
                },
                "train": {"loss": train_loss},
                "validation": {"loss": validation_loss},
                "test": {"loss": test_loss},
            }
        )

    aggregate: dict[str, dict[str, object]] = {}
    for split_name in ("train", "validation", "test"):
        labels = np.concatenate(split_labels[split_name])
        probabilities = np.concatenate(split_probabilities[split_name])
        split_count = sum(len(values) for values in split_labels[split_name])
        weighted_loss = sum(
            float(group_result[split_name]["loss"]) * len(split_labels[split_name][index])
            for index, group_result in enumerate(group_results)
        ) / max(split_count, 1)
        aggregate[split_name] = {
            "loss": weighted_loss,
            **classification_metrics(labels, probabilities, threshold),
        }
        for group_result in group_results:
            group_result[split_name].update(
                classification_metrics(
                    split_labels[split_name][int(group_result["group"]) - 1],
                    split_probabilities[split_name][int(group_result["group"]) - 1],
                    threshold,
                )
            )
    return group_results, aggregate["train"], aggregate["validation"], aggregate["test"]


def train_multiple(args: argparse.Namespace) -> dict[str, object]:
    """Train one model sequentially over all configured graph datasets."""
    set_seed(args.seed)
    data_dirs = [Path(path).resolve() for path in args.data_dirs]
    if not data_dirs:
        raise ValueError("At least one GNN data directory is required.")
    arrays_list = [load_training_data(data_dir) for data_dir in data_dirs]
    feature_dimensions = {arrays["node_features"].shape[1] for arrays in arrays_list}
    if len(feature_dimensions) != 1:
        raise ValueError(f"All groups must have the same feature dimension, got {sorted(feature_dimensions)}")
    input_dim = int(next(iter(feature_dimensions)))
    expected_feature_dim = getattr(args, "expected_feature_dim", DEFAULT_EXPECTED_FEATURE_DIM)
    if expected_feature_dim is not None and input_dim != int(expected_feature_dim):
        raise ValueError(
            f"Expected {expected_feature_dim} angle features per node, but the configured GNN data "
            f"contain {input_dim}. The data are probably an old build; run build_gnn_dataset.py "
            f"again after confirming that raw_pic413-3-1n contains 396 images."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_mean, feature_std = compute_global_feature_statistics(arrays_list)
    prepared_groups = [
        prepare_group(data_dir, arrays, feature_mean, feature_std, args, args.seed + group_index)
        for group_index, (data_dir, arrays) in enumerate(zip(data_dirs, arrays_list))
    ]

    model = EdgeGraphSAGE(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_function = nn.BCEWithLogitsLoss()
    history: list[dict[str, object]] = []
    group_best_epochs: list[int] = []

    print(f"Configured graph groups: {len(prepared_groups)}")
    print(f"Model parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    for group_index, group in enumerate(prepared_groups, start=1):
        train_np = group["train_np"]
        validation_np = group["validation_np"]
        train_indices = torch.from_numpy(train_np)
        validation_indices = torch.from_numpy(validation_np)
        group_output = args.output_dir / f"group_{group_index}"
        group_output.mkdir(parents=True, exist_ok=True)
        np.savez(group_output / "split_indices.npz", train=train_np, validation=validation_np, test=group["test_np"])
        np.save(group_output / "labeled_edge_midpoints.npy", group["edge_midpoints"], allow_pickle=False)

        best_state: dict[str, torch.Tensor] | None = None
        best_validation_loss = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        print(
            f"\n=== Training group {group_index}: {group['data_dir']} ===\n"
            f"Training edges: {len(train_np)}, validation: {len(validation_np)}, test: {len(group['test_np'])}"
        )
        for epoch in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad()
            embeddings = model.encode(group["node_features"], group["graph_edge_index"], group["degree"])
            train_logits = model.decode(embeddings, group["labeled_edge_index"][:, train_indices])
            train_loss = loss_function(train_logits, group["labels"][train_indices])
            train_loss.backward()
            optimizer.step()

            validation_loss, validation_metrics, _ = evaluate_split(
                model, group["node_features"], group["graph_edge_index"], group["degree"],
                group["labeled_edge_index"], group["labels"], validation_indices, loss_function
            )
            history.append(
                {
                    "group": group_index,
                    "epoch": epoch,
                    "train_loss": float(train_loss.item()),
                    "validation_loss": validation_loss,
                    "validation_f1": float(validation_metrics["f1"]),
                    "validation_pr_auc": validation_metrics["pr_auc"],
                }
            )
            if validation_loss < best_validation_loss - 1e-5:
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epoch == 1 or epoch % 10 == 0:
                print(
                    f"group={group_index} epoch={epoch:03d} train_loss={train_loss.item():.4f} "
                    f"val_loss={validation_loss:.4f} val_f1={validation_metrics['f1']:.4f} "
                    f"val_pr_auc={validation_metrics['pr_auc']}",
                    flush=True,
                )
            if epochs_without_improvement >= args.patience:
                print(f"Group {group_index} early stopping at epoch {epoch}.")
                break
        if best_state is None:
            raise RuntimeError(f"Training did not produce a valid state for group {group_index}.")
        model.load_state_dict(best_state)
        group_best_epochs.append(best_epoch)

    model.eval()
    with torch.no_grad():
        group_results, train_metrics, validation_metrics, test_metrics = _aggregate_split_metrics(
            prepared_groups, model, loss_function, threshold=0.5
        )
    validation_labels = np.concatenate([group["arrays"]["edge_labels"][group["validation_np"]] for group in prepared_groups])
    validation_probabilities = []
    for group in prepared_groups:
        _, _, probabilities = evaluate_split(
            model, group["node_features"], group["graph_edge_index"], group["degree"],
            group["labeled_edge_index"], group["labels"], torch.from_numpy(group["validation_np"]), loss_function
        )
        validation_probabilities.append(probabilities)
    optimal_threshold, best_validation_f1 = select_f1_threshold(
        validation_labels, np.concatenate(validation_probabilities)
    )
    _, train_metrics, validation_metrics, test_metrics = _aggregate_split_metrics(
        prepared_groups, model, loss_function, threshold=optimal_threshold
    )

    all_probability_blocks: list[np.ndarray] = []
    all_prediction_blocks: list[np.ndarray] = []
    test_probability_blocks: list[np.ndarray] = []
    for group_index, group in enumerate(prepared_groups, start=1):
        with torch.no_grad():
            embeddings = model.encode(group["node_features"], group["graph_edge_index"], group["degree"])
            all_edges = torch.from_numpy(group["arrays"]["undirected_edges"].astype(np.int64).T)
            all_probabilities = torch.sigmoid(model.decode(embeddings, all_edges)).cpu().numpy()
        all_probability_blocks.append(all_probabilities)
        all_prediction_blocks.append((all_probabilities >= optimal_threshold).astype(np.int8))
        _, _, test_probabilities = evaluate_split(
            model, group["node_features"], group["graph_edge_index"], group["degree"],
            group["labeled_edge_index"], group["labels"], torch.from_numpy(group["test_np"]), loss_function
        )
        test_probability_blocks.append(test_probabilities)
        np.save(args.output_dir / f"all_edge_probabilities_group_{group_index}.npy", all_probabilities, allow_pickle=False)
        np.save(args.output_dir / f"all_edge_predictions_group_{group_index}.npy", (all_probabilities >= optimal_threshold).astype(np.int8), allow_pickle=False)
        np.save(args.output_dir / f"test_probabilities_group_{group_index}.npy", test_probabilities, allow_pickle=False)

    np.save(args.output_dir / "all_edge_probabilities.npy", np.concatenate(all_probability_blocks), allow_pickle=False)
    np.save(args.output_dir / "all_edge_predictions.npy", np.concatenate(all_prediction_blocks), allow_pickle=False)
    np.save(args.output_dir / "test_probabilities.npy", np.concatenate(test_probability_blocks), allow_pickle=False)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "best_epoch": group_best_epochs[-1],
        "best_epoch_per_group": group_best_epochs,
        "decision_threshold": optimal_threshold,
        "threshold_selection_metric": "validation_f1",
        "data_dirs": [str(path) for path in data_dirs],
    }
    torch.save(checkpoint, args.output_dir / "best_model.pt")

    with (args.output_dir / "training_history.csv").open("w", encoding="utf-8", newline="") as file:
        fieldnames = ["group", "epoch", "train_loss", "validation_loss", "validation_f1", "validation_pr_auc"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    results: dict[str, object] = {
        "architecture": f"{args.num_layers}-layer mean GraphSAGE + symmetric MLP edge decoder",
        "seed": args.seed,
        "data_groups": [str(path) for path in data_dirs],
        "best_epoch_per_group": group_best_epochs,
        "split_strategy": args.split_strategy,
        "hyperparameters": {
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_epochs_per_group": args.epochs,
            "patience": args.patience,
            "decision_threshold": optimal_threshold,
            "threshold_selection_metric": "validation_f1",
            "best_validation_f1_during_threshold_search": best_validation_f1,
        },
        "groups": group_results,
        "train": train_metrics,
        "validation": validation_metrics,
        "test": test_metrics,
        "warning": (
            "Each group's edge split uses randomized farthest-point sampling, so train/validation/test edges are spatially dispersed; "
            "graphs are still trained transductively and can share nodes across splits."
            if args.split_strategy == "spatial"
            else "Random edge split; spatial clustering may occur because --split-strategy=random was selected."
        ),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def train(args: argparse.Namespace) -> dict[str, object]:
    """Backward-compatible single-directory wrapper."""
    if not hasattr(args, "data_dirs"):
        args.data_dirs = [args.data_dir]
    return train_multiple(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GraphSAGE for SLIC adjacency-edge classification.")
    parser.add_argument(
        "--data-dir",
        dest="data_dirs",
        type=Path,
        action="append",
        default=None,
        help="GNN dataset directory; repeat this option for multiple groups.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split-strategy", choices=("spatial", "random"), default="spatial")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_MAX_EPOCHS,
        help="Safety ceiling for training; early stopping determines the actual epoch count.",
    )
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--expected-feature-dim",
        type=int,
        default=DEFAULT_EXPECTED_FEATURE_DIM,
        help="Required angular feature dimension for every graph group (default: 396).",
    )
    args = parser.parse_args()
    args.data_dirs = [path.resolve() for path in (args.data_dirs or [DEFAULT_DATA_DIR])]
    args.output_dir = args.output_dir.resolve()
    return args


def main() -> None:
    training_results = train_multiple(parse_args())
    print(json.dumps(training_results["test"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
