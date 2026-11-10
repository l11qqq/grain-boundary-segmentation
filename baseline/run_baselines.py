"""Reproducible edge-classification baselines for the SLIC grain graph.

All generated files are constrained to ``baseline/``.  The script reads the
existing SLIC/GNN artifacts without modifying them, trains a one-layer linear
neural network, evaluates a source-matched threshold rule, re-evaluates the
stored GNN predictions on the identical test split, and renders grain maps.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASELINE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASELINE_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str(BASELINE_DIR / ".mplconfig"))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from PIL import Image, ImageDraw
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch import nn


EPS = 1.0e-6
METHOD_ORDER = ["Threshold (20% relative RMSE)", "Linear NN", "GraphSAGE GNN"]
METHOD_SLUGS = {
    "Threshold (20% relative RMSE)": "threshold",
    "Linear NN": "linear_nn",
    "GraphSAGE GNN": "gnn",
}


def configure_method_names(relative_threshold: float) -> None:
    """Keep tables, plots, and reports honest when the threshold is overridden."""
    global METHOD_ORDER, METHOD_SLUGS
    threshold_name = f"Threshold ({relative_threshold:g}% relative RMSE)"
    METHOD_ORDER = [threshold_name, "Linear NN", "GraphSAGE GNN"]
    METHOD_SLUGS = {
        threshold_name: "threshold",
        "Linear NN": "linear_nn",
        "GraphSAGE GNN": "gnn",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=BASELINE_DIR / "results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--relative-threshold", type=float, default=20.0)
    parser.add_argument(
        "--feature-count",
        type=int,
        default=None,
        help=(
            "Use this many uniformly spaced angular channels for the threshold and linear baselines. "
            "The GNN reference remains unchanged. Omit to use all channels."
        ),
    )
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    return parser.parse_args()


def ensure_output_is_local(output_dir: Path) -> Path:
    resolved = output_dir.resolve()
    try:
        resolved.relative_to(BASELINE_DIR.resolve())
    except ValueError as exc:
        raise ValueError(f"Output directory must stay inside {BASELINE_DIR}: {resolved}") from exc
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs() -> dict[str, Any]:
    data_dir = PROJECT_DIR / "outputs" / "gnn_data_3-1"
    slic_dir = PROJECT_DIR / "outputs" / "slic_3-1"
    gnn_dir = PROJECT_DIR / "outputs" / "gnn_results_3-1"
    required = {
        "node_features": data_dir / "node_features.npy",
        "node_ids": data_dir / "node_ids.npy",
        "node_pixel_counts": data_dir / "node_pixel_counts.npy",
        "undirected_edges": data_dir / "undirected_edges.npy",
        "labeled_edges": data_dir / "edge_label_index.npy",
        "edge_labels": data_dir / "edge_labels.npy",
        "all_edge_labels": data_dir / "all_edge_labels.npy",
        "slic_labels": slic_dir / "slic_labels.npy",
        "std_image": slic_dir / "std_image.png",
        "splits": gnn_dir / "group_1" / "split_indices.npz",
        "gnn_test_scores": gnn_dir / "test_probabilities_group_1.npy",
        "gnn_all_scores": gnn_dir / "all_edge_probabilities_group_1.npy",
        "gnn_metrics": gnn_dir / "metrics.json",
        "gnn_checkpoint": gnn_dir / "best_model.pt",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required source files:\n" + "\n".join(missing))

    arrays = {
        key: np.load(path, allow_pickle=False)
        for key, path in required.items()
        if path.suffix == ".npy"
    }
    split_archive = np.load(required["splits"], allow_pickle=False)
    arrays["splits"] = {name: split_archive[name].astype(np.int64) for name in split_archive.files}
    split_archive.close()
    arrays["gnn_metrics"] = json.loads(required["gnn_metrics"].read_text(encoding="utf-8"))
    arrays["paths"] = required
    validate_inputs(arrays)
    return arrays


def validate_inputs(data: dict[str, Any]) -> None:
    node_features = data["node_features"]
    node_ids = data["node_ids"]
    undirected_edges = data["undirected_edges"]
    labeled_edges = data["labeled_edges"]
    labels = data["edge_labels"]
    splits = data["splits"]
    if node_features.ndim != 2 or node_features.shape[0] != len(node_ids):
        raise ValueError("node_features/node_ids shape mismatch")
    if undirected_edges.ndim != 2 or undirected_edges.shape[1] != 2:
        raise ValueError("undirected_edges must have shape [E, 2]")
    if labeled_edges.shape != (2, len(labels)):
        raise ValueError("edge_label_index must have shape [2, number of labels]")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("edge_labels must contain both binary classes")
    if data["gnn_test_scores"].shape != (len(splits["test"]),):
        raise ValueError("Stored GNN test probabilities do not align with the test split")
    if data["gnn_all_scores"].shape != (len(undirected_edges),):
        raise ValueError("Stored GNN all-edge probabilities do not align with undirected_edges")
    all_split_indices = np.concatenate([splits[name] for name in ("train", "validation", "test")])
    if len(np.unique(all_split_indices)) != len(labels) or set(all_split_indices) != set(range(len(labels))):
        raise ValueError("Train/validation/test split is not a disjoint partition of labeled edges")
    if not np.array_equal(node_ids, np.arange(len(node_ids), dtype=node_ids.dtype)):
        raise ValueError("This baseline expects the saved contiguous node IDs used by the GNN data")
    slic_labels = data["slic_labels"]
    if int(slic_labels.min()) != 0 or int(slic_labels.max()) != len(node_ids) - 1:
        raise ValueError("SLIC label values do not align with GNN node rows")

    # Ensure CSV-order labeled pairs really are graph edges.  Split indices and
    # GNN test probabilities index this exact order, so this is a key fairness check.
    edge_set = {tuple(map(int, pair)) for pair in undirected_edges}
    for pair in labeled_edges.T:
        normalized = (int(min(pair)), int(max(pair)))
        if normalized not in edge_set:
            raise ValueError(f"Labeled edge {normalized} is not in the SLIC adjacency graph")


def mean_normalized_node_features(
    node_features: np.ndarray,
    pixel_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Match segment_grains_* NORMALIZE='mean' without rereading 396 images."""
    weights = pixel_counts.astype(np.float64)
    global_image_means = np.average(node_features.astype(np.float64), axis=0, weights=weights)
    normalized = node_features.astype(np.float32) / (global_image_means.astype(np.float32) + EPS)
    return normalized, global_image_means.astype(np.float32)


def relative_rmse_scores(
    normalized_nodes: np.ndarray,
    edges: np.ndarray,
    batch_size: int = 4096,
) -> np.ndarray:
    """Relative vector distance (%) used by the source threshold segmenter."""
    scores = np.empty(len(edges), dtype=np.float32)
    feature_count = normalized_nodes.shape[1]
    for start in range(0, len(edges), batch_size):
        stop = min(start + batch_size, len(edges))
        edge_batch = edges[start:stop]
        source = normalized_nodes[edge_batch[:, 0]]
        target = normalized_nodes[edge_batch[:, 1]]
        distance = np.sqrt(np.sum(np.square(source - target), axis=1) / feature_count)
        source_norm = np.sqrt(np.sum(np.square(source), axis=1) / feature_count)
        target_norm = np.sqrt(np.sum(np.square(target), axis=1) / feature_count)
        scores[start:stop] = distance / (0.5 * (source_norm + target_norm) + EPS) * 100.0
    return scores


def absolute_difference_features(normalized_nodes: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.abs(normalized_nodes[edges[:, 0]] - normalized_nodes[edges[:, 1]]).astype(
        np.float32,
        copy=False,
    )


class LinearBoundaryNet(nn.Module):
    """One affine layer; sigmoid is applied only for probabilities/evaluation."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(1)


def select_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if len(thresholds) == 0:
        return 0.5, 0.0
    f1_values = 2.0 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = int(np.argmax(f1_values))
    return float(thresholds[best]), float(f1_values[best])


def train_linear_model(
    normalized_nodes: np.ndarray,
    labeled_edges: np.ndarray,
    labels: np.ndarray,
    splits: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[LinearBoundaryNet, dict[str, Any], np.ndarray, np.ndarray]:
    edge_features = absolute_difference_features(normalized_nodes, labeled_edges.T)
    train_indices = splits["train"]
    validation_indices = splits["validation"]
    feature_mean = edge_features[train_indices].mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_std = edge_features[train_indices].std(axis=0, dtype=np.float64).astype(np.float32)
    feature_std[feature_std < 1.0e-6] = 1.0
    standardized = (edge_features - feature_mean) / feature_std

    x = torch.from_numpy(standardized)
    y = torch.from_numpy(labels.astype(np.float32))
    train_tensor = torch.from_numpy(train_indices)
    validation_tensor = torch.from_numpy(validation_indices)
    model = LinearBoundaryNet(edge_features.shape[1])
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_validation_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_logits = model(x[train_tensor])
        train_loss = loss_function(train_logits, y[train_tensor])
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_logits = model(x[validation_tensor])
            validation_loss = float(loss_function(validation_logits, y[validation_tensor]).item())
            validation_probabilities = torch.sigmoid(validation_logits).numpy()
            validation_f1 = f1_score(
                labels[validation_indices],
                validation_probabilities >= 0.5,
                zero_division=0,
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss.item()),
                "validation_loss": validation_loss,
                "validation_f1_at_0.5": float(validation_f1),
            }
        )
        if validation_loss < best_validation_loss - 1.0e-7:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            break
    training_seconds = time.perf_counter() - start_time
    if best_state is None:
        raise RuntimeError("Linear model training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        labeled_probabilities = torch.sigmoid(model(x)).numpy().astype(np.float32)
    decision_threshold, best_validation_f1 = select_f1_threshold(
        labels[validation_indices],
        labeled_probabilities[validation_indices],
    )
    training = {
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "best_validation_loss": best_validation_loss,
        "best_validation_f1_after_threshold_search": best_validation_f1,
        "decision_threshold": decision_threshold,
        "training_seconds": training_seconds,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
    }
    return model, training, feature_mean, feature_std


@torch.no_grad()
def predict_linear_all_edges(
    model: LinearBoundaryNet,
    normalized_nodes: np.ndarray,
    edges: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    batch_size: int = 4096,
) -> tuple[np.ndarray, float]:
    probabilities = np.empty(len(edges), dtype=np.float32)
    start_time = time.perf_counter()
    for start in range(0, len(edges), batch_size):
        stop = min(start + batch_size, len(edges))
        descriptors = absolute_difference_features(normalized_nodes, edges[start:stop])
        descriptors = (descriptors - feature_mean) / feature_std
        probabilities[start:stop] = torch.sigmoid(model(torch.from_numpy(descriptors))).numpy()
    return probabilities, time.perf_counter() - start_time


def classification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    predictions = predictions.astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    specificity = tn / max(tn + fp, 1)
    iou = tp / max(tp + fp + fn, 1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "boundary_iou": float(iou),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def fast_binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> tuple[float, float, float, float]:
    labels_bool = labels.astype(bool)
    predictions_bool = predictions.astype(bool)
    tp = int(np.sum(labels_bool & predictions_bool))
    tn = int(np.sum(~labels_bool & ~predictions_bool))
    fp = int(np.sum(~labels_bool & predictions_bool))
    fn = int(np.sum(labels_bool & ~predictions_bool))
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    balanced = 0.5 * (sensitivity + specificity)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    denominator = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
    mcc = (tp * tn - fp * fn) / denominator
    return accuracy, balanced, f1, mcc


def bootstrap_intervals(
    labels: np.ndarray,
    predictions: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    values = np.empty((repeats, 4), dtype=np.float64)
    for repeat in range(repeats):
        indices = rng.integers(0, len(labels), size=len(labels))
        values[repeat] = fast_binary_metrics(labels[indices], predictions[indices])
    names = ["accuracy", "balanced_accuracy", "f1", "mcc"]
    return {
        name: [float(bound) for bound in np.percentile(values[:, column], [2.5, 97.5])]
        for column, name in enumerate(names)
    }


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.uint8)

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[item] != item:
            next_item = int(self.parent[item])
            self.parent[item] = root
            item = next_item
        return root

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


def merge_superpixels(
    num_nodes: int,
    edges: np.ndarray,
    boundary_predictions: np.ndarray,
) -> tuple[np.ndarray, int]:
    union_find = UnionFind(num_nodes)
    for (source, target), is_boundary in zip(edges, boundary_predictions, strict=True):
        if not bool(is_boundary):
            union_find.union(int(source), int(target))
    roots = np.fromiter((union_find.find(node) for node in range(num_nodes)), dtype=np.int64)
    _, grain_ids = np.unique(roots, return_inverse=True)
    inconsistent = int(
        np.sum(
            (boundary_predictions.astype(bool))
            & (grain_ids[edges[:, 0]] == grain_ids[edges[:, 1]])
        )
    )
    return grain_ids.astype(np.int32), inconsistent


def boundary_mask(label_map: np.ndarray) -> np.ndarray:
    result = np.zeros(label_map.shape, dtype=bool)
    horizontal = label_map[:, 1:] != label_map[:, :-1]
    vertical = label_map[1:, :] != label_map[:-1, :]
    result[:, :-1] |= horizontal
    result[:, 1:] |= horizontal
    result[:-1, :] |= vertical
    result[1:, :] |= vertical
    return result


def save_segmentation_artifacts(
    method_dir: Path,
    slic_labels: np.ndarray,
    node_grain_ids: np.ndarray,
    std_image_path: Path,
    seed: int,
) -> dict[str, int]:
    method_dir.mkdir(parents=True, exist_ok=True)
    grain_map = node_grain_ids[slic_labels]
    np.save(method_dir / "grain_labels.npy", grain_map.astype(np.int32), allow_pickle=False)
    boundaries = boundary_mask(grain_map)
    grain_count = int(node_grain_ids.max()) + 1
    rng = np.random.default_rng(seed)
    colors = rng.integers(35, 256, size=(grain_count, 3), dtype=np.uint8)
    color_image = colors[grain_map]
    color_image[boundaries] = 0
    Image.fromarray(color_image, mode="RGB").save(method_dir / "segmentation_color.png")

    std_image = np.asarray(Image.open(std_image_path).convert("L"))
    if std_image.shape != grain_map.shape:
        std_image = cv2.resize(std_image, (grain_map.shape[1], grain_map.shape[0]), interpolation=cv2.INTER_LINEAR)
    background = np.repeat(std_image[:, :, None], 3, axis=2)
    overlay = (0.52 * color_image.astype(np.float32) + 0.48 * background.astype(np.float32)).astype(np.uint8)
    overlay[boundaries] = 0
    Image.fromarray(overlay, mode="RGB").save(method_dir / "segmentation_overlay.png")
    binary = np.full(grain_map.shape, 255, dtype=np.uint8)
    binary[boundaries] = 0
    Image.fromarray(binary, mode="L").save(method_dir / "boundary_map.png")
    return {"grain_count": grain_count, "boundary_pixel_count": int(boundaries.sum())}


def save_training_curve(history: list[dict[str, Any]], path: Path) -> None:
    frame = pd.DataFrame(history)
    frame.to_csv(path.with_suffix(".csv"), index=False)
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.plot(frame["epoch"], frame["train_loss"], label="Train BCE", linewidth=1.8)
    axis.plot(frame["epoch"], frame["validation_loss"], label="Validation BCE", linewidth=1.8)
    axis.set(xlabel="Epoch", ylabel="Loss", title="Linear neural network training")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_metric_bar_chart(metrics_by_method: dict[str, dict[str, Any]], path: Path) -> None:
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "mcc",
        "roc_auc",
        "pr_auc",
    ]
    display_names = [
        "Accuracy",
        "Balanced acc.",
        "Precision",
        "Recall",
        "F1",
        "MCC",
        "ROC-AUC",
        "PR-AUC",
    ]
    x = np.arange(len(metric_names))
    width = 0.25
    figure, axis = plt.subplots(figsize=(13.0, 5.4))
    for index, method in enumerate(METHOD_ORDER):
        values = [metrics_by_method[method][name] for name in metric_names]
        bars = axis.bar(x + (index - 1) * width, values, width, label=method)
        axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=7, rotation=90)
    axis.set_xticks(x, display_names)
    axis.set_ylim(0, 1.09)
    axis.set_ylabel("Test-set score")
    axis.set_title("Baseline comparison on the identical spatial test split (n=1,230)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="lower right", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=190)
    plt.close(figure)


def save_roc_pr_curves(
    labels: np.ndarray,
    scores_by_method: dict[str, np.ndarray],
    metrics_by_method: dict[str, dict[str, Any]],
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.7))
    for method in METHOD_ORDER:
        scores = scores_by_method[method]
        fpr, tpr, _ = roc_curve(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        axes[0].plot(fpr, tpr, linewidth=1.8, label=f"{method} ({metrics_by_method[method]['roc_auc']:.3f})")
        axes[1].plot(recall, precision, linewidth=1.8, label=f"{method} ({metrics_by_method[method]['pr_auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="0.6", linewidth=1)
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC curves")
    positive_rate = float(labels.mean())
    axes[1].axhline(positive_rate, linestyle="--", color="0.6", linewidth=1)
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7.5)
    figure.tight_layout()
    figure.savefig(path, dpi=190)
    plt.close(figure)


def save_confusion_matrices(metrics_by_method: dict[str, dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    for axis, method in zip(axes, METHOD_ORDER, strict=True):
        confusion = metrics_by_method[method]["confusion_matrix"]
        matrix = np.array([[confusion["tn"], confusion["fp"]], [confusion["fn"], confusion["tp"]]])
        axis.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=12)
        axis.set_xticks([0, 1], ["same", "boundary"])
        axis.set_yticks([0, 1], ["same", "boundary"])
        axis.set(xlabel="Predicted", ylabel="Ground truth", title=method)
    figure.tight_layout()
    figure.savefig(path, dpi=190)
    plt.close(figure)


def save_segmentation_comparison(output_dir: Path, path: Path) -> None:
    images: list[Image.Image] = []
    target_width = 850
    title_height = 54
    for method in METHOD_ORDER:
        source = output_dir / METHOD_SLUGS[method] / "segmentation_overlay.png"
        image = Image.open(source).convert("RGB")
        target_height = round(image.height * target_width / image.width)
        image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (target_width, target_height + title_height), "white")
        canvas.paste(image, (0, title_height))
        draw = ImageDraw.Draw(canvas)
        draw.text((14, 17), method, fill="black")
        images.append(canvas)
    comparison = Image.new("RGB", (target_width, sum(image.height for image in images)), "white")
    y = 0
    for image in images:
        comparison.paste(image, (0, y))
        y += image.height
    comparison.save(path)


def format_ci(interval: list[float]) -> str:
    return f"[{interval[0]:.4f}, {interval[1]:.4f}]"


def write_markdown_table(frame: pd.DataFrame, path: Path) -> None:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_change_vs_standard(
    current: pd.DataFrame,
    standard_path: Path,
    output_dir: Path,
) -> None:
    """Document the effect of a reduced-feature stress test against the fair baseline."""
    standard = pd.read_csv(standard_path).set_index("Method")
    current_indexed = current.set_index("Method")
    metric_names = [
        "Accuracy",
        "Balanced Accuracy",
        "Precision",
        "Recall",
        "F1",
        "MCC",
        "ROC-AUC",
        "PR-AUC",
    ]
    rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER[:2]:
        standard_method = method
        if standard_method not in standard.index and method.startswith("Threshold ("):
            threshold_rows = [name for name in standard.index if str(name).startswith("Threshold (")]
            if len(threshold_rows) != 1:
                raise ValueError("Could not identify the standard threshold row")
            standard_method = threshold_rows[0]
        for metric in metric_names:
            standard_value = float(standard.loc[standard_method, metric])
            stress_value = float(current_indexed.loc[method, metric])
            rows.append(
                {
                    "Method": method,
                    "Metric": metric,
                    "Standard baseline": f"{standard_value:.4f}",
                    "8-channel stress test": f"{stress_value:.4f}",
                    "Delta": f"{stress_value - standard_value:+.4f}",
                }
            )
    changes = pd.DataFrame(rows)
    changes.to_csv(output_dir / "change_vs_standard.csv", index=False, encoding="utf-8-sig")
    write_markdown_table(changes, output_dir / "change_vs_standard.md")


def save_threshold_change(
    current: pd.DataFrame,
    reference_path: Path,
    output_dir: Path,
    reference_threshold: float,
    current_threshold: float,
) -> None:
    """Compare two threshold settings while holding the feature budget fixed."""
    reference = pd.read_csv(reference_path)
    reference_rows = reference[reference["Method"].astype(str).str.startswith("Threshold (")]
    current_rows = current[current["Method"].astype(str).str.startswith("Threshold (")]
    if len(reference_rows) != 1 or len(current_rows) != 1:
        raise ValueError("Expected one threshold row in each comparison table")
    reference_row = reference_rows.iloc[0]
    current_row = current_rows.iloc[0]
    metric_names = [
        "Accuracy",
        "Balanced Accuracy",
        "Precision",
        "Recall",
        "Specificity",
        "F1",
        "Boundary IoU",
        "MCC",
        "ROC-AUC",
        "PR-AUC",
    ]
    rows: list[dict[str, str]] = []
    for metric in metric_names:
        reference_value = float(reference_row[metric])
        current_value = float(current_row[metric])
        rows.append(
            {
                "Metric": metric,
                f"{reference_threshold:g}% threshold": f"{reference_value:.4f}",
                f"{current_threshold:g}% threshold": f"{current_value:.4f}",
                "Delta": f"{current_value - reference_value:+.4f}",
            }
        )
    changes = pd.DataFrame(rows)
    changes.to_csv(output_dir / "threshold_change_vs_20pct.csv", index=False, encoding="utf-8-sig")
    write_markdown_table(changes, output_dir / "threshold_change_vs_20pct.md")


def mcnemar_comparisons(
    labels: np.ndarray,
    predictions_by_method: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for first_index, first in enumerate(METHOD_ORDER):
        for second in METHOD_ORDER[first_index + 1 :]:
            first_correct = predictions_by_method[first] == labels
            second_correct = predictions_by_method[second] == labels
            first_only = int(np.sum(first_correct & ~second_correct))
            second_only = int(np.sum(~first_correct & second_correct))
            discordant = first_only + second_only
            p_value = float(binomtest(min(first_only, second_only), discordant, 0.5).pvalue) if discordant else 1.0
            rows.append(
                {
                    "method_a": first,
                    "method_b": second,
                    "a_only_correct": first_only,
                    "b_only_correct": second_only,
                    "discordant_edges": discordant,
                    "exact_mcnemar_p": p_value,
                }
            )
    return pd.DataFrame(rows)


def gnn_parameter_count(checkpoint_path: Path) -> int:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return int(sum(tensor.numel() for tensor in checkpoint["model_state_dict"].values()))


def build_report(
    output_dir: Path,
    comparison: pd.DataFrame,
    pairwise: pd.DataFrame,
    data: dict[str, Any],
    linear_training: dict[str, Any],
    selected_feature_count: int,
    total_feature_count: int,
    bootstrap_repeats: int,
    relative_threshold: float,
) -> None:
    best_f1_row = comparison.iloc[int(comparison["F1"].astype(float).argmax())]
    test_labels = data["edge_labels"][data["splits"]["test"]]
    is_reduced_feature_test = selected_feature_count < total_feature_count
    report_title = (
        "# SLIC 边界分割低信息量压力测试报告"
        if is_reduced_feature_test
        else "# SLIC 边界分割 baseline 对比报告"
    )
    warning_lines = (
        [
            "",
            "## 重要说明",
            "",
            f"本实验是低信息量消融/压力测试：阈值法和线性网络只使用均匀抽取的 {selected_feature_count}/{total_feature_count} 个角度通道，而已有 GNN 仍使用全部 {total_feature_count} 个通道。",
            "因此该结果用于敏感性分析和欠拟合行为展示，不应替代使用相同输入信息量的公平主 baseline。",
        ]
        if is_reduced_feature_test
        else []
    )
    relative_output = output_dir.relative_to(PROJECT_DIR)
    reproduction_parts = [
        "python",
        "baseline\\run_baselines.py",
        "--output-dir",
        str(relative_output).replace("/", "\\"),
    ]
    if selected_feature_count < total_feature_count:
        reproduction_parts.extend(["--feature-count", str(selected_feature_count)])
    if not math.isclose(relative_threshold, 20.0):
        reproduction_parts.extend(["--relative-threshold", f"{relative_threshold:g}"])
    reproduction_command = " ".join(reproduction_parts)
    change_path = output_dir / "change_vs_standard.md"
    change_lines = (
        [
            "",
            "## 相对标准 baseline 的变化",
            "",
            change_path.read_text(encoding="utf-8").strip(),
            "",
            "负值表示低信息量压力测试的指标低于使用全部396通道的标准 baseline。",
        ]
        if change_path.is_file()
        else []
    )
    threshold_change_path = output_dir / "threshold_change_vs_20pct.md"
    threshold_change_lines = (
        [
            "",
            "## 相对20%阈值的变化",
            "",
            threshold_change_path.read_text(encoding="utf-8").strip(),
            "",
            "提高二值判定阈值不会改变连续距离分数的排序，因此 ROC-AUC 和 PR-AUC 理论上保持不变；主要变化体现在 Precision、Recall 及混淆矩阵相关指标。",
        ]
        if threshold_change_path.is_file()
        else []
    )
    lines = [
        report_title,
        *warning_lines,
        "",
        "## 结论",
        "",
        f"三种方法均在同一个空间分散测试集上评估（n={len(test_labels)}，边界边={int(test_labels.sum())}，非边界边={int((test_labels == 0).sum())}）。",
        f"测试集 F1 最高的是 **{best_f1_row['Method']}**（F1={float(best_f1_row['F1']):.4f}，MCC={float(best_f1_row['MCC']):.4f}，PR-AUC={float(best_f1_row['PR-AUC']):.4f}）。",
        "",
        "## 方法",
        "",
        f"- **Threshold**：复现 `segment_grains_downsample_sliding_window_adaptive.py` 的核心距离规则。每个角度通道按全图均值归一化；相邻超像素使用 {selected_feature_count} 维均值向量的相对 RMSE；距离大于 {relative_threshold:g}% 判为晶界。该方法不使用人工标签训练。",
        f"- **Linear NN**：输入为两个相邻超像素 {selected_feature_count} 维归一化向量的绝对差，仅含 `Linear({selected_feature_count}, 1)` 和 sigmoid。只用 train 拟合权重，只用 validation 选择决策阈值，test 不参与调参。",
        "- **GraphSAGE GNN**：直接读取 `outputs/gnn_results` 中的已有测试概率、全边概率和验证集选择的阈值，在相同 test 索引上重新计算全部指标。",
        "",
        "## 测试集指标",
        "",
        (output_dir / "metrics_comparison.md").read_text(encoding="utf-8").strip(),
        "",
        f"其中 Boundary IoU 是正类（晶界边）的 Jaccard 指标；区间为 {bootstrap_repeats:,} 次非参数 bootstrap 的 95% CI。Accuracy 容易受类别比例影响，建议论文正文优先报告 Precision、Recall、F1、MCC、Balanced Accuracy 和 PR-AUC。",
        *threshold_change_lines,
        *change_lines,
        "",
        "## 配对显著性",
        "",
        (output_dir / "pairwise_mcnemar.md").read_text(encoding="utf-8").strip(),
        "",
        "McNemar 为同一批测试边上的双侧精确检验；p 值只描述两方法错误率差异，不代表分割的实际效应大小。",
        "",
        "## 可视化与限制",
        "",
        "- `segmentation_comparison.png`：三种方法把预测为同晶粒的相邻超像素做并查集合并后的结果。",
        "- 每种方法目录包含 `boundary_map.png`、`segmentation_color.png`、`segmentation_overlay.png`、全边分数/预测和 `grain_labels.npy`。",
        "- 当前人工真值是超像素邻接**边标签**，不是完整像素级实例真值，因此不能严谨计算像素 IoU、Boundary F-score、ARI 或 VI；这些指标需要额外的完整人工晶粒 mask。",
        "- 图分割通过局部“同晶粒”边的传递闭包合并，可能出现一条预测晶界被其他同类路径绕过的情况；各方法的 `inconsistent_boundary_edges` 已记录在 CSV/JSON 中。",
        "",
        "## 复现",
        "",
        "```powershell",
        reproduction_command,
        "```",
        "",
        f"线性模型训练到第 {linear_training['epochs_ran']} 轮停止，最佳轮次为 {linear_training['best_epoch']}，实际训练耗时 {linear_training['training_seconds']:.3f} s（本机 CPU）。",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_method_names(args.relative_threshold)
    output_dir = ensure_output_is_local(args.output_dir)
    set_seed(args.seed)
    data = load_inputs()
    labels = data["edge_labels"].astype(np.int64)
    splits = data["splits"]
    test_indices = splits["test"]
    test_labels = labels[test_indices]
    normalized_nodes_full, global_image_means = mean_normalized_node_features(
        data["node_features"], data["node_pixel_counts"]
    )
    total_feature_count = int(normalized_nodes_full.shape[1])
    selected_feature_count = total_feature_count if args.feature_count is None else int(args.feature_count)
    if not 1 <= selected_feature_count <= total_feature_count:
        raise ValueError(f"--feature-count must be between 1 and {total_feature_count}")
    feature_indices = np.linspace(
        0,
        total_feature_count - 1,
        selected_feature_count,
        dtype=np.int64,
    )
    if len(np.unique(feature_indices)) != selected_feature_count:
        raise ValueError("Uniform feature selection produced duplicate indices")
    normalized_nodes = normalized_nodes_full[:, feature_indices]

    # Threshold baseline: source-matched, fixed, and completely unsupervised.
    threshold_start = time.perf_counter()
    threshold_all_scores = relative_rmse_scores(normalized_nodes, data["undirected_edges"])
    threshold_seconds = time.perf_counter() - threshold_start
    threshold_labeled_scores = relative_rmse_scores(normalized_nodes, data["labeled_edges"].T)
    threshold_all_predictions = (threshold_all_scores > args.relative_threshold).astype(np.int8)
    threshold_test_predictions = (threshold_labeled_scores[test_indices] > args.relative_threshold).astype(np.int8)

    # One-layer linear neural network.
    linear_model, linear_training, feature_mean, feature_std = train_linear_model(
        normalized_nodes,
        data["labeled_edges"],
        labels,
        splits,
        args,
    )
    linear_all_scores, linear_inference_seconds = predict_linear_all_edges(
        linear_model,
        normalized_nodes,
        data["undirected_edges"],
        feature_mean,
        feature_std,
    )
    linear_threshold = float(linear_training["decision_threshold"])
    linear_all_predictions = (linear_all_scores >= linear_threshold).astype(np.int8)
    linear_labeled_scores, _ = predict_linear_all_edges(
        linear_model,
        normalized_nodes,
        data["labeled_edges"].T,
        feature_mean,
        feature_std,
    )
    linear_test_predictions = (linear_labeled_scores[test_indices] >= linear_threshold).astype(np.int8)

    # Existing GNN result, evaluated from saved probabilities on the exact same test indices.
    gnn_threshold = float(data["gnn_metrics"]["hyperparameters"]["decision_threshold"])
    gnn_test_scores = data["gnn_test_scores"].astype(np.float32)
    gnn_all_scores = data["gnn_all_scores"].astype(np.float32)
    gnn_test_predictions = (gnn_test_scores >= gnn_threshold).astype(np.int8)
    gnn_all_predictions = (gnn_all_scores >= gnn_threshold).astype(np.int8)

    scores_by_method = {
        METHOD_ORDER[0]: threshold_labeled_scores[test_indices],
        METHOD_ORDER[1]: linear_labeled_scores[test_indices],
        METHOD_ORDER[2]: gnn_test_scores,
    }
    predictions_by_method = {
        METHOD_ORDER[0]: threshold_test_predictions,
        METHOD_ORDER[1]: linear_test_predictions,
        METHOD_ORDER[2]: gnn_test_predictions,
    }
    all_scores_by_method = {
        METHOD_ORDER[0]: threshold_all_scores,
        METHOD_ORDER[1]: linear_all_scores,
        METHOD_ORDER[2]: gnn_all_scores,
    }
    all_predictions_by_method = {
        METHOD_ORDER[0]: threshold_all_predictions,
        METHOD_ORDER[1]: linear_all_predictions,
        METHOD_ORDER[2]: gnn_all_predictions,
    }
    metrics_by_method: dict[str, dict[str, Any]] = {}
    for method_index, method in enumerate(METHOD_ORDER):
        metrics = classification_metrics(test_labels, scores_by_method[method], predictions_by_method[method])
        metrics["bootstrap_95_ci"] = bootstrap_intervals(
            test_labels,
            predictions_by_method[method],
            args.bootstrap_repeats,
            args.seed + 100 + method_index,
        )
        metrics_by_method[method] = metrics

    parameter_counts = {
        METHOD_ORDER[0]: 0,
        METHOD_ORDER[1]: int(linear_training["parameter_count"]),
        METHOD_ORDER[2]: gnn_parameter_count(data["paths"]["gnn_checkpoint"]),
    }
    threshold_descriptions = {
        METHOD_ORDER[0]: f"> {args.relative_threshold:g} relative-RMSE %",
        METHOD_ORDER[1]: f">= {linear_threshold:.6f} probability",
        METHOD_ORDER[2]: f">= {gnn_threshold:.6f} probability",
    }

    segmentation_metadata: dict[str, dict[str, Any]] = {}
    for method_index, method in enumerate(METHOD_ORDER):
        slug = METHOD_SLUGS[method]
        method_dir = output_dir / slug
        method_dir.mkdir(parents=True, exist_ok=True)
        np.save(method_dir / "all_edge_scores.npy", all_scores_by_method[method], allow_pickle=False)
        np.save(method_dir / "all_edge_predictions.npy", all_predictions_by_method[method], allow_pickle=False)
        np.save(method_dir / "test_scores.npy", scores_by_method[method], allow_pickle=False)
        np.save(method_dir / "test_predictions.npy", predictions_by_method[method], allow_pickle=False)
        node_grain_ids, inconsistent = merge_superpixels(
            len(data["node_ids"]),
            data["undirected_edges"],
            all_predictions_by_method[method],
        )
        np.save(method_dir / "node_grain_ids.npy", node_grain_ids, allow_pickle=False)
        visual_metadata = save_segmentation_artifacts(
            method_dir,
            data["slic_labels"],
            node_grain_ids,
            data["paths"]["std_image"],
            args.seed + method_index,
        )
        visual_metadata["inconsistent_boundary_edges"] = inconsistent
        segmentation_metadata[method] = visual_metadata

    torch.save(
        {
            "model_state_dict": linear_model.state_dict(),
            "feature_mean": torch.from_numpy(feature_mean),
            "feature_std": torch.from_numpy(feature_std),
            "input_dim": int(normalized_nodes.shape[1]),
            "decision_threshold": linear_threshold,
            "seed": args.seed,
            "feature_definition": "abs(mean-normalized node_a - mean-normalized node_b)",
        },
        output_dir / "linear_nn" / "best_model.pt",
    )
    np.save(output_dir / "global_image_means.npy", global_image_means, allow_pickle=False)
    np.save(output_dir / "selected_feature_indices.npy", feature_indices, allow_pickle=False)
    save_training_curve(linear_training["history"], output_dir / "linear_nn" / "training_curve.png")

    rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        metrics = metrics_by_method[method]
        ci = metrics["bootstrap_95_ci"]
        rows.append(
            {
                "Method": method,
                "Accuracy": f"{metrics['accuracy']:.4f}",
                "Balanced Accuracy": f"{metrics['balanced_accuracy']:.4f}",
                "Precision": f"{metrics['precision']:.4f}",
                "Recall": f"{metrics['recall']:.4f}",
                "Specificity": f"{metrics['specificity']:.4f}",
                "F1": f"{metrics['f1']:.4f}",
                "Boundary IoU": f"{metrics['boundary_iou']:.4f}",
                "MCC": f"{metrics['mcc']:.4f}",
                "ROC-AUC": f"{metrics['roc_auc']:.4f}",
                "PR-AUC": f"{metrics['pr_auc']:.4f}",
                "F1 95% CI": format_ci(ci["f1"]),
                "MCC 95% CI": format_ci(ci["mcc"]),
                "Decision threshold": threshold_descriptions[method],
                "Trainable parameters": parameter_counts[method],
                "Merged grains": segmentation_metadata[method]["grain_count"],
                "Inconsistent boundary edges": segmentation_metadata[method]["inconsistent_boundary_edges"],
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "metrics_comparison.csv", index=False, encoding="utf-8-sig")
    write_markdown_table(comparison, output_dir / "metrics_comparison.md")
    standard_metrics_path = BASELINE_DIR / "results" / "metrics_comparison.csv"
    if selected_feature_count < total_feature_count and standard_metrics_path.is_file():
        save_change_vs_standard(comparison, standard_metrics_path, output_dir)
    threshold_reference_dir = BASELINE_DIR / "results_test"
    threshold_reference_metrics = threshold_reference_dir / "metrics_comparison.csv"
    threshold_reference_json = threshold_reference_dir / "metrics.json"
    if (
        not math.isclose(args.relative_threshold, 20.0)
        and threshold_reference_metrics.is_file()
        and threshold_reference_json.is_file()
    ):
        reference_metadata = json.loads(threshold_reference_json.read_text(encoding="utf-8"))
        reference_feature_count = int(reference_metadata["evaluation_protocol"]["baseline_feature_count"])
        if reference_feature_count == selected_feature_count:
            save_threshold_change(
                comparison,
                threshold_reference_metrics,
                output_dir,
                reference_threshold=20.0,
                current_threshold=args.relative_threshold,
            )

    pairwise = mcnemar_comparisons(test_labels, predictions_by_method)
    pairwise.to_csv(output_dir / "pairwise_mcnemar.csv", index=False, encoding="utf-8-sig")
    pairwise_display = pairwise.copy()
    pairwise_display["exact_mcnemar_p"] = pairwise_display["exact_mcnemar_p"].map(lambda value: f"{value:.6g}")
    write_markdown_table(pairwise_display, output_dir / "pairwise_mcnemar.md")

    save_metric_bar_chart(metrics_by_method, output_dir / "metrics_comparison.png")
    save_roc_pr_curves(test_labels, scores_by_method, metrics_by_method, output_dir / "roc_pr_curves.png")
    save_confusion_matrices(metrics_by_method, output_dir / "confusion_matrices.png")
    save_segmentation_comparison(output_dir, output_dir / "segmentation_comparison.png")

    stored_gnn_test = data["gnn_metrics"]["test"]
    gnn_metric_deltas = {
        name: float(metrics_by_method[METHOD_ORDER[2]][name] - stored_gnn_test[name])
        for name in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")
    }
    result_json = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_protocol": {
            "split_source": str(data["paths"]["splits"]),
            "split_strategy": data["gnn_metrics"].get("split_strategy"),
            "split_sizes": {name: int(len(indices)) for name, indices in splits.items()},
            "test_class_counts": {"same_grain_0": int(np.sum(test_labels == 0)), "boundary_1": int(np.sum(test_labels == 1))},
            "bootstrap_repeats": args.bootstrap_repeats,
            "seed": args.seed,
            "baseline_feature_count": selected_feature_count,
            "total_available_feature_count": total_feature_count,
            "selected_feature_indices": feature_indices,
            "reduced_feature_stress_test": selected_feature_count < total_feature_count,
        },
        "methods": {
            METHOD_ORDER[0]: {
                "definition": f"mean normalization + relative vector RMSE over {selected_feature_count} uniformly sampled channels",
                "decision_threshold": args.relative_threshold,
                "threshold_unit": "percent",
                "trainable_parameters": 0,
                "all_edge_inference_seconds": threshold_seconds,
            },
            METHOD_ORDER[1]: {
                "definition": f"abs node-feature difference -> Linear({selected_feature_count}, 1) -> sigmoid",
                "decision_threshold": linear_threshold,
                "trainable_parameters": parameter_counts[METHOD_ORDER[1]],
                "training": {key: value for key, value in linear_training.items() if key != "history"},
                "all_edge_inference_seconds": linear_inference_seconds,
            },
            METHOD_ORDER[2]: {
                "definition": data["gnn_metrics"].get("architecture"),
                "decision_threshold": gnn_threshold,
                "trainable_parameters": parameter_counts[METHOD_ORDER[2]],
                "source": str(data["paths"]["gnn_metrics"]),
                "recomputed_minus_stored_metric": gnn_metric_deltas,
            },
        },
        "test_metrics": metrics_by_method,
        "segmentation": segmentation_metadata,
        "limitations": [
            "Ground truth labels SLIC adjacency edges, not all pixels or complete grain instances.",
            "Pixel IoU, boundary F-score, ARI, and VI are therefore not reported.",
            "Merged grain counts are visualization outputs, not ground-truth-validated region metrics.",
        ],
        "provenance": {
            "source_files_sha256": {
                str(path.relative_to(PROJECT_DIR)): sha256(path)
                for path in (
                    data["paths"]["node_features"],
                    data["paths"]["labeled_edges"],
                    data["paths"]["edge_labels"],
                    data["paths"]["slic_labels"],
                    data["paths"]["splits"],
                    data["paths"]["gnn_test_scores"],
                    data["paths"]["gnn_all_scores"],
                    data["paths"]["gnn_metrics"],
                )
            },
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
                "torch": torch.__version__,
                "opencv": cv2.__version__,
                "matplotlib": matplotlib.__version__,
            },
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(json_ready(result_json), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    build_report(
        output_dir,
        comparison,
        pairwise,
        data,
        linear_training,
        selected_feature_count,
        total_feature_count,
        args.bootstrap_repeats,
        args.relative_threshold,
    )
    print(comparison.to_string(index=False))
    print(f"\nAll outputs saved under: {output_dir}")


if __name__ == "__main__":
    main()
