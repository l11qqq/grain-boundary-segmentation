from __future__ import annotations

import gc
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    from numba import njit
except Exception:  # pragma: no cover - used only when numba is unavailable.
    njit = None


"""
滑窗自适应阈值晶粒候选分割

这个版本和 segment_grains_local_window.py 的区别：
- 不再对整幅图的每个像素统一使用同一个阈值。
- 先按 WINDOW_SIZE / WINDOW_STRIDE 在图像上移动检测窗口。
- 每个窗口内部逐像素比较局部范围内的 L 维角度亮度向量。
- 每个窗口内部根据本窗口的距离分布自动估计阈值。
- 窗口内相似像素连边，所有窗口共用一个全局并查集，最后合并连通区域。

注意：
滑窗仍然会覆盖图像，只是检测和阈值估计发生在局部窗口内。
如果 WINDOW_STRIDE < WINDOW_SIZE，窗口会重叠，跨窗口区域可以通过重叠区连接起来。
"""


# =========================
# 可调参数区
# =========================

# 原始图像目录，要求文件名形如 pic_r_phi.bmp。
RAW_DIR = Path(__file__).resolve().parent / "raw_pic413-3-1n"

# 输出目录。
OUT_DIR = Path("output_sliding_window_adaptive")

# 滑窗大小。WINDOW_SIZE=96 表示每次处理 96 x 96 的局部窗口。
WINDOW_SIZE = 256

# 滑窗步长。步长小于窗口尺寸时会有重叠，边界连续性更好，但计算更慢。
WINDOW_STRIDE = 128

# 窗口内部每个像素只和该半径内的局部像素比较。
# LOCAL_COMPARE_RADIUS=5 表示每个像素最多比较 11 x 11 范围内的其它像素。
LOCAL_COMPARE_RADIUS = 9

# 每个角度通道的归一化方式：
# "zscore": 每张图减均值再除以标准差，推荐；
# "mean": 每张图除以均值；
# "none": 不做归一化，直接使用 0..255 灰度值。
NORMALIZE = "zscore"

# 向量距离：
# "rmse": sqrt(mean((vp-vq)^2))，推荐；
# "euclidean": sqrt(sum((vp-vq)^2))。
DISTANCE = "rmse"

# 窗口内阈值模式：
# "adaptive": 每个窗口用 Otsu / percentile 从距离分布估计绝对阈值；
# "relative_percent": 使用相对百分比阈值，distance / mean(norm(vp), norm(vq)) * 100。
THRESHOLD_MODE = "adaptive"

# THRESHOLD_MODE="relative_percent" 时使用。
# 例如 10.0 表示两个像素的角度亮度向量相对差异 <= 10% 就连边。
RELATIVE_DISTANCE_PERCENT = 10.0

# 每个窗口内部自适应阈值估计方法：
# "otsu": 对本窗口的局部距离分布做 Otsu 二分类；
# "percentile": 使用本窗口距离分布的 ADAPTIVE_PERCENTILE 分位数。
ADAPTIVE_THRESHOLD_METHOD = "otsu"
ADAPTIVE_PERCENTILE = 35.0

# 对每个窗口估计出的阈值做整体缩放。区域太碎可调大，区域粘连可调小。
ADAPTIVE_THRESHOLD_SCALE = 1.0

# 可选阈值下限 / 上限。None 表示不限制。
# 使用 zscore + rmse 时，可尝试 MIN_THRESHOLD=0.10 或 0.15 来避免均匀窗口阈值过低。
MIN_THRESHOLD = None
MAX_THRESHOLD = None

# 自动阈值直方图分箱数，仅 ADAPTIVE_THRESHOLD_METHOD="otsu" 时使用。
OTSU_BINS = 256

# 小碎片删除阈值。小于该像素数的区域会被置为背景 0。
MIN_REGION_SIZE = 20

# 是否保存原始灰度堆栈 stack.npy，shape=(L,H,W)，uint8。
SAVE_STACK = False

# 是否保留归一化后的 features.npy，shape=(H,W,L)，float32。
# 默认只作为临时文件使用，流程结束后删除。
KEEP_FEATURES = False

# 伪彩色随机种子，只影响显示颜色，不影响 labels.npy。
SEED = 12345

# 进度打印间隔。
PROGRESS_EVERY_IMAGES = 25
PROGRESS_EVERY_WINDOWS = 25


PIC_RE = re.compile(r"^pic_([-+]?\d+(?:\.\d+)?)_([-+]?\d+(?:\.\d+)?)\.bmp$", re.IGNORECASE)
EPS = 1.0e-6


def validate_config() -> None:
    if WINDOW_SIZE < 2:
        raise ValueError("WINDOW_SIZE must be >= 2")
    if WINDOW_STRIDE < 1:
        raise ValueError("WINDOW_STRIDE must be >= 1")
    if LOCAL_COMPARE_RADIUS < 1:
        raise ValueError("LOCAL_COMPARE_RADIUS must be >= 1")
    if NORMALIZE not in ("zscore", "mean", "none"):
        raise ValueError('NORMALIZE must be "zscore", "mean", or "none"')
    if DISTANCE not in ("rmse", "euclidean"):
        raise ValueError('DISTANCE must be "rmse" or "euclidean"')
    if THRESHOLD_MODE not in ("adaptive", "relative_percent"):
        raise ValueError('THRESHOLD_MODE must be "adaptive" or "relative_percent"')
    if RELATIVE_DISTANCE_PERCENT < 0:
        raise ValueError("RELATIVE_DISTANCE_PERCENT must be non-negative")
    if ADAPTIVE_THRESHOLD_METHOD not in ("otsu", "percentile"):
        raise ValueError('ADAPTIVE_THRESHOLD_METHOD must be "otsu" or "percentile"')
    if not 0.0 <= ADAPTIVE_PERCENTILE <= 100.0:
        raise ValueError("ADAPTIVE_PERCENTILE must be between 0 and 100")
    if ADAPTIVE_THRESHOLD_SCALE <= 0:
        raise ValueError("ADAPTIVE_THRESHOLD_SCALE must be > 0")
    if MIN_THRESHOLD is not None and MIN_THRESHOLD < 0:
        raise ValueError("MIN_THRESHOLD must be None or non-negative")
    if MAX_THRESHOLD is not None and MAX_THRESHOLD < 0:
        raise ValueError("MAX_THRESHOLD must be None or non-negative")
    if MIN_THRESHOLD is not None and MAX_THRESHOLD is not None and MIN_THRESHOLD > MAX_THRESHOLD:
        raise ValueError("MIN_THRESHOLD must be <= MAX_THRESHOLD")
    if OTSU_BINS <= 1:
        raise ValueError("OTSU_BINS must be > 1")
    if MIN_REGION_SIZE < 1:
        raise ValueError("MIN_REGION_SIZE must be >= 1")


def discover_images(raw_dir: Path) -> list[tuple[tuple[float, float], Path]]:
    files: list[tuple[tuple[float, float], Path]] = []
    for path in raw_dir.iterdir():
        match = PIC_RE.match(path.name)
        if not match:
            continue
        r = float(match.group(1))
        phi = float(match.group(2))
        files.append(((r, phi), path))

    if not files:
        raise FileNotFoundError(f"No files matching pic_r_phi.bmp were found in {raw_dir}")

    return sorted(files, key=lambda item: (item[0][0], item[0][1]))


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def normalize_channel(image_u8: np.ndarray) -> np.ndarray:
    image = image_u8.astype(np.float32)
    if NORMALIZE == "none":
        return image
    if NORMALIZE == "mean":
        return image / (float(image.mean()) + EPS)

    mean = float(image.mean())
    std = float(image.std())
    return (image - mean) / (std + EPS)


def build_features(
    image_items: list[tuple[tuple[float, float], Path]],
    out_dir: Path,
) -> tuple[np.memmap, np.ndarray, Path, tuple[int, int, int]]:
    first = read_gray(image_items[0][1])
    height, width = first.shape
    image_count = len(image_items)

    feature_path = out_dir / ("features.npy" if KEEP_FEATURES else "_features_tmp.npy")
    features = np.lib.format.open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float32,
        shape=(height, width, image_count),
    )

    stack = None
    if SAVE_STACK:
        stack = np.lib.format.open_memmap(
            out_dir / "stack.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(image_count, height, width),
        )

    positions = np.empty((image_count, 2), dtype=np.float32)
    for index, (position, path) in enumerate(image_items):
        image = first if index == 0 else read_gray(path)
        if image.shape != (height, width):
            raise ValueError(f"Image shape mismatch for {path}: expected {(height, width)}, got {image.shape}")

        positions[index] = position
        features[:, :, index] = normalize_channel(image)
        if stack is not None:
            stack[index] = image

        if (index + 1) % PROGRESS_EVERY_IMAGES == 0 or index + 1 == image_count:
            print(f"Built features from {index + 1}/{image_count} images")

    features.flush()
    if stack is not None:
        stack.flush()

    return features, positions, feature_path, (image_count, height, width)


def sliding_window_starts(length: int, window_size: int, stride: int) -> list[int]:
    if length <= window_size:
        return [0]

    starts = list(range(0, length - window_size + 1, stride))
    last = length - window_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def sliding_windows(height: int, width: int) -> list[tuple[int, int, int, int]]:
    ys = sliding_window_starts(height, WINDOW_SIZE, WINDOW_STRIDE)
    xs = sliding_window_starts(width, WINDOW_SIZE, WINDOW_STRIDE)
    windows = []
    for y0 in ys:
        for x0 in xs:
            windows.append((y0, min(y0 + WINDOW_SIZE, height), x0, min(x0 + WINDOW_SIZE, width)))
    return windows


def local_offsets(radius: int) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    for dx in range(1, radius + 1):
        offsets.append((0, dx))
    for dy in range(1, radius + 1):
        for dx in range(-radius, radius + 1):
            offsets.append((dy, dx))
    return offsets


def window_pair_slices(
    window_height: int,
    window_width: int,
    dy: int,
    dx: int,
) -> tuple[tuple[slice, slice, slice], tuple[slice, slice, slice]] | None:
    if dy >= window_height or abs(dx) >= window_width:
        return None

    if dy == 0:
        return (
            (slice(0, window_height), slice(0, window_width - dx), slice(None)),
            (slice(0, window_height), slice(dx, window_width), slice(None)),
        )

    if dx >= 0:
        return (
            (slice(0, window_height - dy), slice(0, window_width - dx), slice(None)),
            (slice(dy, window_height), slice(dx, window_width), slice(None)),
        )

    left = -dx
    return (
        (slice(0, window_height - dy), slice(left, window_width), slice(None)),
        (slice(dy, window_height), slice(0, window_width - left), slice(None)),
    )


def distance_for_offset(window_features: np.ndarray, dy: int, dx: int) -> np.ndarray | None:
    window_height, window_width, image_count = window_features.shape
    slices = window_pair_slices(window_height, window_width, dy, dx)
    if slices is None:
        return None

    slice_a, slice_b = slices
    diff = window_features[slice_a] - window_features[slice_b]
    sumsq = np.einsum("ijk,ijk->ij", diff, diff, optimize=True)
    if DISTANCE == "rmse":
        return np.sqrt(sumsq / image_count).astype(np.float32, copy=False)
    return np.sqrt(sumsq).astype(np.float32, copy=False)


def relative_percent_for_offset(window_features: np.ndarray, dy: int, dx: int) -> np.ndarray | None:
    window_height, window_width, image_count = window_features.shape
    slices = window_pair_slices(window_height, window_width, dy, dx)
    if slices is None:
        return None

    slice_a, slice_b = slices
    a = window_features[slice_a]
    b = window_features[slice_b]
    diff = a - b
    sumsq = np.einsum("ijk,ijk->ij", diff, diff, optimize=True)
    norm_a_sumsq = np.einsum("ijk,ijk->ij", a, a, optimize=True)
    norm_b_sumsq = np.einsum("ijk,ijk->ij", b, b, optimize=True)

    if DISTANCE == "rmse":
        distance = np.sqrt(sumsq / image_count)
        norm_a = np.sqrt(norm_a_sumsq / image_count)
        norm_b = np.sqrt(norm_b_sumsq / image_count)
    else:
        distance = np.sqrt(sumsq)
        norm_a = np.sqrt(norm_a_sumsq)
        norm_b = np.sqrt(norm_b_sumsq)

    denominator = 0.5 * (norm_a + norm_b) + EPS
    return (distance / denominator * 100.0).astype(np.float32, copy=False)


def otsu_from_arrays(distance_arrays: list[np.ndarray]) -> float:
    nonempty = [arr for arr in distance_arrays if arr.size]
    finite_min = min(float(np.nanmin(arr)) for arr in nonempty)
    finite_max = max(float(np.nanmax(arr)) for arr in nonempty)
    if finite_min == finite_max:
        return finite_min

    hist = np.zeros(OTSU_BINS, dtype=np.float64)
    for arr in nonempty:
        current, edges = np.histogram(arr, bins=OTSU_BINS, range=(finite_min, finite_max))
        hist += current

    centers = (edges[:-1] + edges[1:]) * 0.5
    total = hist.sum()
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    mean_bg = np.cumsum(hist * centers)
    mean_total = mean_bg[-1]

    valid = (weight_bg > 0) & (weight_fg > 0)
    score = np.zeros_like(centers)
    score[valid] = (mean_total * weight_bg[valid] - mean_bg[valid] * total) ** 2 / (
        weight_bg[valid] * weight_fg[valid]
    )
    return float(centers[int(np.argmax(score))])


def adaptive_threshold(distance_arrays: list[np.ndarray]) -> float:
    if not distance_arrays:
        return 0.0

    if ADAPTIVE_THRESHOLD_METHOD == "percentile":
        values = np.concatenate([arr.ravel() for arr in distance_arrays if arr.size])
        threshold = float(np.percentile(values, ADAPTIVE_PERCENTILE))
    else:
        threshold = otsu_from_arrays(distance_arrays)

    threshold *= ADAPTIVE_THRESHOLD_SCALE
    if MIN_THRESHOLD is not None:
        threshold = max(threshold, float(MIN_THRESHOLD))
    if MAX_THRESHOLD is not None:
        threshold = min(threshold, float(MAX_THRESHOLD))
    return threshold


if njit is not None:

    @njit(cache=True)
    def _find(parent: np.ndarray, index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    @njit(cache=True)
    def _union(parent: np.ndarray, size: np.ndarray, a: int, b: int) -> None:
        root_a = _find(parent, a)
        root_b = _find(parent, b)
        if root_a == root_b:
            return
        if size[root_a] < size[root_b]:
            root_a, root_b = root_b, root_a
        parent[root_b] = root_a
        size[root_a] += size[root_b]

    @njit(cache=True)
    def _union_window_offset_mask(
        parent: np.ndarray,
        size: np.ndarray,
        mask: np.ndarray,
        dy: int,
        dx: int,
        width: int,
        window_y0: int,
        window_x0: int,
    ) -> None:
        for local_y in range(mask.shape[0]):
            y = window_y0 + local_y
            if dy == 0:
                base_a = y * width + window_x0
                base_b = y * width + window_x0 + dx
            elif dx >= 0:
                base_a = y * width + window_x0
                base_b = (y + dy) * width + window_x0 + dx
            else:
                base_a = y * width + window_x0 - dx
                base_b = (y + dy) * width + window_x0

            for local_x in range(mask.shape[1]):
                if mask[local_y, local_x]:
                    _union(parent, size, base_a + local_x, base_b + local_x)

    @njit(cache=True)
    def _flatten(parent: np.ndarray) -> None:
        for index in range(parent.size):
            parent[index] = _find(parent, index)


def union_window_offset_mask(
    parent: np.ndarray,
    size: np.ndarray,
    mask: np.ndarray,
    dy: int,
    dx: int,
    width: int,
    window_y0: int,
    window_x0: int,
) -> None:
    if njit is not None:
        _union_window_offset_mask(parent, size, mask, dy, dx, width, window_y0, window_x0)
        return

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return int(index)

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a == root_b:
            return
        if size[root_a] < size[root_b]:
            root_a, root_b = root_b, root_a
        parent[root_b] = root_a
        size[root_a] += size[root_b]

    ys, xs = np.nonzero(mask)
    for local_y, local_x in zip(ys, xs, strict=True):
        y = window_y0 + int(local_y)
        x = int(local_x)
        if dy == 0:
            a = y * width + window_x0 + x
            b = y * width + window_x0 + dx + x
        elif dx >= 0:
            a = y * width + window_x0 + x
            b = (y + dy) * width + window_x0 + dx + x
        else:
            a = y * width + window_x0 - dx + x
            b = (y + dy) * width + window_x0 + x
        union(a, b)


def flatten_parent(parent: np.ndarray) -> None:
    if njit is not None:
        _flatten(parent)
        return

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return int(index)

    for index in range(parent.size):
        parent[index] = find(index)


def process_window(
    features: np.memmap,
    offsets: list[tuple[int, int]],
    parent: np.ndarray,
    size: np.ndarray,
    window: tuple[int, int, int, int],
) -> float:
    y0, y1, x0, x1 = window
    window_features = np.asarray(features[y0:y1, x0:x1, :])

    distances_by_offset: list[tuple[int, int, np.ndarray]] = []
    threshold_inputs: list[np.ndarray] = []
    for dy, dx in offsets:
        if THRESHOLD_MODE == "relative_percent":
            distances = relative_percent_for_offset(window_features, dy, dx)
        else:
            distances = distance_for_offset(window_features, dy, dx)
        if distances is None:
            continue
        distances_by_offset.append((dy, dx, distances))
        if THRESHOLD_MODE == "adaptive":
            threshold_inputs.append(distances)

    if THRESHOLD_MODE == "relative_percent":
        threshold = float(RELATIVE_DISTANCE_PERCENT)
    else:
        threshold = adaptive_threshold(threshold_inputs)

    for dy, dx, distances in distances_by_offset:
        mask = distances <= threshold
        union_window_offset_mask(parent, size, mask, dy, dx, features.shape[1], y0, x0)

    return threshold


def build_labels(parent: np.ndarray, height: int, width: int) -> tuple[np.ndarray, int, int]:
    flatten_parent(parent)
    _, inverse = np.unique(parent, return_inverse=True)
    counts = np.bincount(inverse)

    keep = counts >= MIN_REGION_SIZE
    removed_components = int((~keep).sum())
    remap = np.zeros(counts.size, dtype=np.int32)
    remap[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)

    labels = remap[inverse].reshape(height, width)
    return labels, int(keep.sum()), removed_components


def make_pseudocolor(labels: np.ndarray, label_count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    colors = rng.integers(0, 256, size=(label_count + 1, 3), dtype=np.uint8)
    colors[0] = np.array([0, 0, 0], dtype=np.uint8)
    return colors[labels]


def main() -> None:
    validate_config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    image_items = discover_images(RAW_DIR)
    features, positions, feature_path, stack_shape = build_features(image_items, OUT_DIR)
    image_count, height, width = stack_shape
    windows = sliding_windows(height, width)
    offsets = local_offsets(LOCAL_COMPARE_RADIUS)

    print(f"Sliding windows: {len(windows)}")
    print(f"Local compare offsets per window: {len(offsets)}")

    parent = np.arange(height * width, dtype=np.int32)
    size = np.ones(height * width, dtype=np.int32)
    thresholds = np.empty(len(windows), dtype=np.float32)

    for index, window in enumerate(windows):
        thresholds[index] = process_window(features, offsets, parent, size, window)
        if (index + 1) % PROGRESS_EVERY_WINDOWS == 0 or index + 1 == len(windows):
            print(
                f"Processed {index + 1}/{len(windows)} windows; "
                f"last threshold = {thresholds[index]:.6g}"
            )

    labels, label_count, removed_components = build_labels(parent, height, width)

    labels_path = OUT_DIR / "labels.npy"
    np.save(labels_path, labels)
    np.save(OUT_DIR / "light_positions.npy", positions)
    np.save(OUT_DIR / "window_thresholds.npy", thresholds)

    color = make_pseudocolor(labels, label_count, SEED)
    color_path = OUT_DIR / "segmentation_pseudocolor.png"
    Image.fromarray(color).save(color_path)

    metadata = {
        "raw_dir": str(RAW_DIR),
        "image_count": image_count,
        "height": height,
        "width": width,
        "stack_shape": [image_count, height, width],
        "features_shape": [height, width, image_count],
        "normalize": NORMALIZE,
        "window_size": WINDOW_SIZE,
        "window_stride": WINDOW_STRIDE,
        "window_count": len(windows),
        "local_compare_radius": LOCAL_COMPARE_RADIUS,
        "offset_count_per_window": len(offsets),
        "distance": DISTANCE,
        "threshold_mode": THRESHOLD_MODE,
        "relative_distance_percent": RELATIVE_DISTANCE_PERCENT,
        "adaptive_threshold_method": ADAPTIVE_THRESHOLD_METHOD,
        "adaptive_percentile": ADAPTIVE_PERCENTILE,
        "adaptive_threshold_scale": ADAPTIVE_THRESHOLD_SCALE,
        "min_threshold": MIN_THRESHOLD,
        "max_threshold": MAX_THRESHOLD,
        "threshold_min": float(thresholds.min()) if thresholds.size else 0.0,
        "threshold_mean": float(thresholds.mean()) if thresholds.size else 0.0,
        "threshold_max": float(thresholds.max()) if thresholds.size else 0.0,
        "min_region_size": MIN_REGION_SIZE,
        "label_count_without_background": label_count,
        "removed_small_components": removed_components,
        "stack_saved": bool(SAVE_STACK),
        "features_saved": bool(KEEP_FEATURES),
        "labels": str(labels_path),
        "pseudocolor": str(color_path),
    }
    with (OUT_DIR / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    del features
    gc.collect()
    if not KEEP_FEATURES and feature_path.exists():
        feature_path.unlink()

    print(f"Saved {labels_path}")
    print(f"Saved {color_path}")
    print(f"Regions without background: {label_count}")
    print(f"Removed small components: {removed_components}")
    print(
        "Window threshold range: "
        f"{metadata['threshold_min']:.6g} - {metadata['threshold_max']:.6g}, "
        f"mean = {metadata['threshold_mean']:.6g}"
    )


if __name__ == "__main__":
    main()
