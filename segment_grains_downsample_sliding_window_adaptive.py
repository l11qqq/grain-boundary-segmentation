from __future__ import annotations

import gc
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import segment_grains_sliding_window_adaptive as sw


"""
先降采样，再做滑窗自适应阈值分割

流程：
1. 从 raw_pic/ 读取 pic_r_phi.bmp，按 (r, phi) 排序。
2. 每张原始图先转灰度，再按 DOWNSAMPLE_FACTOR 降采样。
3. 在降采样图上构建 stack=(L,Hd,Wd) 和 features=(Hd,Wd,L)。
4. 在降采样图上按 WINDOW_SIZE / WINDOW_STRIDE 做滑窗检测。
5. 每个滑窗内部计算自适应阈值，并在窗口内逐像素局部连边。
6. 全部窗口共用一个并查集，最后得到降采样尺寸 labels。
7. 可选将 labels 最近邻上采样回原图尺寸，便于叠加或查看。
"""


# =========================
# 可调参数区
# =========================

# 原始图像目录，要求文件名形如 pic_r_phi.bmp。
RAW_DIR = Path(__file__).resolve().parent / "raw_pic413-3-1n"

# 输出目录。
OUT_DIR = Path("output_downsample_sliding_window_adaptive")

# 降采样倍数。2 表示宽高各缩小为 1/2，像素数约变为 1/4。
# 例如原图 2044 x 1536，DOWNSAMPLE_FACTOR=2 后约为 1022 x 768。
DOWNSAMPLE_FACTOR = 4.0

# 降采样插值方式：
# "area": 推荐用于缩小图像；
# "linear": 双线性；
# "nearest": 最近邻。
DOWNSAMPLE_INTERPOLATION = "area"

# 是否同时保存降采样后的灰度 stack.npy，shape=(L,Hd,Wd)，uint8。
SAVE_DOWNSAMPLED_STACK = False

# 是否保留归一化后的降采样 features.npy，shape=(Hd,Wd,L)，float32。
KEEP_FEATURES = False

# 是否把降采样 labels 最近邻上采样回原图大小。
SAVE_UPSAMPLED_LABELS = True

# 是否保存调试过程图像。
SAVE_DEBUG_IMAGES = True

# 调试图像输出子目录。
DEBUG_DIR_NAME = "debug"

# 滑窗大小和步长，单位是“降采样后的像素”。
WINDOW_SIZE = 64
WINDOW_STRIDE = 32

# 窗口内部逐像素比较半径，单位是“降采样后的像素”。
LOCAL_COMPARE_RADIUS = 9

# 每个角度通道的归一化方式："zscore"、"mean"、"none"。
NORMALIZE = "mean"

# 向量距离："rmse" 或 "euclidean"。
DISTANCE = "rmse"

# 阈值模式：
# "relative_percent": 按相对百分比判断，distance / mean(norm(vp), norm(vq)) * 100；
# "adaptive": 每个窗口用 Otsu / percentile 从距离分布估计绝对阈值。
THRESHOLD_MODE = "relative_percent"

# THRESHOLD_MODE="relative_percent" 时使用。
# 例如 10.0 表示两个像素的角度亮度向量相对差异 <= 10% 就认为相似。
# 413样品20.0 Fe样品22.5
RELATIVE_DISTANCE_PERCENT = 20

# 窗口内自适应阈值方法："otsu" 或 "percentile"。
# 仅 THRESHOLD_MODE="adaptive" 时参与计算。
ADAPTIVE_THRESHOLD_METHOD = "percentile"
ADAPTIVE_PERCENTILE = 50.0  # 仅在 ADAPTIVE_THRESHOLD_METHOD="percentile" 时有效，表示百分位数。

# 阈值缩放：区域太碎可调大，区域粘连可调小。
ADAPTIVE_THRESHOLD_SCALE = 1.0

# 阈值上下限。None 表示不限制。
MIN_THRESHOLD = None
MAX_THRESHOLD = None

# Otsu 直方图分箱数。
OTSU_BINS = 256

# 小碎片删除阈值，单位是“降采样后的像素数”。
MIN_REGION_SIZE = 20

# 伪彩色随机种子，只影响显示颜色，不影响 labels。
SEED = 12345

# 进度打印间隔。
PROGRESS_EVERY_IMAGES = 25
PROGRESS_EVERY_WINDOWS = 25


INTERPOLATION_MAP = {
    "area": cv2.INTER_AREA,
    "linear": cv2.INTER_LINEAR,
    "nearest": cv2.INTER_NEAREST,
}


def sync_base_config() -> None:
    sw.RAW_DIR = RAW_DIR
    sw.OUT_DIR = OUT_DIR
    sw.WINDOW_SIZE = WINDOW_SIZE
    sw.WINDOW_STRIDE = WINDOW_STRIDE
    sw.LOCAL_COMPARE_RADIUS = LOCAL_COMPARE_RADIUS
    sw.NORMALIZE = NORMALIZE
    sw.DISTANCE = DISTANCE
    sw.THRESHOLD_MODE = THRESHOLD_MODE
    sw.RELATIVE_DISTANCE_PERCENT = RELATIVE_DISTANCE_PERCENT
    sw.ADAPTIVE_THRESHOLD_METHOD = ADAPTIVE_THRESHOLD_METHOD
    sw.ADAPTIVE_PERCENTILE = ADAPTIVE_PERCENTILE
    sw.ADAPTIVE_THRESHOLD_SCALE = ADAPTIVE_THRESHOLD_SCALE
    sw.MIN_THRESHOLD = MIN_THRESHOLD
    sw.MAX_THRESHOLD = MAX_THRESHOLD
    sw.OTSU_BINS = OTSU_BINS
    sw.MIN_REGION_SIZE = MIN_REGION_SIZE
    sw.SAVE_STACK = SAVE_DOWNSAMPLED_STACK
    sw.KEEP_FEATURES = KEEP_FEATURES
    sw.SEED = SEED
    sw.PROGRESS_EVERY_IMAGES = PROGRESS_EVERY_IMAGES
    sw.PROGRESS_EVERY_WINDOWS = PROGRESS_EVERY_WINDOWS


def validate_config() -> None:
    if DOWNSAMPLE_FACTOR < 1.0:
        raise ValueError("DOWNSAMPLE_FACTOR must be >= 1.0")
    if DOWNSAMPLE_INTERPOLATION not in INTERPOLATION_MAP:
        raise ValueError('DOWNSAMPLE_INTERPOLATION must be "area", "linear", or "nearest"')
    sync_base_config()
    sw.validate_config()


def downsample_gray(image: np.ndarray) -> np.ndarray:
    if DOWNSAMPLE_FACTOR == 1.0:
        return image

    height, width = image.shape
    target_width = max(1, int(round(width / DOWNSAMPLE_FACTOR)))
    target_height = max(1, int(round(height / DOWNSAMPLE_FACTOR)))
    return cv2.resize(
        image,
        (target_width, target_height),
        interpolation=INTERPOLATION_MAP[DOWNSAMPLE_INTERPOLATION],
    )


def build_downsampled_features(
    image_items: list[tuple[tuple[float, float], Path]],
    out_dir: Path,
) -> tuple[np.memmap, np.ndarray, Path, tuple[int, int, int], tuple[int, int]]:
    first_original = sw.read_gray(image_items[0][1])
    original_height, original_width = first_original.shape
    first = downsample_gray(first_original)
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
    if SAVE_DOWNSAMPLED_STACK:
        stack = np.lib.format.open_memmap(
            out_dir / "stack_downsampled.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(image_count, height, width),
        )

    positions = np.empty((image_count, 2), dtype=np.float32)
    for index, (position, path) in enumerate(image_items):
        image_original = first_original if index == 0 else sw.read_gray(path)
        if image_original.shape != (original_height, original_width):
            raise ValueError(
                f"Image shape mismatch for {path}: "
                f"expected {(original_height, original_width)}, got {image_original.shape}"
            )

        image = first if index == 0 else downsample_gray(image_original)
        if image.shape != (height, width):
            raise ValueError(f"Downsampled shape mismatch for {path}: expected {(height, width)}, got {image.shape}")

        positions[index] = position
        features[:, :, index] = sw.normalize_channel(image)
        if stack is not None:
            stack[index] = image

        if (index + 1) % PROGRESS_EVERY_IMAGES == 0 or index + 1 == image_count:
            print(f"Built downsampled features from {index + 1}/{image_count} images")

    features.flush()
    if stack is not None:
        stack.flush()

    return features, positions, feature_path, (image_count, height, width), (original_height, original_width)


def upsample_labels(labels: np.ndarray, original_shape: tuple[int, int]) -> np.ndarray:
    original_height, original_width = original_shape
    return cv2.resize(
        labels.astype(np.float32),
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.int32)


def labels_before_and_after_min_size(
    parent: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    sw.flatten_parent(parent)
    _, inverse = np.unique(parent, return_inverse=True)
    counts = np.bincount(inverse)

    raw_label_count = counts.size
    labels_raw = (inverse + 1).reshape(height, width).astype(np.int32, copy=False)

    keep = counts >= MIN_REGION_SIZE
    removed_components = int((~keep).sum())
    remap = np.zeros(raw_label_count, dtype=np.int32)
    remap[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)

    labels = remap[inverse].reshape(height, width)
    return labels_raw, labels, int(keep.sum()), removed_components, raw_label_count


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros(image.shape, dtype=np.uint8)

    min_value = float(image[finite].min())
    max_value = float(image[finite].max())
    if max_value <= min_value:
        return np.zeros(image.shape, dtype=np.uint8)

    scaled = (image - min_value) * (255.0 / (max_value - min_value))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def save_colormap(image: np.ndarray, path: Path) -> None:
    gray = normalize_to_uint8(image)
    color_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(color_rgb).save(path)


def window_threshold_map(
    shape: tuple[int, int],
    windows: list[tuple[int, int, int, int]],
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    accum = np.zeros((height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)

    for (y0, y1, x0, x1), threshold in zip(windows, thresholds, strict=True):
        accum[y0:y1, x0:x1] += float(threshold)
        count[y0:y1, x0:x1] += 1.0

    threshold_map = np.divide(accum, count, out=np.zeros_like(accum), where=count > 0)
    return threshold_map, count


def save_debug_outputs(
    image_items: list[tuple[tuple[float, float], Path]],
    features: np.memmap,
    windows: list[tuple[int, int, int, int]],
    thresholds: np.ndarray,
    labels_raw: np.ndarray,
    raw_label_count: int,
    labels: np.ndarray,
    label_count: int,
) -> Path:
    debug_dir = OUT_DIR / DEBUG_DIR_NAME
    debug_dir.mkdir(parents=True, exist_ok=True)

    first_original = sw.read_gray(image_items[0][1])
    first_downsampled = downsample_gray(first_original)
    Image.fromarray(first_original).save(debug_dir / "01_first_original_gray.png")
    Image.fromarray(first_downsampled).save(debug_dir / "02_first_downsampled_gray.png")

    feature_mean = np.asarray(features.mean(axis=2), dtype=np.float32)
    feature_std = np.asarray(features.std(axis=2), dtype=np.float32)
    Image.fromarray(normalize_to_uint8(feature_mean)).save(debug_dir / "03_feature_mean.png")
    Image.fromarray(normalize_to_uint8(feature_std)).save(debug_dir / "04_feature_std.png")

    threshold_map, coverage_map = window_threshold_map(labels.shape, windows, thresholds)
    np.save(debug_dir / "05_window_threshold_map.npy", threshold_map)
    save_colormap(threshold_map, debug_dir / "05_window_threshold_map.png")
    Image.fromarray(normalize_to_uint8(coverage_map)).save(debug_dir / "06_window_coverage.png")

    raw_color = sw.make_pseudocolor(labels_raw, raw_label_count, SEED)
    Image.fromarray(raw_color).save(debug_dir / "07_labels_before_min_size.png")

    removed_mask = ((labels == 0) & (labels_raw > 0)).astype(np.uint8) * 255
    Image.fromarray(removed_mask).save(debug_dir / "08_removed_small_regions_mask.png")

    final_color = sw.make_pseudocolor(labels, label_count, SEED)
    Image.fromarray(final_color).save(debug_dir / "09_labels_after_min_size.png")
    return debug_dir


def main() -> None:
    validate_config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    image_items = sw.discover_images(RAW_DIR)
    features, positions, feature_path, stack_shape, original_shape = build_downsampled_features(image_items, OUT_DIR)
    image_count, height, width = stack_shape
    original_height, original_width = original_shape
    windows = sw.sliding_windows(height, width)
    offsets = sw.local_offsets(LOCAL_COMPARE_RADIUS)

    print(f"Original size: {original_width} x {original_height}")
    print(f"Downsampled size: {width} x {height}")
    print(f"Sliding windows on downsampled image: {len(windows)}")
    print(f"Local compare offsets per window: {len(offsets)}")

    parent = np.arange(height * width, dtype=np.int32)
    size = np.ones(height * width, dtype=np.int32)
    thresholds = np.empty(len(windows), dtype=np.float32)

    for index, window in enumerate(windows):
        thresholds[index] = sw.process_window(features, offsets, parent, size, window)
        if (index + 1) % PROGRESS_EVERY_WINDOWS == 0 or index + 1 == len(windows):
            print(
                f"Processed {index + 1}/{len(windows)} windows; "
                f"last threshold = {thresholds[index]:.6g}"
            )

    labels_raw, labels, label_count, removed_components, raw_label_count = labels_before_and_after_min_size(
        parent,
        height,
        width,
    )

    labels_path = OUT_DIR / "labels.npy"
    np.save(labels_path, labels)
    np.save(OUT_DIR / "light_positions.npy", positions)
    np.save(OUT_DIR / "window_thresholds.npy", thresholds)
    np.save(OUT_DIR / "labels_before_min_size.npy", labels_raw)

    color = sw.make_pseudocolor(labels, label_count, SEED)
    color_path = OUT_DIR / "segmentation_pseudocolor.png"
    Image.fromarray(color).save(color_path)

    upsampled_labels_path = None
    upsampled_color_path = None
    if SAVE_UPSAMPLED_LABELS:
        labels_full = upsample_labels(labels, original_shape)
        upsampled_labels_path = OUT_DIR / "labels_upsampled.npy"
        np.save(upsampled_labels_path, labels_full)

        color_full = sw.make_pseudocolor(labels_full, label_count, SEED)
        upsampled_color_path = OUT_DIR / "segmentation_pseudocolor_upsampled.png"
        Image.fromarray(color_full).save(upsampled_color_path)

    debug_dir = None
    if SAVE_DEBUG_IMAGES:
        debug_dir = save_debug_outputs(
            image_items,
            features,
            windows,
            thresholds,
            labels_raw,
            raw_label_count,
            labels,
            label_count,
        )

    metadata = {
        "raw_dir": str(RAW_DIR),
        "image_count": image_count,
        "original_height": original_height,
        "original_width": original_width,
        "downsampled_height": height,
        "downsampled_width": width,
        "downsample_factor": DOWNSAMPLE_FACTOR,
        "downsample_interpolation": DOWNSAMPLE_INTERPOLATION,
        "stack_shape_downsampled": [image_count, height, width],
        "features_shape_downsampled": [height, width, image_count],
        "normalize": NORMALIZE,
        "window_size_downsampled_pixels": WINDOW_SIZE,
        "window_stride_downsampled_pixels": WINDOW_STRIDE,
        "window_count": len(windows),
        "local_compare_radius_downsampled_pixels": LOCAL_COMPARE_RADIUS,
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
        "min_region_size_downsampled_pixels": MIN_REGION_SIZE,
        "label_count_without_background": label_count,
        "raw_label_count_before_min_size": raw_label_count,
        "removed_small_components": removed_components,
        "downsampled_stack_saved": bool(SAVE_DOWNSAMPLED_STACK),
        "features_saved": bool(KEEP_FEATURES),
        "upsampled_labels_saved": bool(SAVE_UPSAMPLED_LABELS),
        "debug_images_saved": bool(SAVE_DEBUG_IMAGES),
        "debug_dir": str(debug_dir) if debug_dir else None,
        "labels_downsampled": str(labels_path),
        "labels_before_min_size": str(OUT_DIR / "labels_before_min_size.npy"),
        "pseudocolor_downsampled": str(color_path),
        "labels_upsampled": str(upsampled_labels_path) if upsampled_labels_path else None,
        "pseudocolor_upsampled": str(upsampled_color_path) if upsampled_color_path else None,
    }
    with (OUT_DIR / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    del features
    gc.collect()
    if not KEEP_FEATURES and feature_path.exists():
        feature_path.unlink()

    print(f"Saved {labels_path}")
    print(f"Saved {color_path}")
    if upsampled_labels_path is not None:
        print(f"Saved {upsampled_labels_path}")
    if upsampled_color_path is not None:
        print(f"Saved {upsampled_color_path}")
    if debug_dir is not None:
        print(f"Saved debug images in {debug_dir}")
    print(f"Regions without background: {label_count}")
    print(f"Removed small components: {removed_components}")
    print(
        "Window threshold range: "
        f"{metadata['threshold_min']:.6g} - {metadata['threshold_max']:.6g}, "
        f"mean = {metadata['threshold_mean']:.6g}"
    )


if __name__ == "__main__":
    main()
