"""Interactive same-crop comparison tool for crossing-boundary superpixels.

Run with::

    python fig3/final_evaluation/superpixel_manual_evaluator.py

The default candidate label map is ``outputs/slic_3-1/slic_labels.npy``.  The
default reference grains come from the LabelMe grain polygons stored beside
that label map.  Users can switch candidate label maps while the crop remains
fixed in original-image pixel coordinates.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from superpixel_evaluation_core import (
    CropEvaluation,
    CropRect,
    ReferenceData,
    compute_auto_crossing_ids,
    compute_boundaries,
    evaluate_crop,
    load_label_array,
    load_reference_data,
    normalize_id_set,
    save_lossless_crop_png,
)


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parents[1]
DEFAULT_DATA_DIR = PROJECT_DIR / "outputs" / "slic_3-1"
DEFAULT_CANDIDATE = DEFAULT_DATA_DIR / "slic_labels.npy"
DEFAULT_REFERENCE = DEFAULT_DATA_DIR / "angular_boundary_strength.json"
DEFAULT_STD_BACKGROUND = DEFAULT_DATA_DIR / "std_image.png"
DEFAULT_FIRST_MEAN_BACKGROUND = APP_DIR / "coarse" / "first_angular_mean_background.png"
DEFAULT_BACKGROUND = (
    DEFAULT_FIRST_MEAN_BACKGROUND
    if DEFAULT_FIRST_MEAN_BACKGROUND.is_file()
    else DEFAULT_STD_BACKGROUND
)
STATE_PATH = APP_DIR / "superpixel_manual_evaluation_state.json"
RESULTS_JSON = APP_DIR / "superpixel_manual_evaluation_results.json"
RESULTS_CSV = APP_DIR / "superpixel_manual_evaluation_results.csv"
RESULT_FIELDS = (
    "timestamp",
    "candidate_name",
    "candidate_path",
    "reference_path",
    "crop_x",
    "crop_y",
    "crop_width",
    "crop_height",
    "total_superpixels",
    "border_touching_superpixels",
    "reference_decidable_superpixels",
    "reference_unknown_superpixels",
    "auto_marked_superpixels",
    "final_marked_superpixels",
    "crossing_ratio",
    "crossing_ratio_pct",
)

LABEL_FILE_TYPES = [
    ("超像素标签", "*.npy *.npz"),
    ("NumPy NPY", "*.npy"),
    ("NumPy NPZ", "*.npz"),
    ("所有文件", "*.*"),
]
REFERENCE_FILE_TYPES = [
    ("参考晶粒标签", "*.json *.npy *.npz"),
    ("LabelMe JSON", "*.json"),
    ("NumPy标签", "*.npy *.npz"),
    ("所有文件", "*.*"),
]
IMAGE_FILE_TYPES = [
    ("图像", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
    ("所有文件", "*.*"),
]


def _read_json(path: Path, default: object) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _path_key(path: Path) -> str:
    return str(path.resolve()).casefold()


class SuperpixelManualEvaluatorApp:
    HANDLE_RADIUS = 6
    HANDLE_HIT_RADIUS = 12
    MIN_CROP_SIZE = 4

    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root = root
        self.args = args
        self.root.title("同区域超像素跨晶界人工评估")
        self.root.geometry("1520x940")
        self.root.minsize(1180, 760)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        saved = _read_json(STATE_PATH, {})
        self.saved_state = saved if isinstance(saved, dict) else {}
        corrections = self.saved_state.get("candidate_corrections", {})
        self.corrections_by_candidate = corrections if isinstance(corrections, dict) else {}
        saved_results = _read_json(RESULTS_JSON, [])
        self.result_history = saved_results if isinstance(saved_results, list) else []

        self.reference_path: Path | None = None
        self.reference: ReferenceData | None = None
        self.candidate_path: Path | None = None
        self.candidate_labels: np.ndarray | None = None
        self.candidate_boundaries: np.ndarray | None = None
        self.auto_crossing_ids: set[int] = set()
        self.auto_diagnostics: dict[int, dict[str, int]] = {}
        self.manual_marked_ids: set[int] = set()
        self.manual_unmarked_ids: set[int] = set()
        self.background_path: Path | None = None
        self.background_image: Image.Image | None = None
        self.background_rgb: np.ndarray | None = None
        self.current_evaluation: CropEvaluation | None = None

        self.crop: CropRect | None = None
        self.crop_aspect_ratio = 1.0
        self.scale = float(np.clip(self.saved_state.get("scale", args.scale), 0.2, 4.0))
        self.photo: ImageTk.PhotoImage | None = None

        self.drag_action: str | None = None
        self.drag_start = (0.0, 0.0)
        self.drag_original: CropRect | None = None
        self.drag_handle: str | None = None
        self.drag_moved = False

        self.reference_var = tk.StringVar(value="尚未加载参考晶粒标签")
        self.candidate_var = tk.StringVar(value="尚未加载候选超像素label")
        self.background_var = tk.StringVar(value="尚未加载底图")
        self.status_var = tk.StringVar(value="正在加载默认数据……")
        self.mode_var = tk.StringVar(value="mark")
        self.lock_aspect_var = tk.BooleanVar(
            value=bool(self.saved_state.get("lock_aspect", True))
        )
        self.zero_unlabeled_var = tk.BooleanVar(value=False)
        self.show_candidate_boundaries_var = tk.BooleanVar(
            value=bool(self.saved_state.get("show_candidate_boundaries", True))
        )
        if "hide_reference_label" in self.saved_state:
            hide_reference_label = bool(self.saved_state["hide_reference_label"])
        elif "show_reference_boundaries" in self.saved_state:
            # Read state written by an earlier version of the software.
            hide_reference_label = not bool(self.saved_state["show_reference_boundaries"])
        else:
            hide_reference_label = True
        self.hide_reference_label_var = tk.BooleanVar(value=hide_reference_label)
        self.scale_var = tk.StringVar(value=f"{self.scale:.2f}×")
        self.crop_x_var = tk.StringVar()
        self.crop_y_var = tk.StringVar()
        self.crop_w_var = tk.StringVar()
        self.crop_h_var = tk.StringVar()
        self.total_var = tk.StringVar(value="—")
        self.border_var = tk.StringVar(value="—")
        self.decidable_var = tk.StringVar(value="—")
        self.unknown_var = tk.StringVar(value="—")
        self.auto_var = tk.StringVar(value="—")
        self.marked_var = tk.StringVar(value="—")
        self.ratio_var = tk.StringVar(value="—")
        self.reference_info_var = tk.StringVar(value="")
        self.clicked_id_var = tk.StringVar(value="当前超像素：—")

        self._build_ui()
        self._bind_events()
        self._load_initial_data()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=8)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(main)
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 7))
        ttk.Radiobutton(
            toolbar, text="人工标记", variable=self.mode_var, value="mark", command=self._mode_changed
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            toolbar, text="新建裁剪框", variable=self.mode_var, value="new_crop", command=self._mode_changed
        ).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Radiobutton(
            toolbar, text="移动/缩放裁剪框", variable=self.mode_var, value="edit_crop", command=self._mode_changed
        ).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=12)
        ttk.Checkbutton(
            toolbar,
            text="缩放时锁定当前宽高比",
            variable=self.lock_aspect_var,
            command=self._lock_aspect_changed,
        ).pack(side=tk.LEFT)
        ttk.Label(
            toolbar,
            text="标记模式：单击超像素切换蓝色；裁剪框边缘相交的超像素按完整1个计数",
            foreground="#555555",
        ).pack(side=tk.LEFT, padx=(16, 0))

        canvas_panel = ttk.Frame(main, width=1080, height=800)
        canvas_panel.grid(row=1, column=0, sticky="nsew")
        canvas_panel.grid_propagate(False)
        canvas_panel.columnconfigure(0, weight=1)
        canvas_panel.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(
            canvas_panel,
            width=1040,
            height=750,
            bg="#17191c",
            highlightthickness=1,
            highlightbackground="#777777",
        )
        xbar = ttk.Scrollbar(canvas_panel, orient=tk.HORIZONTAL, command=self.canvas.xview)
        ybar = ttk.Scrollbar(canvas_panel, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=xbar.set, yscrollcommand=ybar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        xbar.grid(row=1, column=0, sticky="ew")

        side_container = ttk.Frame(main, width=390)
        side_container.grid(row=1, column=1, sticky="ns", padx=(10, 0))
        side_container.grid_propagate(False)
        side_canvas = tk.Canvas(side_container, width=370, highlightthickness=0)
        side_scroll = ttk.Scrollbar(side_container, orient=tk.VERTICAL, command=side_canvas.yview)
        side_canvas.configure(yscrollcommand=side_scroll.set)
        side_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        side_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        side = ttk.Frame(side_canvas, padding=(4, 0, 7, 4))
        side_window = side_canvas.create_window((0, 0), window=side, anchor="nw")
        side.bind(
            "<Configure>",
            lambda _event: side_canvas.configure(scrollregion=side_canvas.bbox("all")),
        )
        side_canvas.bind(
            "<Configure>",
            lambda event: side_canvas.itemconfigure(side_window, width=event.width),
        )

        self._build_source_panel(side)
        self._build_crop_panel(side)
        self._build_stats_panel(side)
        self._build_history_panel(side)

        ttk.Label(
            main,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=(7, 4),
        ).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(7, 0))

    def _build_source_panel(self, side: ttk.Frame) -> None:
        panel = ttk.LabelFrame(side, text="数据源", padding=7)
        panel.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(panel, text="参考晶粒标注", font=("Microsoft YaHei UI", 9, "bold")).pack(anchor=tk.W)
        ttk.Label(panel, textvariable=self.reference_var, wraplength=335, justify=tk.LEFT).pack(
            anchor=tk.W, fill=tk.X, pady=(2, 3)
        )
        ref_buttons = ttk.Frame(panel)
        ref_buttons.pack(fill=tk.X)
        ttk.Button(ref_buttons, text="导入参考标注", command=self.choose_reference).pack(side=tk.LEFT)
        ttk.Checkbutton(
            ref_buttons,
            text="NPY中0为无标签",
            variable=self.zero_unlabeled_var,
            command=self.reload_reference,
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(
            panel,
            textvariable=self.reference_info_var,
            foreground="#725000",
            wraplength=335,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, fill=tk.X, pady=(4, 8))

        ttk.Label(panel, text="当前对比超像素label", font=("Microsoft YaHei UI", 9, "bold")).pack(anchor=tk.W)
        ttk.Label(panel, textvariable=self.candidate_var, wraplength=335, justify=tk.LEFT).pack(
            anchor=tk.W, fill=tk.X, pady=(2, 3)
        )
        ttk.Button(panel, text="导入/切换对比超像素", command=self.choose_candidate).pack(anchor=tk.W)

        ttk.Label(panel, text="统一可视化底图", font=("Microsoft YaHei UI", 9, "bold")).pack(
            anchor=tk.W, pady=(9, 0)
        )
        ttk.Label(panel, textvariable=self.background_var, wraplength=335, justify=tk.LEFT).pack(
            anchor=tk.W, fill=tk.X, pady=(2, 3)
        )
        ttk.Button(panel, text="更换底图", command=self.choose_background).pack(anchor=tk.W)

    def _build_crop_panel(self, side: ttk.Frame) -> None:
        panel = ttk.LabelFrame(side, text="裁剪框（原图像素坐标）", padding=7)
        panel.pack(fill=tk.X, pady=(0, 8))
        entries = ttk.Frame(panel)
        entries.pack(fill=tk.X)
        for column, (label, variable) in enumerate(
            (("X", self.crop_x_var), ("Y", self.crop_y_var), ("宽", self.crop_w_var), ("高", self.crop_h_var))
        ):
            ttk.Label(entries, text=label).grid(row=0, column=column, sticky="w", padx=(0 if column == 0 else 5, 0))
            ttk.Entry(entries, textvariable=variable, width=8).grid(
                row=1, column=column, sticky="ew", padx=(0 if column == 0 else 5, 0)
            )
            entries.columnconfigure(column, weight=1)
        crop_buttons = ttk.Frame(panel)
        crop_buttons.pack(fill=tk.X, pady=(7, 0))
        ttk.Button(crop_buttons, text="应用坐标", command=self.apply_crop_entries).pack(side=tk.LEFT)
        ttk.Button(crop_buttons, text="整幅图像", command=self.select_full_image).pack(side=tk.LEFT, padx=(6, 0))

        zoom = ttk.Frame(panel)
        zoom.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(zoom, text="显示缩放").pack(side=tk.LEFT)
        ttk.Button(zoom, text="−", width=3, command=lambda: self.set_scale(self.scale / 1.25)).pack(
            side=tk.LEFT, padx=(7, 0)
        )
        ttk.Button(zoom, text="+", width=3, command=lambda: self.set_scale(self.scale * 1.25)).pack(
            side=tk.LEFT, padx=(4, 0)
        )
        ttk.Button(zoom, text="100%", command=lambda: self.set_scale(1.0)).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(zoom, textvariable=self.scale_var).pack(side=tk.LEFT, padx=(7, 0))

        display = ttk.LabelFrame(panel, text="显示图层", padding=(6, 4))
        display.pack(fill=tk.X, pady=(8, 0))
        ttk.Checkbutton(
            display,
            text="显示当前候选超像素边界（红）",
            variable=self.show_candidate_boundaries_var,
            command=self._display_layers_changed,
        ).pack(anchor=tk.W)
        ttk.Checkbutton(
            display,
            text="隐藏参考 Label / 晶界层（避免重叠）",
            variable=self.hide_reference_label_var,
            command=self._display_layers_changed,
        ).pack(anchor=tk.W, pady=(3, 0))
        ttk.Label(
            display,
            text="默认隐藏黄色参考层；取消勾选可临时核对。",
            foreground="#666666",
        ).pack(anchor=tk.W, pady=(2, 0))
        ttk.Button(
            panel,
            text="保存裁剪区域图片（原分辨率无损 PNG）",
            command=self.save_crop_image,
        ).pack(fill=tk.X, pady=(8, 0))

    def _build_stats_panel(self, side: ttk.Frame) -> None:
        panel = ttk.LabelFrame(side, text="当前裁剪框统计", padding=8)
        panel.pack(fill=tk.X, pady=(0, 8))
        rows = (
            ("全部超像素", self.total_var),
            ("接触裁剪边框", self.border_var),
            ("有参考像素可初判", self.decidable_var),
            ("无参考像素未知", self.unknown_var),
            ("自动初标蓝色", self.auto_var),
            ("人工修正后蓝色", self.marked_var),
            ("跨晶界比例", self.ratio_var),
        )
        for row_index, (label, variable) in enumerate(rows):
            ttk.Label(panel, text=label).grid(row=row_index, column=0, sticky="w", pady=2)
            font = ("Microsoft YaHei UI", 12, "bold") if label == "跨晶界比例" else ("Microsoft YaHei UI", 10, "bold")
            ttk.Label(panel, textvariable=variable, font=font, foreground="#0057a8").grid(
                row=row_index, column=1, sticky="e", pady=2
            )
        panel.columnconfigure(1, weight=1)
        ttk.Label(panel, textvariable=self.clicked_id_var, foreground="#444444").grid(
            row=len(rows), column=0, columnspan=2, sticky="w", pady=(7, 2)
        )
        buttons = ttk.Frame(panel)
        buttons.grid(row=len(rows) + 1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(buttons, text="重新自动初标", command=self.recompute_automatic_marks).pack(side=tk.LEFT)
        ttk.Button(buttons, text="清除人工修改", command=self.reset_manual_corrections).pack(
            side=tk.LEFT, padx=(6, 0)
        )
        ttk.Button(panel, text="记录当前结果", command=self.record_current_result).grid(
            row=len(rows) + 2, column=0, columnspan=2, sticky="ew", pady=(7, 0)
        )
        ttk.Label(
            panel,
            text=(
                "自动初标依据完整超像素与参考晶粒的重叠关系；裁剪框只决定哪些超像素进入分母。"
                "单击可取消蓝色，也可补加漏标。"
            ),
            wraplength=330,
            justify=tk.LEFT,
            foreground="#555555",
        ).grid(row=len(rows) + 3, column=0, columnspan=2, sticky="ew", pady=(7, 0))

    def _build_history_panel(self, side: ttk.Frame) -> None:
        panel = ttk.LabelFrame(side, text="已记录对比结果", padding=6)
        panel.pack(fill=tk.BOTH, expand=True)
        columns = ("name", "total", "marked", "ratio")
        self.history_tree = ttk.Treeview(panel, columns=columns, show="headings", height=6)
        self.history_tree.heading("name", text="文件")
        self.history_tree.heading("total", text="总数")
        self.history_tree.heading("marked", text="蓝色")
        self.history_tree.heading("ratio", text="比例")
        self.history_tree.column("name", width=145, stretch=True)
        self.history_tree.column("total", width=46, anchor=tk.CENTER)
        self.history_tree.column("marked", width=46, anchor=tk.CENTER)
        self.history_tree.column("ratio", width=65, anchor=tk.E)
        self.history_tree.pack(fill=tk.BOTH, expand=True)
        ttk.Button(panel, text="清空记录表", command=self.clear_result_history).pack(anchor=tk.E, pady=(5, 0))
        self._refresh_history_tree()

    def _bind_events(self) -> None:
        self.canvas.bind("<ButtonPress-1>", self.on_left_press)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_release)
        self.canvas.bind("<Control-MouseWheel>", self.on_ctrl_mousewheel)
        self.canvas.bind("<Shift-MouseWheel>", self.on_shift_mousewheel)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel_scroll)

    # -------------------------------------------------------------- loading
    def _load_initial_data(self) -> None:
        reference = Path(
            self.args.reference
            or self.saved_state.get("reference_path")
            or DEFAULT_REFERENCE
        )
        candidate = Path(
            self.args.candidate
            or self.saved_state.get("candidate_path")
            or DEFAULT_CANDIDATE
        )
        background = Path(
            self.args.background
            or self.saved_state.get("background_path")
            or DEFAULT_BACKGROUND
        )
        errors: list[str] = []
        try:
            self.load_reference(reference)
        except Exception as exc:
            errors.append(f"参考标注：{exc}")
        try:
            self.load_background(background)
        except Exception as exc:
            errors.append(f"底图：{exc}")
        try:
            self.load_candidate(candidate)
        except Exception as exc:
            errors.append(f"候选label：{exc}")
        if errors:
            self.status_var.set("；".join(errors))

    def choose_reference(self) -> None:
        initial = self.reference_path.parent if self.reference_path else DEFAULT_DATA_DIR
        filename = filedialog.askopenfilename(
            parent=self.root,
            title="选择参考晶粒标注",
            initialdir=initial,
            filetypes=REFERENCE_FILE_TYPES,
        )
        if filename:
            try:
                self.load_reference(Path(filename))
            except Exception as exc:
                messagebox.showerror("参考标注加载失败", str(exc), parent=self.root)

    def choose_candidate(self) -> None:
        initial = self.candidate_path.parent if self.candidate_path else APP_DIR
        filename = filedialog.askopenfilename(
            parent=self.root,
            title="选择待比较的超像素label",
            initialdir=initial,
            filetypes=LABEL_FILE_TYPES,
        )
        if filename:
            try:
                self.load_candidate(Path(filename))
            except Exception as exc:
                messagebox.showerror("超像素label加载失败", str(exc), parent=self.root)

    def choose_background(self) -> None:
        initial = self.background_path.parent if self.background_path else APP_DIR
        filename = filedialog.askopenfilename(
            parent=self.root,
            title="选择统一可视化底图",
            initialdir=initial,
            filetypes=IMAGE_FILE_TYPES,
        )
        if filename:
            try:
                self.load_background(Path(filename))
            except Exception as exc:
                messagebox.showerror("底图加载失败", str(exc), parent=self.root)

    def load_reference(self, path: Path) -> None:
        reference = load_reference_data(path, zero_is_unlabeled=bool(self.zero_unlabeled_var.get()))
        self._assert_shape_compatible(reference.labels.shape, "参考标注")
        self.reference_path = Path(path).resolve()
        self.reference = reference
        self.reference_var.set(str(self.reference_path))
        metadata = reference.metadata
        regions = metadata.get("observed_grain_count", metadata.get("reference_region_count", "?"))
        valid_pixels = int(metadata.get("valid_labeled_pixels", 0))
        coverage = 100.0 * valid_pixels / reference.labels.size
        self.reference_info_var.set(f"参考区域 {regions}；有效像素 {valid_pixels:,}（{coverage:.2f}%）")
        self.recompute_automatic_marks(render_after=False)
        self._ensure_crop()
        self.refresh_evaluation_and_render()
        self.status_var.set(f"已加载参考标注：{self.reference_path.name}")
        self.save_state()

    def reload_reference(self) -> None:
        if self.reference_path is None:
            return
        try:
            self.load_reference(self.reference_path)
        except Exception as exc:
            messagebox.showerror("重新加载参考标注失败", str(exc), parent=self.root)

    def load_candidate(self, path: Path) -> None:
        labels = load_label_array(path)
        self._assert_shape_compatible(labels.shape, "候选超像素label")
        self._save_current_corrections()
        self.candidate_path = Path(path).resolve()
        self.candidate_labels = labels
        self.candidate_boundaries = compute_boundaries(labels)
        saved = self.corrections_by_candidate.get(_path_key(self.candidate_path), {})
        if not isinstance(saved, dict):
            saved = {}
        self.manual_marked_ids = normalize_id_set(saved.get("manual_marked_ids", []))
        self.manual_unmarked_ids = normalize_id_set(saved.get("manual_unmarked_ids", []))
        self.candidate_var.set(str(self.candidate_path))
        self.recompute_automatic_marks(render_after=False)
        self._ensure_crop()
        self.refresh_evaluation_and_render()
        self.status_var.set(
            f"已加载 {self.candidate_path.name}；全图 {np.unique(labels).size:,} 个超像素；裁剪框已保留"
        )
        self.save_state()

    def load_background(self, path: Path) -> None:
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"底图不存在：{path}")
        with Image.open(path) as source:
            image = source.convert("RGB").copy()
        shape = (image.height, image.width)
        self._assert_shape_compatible(shape, "底图")
        self.background_path = path
        self.background_image = image
        self.background_rgb = np.asarray(image, dtype=np.uint8).copy()
        self.background_var.set(str(path))
        self._ensure_crop()
        self.refresh_evaluation_and_render()
        self.status_var.set(f"已加载底图：{path.name}（{image.width}×{image.height}，100%为原分辨率）")
        self.save_state()

    def _known_shape(self) -> tuple[int, int] | None:
        if self.candidate_labels is not None:
            return self.candidate_labels.shape
        if self.reference is not None:
            return self.reference.labels.shape
        if self.background_image is not None:
            return (self.background_image.height, self.background_image.width)
        return None

    def _assert_shape_compatible(self, shape: tuple[int, int], source_name: str) -> None:
        known = self._known_shape()
        if known is not None and tuple(shape) != tuple(known):
            raise ValueError(f"{source_name}尺寸 {tuple(shape)} 与当前固定原图尺寸 {tuple(known)} 不一致")

    # ------------------------------------------------------------ evaluation
    def _ensure_crop(self) -> None:
        shape = self._known_shape()
        if shape is None:
            return
        if self.crop is None:
            saved_crop = self.saved_state.get("crop")
            if isinstance(saved_crop, list) and len(saved_crop) == 4:
                try:
                    self.crop = CropRect(*(int(value) for value in saved_crop)).normalized(shape)
                except (TypeError, ValueError):
                    self.crop = None
            if self.crop is None:
                height, width = shape
                if self.reference is not None and np.any(self.reference.valid_mask):
                    ys, xs = np.nonzero(self.reference.valid_mask)
                    self.crop = CropRect(
                        int(xs.min()),
                        int(ys.min()),
                        int(xs.max()) + 1,
                        int(ys.max()) + 1,
                    )
                else:
                    self.crop = CropRect(width // 4, height // 4, 3 * width // 4, 3 * height // 4)
        self.crop = self.crop.normalized(shape, self.MIN_CROP_SIZE)
        self.crop_aspect_ratio = self.crop.aspect_ratio
        self._update_crop_entries()

    def recompute_automatic_marks(self, render_after: bool = True) -> None:
        if self.candidate_labels is None or self.reference is None:
            self.auto_crossing_ids = set()
            self.auto_diagnostics = {}
        else:
            self.auto_crossing_ids, self.auto_diagnostics = compute_auto_crossing_ids(
                self.candidate_labels,
                self.reference,
            )
        if render_after:
            self.refresh_evaluation_and_render()
            self.status_var.set(f"已重新计算自动初标：全图候选 {len(self.auto_crossing_ids):,} 个")

    def refresh_evaluation(self) -> None:
        if self.candidate_labels is None or self.crop is None:
            self.current_evaluation = None
            self.total_var.set("—")
            self.border_var.set("—")
            self.decidable_var.set("—")
            self.unknown_var.set("—")
            self.auto_var.set("—")
            self.marked_var.set("—")
            self.ratio_var.set("—")
            return
        self.current_evaluation = evaluate_crop(
            self.candidate_labels,
            self.crop,
            self.auto_crossing_ids,
            self.manual_marked_ids,
            self.manual_unmarked_ids,
        )
        evaluation = self.current_evaluation
        self.total_var.set(f"{evaluation.total_superpixels:,}")
        self.border_var.set(f"{evaluation.border_touching_superpixels:,}")
        decidable = len(evaluation.crop_superpixel_ids.intersection(self.auto_diagnostics))
        unknown = evaluation.total_superpixels - decidable
        self.decidable_var.set(f"{decidable:,}")
        self.unknown_var.set(f"{unknown:,}")
        self.auto_var.set(f"{evaluation.auto_marked_superpixels:,}")
        self.marked_var.set(f"{evaluation.final_marked_superpixels:,}")
        self.ratio_var.set(f"{100.0 * evaluation.marked_ratio:.4f}%")
        self._update_crop_entries()

    def refresh_evaluation_and_render(self) -> None:
        self.refresh_evaluation()
        self.render()

    def toggle_superpixel(self, superpixel_id: int) -> None:
        evaluation = self.current_evaluation
        if evaluation is None or superpixel_id not in evaluation.crop_superpixel_ids:
            self.status_var.set(f"超像素 {superpixel_id} 不与当前裁剪框相交，未纳入本次统计")
            return
        if superpixel_id in evaluation.final_marked_ids:
            self.manual_marked_ids.discard(superpixel_id)
            if superpixel_id in evaluation.auto_marked_ids:
                self.manual_unmarked_ids.add(superpixel_id)
            action = "取消蓝色"
        else:
            self.manual_unmarked_ids.discard(superpixel_id)
            self.manual_marked_ids.add(superpixel_id)
            action = "补充蓝色"
        self._save_current_corrections()
        self.refresh_evaluation_and_render()
        self.status_var.set(f"超像素 {superpixel_id}：{action}；结果已自动保存")
        self.save_state()

    def reset_manual_corrections(self) -> None:
        if self.candidate_path is None:
            return
        if not self.manual_marked_ids and not self.manual_unmarked_ids:
            return
        self.manual_marked_ids.clear()
        self.manual_unmarked_ids.clear()
        self._save_current_corrections()
        self.refresh_evaluation_and_render()
        self.save_state()
        self.status_var.set("已清除当前候选文件的人工修改，恢复自动初标")

    # ---------------------------------------------------------------- render
    def _make_display_rgb(self) -> np.ndarray | None:
        if self.candidate_labels is None:
            return self.background_rgb.copy() if self.background_rgb is not None else None
        if self.background_rgb is None:
            gray = np.zeros(self.candidate_labels.shape, dtype=np.uint8)
            base = np.repeat(gray[:, :, None], 3, axis=2)
        else:
            base = self.background_rgb.copy()
        if base.shape[:2] != self.candidate_labels.shape:
            return None

        evaluation = self.current_evaluation
        if evaluation is not None and evaluation.final_marked_ids:
            marked_mask = np.isin(self.candidate_labels, list(evaluation.final_marked_ids))
            base[marked_mask] = (
                0.30 * base[marked_mask].astype(np.float32)
                + 0.70 * np.array([0, 90, 255], dtype=np.float32)
            ).astype(np.uint8)
        if bool(self.show_candidate_boundaries_var.get()) and self.candidate_boundaries is not None:
            base[self.candidate_boundaries] = np.array([255, 45, 45], dtype=np.uint8)
        if not bool(self.hide_reference_label_var.get()) and self.reference is not None:
            base[self.reference.boundary_mask] = np.array([255, 215, 0], dtype=np.uint8)
        return base

    def _display_layers_changed(self) -> None:
        self.render()
        self.save_state()

    def render(self) -> None:
        rgb = self._make_display_rgb()
        if rgb is None:
            return
        height, width = rgb.shape[:2]
        display_size = (
            max(1, int(round(width * self.scale))),
            max(1, int(round(height * self.scale))),
        )
        image = Image.fromarray(rgb)
        if display_size != image.size:
            resampling = Image.Resampling.NEAREST if self.scale >= 1.0 else Image.Resampling.BILINEAR
            image = image.resize(display_size, resampling)
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo, tags=("image",))
        self.canvas.configure(scrollregion=(0, 0, display_size[0], display_size[1]))
        self._draw_crop()

    def _draw_crop(self) -> None:
        self.canvas.delete("crop")
        self.canvas.delete("crop_handle")
        if self.crop is None:
            return
        x0, y0, x1, y1 = [value * self.scale for value in self.crop.as_list()]
        self.canvas.create_rectangle(
            x0,
            y0,
            x1,
            y1,
            outline="#39ff88",
            width=3,
            tags=("crop",),
        )
        if self.mode_var.get() == "edit_crop":
            radius = self.HANDLE_RADIUS
            for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
                self.canvas.create_rectangle(
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius,
                    fill="#ffffff",
                    outline="#007f45",
                    width=2,
                    tags=("crop_handle",),
                )

    # ----------------------------------------------------------- crop events
    def _event_image_point(self, event: tk.Event) -> tuple[float, float]:
        x = self.canvas.canvasx(event.x) / self.scale
        y = self.canvas.canvasy(event.y) / self.scale
        shape = self._known_shape()
        if shape is None:
            return x, y
        return float(np.clip(x, 0, shape[1])), float(np.clip(y, 0, shape[0]))

    def on_left_press(self, event: tk.Event) -> None:
        if self.candidate_labels is None:
            return
        point = self._event_image_point(event)
        self.drag_start = point
        self.drag_original = self.crop
        self.drag_moved = False
        mode = self.mode_var.get()
        if mode == "mark":
            self.drag_action = "mark"
            return
        if mode == "new_crop":
            self.drag_action = "new"
            self.crop = CropRect(int(point[0]), int(point[1]), int(point[0]) + 2, int(point[1]) + 2)
            self._draw_crop()
            return
        if mode == "edit_crop" and self.crop is not None:
            handle = self._hit_crop_handle(*point)
            if handle is not None:
                self.drag_action = "resize"
                self.drag_handle = handle
            elif self._inside_crop(*point):
                self.drag_action = "move"
            else:
                self.drag_action = None

    def on_left_drag(self, event: tk.Event) -> None:
        if self.drag_action is None or self.candidate_labels is None:
            return
        point = self._event_image_point(event)
        if abs(point[0] - self.drag_start[0]) + abs(point[1] - self.drag_start[1]) > 2.0:
            self.drag_moved = True
        shape = self.candidate_labels.shape
        if self.drag_action == "new":
            self.crop = self._new_crop_from_points(self.drag_start, point).normalized(
                shape, self.MIN_CROP_SIZE
            )
            self._draw_crop()
        elif self.drag_action == "move" and self.drag_original is not None:
            dx = int(round(point[0] - self.drag_start[0]))
            dy = int(round(point[1] - self.drag_start[1]))
            box = self.drag_original
            width, height = box.width, box.height
            x0 = min(max(box.x0 + dx, 0), shape[1] - width)
            y0 = min(max(box.y0 + dy, 0), shape[0] - height)
            self.crop = CropRect(x0, y0, x0 + width, y0 + height)
            self._draw_crop()
        elif self.drag_action == "resize" and self.drag_original is not None and self.drag_handle:
            self.crop = self._resize_crop_corner(self.drag_original, self.drag_handle, point).normalized(
                shape, self.MIN_CROP_SIZE
            )
            self._draw_crop()

    def on_left_release(self, event: tk.Event) -> None:
        if self.drag_action == "mark" and not self.drag_moved and self.candidate_labels is not None:
            x, y = self._event_image_point(event)
            xi = min(max(int(x), 0), self.candidate_labels.shape[1] - 1)
            yi = min(max(int(y), 0), self.candidate_labels.shape[0] - 1)
            superpixel_id = int(self.candidate_labels[yi, xi])
            self.clicked_id_var.set(f"当前超像素：{superpixel_id}")
            self.toggle_superpixel(superpixel_id)
        elif self.drag_action in {"new", "move", "resize"} and self.crop is not None:
            self.crop = self.crop.normalized(self.candidate_labels.shape, self.MIN_CROP_SIZE)
            self.crop_aspect_ratio = self.crop.aspect_ratio
            self.refresh_evaluation_and_render()
            self.save_state()
            self.status_var.set(
                f"裁剪框已更新：x={self.crop.x0}, y={self.crop.y0}, "
                f"w={self.crop.width}, h={self.crop.height}；切换候选label时将保留"
            )
        self.drag_action = None
        self.drag_handle = None
        self.drag_original = None

    def _new_crop_from_points(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> CropRect:
        x0, y0 = start
        x1, y1 = end
        if bool(self.lock_aspect_var.get()):
            ratio = max(self.crop_aspect_ratio, 1e-6)
            dx, dy = x1 - x0, y1 - y0
            width, height = abs(dx), abs(dy)
            if height < 1e-6:
                height = width / ratio
            elif width / height > ratio:
                height = width / ratio
            else:
                width = height * ratio
            x1 = x0 + (-width if dx < 0 else width)
            y1 = y0 + (-height if dy < 0 else height)
        return CropRect(int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1)))

    def _resize_crop_corner(
        self,
        box: CropRect,
        handle: str,
        point: tuple[float, float],
    ) -> CropRect:
        anchors = {
            "nw": (box.x1, box.y1, -1, -1),
            "ne": (box.x0, box.y1, 1, -1),
            "se": (box.x0, box.y0, 1, 1),
            "sw": (box.x1, box.y0, -1, 1),
        }
        anchor_x, anchor_y, sign_x, sign_y = anchors[handle]
        width = max(abs(point[0] - anchor_x), self.MIN_CROP_SIZE)
        height = max(abs(point[1] - anchor_y), self.MIN_CROP_SIZE)
        if bool(self.lock_aspect_var.get()):
            ratio = max(self.crop_aspect_ratio, 1e-6)
            if width / height > ratio:
                height = width / ratio
            else:
                width = height * ratio
        moving_x = anchor_x + sign_x * width
        moving_y = anchor_y + sign_y * height
        return CropRect(int(round(anchor_x)), int(round(anchor_y)), int(round(moving_x)), int(round(moving_y)))

    def _hit_crop_handle(self, x: float, y: float) -> str | None:
        if self.crop is None:
            return None
        tolerance = self.HANDLE_HIT_RADIUS / self.scale
        points = {
            "nw": (self.crop.x0, self.crop.y0),
            "ne": (self.crop.x1, self.crop.y0),
            "se": (self.crop.x1, self.crop.y1),
            "sw": (self.crop.x0, self.crop.y1),
        }
        for name, (hx, hy) in points.items():
            if abs(x - hx) <= tolerance and abs(y - hy) <= tolerance:
                return name
        return None

    def _inside_crop(self, x: float, y: float) -> bool:
        return bool(
            self.crop is not None
            and self.crop.x0 <= x <= self.crop.x1
            and self.crop.y0 <= y <= self.crop.y1
        )

    def apply_crop_entries(self) -> None:
        shape = self._known_shape()
        if shape is None:
            return
        try:
            x = int(self.crop_x_var.get())
            y = int(self.crop_y_var.get())
            width = int(self.crop_w_var.get())
            height = int(self.crop_h_var.get())
            if width < self.MIN_CROP_SIZE or height < self.MIN_CROP_SIZE:
                raise ValueError(f"宽和高至少为 {self.MIN_CROP_SIZE} pixel")
            crop = CropRect(x, y, x + width, y + height).normalized(shape, self.MIN_CROP_SIZE)
        except ValueError as exc:
            messagebox.showerror("裁剪框坐标无效", str(exc), parent=self.root)
            return
        self.crop = crop
        self.crop_aspect_ratio = crop.aspect_ratio
        self.refresh_evaluation_and_render()
        self.save_state()
        self.status_var.set("已应用裁剪框坐标；切换超像素文件时保持不变")

    def select_full_image(self) -> None:
        shape = self._known_shape()
        if shape is None:
            return
        self.crop = CropRect(0, 0, shape[1], shape[0])
        self.crop_aspect_ratio = self.crop.aspect_ratio
        self.refresh_evaluation_and_render()
        self.save_state()

    def _update_crop_entries(self) -> None:
        if self.crop is None:
            return
        self.crop_x_var.set(str(self.crop.x0))
        self.crop_y_var.set(str(self.crop.y0))
        self.crop_w_var.set(str(self.crop.width))
        self.crop_h_var.set(str(self.crop.height))

    # -------------------------------------------------------------- view mode
    def _mode_changed(self) -> None:
        cursor = "hand2" if self.mode_var.get() == "mark" else "crosshair"
        self.canvas.configure(cursor=cursor)
        self._draw_crop()

    def _lock_aspect_changed(self) -> None:
        if self.crop is not None:
            self.crop_aspect_ratio = self.crop.aspect_ratio
        self.save_state()

    def set_scale(self, value: float) -> None:
        self.scale = float(np.clip(value, 0.2, 4.0))
        self.scale_var.set(f"{self.scale:.2f}×")
        self.render()
        self.save_state()

    def on_ctrl_mousewheel(self, event: tk.Event) -> str:
        self.set_scale(self.scale * (1.15 if event.delta > 0 else 1.0 / 1.15))
        return "break"

    def on_mousewheel_scroll(self, event: tk.Event) -> str:
        self.canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
        return "break"

    def on_shift_mousewheel(self, event: tk.Event) -> str:
        self.canvas.xview_scroll(-1 if event.delta > 0 else 1, "units")
        return "break"

    def save_crop_image(self) -> None:
        """Export the current visible-layer composition at exact source pixels."""
        if self.crop is None:
            messagebox.showwarning("无法保存", "请先设置裁剪框。", parent=self.root)
            return
        rgb = self._make_display_rgb()
        if rgb is None:
            messagebox.showwarning("无法保存", "请先加载可视化底图或候选label。", parent=self.root)
            return
        crop = self.crop.normalized(rgb.shape[:2], self.MIN_CROP_SIZE)
        source_name = (
            self.candidate_path.parent.name
            if self.candidate_path is not None and self.candidate_path.parent.name
            else "superpixel"
        )
        safe_name = "".join(
            character if character.isalnum() or character in "-_." else "_"
            for character in source_name
        )
        suggested_name = (
            f"{safe_name}_crop_x{crop.x0}_y{crop.y0}_"
            f"w{crop.width}_h{crop.height}.png"
        )
        filename = filedialog.asksaveasfilename(
            parent=self.root,
            title="保存原始分辨率裁剪图",
            initialdir=APP_DIR,
            initialfile=suggested_name,
            defaultextension=".png",
            filetypes=[("无损 PNG", "*.png")],
        )
        if not filename:
            return
        try:
            crop_rgb = save_lossless_crop_png(Path(filename), rgb, crop)
        except (OSError, ValueError) as exc:
            messagebox.showerror("保存失败", str(exc), parent=self.root)
            return
        self.status_var.set(
            f"已无损保存裁剪图：{crop_rgb.shape[1]}×{crop_rgb.shape[0]} pixel；"
            "未缩放、未重采样，不含绿色裁剪操作框"
        )

    # -------------------------------------------------------------- persistence
    def _save_current_corrections(self) -> None:
        if self.candidate_path is None:
            return
        self.corrections_by_candidate[_path_key(self.candidate_path)] = {
            "candidate_path": str(self.candidate_path),
            "manual_marked_ids": sorted(self.manual_marked_ids),
            "manual_unmarked_ids": sorted(self.manual_unmarked_ids),
        }

    def save_state(self) -> None:
        self._save_current_corrections()
        payload = {
            "version": 2,
            "reference_path": str(self.reference_path) if self.reference_path else None,
            "candidate_path": str(self.candidate_path) if self.candidate_path else None,
            "background_path": str(self.background_path) if self.background_path else None,
            "crop": self.crop.as_list() if self.crop else None,
            "lock_aspect": bool(self.lock_aspect_var.get()),
            "scale": self.scale,
            "show_candidate_boundaries": bool(self.show_candidate_boundaries_var.get()),
            "hide_reference_label": bool(self.hide_reference_label_var.get()),
            "candidate_corrections": self.corrections_by_candidate,
        }
        STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def record_current_result(self) -> None:
        if self.current_evaluation is None or self.candidate_path is None or self.crop is None:
            messagebox.showwarning("无法记录", "请先加载候选label并设置裁剪框。", parent=self.root)
            return
        evaluation = self.current_evaluation
        decidable = len(evaluation.crop_superpixel_ids.intersection(self.auto_diagnostics))
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "candidate_name": self.candidate_path.parent.name or self.candidate_path.stem,
            "candidate_path": str(self.candidate_path),
            "reference_path": str(self.reference_path) if self.reference_path else "",
            "crop_x": self.crop.x0,
            "crop_y": self.crop.y0,
            "crop_width": self.crop.width,
            "crop_height": self.crop.height,
            "total_superpixels": evaluation.total_superpixels,
            "border_touching_superpixels": evaluation.border_touching_superpixels,
            "reference_decidable_superpixels": decidable,
            "reference_unknown_superpixels": evaluation.total_superpixels - decidable,
            "auto_marked_superpixels": evaluation.auto_marked_superpixels,
            "final_marked_superpixels": evaluation.final_marked_superpixels,
            "crossing_ratio": evaluation.marked_ratio,
            "crossing_ratio_pct": 100.0 * evaluation.marked_ratio,
        }
        self.result_history.append(row)
        self._write_results()
        self._refresh_history_tree()
        self.status_var.set(
            f"已记录 {row['candidate_name']}：{row['final_marked_superpixels']}/"
            f"{row['total_superpixels']} = {row['crossing_ratio_pct']:.4f}%"
        )

    def _write_results(self) -> None:
        RESULTS_JSON.write_text(
            json.dumps(self.result_history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if self.result_history:
            with RESULTS_CSV.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=list(RESULT_FIELDS),
                    extrasaction="ignore",
                )
                writer.writeheader()
                for row in self.result_history:
                    writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})
        else:
            RESULTS_CSV.write_text("", encoding="utf-8-sig")

    def _refresh_history_tree(self) -> None:
        if not hasattr(self, "history_tree"):
            return
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        for row in self.result_history[-100:]:
            self.history_tree.insert(
                "",
                tk.END,
                values=(
                    row.get("candidate_name", ""),
                    row.get("total_superpixels", ""),
                    row.get("final_marked_superpixels", ""),
                    f"{float(row.get('crossing_ratio_pct', 0.0)):.3f}%",
                ),
            )

    def clear_result_history(self) -> None:
        if not self.result_history:
            return
        if not messagebox.askyesno("清空记录", "确定清空已记录的对比结果吗？", parent=self.root):
            return
        self.result_history.clear()
        self._write_results()
        self._refresh_history_tree()
        self.status_var.set("已清空结果记录；超像素人工修改和裁剪框不受影响")

    def on_close(self) -> None:
        self.save_state()
        self.root.destroy()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive same-crop superpixel evaluator")
    parser.add_argument("--candidate", type=Path, default=None)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--background", type=Path, default=None)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--ui-smoke-test", action="store_true")
    return parser.parse_args(argv)


def run_headless_smoke_test(args: argparse.Namespace) -> None:
    candidate_path = Path(args.candidate or DEFAULT_CANDIDATE)
    reference_path = Path(args.reference or DEFAULT_REFERENCE)
    candidate = load_label_array(candidate_path)
    reference = load_reference_data(reference_path)
    crossing_ids, _diagnostics = compute_auto_crossing_ids(candidate, reference)
    height, width = candidate.shape
    if np.any(reference.valid_mask):
        ys, xs = np.nonzero(reference.valid_mask)
        crop = CropRect(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    else:
        crop = CropRect(width // 4, height // 4, 3 * width // 4, 3 * height // 4)
    result = evaluate_crop(candidate, crop, crossing_ids)
    print(
        json.dumps(
            {
                "candidate": str(candidate_path.resolve()),
                "reference": str(reference_path.resolve()),
                "shape": list(candidate.shape),
                "crop": crop.as_list(),
                "total_superpixels": result.total_superpixels,
                "auto_marked_superpixels": result.auto_marked_superpixels,
                "ratio_pct": 100.0 * result.marked_ratio,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.smoke_test:
        run_headless_smoke_test(args)
        return
    root = tk.Tk()
    root.option_add("*Font", ("Microsoft YaHei UI", 9))
    app = SuperpixelManualEvaluatorApp(root, args)
    if args.ui_smoke_test:
        root.withdraw()
        root.update_idletasks()
        root.after(250, app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
