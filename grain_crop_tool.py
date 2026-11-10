"""晶粒统计、可视化裁剪、尺度标定与EBSD控制点配准工具。

功能：
1. 打开常见显微图像并可视化框选裁剪区域；
2. 拖动裁剪框，拖动四角控制点并保持当前长宽比缩放；
3. 通过裁剪框实际宽/高，或图上比例尺线段完成 um/px 标定；
4. 自动排除裁剪边缘的不完整晶粒，计算面积、周长和等效圆直径；
5. 保存原始分辨率裁剪图、标定元数据和 Excel 晶粒统计表。
6. 通过对应控制点将待对齐图仿射配准到EBSD参考图并保留彩色内容导出；
7. 对随机色晶粒实例图使用最近邻配准，计算EBSD晶粒与预测晶粒的面积覆盖率，并导出Excel报告。

EBSD配准模式可单独导出对齐图，也可对随机色晶粒实例图执行重叠评估。

运行：
    D:\python\python3.11.9\python.exe grain_crop_tool.py
"""

from __future__ import annotations

import json
import math
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk
from skimage import measure


APP_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = APP_DIR / "cropped_images"
DEFAULT_STATISTICS_DIR = APP_DIR / "grain_statistics"


def _first_existing_path(candidates: list[Path]) -> Path:
    """Resolve a bundled component while retaining a useful fallback for diagnostics."""
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


RUNTIME_DIR_CANDIDATES = [
    APP_DIR / "grain_statistics_runtime",
    APP_DIR / "grain_statistics" / "grain_statistics_runtime",
]
GRAIN_STATISTICS_RUNTIME_DIR = _first_existing_path(RUNTIME_DIR_CANDIDATES)
XLSX_EXPORTER = GRAIN_STATISTICS_RUNTIME_DIR / "export_grain_statistics.mjs"
ALIGNMENT_XLSX_EXPORTER = GRAIN_STATISTICS_RUNTIME_DIR / "export_alignment_overlap_report.mjs"

_configured_node = os.environ.get("GRAIN_CROP_NODE")
_node_from_path = shutil.which("node")
NODE_EXECUTABLE_CANDIDATES = []
if _configured_node:
    NODE_EXECUTABLE_CANDIDATES.append(Path(_configured_node))
NODE_EXECUTABLE_CANDIDATES.append(
    Path(r"C:\Users\liangqin\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe")
)
if _node_from_path:
    NODE_EXECUTABLE_CANDIDATES.append(Path(_node_from_path))
NODE_EXECUTABLE = _first_existing_path(NODE_EXECUTABLE_CANDIDATES)
IMAGE_TYPES = [
    ("图像文件", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
    ("PNG", "*.png"),
    ("TIFF", "*.tif *.tiff"),
    ("JPEG", "*.jpg *.jpeg"),
    ("所有文件", "*.*"),
]


class GrainCropTool:
    HANDLE_RADIUS = 6
    HANDLE_HIT_RADIUS = 12
    MIN_CROP_PIXELS = 4

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("晶粒图像裁剪、尺度标定与EBSD对齐")
        self.root.geometry("1360x860")
        self.root.minsize(1050, 680)

        self.image_path: Path | None = None
        self.image: Image.Image | None = None
        # image 是当前显示图；neural_image 始终保存“打开图像”加载的原始神经网络结果。
        self.neural_image: Image.Image | None = None
        self.ebsd_image_path: Path | None = None
        self.ebsd_image: Image.Image | None = None
        self.registration_view = "neural"
        self.registration_points: list[dict[str, tuple[float, float] | None]] = []
        self.registration_point_buttons: list[ttk.Button] = []
        self.active_registration_point: int | None = None
        self.photo: ImageTk.PhotoImage | None = None

        # 视图变换：图像坐标 -> Canvas 坐标。
        self.view_scale = 1.0
        self.view_left = 0.0
        self.view_top = 0.0

        # 裁剪框为图像坐标下的 (left, top, right, bottom)，右/下为连续坐标。
        self.crop_box: list[float] | None = None
        self.locked_aspect: float | None = None
        self.action: str | None = None
        self.action_start = (0.0, 0.0)
        self.action_box: list[float] | None = None
        self.active_handle: str | None = None

        # 比例尺线段使用图像坐标。
        self.scale_line: list[float] | None = None
        self.scale_line_drawing = False

        # 可独立于显示图像加载的晶粒实例标签/晶界图。
        self.statistics_source_path: Path | None = None
        self.statistics_labels: np.ndarray | None = None
        self.statistics_label_method: str | None = None
        self._statistics_export_queue: queue.Queue[dict] = queue.Queue()
        self._statistics_export_thread: threading.Thread | None = None
        self._statistics_export_started_at = 0.0
        self._statistics_export_context: dict | None = None

        self.lock_aspect_var = tk.BooleanVar(value=True)
        self.calibration_mode = tk.StringVar(value="fov")
        self.fov_width_var = tk.StringVar()
        self.fov_height_var = tk.StringVar()
        self.scale_pixels_var = tk.StringVar()
        self.scale_value_var = tk.StringVar()
        self.crop_info_var = tk.StringVar(value="请先打开图像")
        self.calibration_info_var = tk.StringVar(value="尚未标定（只能统计像素单位）")
        self.statistics_source_var = tk.StringVar(value="尚未加载；打开后处理结果图时将自动查找实例标签")
        self.minimum_area_var = tk.StringVar(value="1")
        # 标定后默认按 5 μm 分组；未标定时同一输入表示 5 pixel。
        self.histogram_bin_width_var = tk.StringVar(value="5")
        self.status_var = tk.StringVar(value="打开图像后，在图像上拖动鼠标建立裁剪框。")
        self.alignment_info_var = tk.StringVar(value="尚未导入EBSD参考图。")
        self.switch_image_button_text = tk.StringVar(value="切换到 EBSD参考图")

        self._build_ui()
        self._bind_events()

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self.root, padding=(8, 7))
        toolbar.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(toolbar, text="打开图像", command=self.open_image).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="整幅图像", command=self.select_full_image).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(toolbar, text="清除裁剪框", command=self.clear_crop).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(toolbar, text="导入EBSD参考图", command=self.load_ebsd_image).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=12)
        ttk.Checkbutton(
            toolbar,
            text="缩放时锁定当前长宽比",
            variable=self.lock_aspect_var,
            command=self._aspect_lock_changed,
        ).pack(side=tk.LEFT)
        ttk.Label(
            toolbar,
            text="提示：框内拖动＝移动；拖四角＝缩放",
            foreground="#555555",
        ).pack(side=tk.LEFT, padx=16)
        ttk.Button(toolbar, text="保存裁剪图与标定信息", command=self.save_crop).pack(side=tk.RIGHT)

        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True)

        canvas_frame = ttk.Frame(body)
        side_container = ttk.Frame(body, width=355)
        body.add(canvas_frame, weight=5)
        body.add(side_container, weight=0)

        self.canvas = tk.Canvas(canvas_frame, bg="#202124", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 右侧内容较多，使用滚动面板以兼容较小屏幕。
        side_canvas = tk.Canvas(side_container, width=335, highlightthickness=0, bg="#f0f0f0")
        side_scrollbar = ttk.Scrollbar(side_container, orient=tk.VERTICAL, command=side_canvas.yview)
        side_canvas.configure(yscrollcommand=side_scrollbar.set)
        side_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        side_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        side = ttk.Frame(side_canvas, padding=10)
        side_window = side_canvas.create_window((0, 0), window=side, anchor=tk.NW)
        side.bind(
            "<Configure>",
            lambda _event: side_canvas.configure(scrollregion=side_canvas.bbox("all")),
        )
        side_canvas.bind(
            "<Configure>",
            lambda event: side_canvas.itemconfigure(side_window, width=event.width),
        )

        ttk.Label(side, text="裁剪区域", font=("Microsoft YaHei UI", 11, "bold")).pack(anchor=tk.W)
        ttk.Label(
            side,
            textvariable=self.crop_info_var,
            justify=tk.LEFT,
            wraplength=310,
        ).pack(anchor=tk.W, fill=tk.X, pady=(6, 14))

        ttk.Separator(side).pack(fill=tk.X, pady=(0, 12))
        ttk.Label(side, text="尺度标定", font=("Microsoft YaHei UI", 11, "bold")).pack(anchor=tk.W)
        ttk.Label(
            side,
            text="物理面积、周长和等效晶粒直径需要尺度标定。单位固定为 μm。",
            wraplength=310,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(5, 9))

        fov_radio = ttk.Radiobutton(
            side,
            text="按裁剪框实际宽高标定",
            variable=self.calibration_mode,
            value="fov",
            command=self._calibration_changed,
        )
        fov_radio.pack(anchor=tk.W)

        fov_frame = ttk.Frame(side, padding=(20, 5, 0, 8))
        fov_frame.pack(fill=tk.X)
        ttk.Label(fov_frame, text="裁剪框实际宽度 (μm)").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(fov_frame, textvariable=self.fov_width_var, width=14).grid(row=0, column=1, padx=(8, 0))
        ttk.Label(fov_frame, text="裁剪框实际高度 (μm)").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        ttk.Entry(fov_frame, textvariable=self.fov_height_var, width=14).grid(row=1, column=1, padx=(8, 0), pady=(5, 0))
        ttk.Label(
            fov_frame,
            text="可只填一项；只填一项时按方形像素换算另一方向。",
            foreground="#666666",
            wraplength=270,
            justify=tk.LEFT,
        ).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))

        ttk.Radiobutton(
            side,
            text="按自定义比例尺标定",
            variable=self.calibration_mode,
            value="scale_bar",
            command=self._calibration_changed,
        ).pack(anchor=tk.W)

        scale_frame = ttk.Frame(side, padding=(20, 5, 0, 8))
        scale_frame.pack(fill=tk.X)
        ttk.Button(scale_frame, text="绘制水平比例尺线", command=self.start_scale_line).grid(
            row=0, column=0, columnspan=2, sticky=tk.W
        )
        ttk.Label(scale_frame, text="自动测得长度 (pixel)").grid(row=1, column=0, sticky=tk.W, pady=(6, 0))
        ttk.Entry(scale_frame, textvariable=self.scale_pixels_var, width=14, state="readonly").grid(
            row=1, column=1, padx=(8, 0), pady=(6, 0)
        )
        ttk.Label(scale_frame, text="对应实际长度 (μm)").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        ttk.Entry(scale_frame, textvariable=self.scale_value_var, width=14).grid(
            row=2, column=1, padx=(8, 0), pady=(5, 0)
        )
        ttk.Label(
            scale_frame,
            text="线长按打开图像的原始像素坐标计算，与界面缩放比例无关。",
            foreground="#666666",
            wraplength=270,
            justify=tk.LEFT,
        ).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))

        ttk.Button(side, text="计算/刷新标定", command=self.update_information).pack(anchor=tk.W, pady=(1, 8))
        ttk.Label(
            side,
            textvariable=self.calibration_info_var,
            foreground="#145a86",
            wraplength=310,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, fill=tk.X)

        ttk.Separator(side).pack(fill=tk.X, pady=14)
        ttk.Label(side, text="保存内容", font=("Microsoft YaHei UI", 10, "bold")).pack(anchor=tk.W)
        ttk.Label(
            side,
            text=(
                "• 原始分辨率裁剪图\n"
                "• 裁剪框在原图中的坐标\n"
                "• x/y 方向 μm/pixel\n"
                "• 裁剪后实际视场宽和高\n"
                "• 后续晶粒统计所需 JSON 元数据"
            ),
            justify=tk.LEFT,
            wraplength=310,
        ).pack(anchor=tk.W, pady=(5, 0))

        self.statistics_section = ttk.Frame(side)
        self.statistics_section.pack(fill=tk.X)
        ttk.Separator(self.statistics_section).pack(fill=tk.X, pady=14)
        ttk.Label(
            self.statistics_section,
            text="等效晶粒尺寸统计",
            font=("Microsoft YaHei UI", 11, "bold"),
        ).pack(anchor=tk.W)
        ttk.Label(
            self.statistics_section,
            text=(
                "优先加载 .npy 实例标签；也可加载纯黑白晶界图。"
                "凡与裁剪框边缘相接的区域均作为不完整晶粒排除。"
            ),
            wraplength=310,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(5, 7))
        ttk.Button(
            self.statistics_section,
            text="加载实例标签 / 晶界图",
            command=self.load_statistics_source,
        ).pack(anchor=tk.W)
        ttk.Label(
            self.statistics_section,
            textvariable=self.statistics_source_var,
            foreground="#5b3b8c",
            wraplength=310,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, fill=tk.X, pady=(5, 8))

        statistics_options = ttk.Frame(self.statistics_section)
        statistics_options.pack(fill=tk.X)
        ttk.Label(statistics_options, text="最小保留面积 (pixel²)").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(statistics_options, textvariable=self.minimum_area_var, width=12).grid(row=0, column=1, padx=(8, 0))
        ttk.Label(statistics_options, text="直径分组步长（μm/pixel；留空自动）").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        ttk.Entry(statistics_options, textvariable=self.histogram_bin_width_var, width=12).grid(
            row=1, column=1, padx=(8, 0), pady=(5, 0)
        )
        self.statistics_export_button = ttk.Button(
            self.statistics_section,
            text="计算并导出 Excel",
            command=self.calculate_and_export_statistics,
        )
        self.statistics_export_button.pack(anchor=tk.W, pady=(10, 5))
        ttk.Label(
            self.statistics_section,
            text=(
                "Excel 包含标定摘要、完整晶粒逐粒数据和一张等效圆直径数量分布图。"
                "横轴为等效圆直径，纵轴为晶粒个数 Count；接触裁剪框边缘的晶粒不计入。"
            ),
            wraplength=310,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, 8))

        self.registration_section = ttk.Frame(side)
        ttk.Separator(self.registration_section).pack(fill=tk.X, pady=14)
        ttk.Label(
            self.registration_section,
            text="待对齐图 / EBSD参考图对齐",
            font=("Microsoft YaHei UI", 11, "bold"),
        ).pack(anchor=tk.W)
        ttk.Label(
            self.registration_section,
            text=(
                "添加对应点后，点击“点1、点2……”进入标点模式。"
                "在当前图像上单击可设置位置，再次单击会重置该点。"
                "至少需要3个分布均匀且不共线的完整对应点。"
            ),
            wraplength=310,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(5, 8))
        ttk.Button(self.registration_section, text="取消对齐", command=self.cancel_alignment).pack(anchor=tk.W)

        point_header = ttk.Frame(self.registration_section)
        point_header.pack(fill=tk.X, pady=(10, 4))
        ttk.Label(point_header, text="对应控制点", font=("Microsoft YaHei UI", 10, "bold")).pack(side=tk.LEFT)
        ttk.Button(point_header, text="添加点", command=self.add_registration_point).pack(side=tk.RIGHT)

        self.registration_point_box = ttk.LabelFrame(self.registration_section, text="点列表", padding=6)
        self.registration_point_box.pack(fill=tk.X)
        self.registration_points_frame = ttk.Frame(self.registration_point_box)
        self.registration_points_frame.pack(fill=tk.X)
        ttk.Label(
            self.registration_points_frame,
            text="点击“添加点”建立点1。",
            foreground="#666666",
        ).grid(row=0, column=0, sticky=tk.W)

        ttk.Button(
            self.registration_section,
            textvariable=self.switch_image_button_text,
            command=self.toggle_registration_image,
        ).pack(fill=tk.X, pady=(8, 5))
        ttk.Button(
            self.registration_section,
            text="导出对齐后的图像",
            command=self.export_aligned_image,
        ).pack(fill=tk.X)
        ttk.Button(
            self.registration_section,
            text="评估晶粒覆盖率并导出 Excel",
            command=self.export_alignment_overlap_report,
        ).pack(fill=tk.X, pady=(6, 0))
        ttk.Label(
            self.registration_section,
            text=(
                "“导出对齐图”仅进行仿射配准。“评估晶粒覆盖率”要求"
                "待对齐图为无晶界的随机色晶粒实例图，并且已完成 μm 尺度标定。"
            ),
            wraplength=310,
            justify=tk.LEFT,
            foreground="#555555",
        ).pack(anchor=tk.W, fill=tk.X, pady=(7, 0))
        ttk.Label(
            self.registration_section,
            textvariable=self.alignment_info_var,
            foreground="#145a86",
            wraplength=310,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, fill=tk.X, pady=(8, 4))

        status = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=(7, 4))
        status.pack(side=tk.BOTTOM, fill=tk.X)

    def _bind_events(self) -> None:
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<ButtonPress-1>", self._on_left_press)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_release)
        for variable in (
            self.fov_width_var,
            self.fov_height_var,
            self.scale_pixels_var,
            self.scale_value_var,
        ):
            variable.trace_add("write", lambda *_args: self.update_information(silent=True))

    def open_image(self) -> None:
        filename = filedialog.askopenfilename(
            title="选择待对齐图或其他待裁剪图像",
            initialdir=APP_DIR,
            filetypes=IMAGE_TYPES,
        )
        if not filename:
            return
        try:
            with Image.open(filename) as source:
                source.seek(0)
                self.neural_image = source.copy()
        except Exception as exc:
            messagebox.showerror("打开失败", f"无法读取图像：\n{exc}")
            return

        self.image = self.neural_image
        self.image_path = Path(filename).resolve()
        self._reset_alignment_state(show_statistics=True)
        self.crop_box = None
        self.locked_aspect = None
        self.scale_line = None
        self.scale_pixels_var.set("")
        self.action = None
        self.statistics_source_path = None
        self.statistics_labels = None
        self.statistics_label_method = None
        self.statistics_source_var.set("正在查找同目录实例标签……")
        self._auto_load_statistics_source()
        self._render()
        self.update_information(silent=True)
        self.status_var.set(
            f"已打开神经网络/待裁剪图：{self.image_path.name}，尺寸 {self.image.width} × {self.image.height} pixel。"
        )

    def _source_image(self) -> Image.Image | None:
        """返回裁剪、统计和尺度标定所对应的神经网络原图。"""
        return self.neural_image if self.neural_image is not None else self.image

    def _reset_alignment_state(self, show_statistics: bool) -> None:
        self.ebsd_image_path = None
        self.ebsd_image = None
        self.registration_view = "neural"
        self.registration_points.clear()
        self.registration_point_buttons.clear()
        self.active_registration_point = None
        self.switch_image_button_text.set("切换到 EBSD参考图")
        self.alignment_info_var.set("尚未导入EBSD参考图。")
        if self.neural_image is not None:
            self.image = self.neural_image
        if hasattr(self, "registration_points_frame"):
            self._refresh_registration_point_buttons()
        if hasattr(self, "registration_section"):
            self.registration_section.pack_forget()
        if show_statistics and hasattr(self, "statistics_section") and not self.statistics_section.winfo_manager():
            self.statistics_section.pack(fill=tk.X)

    def load_ebsd_image(self) -> None:
        if self.neural_image is None or self.image_path is None:
            messagebox.showinfo("提示", "请先点击“打开图像”导入待对齐图。")
            return
        filename = filedialog.askopenfilename(
            title="选择EBSD参考图（支持彩色图像）",
            initialdir=self.image_path.parent,
            filetypes=IMAGE_TYPES,
        )
        if not filename:
            return
        try:
            with Image.open(filename) as source:
                source.seek(0)
                # 配准点应在参考图原始视觉信息上选择。灰度图保持灰度，彩色图统一为RGB，
                # 不再为晶粒统计强制转换成灰度图。
                if source.mode in ("1", "L", "I", "I;16", "F"):
                    ebsd_image = source.convert("L").copy()
                else:
                    ebsd_image = source.convert("RGB").copy()
        except Exception as exc:
            messagebox.showerror("导入失败", f"无法读取EBSD参考图：\n{exc}")
            return

        self.ebsd_image_path = Path(filename).resolve()
        self.ebsd_image = ebsd_image
        self.registration_points.clear()
        self.active_registration_point = None
        self.registration_view = "neural"
        self.image = self.neural_image
        self.switch_image_button_text.set("切换到 EBSD参考图")
        self.statistics_section.pack_forget()
        if not self.registration_section.winfo_manager():
            self.registration_section.pack(fill=tk.X)
        self._refresh_registration_point_buttons()
        self._refresh_alignment_information()
        self._render()
        self.status_var.set(
            f"已导入EBSD参考图：{self.ebsd_image_path.name}（{self.ebsd_image.mode}模式，保留彩色信息）。"
            "请添加对应点。"
        )

    def cancel_alignment(self) -> None:
        if self.ebsd_image is None:
            return
        self._reset_alignment_state(show_statistics=True)
        self._render()
        self.update_information(silent=True)
        self.status_var.set("已取消图像对齐并卸载EBSD参考图，恢复原有裁剪与晶粒统计功能。")

    def _refresh_registration_point_buttons(self) -> None:
        if not hasattr(self, "registration_points_frame"):
            return
        for widget in self.registration_points_frame.winfo_children():
            widget.destroy()
        self.registration_point_buttons.clear()
        if not self.registration_points:
            ttk.Label(
                self.registration_points_frame,
                text="点击“添加点”建立点1。",
                foreground="#666666",
            ).grid(row=0, column=0, sticky=tk.W)
            return
        for index, point_pair in enumerate(self.registration_points):
            neural_mark = "✓" if point_pair["neural"] is not None else "○"
            ebsd_mark = "✓" if point_pair["ebsd"] is not None else "○"
            prefix = "▶ " if index == self.active_registration_point else ""
            text = f"{prefix}点{index + 1}  神{neural_mark}/E{ebsd_mark}"
            button = ttk.Button(
                self.registration_points_frame,
                text=text,
                command=lambda point_index=index: self.activate_registration_point(point_index),
            )
            button.grid(row=index // 2, column=index % 2, sticky=tk.EW, padx=2, pady=2)
            self.registration_point_buttons.append(button)
        self.registration_points_frame.columnconfigure(0, weight=1)
        self.registration_points_frame.columnconfigure(1, weight=1)

    def add_registration_point(self) -> None:
        if self.ebsd_image is None:
            messagebox.showinfo("提示", "请先导入EBSD参考图。")
            return
        self.registration_points.append({"neural": None, "ebsd": None})
        self.active_registration_point = None
        self._refresh_registration_point_buttons()
        point_number = len(self.registration_points)
        self.status_var.set(f"已添加点{point_number}；请点击点{point_number}按钮进入标点模式。")
        self._refresh_alignment_information()

    def activate_registration_point(self, index: int) -> None:
        if not 0 <= index < len(self.registration_points):
            return
        if self.active_registration_point == index:
            self.active_registration_point = None
            self._refresh_registration_point_buttons()
            self._draw_overlays()
            self.status_var.set(f"已退出点{index + 1}标点模式。")
            return
        self.active_registration_point = index
        self.scale_line_drawing = False
        self.action = None
        self._refresh_registration_point_buttons()
        self._draw_overlays()
        image_name = "待对齐图" if self.registration_view == "neural" else "EBSD参考图"
        self.status_var.set(
            f"点{index + 1}标点模式：请在{image_name}上单击；再次单击会重置该点位置。"
        )

    def toggle_registration_image(self) -> None:
        if self.ebsd_image is None or self.neural_image is None:
            messagebox.showinfo("提示", "请先导入待对齐图和EBSD参考图。")
            return
        self.action = None
        if self.registration_view == "neural":
            self.registration_view = "ebsd"
            self.image = self.ebsd_image
            self.switch_image_button_text.set("切换到 待对齐图")
            shown = "EBSD参考图"
        else:
            self.registration_view = "neural"
            self.image = self.neural_image
            self.switch_image_button_text.set("切换到 EBSD参考图")
            shown = "待对齐图"
        self._render()
        if self.active_registration_point is None:
            self.status_var.set(f"当前显示：{shown}。请选择一个点按钮后进行标点。")
        else:
            self.status_var.set(f"当前显示：{shown}。请标记点{self.active_registration_point + 1}。")

    def _mark_active_registration_point(self, x: float, y: float) -> None:
        if self.active_registration_point is None or self.ebsd_image is None:
            return
        key = "neural" if self.registration_view == "neural" else "ebsd"
        self.registration_points[self.active_registration_point][key] = (float(x), float(y))
        point_number = self.active_registration_point + 1
        self._refresh_registration_point_buttons()
        self._draw_overlays()
        self._refresh_alignment_information()
        image_name = "待对齐图" if key == "neural" else "EBSD参考图"
        self.status_var.set(
            f"已设置{image_name}的点{point_number}：({x:.2f}, {y:.2f})；再次单击可重置位置。"
        )

    def _estimate_alignment(self) -> dict:
        complete = [
            (index, pair["neural"], pair["ebsd"])
            for index, pair in enumerate(self.registration_points)
            if pair["neural"] is not None and pair["ebsd"] is not None
        ]
        if len(complete) < 3:
            raise ValueError("至少需要3个同时在两张图中标记的对应点。")

        source = np.asarray([item[1] for item in complete], dtype=np.float64)
        target = np.asarray([item[2] for item in complete], dtype=np.float64)
        design = np.zeros((source.shape[0] * 2, 6), dtype=np.float64)
        values = np.zeros(source.shape[0] * 2, dtype=np.float64)
        design[0::2, 0:3] = np.column_stack((source, np.ones(source.shape[0])))
        design[1::2, 3:6] = np.column_stack((source, np.ones(source.shape[0])))
        values[0::2] = target[:, 0]
        values[1::2] = target[:, 1]
        parameters, _residuals, rank, _singular_values = np.linalg.lstsq(design, values, rcond=None)
        if rank < 6:
            raise ValueError("控制点接近共线，无法稳定计算仿射变换；请重新选择分布更均匀的点。")
        matrix = np.asarray(
            [
                [parameters[0], parameters[1], parameters[2]],
                [parameters[3], parameters[4], parameters[5]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        determinant = float(np.linalg.det(matrix[:2, :2]))
        if not math.isfinite(determinant) or abs(determinant) < 1e-10:
            raise ValueError("计算得到的仿射变换不可逆，请检查控制点。")
        predicted = np.column_stack((source, np.ones(source.shape[0]))) @ matrix.T
        residual_vectors = predicted[:, :2] - target
        residual_pixels = np.hypot(residual_vectors[:, 0], residual_vectors[:, 1])
        rms = float(math.sqrt(float(np.mean(np.square(residual_pixels)))))
        return {
            "matrix": matrix,
            "inverse_matrix": np.linalg.inv(matrix),
            "point_indices": [item[0] for item in complete],
            "source_points": source,
            "target_points": target,
            "residual_vectors": residual_vectors,
            "residual_pixels": residual_pixels,
            "rms_pixels": rms,
            "determinant": determinant,
        }

    def _aligned_scale_payload(self, matrix: np.ndarray) -> dict:
        inverse_linear = np.linalg.inv(matrix[:2, :2])
        source_per_aligned_x = float(np.hypot(inverse_linear[0, 0], inverse_linear[1, 0]))
        source_per_aligned_y = float(np.hypot(inverse_linear[0, 1], inverse_linear[1, 1]))
        payload: dict[str, float | str | None] = {
            "method": "affine_transform_from_neural_to_ebsd",
            "source_pixels_per_aligned_pixel_x": source_per_aligned_x,
            "source_pixels_per_aligned_pixel_y": source_per_aligned_y,
            "source_pixel_area_per_aligned_pixel": abs(float(np.linalg.det(inverse_linear))),
        }
        calibration = self._calibration()
        if calibration is None:
            payload.update(
                {
                    "micrometers_per_aligned_pixel_x": None,
                    "micrometers_per_aligned_pixel_y": None,
                    "square_micrometers_per_aligned_pixel": None,
                }
            )
            return payload
        sx = float(calibration["micrometers_per_pixel_x"])
        sy = float(calibration["micrometers_per_pixel_y"])
        # 对齐图像素位移 [dx_aligned, dy_aligned] 到物理位移 [dx_um, dy_um]。
        # 保留完整2×2矩阵，避免旋转/剪切存在时仅用x、y尺度乘积近似面积。
        physical_matrix = np.asarray(
            [
                [inverse_linear[0, 0] * sx, inverse_linear[0, 1] * sx],
                [inverse_linear[1, 0] * sy, inverse_linear[1, 1] * sy],
            ],
            dtype=np.float64,
        )
        physical_x = float(np.hypot(physical_matrix[0, 0], physical_matrix[1, 0]))
        physical_y = float(np.hypot(physical_matrix[0, 1], physical_matrix[1, 1]))
        physical_area = abs(float(np.linalg.det(physical_matrix)))
        payload.update(
            {
                "micrometers_per_aligned_pixel_x": physical_x,
                "micrometers_per_aligned_pixel_y": physical_y,
                "square_micrometers_per_aligned_pixel": physical_area,
                "aligned_pixel_to_physical_um_matrix": physical_matrix.tolist(),
            }
        )
        return payload

    def _refresh_alignment_information(self) -> None:
        if self.ebsd_image is None or self.ebsd_image_path is None:
            self.alignment_info_var.set("尚未导入EBSD参考图。")
            return
        complete_count = sum(
            pair["neural"] is not None and pair["ebsd"] is not None for pair in self.registration_points
        )
        lines = [
            f"EBSD参考图：{self.ebsd_image_path.name}",
            f"图像模式：{self.ebsd_image.mode}",
            f"尺寸：{self.ebsd_image.width} × {self.ebsd_image.height} pixel",
            f"完整对应点：{complete_count}/{len(self.registration_points)}",
        ]
        if complete_count >= 3:
            try:
                alignment = self._estimate_alignment()
                lines.append(f"仿射配准 RMS：{alignment['rms_pixels']:.4g} EBSD pixel")
                if complete_count == 3:
                    lines.append("提示：3点为精确拟合，建议增加分散控制点检验误差。")
                scale = self._aligned_scale_payload(alignment["matrix"])
                scale_x = scale["micrometers_per_aligned_pixel_x"]
                scale_y = scale["micrometers_per_aligned_pixel_y"]
                if scale_x is not None and scale_y is not None:
                    lines.append(f"对齐后x尺度：{float(scale_x):.8g} μm/pixel")
                    lines.append(f"对齐后y尺度：{float(scale_y):.8g} μm/pixel")
                    area_scale = scale.get("square_micrometers_per_aligned_pixel")
                    if area_scale is not None:
                        lines.append(f"对齐后单像素面积：{float(area_scale):.8g} μm²")
                else:
                    lines.append("完成原图尺度标定后可自动计算对齐图 μm/pixel。")
            except ValueError as exc:
                lines.append(f"当前点无法配准：{exc}")
        self.alignment_info_var.set("\n".join(lines))

    @staticmethod
    def _image_fill_color(image: Image.Image) -> int | tuple[int, int, int]:
        if image.mode == "L":
            histogram = image.histogram()
            return int(max(range(len(histogram)), key=histogram.__getitem__))
        rgb = image.convert("RGB")
        corners = [
            rgb.getpixel((0, 0)),
            rgb.getpixel((rgb.width - 1, 0)),
            rgb.getpixel((0, rgb.height - 1)),
            rgb.getpixel((rgb.width - 1, rgb.height - 1)),
        ]
        return tuple(int(round(sum(value[channel] for value in corners) / len(corners))) for channel in range(3))

    def _aligned_neural_image(self, alignment: dict) -> Image.Image:
        """将待对齐图变换到EBSD参考图像素坐标系，并保留彩色内容。"""
        if self.neural_image is None or self.ebsd_image is None:
            raise ValueError("尚未同时加载待对齐图和EBSD参考图。")
        source = self.neural_image if self.neural_image.mode in ("L", "RGB") else self.neural_image.convert("RGB")
        # 二值/灰度晶界图使用最近邻以保持离散标签；彩色图使用双三次插值，
        # 避免旋转或缩放后产生明显锯齿，同时保持RGB通道。
        resample = Image.Resampling.BICUBIC if source.mode == "RGB" else Image.Resampling.NEAREST
        inverse = alignment["inverse_matrix"]
        affine_data = (
            float(inverse[0, 0]),
            float(inverse[0, 1]),
            float(inverse[0, 2]),
            float(inverse[1, 0]),
            float(inverse[1, 1]),
            float(inverse[1, 2]),
        )
        return source.transform(
            (self.ebsd_image.width, self.ebsd_image.height),
            Image.Transform.AFFINE,
            affine_data,
            resample=resample,
            fillcolor=self._image_fill_color(source),
        )

    @staticmethod
    def _boundary_mask_from_grayscale(gray: np.ndarray) -> np.ndarray:
        finite = np.asarray(gray, dtype=np.float64)
        if not np.all(np.isfinite(finite)):
            raise ValueError("晶界图包含非有限数值。")
        if float(finite.max()) <= float(finite.min()):
            raise ValueError("图像没有灰度差异，无法区分晶界和晶粒内部。")
        midpoint = (float(finite.min()) + float(finite.max())) / 2.0
        low = finite <= midpoint
        return low if int(low.sum()) <= low.size // 2 else ~low

    @staticmethod
    def _physical_perimeter_affine(mask: np.ndarray, matrix: np.ndarray) -> float:
        """使用完整仿射物理矩阵计算轮廓长度，兼容旋转、非等比缩放和剪切。"""
        physical_matrix = np.asarray(matrix, dtype=np.float64)
        if physical_matrix.shape != (2, 2) or not np.all(np.isfinite(physical_matrix)):
            raise ValueError("对齐像素到物理坐标的2×2矩阵无效。")
        padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
        perimeter = 0.0
        for contour in measure.find_contours(padded, 0.5):
            if contour.shape[0] < 2:
                continue
            delta_row_col = np.diff(contour, axis=0)
            aligned_xy = np.column_stack((delta_row_col[:, 1], delta_row_col[:, 0]))
            physical_xy = aligned_xy @ physical_matrix.T
            perimeter += float(np.hypot(physical_xy[:, 0], physical_xy[:, 1]).sum())
        return perimeter

    @staticmethod
    def _color_instance_labels(image: Image.Image) -> tuple[np.ndarray, np.ndarray, int]:
        """将随机色图转换为离散实例标签。

        先按精确RGB值区分颜色，再按四连通性分开偶然使用了同一颜色的
        不相连区域。这一步在仿射变换前完成，避免插值产生伪颜色。
        """
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        channel_spread = np.max(rgb, axis=2) - np.min(rgb, axis=2)
        if float(np.mean(channel_spread <= 3)) >= 0.995:
            raise ValueError(
                "待对齐图基本为灰度/黑白图。覆盖率评估必须输入无晶界的"
                "predicted_grains_random_color.png 随机色实例图。"
            )
        colors, inverse = np.unique(rgb.reshape(-1, 3), axis=0, return_inverse=True)
        if colors.shape[0] < 2:
            raise ValueError("待对齐图只有一种颜色，无法识别晶粒实例。")
        if colors.shape[0] > 20000:
            raise ValueError(
                f"检测到 {colors.shape[0]} 种颜色，图像像是照片或经过抗锯齿/压缩，"
                "请输入PNG格式的无晶界随机色实例图。"
            )
        color_index = inverse.reshape(rgb.shape[:2]).astype(np.int32)
        component_labels = measure.label(color_index, background=-1, connectivity=1).astype(np.int32)
        unique_labels, first_indices = np.unique(component_labels.reshape(-1), return_index=True)
        palette = np.zeros((int(component_labels.max()) + 1, 3), dtype=np.uint8)
        palette[unique_labels] = rgb.reshape(-1, 3)[first_indices]
        return component_labels, palette, int(colors.shape[0])

    @staticmethod
    def _warp_instance_labels(
        source_labels: np.ndarray,
        source_palette: np.ndarray,
        matrix: np.ndarray,
        output_size: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """用最近邻插值将实例标签配准到EBSD坐标系。"""
        width, height = output_size
        affine = np.asarray(matrix[:2, :], dtype=np.float64)
        warped = cv2.warpAffine(
            source_labels.astype(np.float32),
            affine,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        valid_mask = cv2.warpAffine(
            np.ones(source_labels.shape, dtype=np.uint8),
            affine,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)
        warped_source = np.rint(warped).astype(np.int32)
        warped_source[~valid_mask] = 0
        source_ids = np.unique(warped_source[valid_mask & (warped_source > 0)])
        if source_ids.size == 0:
            raise ValueError("仿射变换后没有晶粒落入EBSD参考图，请检查对应点。")

        aligned_labels = np.zeros((height, width), dtype=np.int32)
        positive = valid_mask & (warped_source > 0)
        aligned_labels[positive] = np.searchsorted(source_ids, warped_source[positive]).astype(np.int32) + 1
        aligned_labels[~valid_mask] = -1

        output_palette = np.full((source_ids.size + 1, 3), 255, dtype=np.uint8)
        output_palette[1:] = source_palette[source_ids]
        aligned_rgb = np.full((height, width, 3), 255, dtype=np.uint8)
        aligned_rgb[positive] = output_palette[aligned_labels[positive]]

        records: list[dict] = []
        for region in measure.regionprops(np.where(aligned_labels > 0, aligned_labels, 0)):
            source_component_id = int(source_ids[region.label - 1])
            color = output_palette[region.label]
            records.append(
                {
                    "predicted_grain_id": int(region.label),
                    "source_component_id": source_component_id,
                    "rgb": [int(color[0]), int(color[1]), int(color[2])],
                    "area_pixels": int(region.area),
                    "centroid_x_pixels": float(region.centroid[1]),
                    "centroid_y_pixels": float(region.centroid[0]),
                }
            )
        return aligned_labels, aligned_rgb, valid_mask, records

    @classmethod
    def _ebsd_boundary_and_regions(
        cls,
        image: Image.Image,
        valid_prediction_mask: np.ndarray,
        minimum_area: int = 4,
    ) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
        """从EBSD参考图提取晶界并只保留闭合、完整落入预测覆盖区的晶粒。"""
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        rgb_float = rgb.astype(np.float64)
        spread = rgb_float.max(axis=2) - rgb_float.min(axis=2)
        grayscale_like = float(np.mean(spread <= 3.0)) >= 0.995
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if grayscale_like:
            boundary = cls._boundary_mask_from_grayscale(gray)
            method = "grayscale_minority_boundary"
        else:
            black = np.max(rgb, axis=2) <= 20
            black_fraction = float(np.mean(black))
            if 0.001 <= black_fraction <= 0.45:
                boundary = black
                method = "near_black_rgb_boundary"
            else:
                _threshold, low = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                boundary = low.astype(bool)
                if int(boundary.sum()) > boundary.size // 2:
                    boundary = ~boundary
                method = "otsu_dark_minority_boundary"

        # EBSD是参考真值，不对其执行膨胀、闭运算或断线修补，
        # 避免后处理改变标准晶界的面积与拓扑结构。
        components = measure.label(~boundary, connectivity=1).astype(np.int32)
        border_ids = set(
            int(value)
            for value in np.unique(
                np.concatenate((components[0, :], components[-1, :], components[:, 0], components[:, -1]))
            )
            if value > 0
        )
        output = np.zeros(components.shape, dtype=np.int32)
        records: list[dict] = []
        excluded_border = 0
        excluded_small = 0
        excluded_incomplete_prediction = 0
        for region in measure.regionprops(components):
            if region.label in border_ids:
                excluded_border += 1
                continue
            if int(region.area) < minimum_area:
                excluded_small += 1
                continue
            coords = region.coords
            valid_fraction = float(np.mean(valid_prediction_mask[coords[:, 0], coords[:, 1]]))
            if valid_fraction < 0.99:
                excluded_incomplete_prediction += 1
                continue
            grain_id = len(records) + 1
            output[coords[:, 0], coords[:, 1]] = grain_id
            records.append(
                {
                    "ebsd_grain_id": grain_id,
                    "source_component_id": int(region.label),
                    "area_pixels": int(region.area),
                    "centroid_x_pixels": float(region.centroid[1]),
                    "centroid_y_pixels": float(region.centroid[0]),
                    "prediction_valid_fraction": valid_fraction,
                }
            )
        metadata = {
            "boundary_method": method,
            "boundary_pixel_count": int(boundary.sum()),
            "boundary_fraction": float(boundary.mean()),
            "minimum_ebsd_grain_area_pixels": int(minimum_area),
            "excluded_border_regions": excluded_border,
            "excluded_small_regions": excluded_small,
            "excluded_incomplete_prediction_regions": excluded_incomplete_prediction,
        }
        return boundary, output, records, metadata

    @staticmethod
    def _recovery_rule(diameter_um: float) -> tuple[str, float]:
        if diameter_um < 5.0:
            return "<5 μm", 0.40
        if diameter_um < 10.0:
            return "5–10 μm", 0.75
        if diameter_um < 20.0:
            return "10–20 μm", 0.90
        return "≥20 μm", 0.95

    @classmethod
    def _evaluate_instance_overlap(
        cls,
        predicted_labels: np.ndarray,
        predicted_records: list[dict],
        ebsd_labels: np.ndarray,
        ebsd_records: list[dict],
        pixel_area_um2: float,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """按EBSD晶粒内占比最大的预测实例计算覆盖率，并执行一对一去重。"""
        if not math.isfinite(pixel_area_um2) or pixel_area_um2 <= 0:
            raise ValueError("对齐后单像素物理面积无效。")
        pred_areas = np.bincount(predicted_labels[predicted_labels > 0].astype(np.int64))
        maximum_predicted_id = int(predicted_labels.max())
        intersections: dict[int, list[tuple[int, int]]] = {}
        valid_pairs = (ebsd_labels > 0) & (predicted_labels > 0)
        if np.any(valid_pairs):
            keys = (
                ebsd_labels[valid_pairs].astype(np.int64) * (maximum_predicted_id + 1)
                + predicted_labels[valid_pairs].astype(np.int64)
            )
            unique_keys, counts = np.unique(keys, return_counts=True)
            for key, count in zip(unique_keys, counts, strict=True):
                ebsd_id = int(key // (maximum_predicted_id + 1))
                predicted_id = int(key % (maximum_predicted_id + 1))
                intersections.setdefault(ebsd_id, []).append((predicted_id, int(count)))

        grain_rows: list[dict] = []
        candidates_by_predicted: dict[int, list[dict]] = {}
        for ebsd in ebsd_records:
            ebsd_id = int(ebsd["ebsd_grain_id"])
            area_pixels = int(ebsd["area_pixels"])
            area_um2 = area_pixels * pixel_area_um2
            diameter_um = 2.0 * math.sqrt(area_um2 / math.pi)
            size_group, required_coverage = cls._recovery_rule(diameter_um)
            candidates = intersections.get(ebsd_id, [])
            if candidates:
                predicted_id, intersection_pixels = max(candidates, key=lambda item: (item[1], -item[0]))
                predicted_area = int(pred_areas[predicted_id]) if predicted_id < pred_areas.size else 0
                coverage = float(intersection_pixels / predicted_area) if predicted_area else 0.0
                ebsd_fill = float(intersection_pixels / area_pixels) if area_pixels else 0.0
            else:
                predicted_id = 0
                intersection_pixels = 0
                predicted_area = 0
                coverage = 0.0
                ebsd_fill = 0.0
            row = {
                **ebsd,
                "area_um2": area_um2,
                "equivalent_diameter_um": diameter_um,
                "size_group": size_group,
                "required_coverage": required_coverage,
                "dominant_predicted_grain_id": predicted_id if predicted_id > 0 else None,
                "intersection_pixels": intersection_pixels,
                "predicted_grain_total_pixels": predicted_area,
                "coverage": coverage,
                "ebsd_fill_fraction": ebsd_fill,
                "unique_assignment": False,
                "correctly_recovered": False,
                "status": "no_predicted_instance" if predicted_id == 0 else "candidate",
            }
            grain_rows.append(row)
            if predicted_id > 0:
                candidates_by_predicted.setdefault(predicted_id, []).append(row)

        for predicted_id, candidates in candidates_by_predicted.items():
            winner = max(
                candidates,
                key=lambda row: (row["intersection_pixels"], row["coverage"], -row["ebsd_grain_id"]),
            )
            for row in candidates:
                if row is winner:
                    row["unique_assignment"] = True
                    row["correctly_recovered"] = bool(row["coverage"] >= row["required_coverage"])
                    row["status"] = "recovered" if row["correctly_recovered"] else "below_coverage_threshold"
                else:
                    row["status"] = "duplicate_predicted_instance"

        predicted_by_id = {int(row["predicted_grain_id"]): dict(row) for row in predicted_records}
        for predicted_id, predicted in predicted_by_id.items():
            predicted["area_um2"] = int(predicted["area_pixels"]) * pixel_area_um2
            predicted["equivalent_diameter_um"] = 2.0 * math.sqrt(predicted["area_um2"] / math.pi)
            candidates = candidates_by_predicted.get(predicted_id, [])
            winner = next((row for row in candidates if row["unique_assignment"]), None)
            predicted["matched_ebsd_grain_id"] = winner["ebsd_grain_id"] if winner else None
            predicted["intersection_pixels"] = winner["intersection_pixels"] if winner else 0
            predicted["coverage"] = winner["coverage"] if winner else 0.0
            predicted["correctly_recovered"] = bool(winner and winner["correctly_recovered"])

        size_group_order = [("<5 μm", 0.40), ("5–10 μm", 0.75), ("10–20 μm", 0.90), ("≥20 μm", 0.95)]
        summary: list[dict] = []
        for group, threshold in size_group_order:
            rows = [row for row in grain_rows if row["size_group"] == group]
            recovered = sum(bool(row["correctly_recovered"]) for row in rows)
            summary.append(
                {
                    "size_group": group,
                    "required_coverage": threshold,
                    "ebsd_grain_count": len(rows),
                    "correctly_recovered": recovered,
                    "recovery_rate": float(recovered / len(rows)) if rows else None,
                    "mean_coverage": float(np.mean([row["coverage"] for row in rows])) if rows else None,
                    "median_coverage": float(np.median([row["coverage"] for row in rows])) if rows else None,
                }
            )
        return grain_rows, list(predicted_by_id.values()), summary

    @staticmethod
    def _numbered_instance_image(
        rgb: np.ndarray,
        predicted_records: list[dict],
        boundary_mask: np.ndarray | None = None,
    ) -> Image.Image:
        canvas = np.asarray(rgb, dtype=np.uint8).copy()
        if boundary_mask is not None:
            canvas[np.asarray(boundary_mask, dtype=bool)] = (0, 0, 0)
        image = Image.fromarray(canvas, mode="RGB")
        draw = ImageDraw.Draw(image)
        font_size = max(9, min(18, int(round(min(image.size) / 38))))
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()
        for record in predicted_records:
            x = float(record["centroid_x_pixels"])
            y = float(record["centroid_y_pixels"])
            red, green, blue = record["rgb"]
            luminance = 0.299 * red + 0.587 * green + 0.114 * blue
            fill = "#FFFFFF" if luminance < 125 else "#000000"
            stroke = "#000000" if fill == "#FFFFFF" else "#FFFFFF"
            draw.text(
                (x, y),
                str(record["predicted_grain_id"]),
                fill=fill,
                font=font,
                anchor="mm",
                stroke_width=1,
                stroke_fill=stroke,
            )
        return image

    @staticmethod
    def _run_excel_exporter(
        exporter: Path,
        payload: dict,
        output_path: Path,
        timeout_seconds: int,
        temporary_prefix: str,
    ) -> None:
        """Run the Node exporter atomically and validate the resulting XLSX container."""
        missing: list[str] = []
        if not NODE_EXECUTABLE.is_file():
            searched = "\n".join(f"  - {path}" for path in NODE_EXECUTABLE_CANDIDATES)
            missing.append(f"Node.js 未找到，已检查：\n{searched}")
        if not exporter.is_file():
            searched = "\n".join(f"  - {path / exporter.name}" for path in RUNTIME_DIR_CANDIDATES)
            missing.append(f"Excel 导出脚本未找到，已检查：\n{searched}")
        if missing:
            raise RuntimeError("\n\n".join(missing))

        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            raise FileExistsError(f"目标文件已存在，为避免覆盖已停止导出：{output_path}")

        # Write to a hidden sibling first.  A failed/terminated exporter can no
        # longer leave a zero-byte or half-written file under the chosen name.
        temporary_output = output_path.with_name(
            f".{output_path.stem}.{os.getpid()}.{threading.get_ident()}.partial.xlsx"
        )
        temporary_output.unlink(missing_ok=True)
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            with tempfile.TemporaryDirectory(prefix=temporary_prefix) as temporary_directory:
                input_json = Path(temporary_directory) / "payload.json"
                input_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                try:
                    completed = subprocess.run(
                        [str(NODE_EXECUTABLE), str(exporter), str(input_json), str(temporary_output)],
                        cwd=exporter.parent,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=timeout_seconds,
                        check=False,
                        creationflags=creation_flags,
                    )
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError(
                        f"Excel 导出超过 {timeout_seconds} 秒，已终止。"
                        "请减小统计区域；若已设置 GRAIN_EXPORT_VERIFY=1，请取消该设置后重试。"
                    ) from exc

            if completed.returncode != 0:
                details = completed.stderr.strip() or completed.stdout.strip() or "未返回诊断信息"
                raise RuntimeError(f"Excel 导出进程失败（退出码 {completed.returncode}）：\n{details}")
            if not temporary_output.is_file() or temporary_output.stat().st_size == 0:
                details = completed.stdout.strip() or completed.stderr.strip() or "未返回诊断信息"
                raise RuntimeError(f"导出进程未生成工作簿：\n{details}")
            if not zipfile.is_zipfile(temporary_output):
                raise RuntimeError("导出结果不是有效的 .xlsx 文件。")
            with zipfile.ZipFile(temporary_output, "r") as archive:
                required_entries = {"[Content_Types].xml", "xl/workbook.xml"}
                missing_entries = required_entries.difference(archive.namelist())
                if missing_entries:
                    raise RuntimeError(
                        "导出的 .xlsx 缺少必要结构：" + ", ".join(sorted(missing_entries))
                    )
            if output_path.exists():
                raise FileExistsError(f"导出期间目标文件已被其他程序创建：{output_path}")
            temporary_output.replace(output_path)
        finally:
            temporary_output.unlink(missing_ok=True)

    def _write_alignment_overlap_workbook(self, payload: dict, output_path: Path) -> None:
        self._run_excel_exporter(
            ALIGNMENT_XLSX_EXPORTER,
            payload,
            output_path,
            timeout_seconds=300,
            temporary_prefix="ebsd_overlap_",
        )

    def export_aligned_image(self) -> None:
        if self.neural_image is None or self.image_path is None or self.ebsd_image is None or self.ebsd_image_path is None:
            messagebox.showinfo("提示", "请先导入待对齐图和EBSD参考图。")
            return
        try:
            alignment = self._estimate_alignment()
        except ValueError as exc:
            messagebox.showerror("无法导出", str(exc))
            return

        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        filename = filedialog.asksaveasfilename(
            title="保存对齐后的图像",
            initialdir=DEFAULT_OUTPUT_DIR,
            initialfile=f"{self.image_path.stem}_aligned_to_{self.ebsd_image_path.stem}.png",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("TIFF", "*.tif *.tiff"), ("JPEG", "*.jpg *.jpeg")],
        )
        if not filename:
            return
        requested_path = Path(filename).resolve()
        output_path = self._non_overwriting_path(requested_path)
        metadata_path = self._non_overwriting_path(output_path.with_suffix(".alignment.json"))

        try:
            aligned = self._aligned_neural_image(alignment)
            if output_path.suffix.lower() in (".jpg", ".jpeg") and aligned.mode != "RGB":
                aligned = aligned.convert("RGB")
            aligned.save(output_path)

            aligned_scale = self._aligned_scale_payload(alignment["matrix"])
            point_rows = []
            for local_index, original_index in enumerate(alignment["point_indices"]):
                point_rows.append(
                    {
                        "point_id": original_index + 1,
                        "neural_xy_pixels": alignment["source_points"][local_index].tolist(),
                        "ebsd_xy_pixels": alignment["target_points"][local_index].tolist(),
                        "residual_xy_pixels": alignment["residual_vectors"][local_index].tolist(),
                        "residual_distance_pixels": float(alignment["residual_pixels"][local_index]),
                    }
                )
            metadata = {
                "format_version": 1,
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "transform_type": "affine",
                "moving_image": str(self.image_path),
                "moving_image_size_pixels": {
                    "width": self.neural_image.width,
                    "height": self.neural_image.height,
                },
                "reference_ebsd_image": str(self.ebsd_image_path),
                "moving_image_mode": self.neural_image.mode,
                "reference_ebsd_image_mode": self.ebsd_image.mode,
                "reference_ebsd_color_preserved": self.ebsd_image.mode == "RGB",
                "aligned_output_size_pixels": {
                    "width": self.ebsd_image.width,
                    "height": self.ebsd_image.height,
                },
                "neural_to_ebsd_matrix": alignment["matrix"].tolist(),
                "ebsd_to_neural_inverse_matrix": alignment["inverse_matrix"].tolist(),
                "affine_linear_determinant": alignment["determinant"],
                "rms_registration_error_ebsd_pixels": alignment["rms_pixels"],
                "control_points": point_rows,
                "source_calibration": self._calibration(),
                "aligned_calibration": aligned_scale,
                "resampling": "bicubic" if aligned.mode == "RGB" else "nearest",
                "notes": (
                    "输出图位于EBSD参考图像素坐标系；彩色图采用双三次插值，"
                    "灰度/二值图采用最近邻插值。配准流程不执行晶粒识别或尺寸统计。"
                ),
            }
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            messagebox.showerror("导出失败", f"无法生成对齐图或元数据：\n{exc}")
            return

        scale_x = aligned_scale["micrometers_per_aligned_pixel_x"]
        scale_y = aligned_scale["micrometers_per_aligned_pixel_y"]
        scale_note = (
            f"对齐后尺度：x={float(scale_x):.8g}、y={float(scale_y):.8g} μm/pixel"
            if scale_x is not None and scale_y is not None
            else "尚未完成物理尺度标定；元数据中已保存相对像素尺度。"
        )
        messagebox.showinfo(
            "导出完成",
            f"对齐图：\n{output_path}\n\n配准元数据：\n{metadata_path}\n\n"
            f"RMS误差：{alignment['rms_pixels']:.4g} EBSD pixel\n{scale_note}",
        )
        self.status_var.set(f"对齐图导出完成：{output_path.name}")

    def export_alignment_overlap_report(self) -> None:
        """配准随机色晶粒实例，与EBSD闭合晶粒匹配并导出覆盖率报告。"""
        if self.neural_image is None or self.image_path is None or self.ebsd_image is None or self.ebsd_image_path is None:
            messagebox.showinfo("提示", "请先导入无晶界随机色晶粒图和EBSD参考图。")
            return
        calibration = self._calibration()
        if calibration is None:
            messagebox.showinfo(
                "需要尺度标定",
                "请先在待对齐原图上填写原始视场尺寸，或绘制比例尺线并填写其实际长度。\n"
                "尺寸分组和等效圆直径必须使用 μm 标定。",
            )
            return
        try:
            alignment = self._estimate_alignment()
            aligned_scale = self._aligned_scale_payload(alignment["matrix"])
            pixel_area_um2 = aligned_scale.get("square_micrometers_per_aligned_pixel")
            if pixel_area_um2 is None or not math.isfinite(float(pixel_area_um2)) or float(pixel_area_um2) <= 0:
                raise ValueError("无法根据当前标定和仿射矩阵计算对齐后像素面积。")
            self.status_var.set("正在解析随机色晶粒并使用最近邻插值配准……")
            self.root.update_idletasks()
            source_labels, source_palette, source_color_count = self._color_instance_labels(self.neural_image)
            predicted_labels, aligned_rgb, valid_mask, predicted_records = self._warp_instance_labels(
                source_labels,
                source_palette,
                alignment["matrix"],
                (self.ebsd_image.width, self.ebsd_image.height),
            )
            boundary_mask, ebsd_labels, ebsd_records, ebsd_metadata = self._ebsd_boundary_and_regions(
                self.ebsd_image,
                valid_mask,
            )
            if not ebsd_records:
                raise ValueError(
                    "EBSD参考图中未检测到同时满足闭合、面积阈值且完整位于预测覆盖区的晶粒。"
                )
            ebsd_rows, predicted_rows, summary_rows = self._evaluate_instance_overlap(
                predicted_labels,
                predicted_records,
                ebsd_labels,
                ebsd_records,
                float(pixel_area_um2),
            )
        except Exception as exc:
            messagebox.showerror("无法评估", str(exc))
            return

        DEFAULT_STATISTICS_DIR.mkdir(parents=True, exist_ok=True)
        filename = filedialog.asksaveasfilename(
            title="保存EBSD晶粒覆盖率报告",
            initialdir=DEFAULT_STATISTICS_DIR,
            initialfile=f"{self.image_path.stem}_aligned_ebsd_overlap_report.xlsx",
            defaultextension=".xlsx",
            filetypes=[("Excel 工作簿", "*.xlsx")],
        )
        if not filename:
            return
        workbook_path = self._non_overwriting_path(Path(filename).resolve())
        output_directory = workbook_path.parent
        output_stem = workbook_path.stem.removesuffix("_report")
        aligned_path = self._non_overwriting_path(output_directory / f"{output_stem}_instances.png")
        numbered_path = self._non_overwriting_path(output_directory / f"{output_stem}_instances_numbered.png")
        overlay_path = self._non_overwriting_path(output_directory / f"{output_stem}_with_ebsd_boundaries.png")
        numbered_overlay_path = self._non_overwriting_path(
            output_directory / f"{output_stem}_with_ebsd_boundaries_numbered.png"
        )
        label_path = self._non_overwriting_path(output_directory / f"{output_stem}_instance_labels.npy")
        metadata_path = self._non_overwriting_path(output_directory / f"{output_stem}_overlap.json")

        point_rows = []
        for local_index, original_index in enumerate(alignment["point_indices"]):
            point_rows.append(
                {
                    "point_id": original_index + 1,
                    "moving_xy_pixels": alignment["source_points"][local_index].tolist(),
                    "ebsd_xy_pixels": alignment["target_points"][local_index].tolist(),
                    "residual_xy_pixels": alignment["residual_vectors"][local_index].tolist(),
                    "residual_distance_pixels": float(alignment["residual_pixels"][local_index]),
                }
            )
        payload = {
            "format_version": 1,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "moving_random_color_image": str(self.image_path),
            "reference_ebsd_image": str(self.ebsd_image_path),
            "moving_image_size_pixels": {"width": self.neural_image.width, "height": self.neural_image.height},
            "aligned_output_size_pixels": {"width": self.ebsd_image.width, "height": self.ebsd_image.height},
            "source_exact_color_count": source_color_count,
            "source_connected_instance_count": int(source_labels.max()),
            "aligned_predicted_instance_count": len(predicted_rows),
            "evaluated_ebsd_grain_count": len(ebsd_rows),
            "registration": {
                "transform_type": "affine_least_squares",
                "neural_to_ebsd_matrix": alignment["matrix"].tolist(),
                "ebsd_to_neural_inverse_matrix": alignment["inverse_matrix"].tolist(),
                "affine_linear_determinant": alignment["determinant"],
                "rms_registration_error_ebsd_pixels": alignment["rms_pixels"],
                "control_points": point_rows,
            },
            "source_calibration": calibration,
            "aligned_calibration": aligned_scale,
            "ebsd_segmentation": ebsd_metadata,
            "matching_definition": {
                "dominant_instance": "predicted instance with the largest pixel count inside each EBSD grain",
                "coverage": "dominant predicted pixels inside EBSD grain / total pixels of that predicted instance in aligned frame",
                "one_to_one_rule": (
                    "if one predicted instance dominates multiple EBSD grains, only the EBSD grain with the largest "
                    "intersection can be counted as recovered"
                ),
                "thresholds": {"<5 μm": 0.40, "5–10 μm": 0.75, "10–20 μm": 0.90, "≥20 μm": 0.95},
                "size_basis": "EBSD equivalent circle diameter",
            },
            "summary": summary_rows,
            "ebsd_grains": ebsd_rows,
            "predicted_grains": predicted_rows,
            "outputs": {
                "excel": str(workbook_path),
                "aligned_instances": str(aligned_path),
                "numbered_instances": str(numbered_path),
                "ebsd_boundary_overlay": str(overlay_path),
                "numbered_ebsd_boundary_overlay": str(numbered_overlay_path),
                "aligned_instance_labels": str(label_path),
                "metadata_json": str(metadata_path),
            },
        }

        try:
            self.status_var.set("正在保存离散实例图、编号图、EBSD叠加图和 Excel……")
            self.root.update_idletasks()
            Image.fromarray(aligned_rgb, mode="RGB").save(aligned_path)
            self._numbered_instance_image(aligned_rgb, predicted_rows).save(numbered_path)
            overlay_rgb = aligned_rgb.copy()
            overlay_rgb[boundary_mask] = (0, 0, 0)
            Image.fromarray(overlay_rgb, mode="RGB").save(overlay_path)
            self._numbered_instance_image(aligned_rgb, predicted_rows, boundary_mask).save(numbered_overlay_path)
            np.save(label_path, predicted_labels.astype(np.int32), allow_pickle=False)
            metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            self._write_alignment_overlap_workbook(payload, workbook_path)
        except Exception as exc:
            messagebox.showerror("导出失败", f"无法完成重叠评估导出：\n{exc}")
            return

        total_ebsd = len(ebsd_rows)
        recovered = sum(bool(row["correctly_recovered"]) for row in ebsd_rows)
        recovery_rate = recovered / total_ebsd if total_ebsd else 0.0
        messagebox.showinfo(
            "覆盖率评估完成",
            f"EBSD闭合晶粒：{total_ebsd} 个\n"
            f"正确恢复：{recovered} 个\n"
            f"总恢复率：{recovery_rate:.2%}\n"
            f"配准RMS：{alignment['rms_pixels']:.4g} EBSD pixel\n\n"
            f"Excel：\n{workbook_path}\n\n"
            f"编号叠加图：\n{numbered_overlay_path}",
        )
        self.status_var.set(f"EBSD重叠评估完成：{recovered}/{total_ebsd}（{recovery_rate:.2%}）")

    def _on_canvas_resize(self, _event: tk.Event) -> None:
        if self.image is not None:
            self._render()

    def _render(self) -> None:
        self.canvas.delete("all")
        if self.image is None:
            self.canvas.create_text(
                max(self.canvas.winfo_width() / 2, 100),
                max(self.canvas.winfo_height() / 2, 100),
                text="请先打开神经网络晶界图、原始图或其他待裁剪图像",
                fill="#d0d0d0",
                font=("Microsoft YaHei UI", 14),
            )
            return

        cw = max(self.canvas.winfo_width(), 2)
        ch = max(self.canvas.winfo_height(), 2)
        margin = 18
        self.view_scale = min((cw - 2 * margin) / self.image.width, (ch - 2 * margin) / self.image.height)
        self.view_scale = max(self.view_scale, 1e-6)
        display_w = max(1, round(self.image.width * self.view_scale))
        display_h = max(1, round(self.image.height * self.view_scale))
        self.view_left = (cw - display_w) / 2
        self.view_top = (ch - display_h) / 2

        display = self.image.resize((display_w, display_h), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(display)
        self.canvas.create_image(self.view_left, self.view_top, image=self.photo, anchor=tk.NW, tags="base_image")
        self.canvas.create_rectangle(
            self.view_left,
            self.view_top,
            self.view_left + display_w,
            self.view_top + display_h,
            outline="#6f7378",
        )
        self._draw_overlays()

    def _draw_overlays(self) -> None:
        self.canvas.delete("overlay")
        if self.image is None:
            return

        show_source_overlays = self.ebsd_image is None or self.registration_view == "neural"
        if show_source_overlays and self.crop_box is not None:
            l, t, r, b = self.crop_box
            x1, y1 = self._image_to_canvas(l, t)
            x2, y2 = self._image_to_canvas(r, b)
            # 框外半透明遮罩在 Tk Canvas 中不可移植，用加粗轮廓保持清晰。
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="#00ff84", width=3, tags="overlay")
            handles = {"nw": (x1, y1), "ne": (x2, y1), "sw": (x1, y2), "se": (x2, y2)}
            for x, y in handles.values():
                d = self.HANDLE_RADIUS
                self.canvas.create_rectangle(
                    x - d, y - d, x + d, y + d, fill="#ffffff", outline="#008f4c", width=2, tags="overlay"
                )
            info = f"{round(r - l)} × {round(b - t)} px"
            self.canvas.create_text(
                x1 + 5,
                max(self.view_top + 10, y1 - 10),
                text=info,
                fill="#00ff84",
                anchor=tk.SW,
                font=("Microsoft YaHei UI", 10, "bold"),
                tags="overlay",
            )

        if show_source_overlays and self.scale_line is not None:
            x1, y1 = self._image_to_canvas(self.scale_line[0], self.scale_line[1])
            x2, y2 = self._image_to_canvas(self.scale_line[2], self.scale_line[3])
            self.canvas.create_line(x1, y1, x2, y2, fill="#ffd400", width=4, tags="overlay")
            self.canvas.create_oval(x1 - 5, y1 - 5, x1 + 5, y1 + 5, fill="#ffd400", tags="overlay")
            self.canvas.create_oval(x2 - 5, y2 - 5, x2 + 5, y2 + 5, fill="#ffd400", tags="overlay")
            # 比例尺只允许沿原图水平方向绘制，因此像素长度就是原图x坐标之差。
            length = abs(self.scale_line[2] - self.scale_line[0])
            self.canvas.create_text(
                (x1 + x2) / 2,
                (y1 + y2) / 2 - 11,
                text=f"{length:.2f} px",
                fill="#ffd400",
                font=("Microsoft YaHei UI", 10, "bold"),
                tags="overlay",
            )

        if self.ebsd_image is not None:
            point_key = "neural" if self.registration_view == "neural" else "ebsd"
            for index, point_pair in enumerate(self.registration_points):
                point = point_pair[point_key]
                if point is None:
                    continue
                cx, cy = self._image_to_canvas(point[0], point[1])
                active = index == self.active_registration_point
                color = "#ffd400" if active else "#00e5ff"
                radius = 7 if active else 6
                self.canvas.create_oval(
                    cx - radius,
                    cy - radius,
                    cx + radius,
                    cy + radius,
                    fill=color,
                    outline="#202124",
                    width=2,
                    tags="overlay",
                )
                self.canvas.create_text(
                    cx + 10,
                    cy - 10,
                    text=str(index + 1),
                    fill=color,
                    anchor=tk.SW,
                    font=("Microsoft YaHei UI", 11, "bold"),
                    tags="overlay",
                )

    def _image_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        return self.view_left + x * self.view_scale, self.view_top + y * self.view_scale

    def _canvas_to_image(self, x: float, y: float) -> tuple[float, float]:
        if self.image is None:
            return 0.0, 0.0
        ix = (x - self.view_left) / self.view_scale
        iy = (y - self.view_top) / self.view_scale
        return self._clamp_point(ix, iy)

    def _clamp_point(self, x: float, y: float) -> tuple[float, float]:
        assert self.image is not None
        return min(max(x, 0.0), float(self.image.width)), min(max(y, 0.0), float(self.image.height))

    def _point_in_displayed_image(self, canvas_x: float, canvas_y: float) -> bool:
        if self.image is None:
            return False
        x, y = self._image_to_canvas(self.image.width, self.image.height)
        return self.view_left <= canvas_x <= x and self.view_top <= canvas_y <= y

    def _hit_handle(self, canvas_x: float, canvas_y: float) -> str | None:
        if self.crop_box is None:
            return None
        l, t, r, b = self.crop_box
        positions = {
            "nw": self._image_to_canvas(l, t),
            "ne": self._image_to_canvas(r, t),
            "sw": self._image_to_canvas(l, b),
            "se": self._image_to_canvas(r, b),
        }
        for name, (x, y) in positions.items():
            if math.hypot(canvas_x - x, canvas_y - y) <= self.HANDLE_HIT_RADIUS:
                return name
        return None

    def _inside_crop(self, x: float, y: float) -> bool:
        return self.crop_box is not None and self.crop_box[0] <= x <= self.crop_box[2] and self.crop_box[1] <= y <= self.crop_box[3]

    def _on_left_press(self, event: tk.Event) -> None:
        if self.image is None or not self._point_in_displayed_image(event.x, event.y):
            return
        x, y = self._canvas_to_image(event.x, event.y)

        if self.ebsd_image is not None and self.active_registration_point is not None:
            self._mark_active_registration_point(x, y)
            self.action = None
            return
        if self.ebsd_image is not None and self.registration_view == "ebsd":
            self.status_var.set("当前为EBSD图；请先点击一个点按钮进入标点模式。")
            return

        if self.scale_line_drawing:
            self.action = "scale_line"
            self.scale_line = [x, y, x, y]
            self._draw_overlays()
            return

        handle = self._hit_handle(event.x, event.y)
        if handle is not None:
            self.action = "resize"
            self.active_handle = handle
            self.action_box = list(self.crop_box) if self.crop_box else None
        elif self._inside_crop(x, y):
            self.action = "move"
            self.action_start = (x, y)
            self.action_box = list(self.crop_box) if self.crop_box else None
        else:
            self.action = "create"
            self.action_start = (x, y)
            self.crop_box = [x, y, x, y]
            self.locked_aspect = None
        self._draw_overlays()

    def _on_left_drag(self, event: tk.Event) -> None:
        if self.image is None or self.action is None:
            return
        x, y = self._canvas_to_image(event.x, event.y)

        if self.action == "scale_line" and self.scale_line is not None:
            # 鼠标纵向移动不改变终点高度，确保标定线在原图坐标中严格水平。
            self.scale_line[2:] = [x, self.scale_line[1]]
        elif self.action == "create":
            sx, sy = self.action_start
            self.crop_box = [min(sx, x), min(sy, y), max(sx, x), max(sy, y)]
        elif self.action == "move" and self.action_box is not None:
            sx, sy = self.action_start
            dx, dy = x - sx, y - sy
            l, t, r, b = self.action_box
            dx = min(max(dx, -l), self.image.width - r)
            dy = min(max(dy, -t), self.image.height - b)
            self.crop_box = [l + dx, t + dy, r + dx, b + dy]
        elif self.action == "resize" and self.action_box is not None and self.active_handle is not None:
            self.crop_box = self._resized_box(x, y, self.active_handle, self.action_box)

        self._draw_overlays()
        self.update_information(silent=True)

    def _on_left_release(self, _event: tk.Event) -> None:
        if self.action == "scale_line" and self.scale_line is not None:
            length = abs(self.scale_line[2] - self.scale_line[0])
            if length >= 1.0:
                self.scale_pixels_var.set(f"{length:.6f}")
                actual_length = self._positive_float(self.scale_value_var.get())
                if actual_length is None:
                    self.status_var.set(
                        f"水平比例尺线已测得 {length:.6f} pixel，请填写对应的实际长度（μm）。"
                    )
                else:
                    self.status_var.set(
                        f"水平比例尺线：{length:.6f} pixel；自动标定为 "
                        f"{actual_length / length:.8g} μm/pixel。"
                    )
            else:
                self.scale_line = None
                self.scale_pixels_var.set("")
                self.status_var.set("比例尺线过短，请重新绘制长度至少为1 pixel的水平线。")
            self.scale_line_drawing = False

        if self.crop_box is not None:
            l, t, r, b = self.crop_box
            if r - l < self.MIN_CROP_PIXELS or b - t < self.MIN_CROP_PIXELS:
                self.crop_box = None
                self.locked_aspect = None
            else:
                self.locked_aspect = (r - l) / (b - t)

        self.action = None
        self.active_handle = None
        self.action_box = None
        self._draw_overlays()
        self.update_information(silent=True)

    def _resized_box(self, x: float, y: float, handle: str, original: list[float]) -> list[float]:
        assert self.image is not None
        l, t, r, b = original
        anchors = {"nw": (r, b), "ne": (l, b), "sw": (r, t), "se": (l, t)}
        ax, ay = anchors[handle]
        sx = -1.0 if handle in ("nw", "sw") else 1.0
        sy = -1.0 if handle in ("nw", "ne") else 1.0
        raw_w = max(abs(x - ax), self.MIN_CROP_PIXELS)
        raw_h = max(abs(y - ay), self.MIN_CROP_PIXELS)

        if self.lock_aspect_var.get():
            ratio = self.locked_aspect or max((r - l) / max(b - t, 1e-9), 1e-9)
            target_w = max(raw_w, raw_h * ratio)
            max_w = ax if sx < 0 else self.image.width - ax
            max_h = ay if sy < 0 else self.image.height - ay
            width = min(target_w, max_w, max_h * ratio)
            height = width / ratio
        else:
            max_w = ax if sx < 0 else self.image.width - ax
            max_h = ay if sy < 0 else self.image.height - ay
            width = min(raw_w, max_w)
            height = min(raw_h, max_h)

        nx, ny = ax + sx * width, ay + sy * height
        return [min(ax, nx), min(ay, ny), max(ax, nx), max(ay, ny)]

    def select_full_image(self) -> None:
        source_image = self._source_image()
        if source_image is None:
            messagebox.showinfo("提示", "请先打开图像。")
            return
        if self.ebsd_image is not None and self.registration_view != "neural":
            self.registration_view = "neural"
            self.image = source_image
            self.switch_image_button_text.set("切换到 EBSD参考图")
            self._render()
        self.crop_box = [0.0, 0.0, float(source_image.width), float(source_image.height)]
        self.locked_aspect = source_image.width / source_image.height
        self._draw_overlays()
        self.update_information(silent=True)

    def clear_crop(self) -> None:
        self.crop_box = None
        self.locked_aspect = None
        self._draw_overlays()
        self.update_information(silent=True)

    def start_scale_line(self) -> None:
        source_image = self._source_image()
        if source_image is None:
            messagebox.showinfo("提示", "请先打开图像。")
            return
        if self.ebsd_image is not None and self.registration_view != "neural":
            self.registration_view = "neural"
            self.image = source_image
            self.switch_image_button_text.set("切换到 EBSD参考图")
            self._render()
        self.active_registration_point = None
        self._refresh_registration_point_buttons()
        self._draw_overlays()
        self.calibration_mode.set("scale_bar")
        self.scale_line_drawing = True
        self.canvas.configure(cursor="crosshair")
        self.status_var.set(
            "请在打开图像上按住鼠标左键，从比例尺一端拖到另一端；线段将自动锁定为水平线。"
        )
        self._calibration_changed()

    def _aspect_lock_changed(self) -> None:
        if self.lock_aspect_var.get() and self.crop_box is not None:
            l, t, r, b = self.crop_box
            if b > t:
                self.locked_aspect = (r - l) / (b - t)

    def _calibration_changed(self) -> None:
        self.update_information(silent=True)

    @staticmethod
    def _positive_float(text: str) -> float | None:
        try:
            value = float(text.strip())
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) and value > 0 else None

    def _calibration(self) -> dict | None:
        source_image = self._source_image()
        if source_image is None:
            return None

        if self.calibration_mode.get() == "fov":
            if self.crop_box is None:
                return None
            width_um = self._positive_float(self.fov_width_var.get())
            height_um = self._positive_float(self.fov_height_var.get())
            if width_um is None and height_um is None:
                return None
            left, top, right, bottom = self._integer_crop_box()
            crop_width_pixels = right - left
            crop_height_pixels = bottom - top
            if width_um is not None:
                scale_x = width_um / crop_width_pixels
            else:
                scale_x = height_um / crop_height_pixels
            if height_um is not None:
                scale_y = height_um / crop_height_pixels
            else:
                scale_y = width_um / crop_width_pixels
            return {
                "method": "crop_field_of_view",
                "crop_field_width_um": width_um,
                "crop_field_height_um": height_um,
                "crop_width_pixels": crop_width_pixels,
                "crop_height_pixels": crop_height_pixels,
                "micrometers_per_pixel_x": scale_x,
                "micrometers_per_pixel_y": scale_y,
                "square_micrometers_per_pixel": scale_x * scale_y,
            }

        pixels = self._positive_float(self.scale_pixels_var.get())
        value_um = self._positive_float(self.scale_value_var.get())
        if pixels is None or value_um is None:
            return None
        scale = value_um / pixels
        return {
            "method": "scale_bar",
            "scale_bar_length_pixels": pixels,
            "scale_bar_value_um": value_um,
            "scale_bar_line_image_coordinates": self.scale_line,
            "micrometers_per_pixel_x": scale,
            "micrometers_per_pixel_y": scale,
            "square_micrometers_per_pixel": scale * scale,
        }

    def update_information(self, silent: bool = False) -> None:
        source_image = self._source_image()
        if source_image is None:
            self.crop_info_var.set("请先打开图像")
            self.calibration_info_var.set("尚未标定（只能统计像素单位）")
            return

        if self.crop_box is None:
            self.crop_info_var.set(f"原图：{source_image.width} × {source_image.height} pixel\n尚未选择裁剪区域")
        else:
            l, t, r, b = self._integer_crop_box()
            self.crop_info_var.set(
                f"原图坐标：left={l}, top={t}, right={r}, bottom={b}\n"
                f"裁剪尺寸：{r - l} × {b - t} pixel"
            )

        calibration = self._calibration()
        if calibration is None:
            self.calibration_info_var.set("尚未完成有效标定（只能统计 pixel、pixel²）。")
            if self.ebsd_image is not None:
                self._refresh_alignment_information()
            if not silent:
                messagebox.showinfo("标定", "请填写有效的正数标定值。")
            return

        sx = calibration["micrometers_per_pixel_x"]
        sy = calibration["micrometers_per_pixel_y"]
        lines = [f"x方向：{sx:.8g} μm/pixel", f"y方向：{sy:.8g} μm/pixel"]
        if calibration["method"] == "crop_field_of_view":
            width_um = calibration.get("crop_field_width_um")
            height_um = calibration.get("crop_field_height_um")
            entered = []
            if width_um is not None:
                entered.append(f"宽 {width_um:.8g} μm")
            if height_um is not None:
                entered.append(f"高 {height_um:.8g} μm")
            lines.insert(0, f"裁剪框标定：{'，'.join(entered)}")
        elif calibration["method"] == "scale_bar":
            lines.insert(
                0,
                f"水平比例尺：{calibration['scale_bar_length_pixels']:.6g} pixel = "
                f"{calibration['scale_bar_value_um']:.6g} μm",
            )
        if self.crop_box is not None:
            l, t, r, b = self._integer_crop_box()
            lines.append(f"裁剪后实际宽度：{(r - l) * sx:.8g} μm")
            lines.append(f"裁剪后实际高度：{(b - t) * sy:.8g} μm")
        mismatch = abs(sx - sy) / max((sx + sy) / 2, 1e-12)
        if mismatch > 0.01:
            lines.append("注意：x/y 像素尺度相差超过 1%，将分别用于面积和周长换算。")
        self.calibration_info_var.set("\n".join(lines))
        if self.ebsd_image is not None:
            self._refresh_alignment_information()

    def _integer_crop_box(self) -> tuple[int, int, int, int]:
        source_image = self._source_image()
        assert source_image is not None and self.crop_box is not None
        l, t, r, b = self.crop_box
        left = max(0, min(source_image.width - 1, int(round(l))))
        top = max(0, min(source_image.height - 1, int(round(t))))
        right = max(left + 1, min(source_image.width, int(round(r))))
        bottom = max(top + 1, min(source_image.height, int(round(b))))
        return left, top, right, bottom

    @staticmethod
    def _non_overwriting_path(path: Path) -> Path:
        if not path.exists():
            return path
        for index in range(2, 10000):
            candidate = path.with_name(f"{path.stem}_v{index}{path.suffix}")
            if not candidate.exists():
                return candidate
        raise RuntimeError("无法生成不重复的输出文件名。")

    def _auto_load_statistics_source(self) -> None:
        if self.image_path is None or self.image is None:
            return
        candidates = [
            self.image_path.parent / "merged_instance_labels.npy",
            self.image_path.parent / "watershed_instance_labels.npy",
        ]
        for candidate in candidates:
            if candidate.exists() and self._load_statistics_path(candidate, show_errors=False):
                return
        self.statistics_source_var.set("未自动找到实例标签；请手动加载 .npy 或黑白晶界图。")

    def load_statistics_source(self) -> None:
        initial_dir = self.image_path.parent if self.image_path is not None else APP_DIR
        filename = filedialog.askopenfilename(
            title="选择晶粒实例标签或纯晶界图",
            initialdir=initial_dir,
            filetypes=[
                ("实例标签 / 晶界图", "*.npy *.png *.tif *.tiff *.bmp"),
                ("NumPy 实例标签", "*.npy"),
                ("图像", "*.png *.tif *.tiff *.bmp"),
                ("所有文件", "*.*"),
            ],
        )
        if filename:
            self._load_statistics_path(Path(filename).resolve(), show_errors=True)

    def _load_statistics_path(self, path: Path, show_errors: bool) -> bool:
        source_image = self._source_image()
        if source_image is None:
            if show_errors:
                messagebox.showinfo("提示", "请先打开用于显示和裁剪的图像。")
            return False
        try:
            if path.suffix.lower() == ".npy":
                raw = np.load(path, allow_pickle=False)
                if raw.ndim != 2 or not np.issubdtype(raw.dtype, np.number):
                    raise ValueError(".npy 必须是二维数值型实例标签矩阵。")
                labels = raw.astype(np.int64, copy=False)
                method = "NumPy实例标签"
            else:
                with Image.open(path) as source:
                    raw = np.asarray(source.copy())
                labels, method = self._labels_from_boundary_or_color_image(raw)

            expected = (source_image.height, source_image.width)
            if labels.shape != expected:
                raise ValueError(f"标签尺寸为 {labels.shape[1]}×{labels.shape[0]}，显示图像尺寸为 {expected[1]}×{expected[0]}。")
            if not np.any(labels >= 0):
                raise ValueError("没有检测到可统计的晶粒区域。")
        except Exception as exc:
            if show_errors:
                messagebox.showerror("加载失败", f"无法作为晶粒标签读取：\n{exc}")
            return False

        self.statistics_source_path = path
        self.statistics_labels = labels
        self.statistics_label_method = method
        valid_count = int(np.unique(labels[labels >= 0]).size)
        self.statistics_source_var.set(f"{path.name}\n解析方式：{method}；候选区域 {valid_count} 个")
        self.status_var.set(f"已加载晶粒统计数据：{path.name}")
        return True

    @staticmethod
    def _labels_from_grayscale(gray: np.ndarray) -> np.ndarray:
        # 晶界线通常是两类像素中数量较少的一类，兼容黑线白底和白线黑底。
        boundary = GrainCropTool._boundary_mask_from_grayscale(gray)
        interiors = ~boundary
        components = measure.label(interiors, connectivity=1).astype(np.int64)
        return np.where(boundary, -1, components)

    @staticmethod
    def _labels_from_boundary_or_color_image(raw: np.ndarray) -> tuple[np.ndarray, str]:
        if raw.ndim == 2 or (raw.ndim == 3 and raw.shape[2] == 1):
            gray = raw if raw.ndim == 2 else raw[..., 0]
            return GrainCropTool._labels_from_grayscale(gray), "黑白晶界图连通区域"

        if raw.ndim == 3 and raw.shape[2] >= 3:
            rgb_source = np.asarray(raw[..., :3])
            rgb_float = np.asarray(rgb_source, dtype=np.float64)
            if not np.all(np.isfinite(rgb_float)):
                raise ValueError("晶界图包含非有限数值。")

            # 很多EBSD软件把黑白晶界图保存为RGB BMP/PNG。若三个通道基本一致，
            # 仅在内存中转换为灰度后解析，不生成或保存中间灰度图。
            channel_spread = np.max(rgb_float, axis=2) - np.min(rgb_float, axis=2)
            grayscale_like_fraction = float(np.mean(channel_spread <= 3.0))
            if grayscale_like_fraction >= 0.995:
                gray = (
                    0.299 * rgb_float[..., 0]
                    + 0.587 * rgb_float[..., 1]
                    + 0.114 * rgb_float[..., 2]
                )
                labels = GrainCropTool._labels_from_grayscale(gray)
                return labels, "RGB自动转灰度→黑白晶界图连通区域"

            rgb = np.asarray(rgb_source, dtype=np.uint8)
            flat = rgb.reshape(-1, 3)
            colors, inverse = np.unique(flat, axis=0, return_inverse=True)
            if colors.shape[0] <= 2:
                counts = np.bincount(inverse)
                boundary_index = int(np.argmin(counts))
                boundary = inverse.reshape(rgb.shape[:2]) == boundary_index
                components = measure.label(~boundary, connectivity=1).astype(np.int64)
                return np.where(boundary, -1, components), "二值彩色晶界图连通区域"
            if colors.shape[0] > 10000:
                raise ValueError("颜色数量过多，像是原始照片而不是随机色实例图或纯晶界图。")
            return inverse.reshape(rgb.shape[:2]).astype(np.int64), "随机颜色实例图"

        raise ValueError(f"不支持的图像数组形状：{raw.shape}")

    @staticmethod
    def _physical_perimeter(mask: np.ndarray, scale_x: float, scale_y: float) -> float:
        padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
        perimeter = 0.0
        for contour in measure.find_contours(padded, 0.5):
            if contour.shape[0] < 2:
                continue
            delta = np.diff(contour, axis=0)
            perimeter += float(np.hypot(delta[:, 0] * scale_y, delta[:, 1] * scale_x).sum())
        return perimeter

    @staticmethod
    def _nice_bin_width(values: np.ndarray) -> float:
        if values.size <= 1:
            return max(float(values[0]) * 0.1 if values.size else 1.0, 1e-6)
        data_range = float(np.ptp(values))
        q25, q75 = np.percentile(values, [25, 75])
        raw = 2.0 * float(q75 - q25) / np.cbrt(values.size)
        if not math.isfinite(raw) or raw <= 0:
            raw = data_range / 10.0 if data_range > 0 else max(float(values[0]) * 0.1, 1e-6)
        exponent = math.floor(math.log10(raw))
        fraction = raw / (10**exponent)
        nice_fraction = 1.0 if fraction <= 1 else 2.0 if fraction <= 2 else 5.0 if fraction <= 5 else 10.0
        return nice_fraction * (10**exponent)

    @staticmethod
    def _grain_size_values(
        area_pixels: int,
        scale_x: float | None,
        scale_y: float | None,
    ) -> dict[str, float | None]:
        """Return pixel/physical area and equivalent-circle diameters from one area."""
        equivalent_diameter_pixels = 2.0 * math.sqrt(float(area_pixels) / math.pi)
        if scale_x is None or scale_y is None:
            return {
                "area_um2": None,
                "equivalent_diameter_um": None,
                "equivalent_diameter_pixels": equivalent_diameter_pixels,
            }
        area_um2 = float(area_pixels) * float(scale_x) * float(scale_y)
        return {
            "area_um2": area_um2,
            "equivalent_diameter_um": 2.0 * math.sqrt(area_um2 / math.pi),
            "equivalent_diameter_pixels": equivalent_diameter_pixels,
        }

    def _measure_grains(self, minimum_area: int) -> tuple[list[dict], int, int]:
        if self.statistics_labels is None or self.crop_box is None:
            raise ValueError("尚未加载晶粒标签或尚未选择裁剪区域。")
        left, top, right, bottom = self._integer_crop_box()
        crop_labels = self.statistics_labels[top:bottom, left:right]
        valid = crop_labels >= 0
        source_ids = np.unique(crop_labels[valid])
        if source_ids.size == 0:
            return [], 0, 0

        # 重映射到从1开始的连续编号，避免源标签0被 regionprops 当作背景。
        mapped = np.zeros(crop_labels.shape, dtype=np.int32)
        mapped[valid] = np.searchsorted(source_ids, crop_labels[valid]).astype(np.int32) + 1
        border_mapped_ids = np.unique(
            np.concatenate((mapped[0, :], mapped[-1, :], mapped[:, 0], mapped[:, -1]))
        )
        border_mapped_ids = set(int(value) for value in border_mapped_ids if value > 0)

        calibration = self._calibration()
        scale_x = calibration["micrometers_per_pixel_x"] if calibration else None
        scale_y = calibration["micrometers_per_pixel_y"] if calibration else None
        grains: list[dict] = []
        excluded_border = 0
        excluded_small = 0
        next_id = 1
        for region in measure.regionprops(mapped):
            if region.label in border_mapped_ids:
                excluded_border += 1
                continue
            area_pixels = int(region.area)
            if area_pixels < minimum_area:
                excluded_small += 1
                continue
            local_mask = region.image
            perimeter_pixels = float(measure.perimeter_crofton(local_mask, directions=4))
            perimeter_um = (
                self._physical_perimeter(local_mask, float(scale_x), float(scale_y))
                if scale_x is not None and scale_y is not None
                else None
            )
            size_values = self._grain_size_values(area_pixels, scale_x, scale_y)
            grains.append(
                {
                    "grain_id": next_id,
                    "source_label_id": int(source_ids[region.label - 1]),
                    "area_pixels": area_pixels,
                    "perimeter_pixels": perimeter_pixels,
                    "perimeter_um": perimeter_um,
                    "centroid_source_x_pixels": float(left + region.centroid[1]),
                    "centroid_source_y_pixels": float(top + region.centroid[0]),
                    **size_values,
                }
            )
            next_id += 1
        return grains, excluded_border, excluded_small

    @classmethod
    def _measure_full_label_array(
        cls,
        labels: np.ndarray,
        minimum_area: int,
        calibration: dict,
    ) -> tuple[list[dict], int, int]:
        """统计整幅对齐实例标签；接触输出图边缘的区域不计入结果。"""
        label_array = np.asarray(labels)
        if label_array.ndim != 2:
            raise ValueError("对齐实例标签必须是二维数组。")
        valid = label_array >= 0
        source_ids = np.unique(label_array[valid])
        if source_ids.size == 0:
            return [], 0, 0
        mapped = np.zeros(label_array.shape, dtype=np.int32)
        mapped[valid] = np.searchsorted(source_ids, label_array[valid]).astype(np.int32) + 1
        border_ids = set(
            int(value)
            for value in np.unique(
                np.concatenate((mapped[0, :], mapped[-1, :], mapped[:, 0], mapped[:, -1]))
            )
            if value > 0
        )
        scale_x = float(calibration["micrometers_per_pixel_x"])
        scale_y = float(calibration["micrometers_per_pixel_y"])
        affine_physical = calibration.get("aligned_pixel_to_physical_um_matrix")
        grains: list[dict] = []
        excluded_border = 0
        excluded_small = 0
        for region in measure.regionprops(mapped):
            if region.label in border_ids:
                excluded_border += 1
                continue
            area_pixels = int(region.area)
            if area_pixels < minimum_area:
                excluded_small += 1
                continue
            if affine_physical is not None:
                perimeter_um = cls._physical_perimeter_affine(region.image, np.asarray(affine_physical))
            else:
                perimeter_um = cls._physical_perimeter(region.image, scale_x, scale_y)
            size_values = cls._grain_size_values(area_pixels, scale_x, scale_y)
            grains.append(
                {
                    "grain_id": len(grains) + 1,
                    "source_label_id": int(source_ids[region.label - 1]),
                    "area_pixels": area_pixels,
                    "perimeter_pixels": float(measure.perimeter_crofton(region.image, directions=4)),
                    "perimeter_um": perimeter_um,
                    "centroid_source_x_pixels": float(region.centroid[1]),
                    "centroid_source_y_pixels": float(region.centroid[0]),
                    **size_values,
                }
            )
        return grains, excluded_border, excluded_small

    def _histogram_payload(
        self,
        grains: list[dict],
        calibration: dict | None,
        width_text: str | None = None,
    ) -> list[dict]:
        if not grains:
            return []
        if calibration:
            values = np.asarray([float(grain["equivalent_diameter_um"]) for grain in grains])
        else:
            values = np.asarray([float(grain["equivalent_diameter_pixels"]) for grain in grains])

        requested_width = self._positive_float(
            self.histogram_bin_width_var.get() if width_text is None else width_text
        )
        width = requested_width or self._nice_bin_width(values)
        lower = 0.0
        upper = math.ceil(float(values.max()) / width) * width
        if upper <= lower:
            upper = lower + width
        bin_count = max(1, int(round((upper - lower) / width)))
        edges = lower + np.arange(bin_count + 1, dtype=np.float64) * width
        counts, edges = np.histogram(values, bins=edges)
        rows = []
        for index, count in enumerate(counts):
            left_edge, right_edge = float(edges[index]), float(edges[index + 1])
            closing = "]" if index == len(counts) - 1 else ")"
            rows.append(
                {
                    "lower": left_edge,
                    "upper": right_edge,
                    "center": (left_edge + right_edge) / 2.0,
                    "label": f"[{left_edge:.4g}, {right_edge:.4g}{closing}",
                    "count": int(count),
                }
            )
        return rows

    @classmethod
    def _area_fraction_distribution(
        cls,
        labels: np.ndarray,
        crop_box: tuple[int, int, int, int],
        calibration: dict | None,
        minimum_area: int,
    ) -> dict:
        """按1 μm（未标定时1 pixel）分箱统计包含边缘晶粒的面积占整图比例。"""
        left, top, right, bottom = crop_box
        label_array = np.asarray(labels)
        if label_array.ndim != 2:
            raise ValueError("面积占比分布要求二维实例标签。")
        if not (0 <= left < right <= label_array.shape[1] and 0 <= top < bottom <= label_array.shape[0]):
            raise ValueError("面积占比分布的统计区域超出标签尺寸。")
        crop_labels = label_array[top:bottom, left:right]
        total_pixels = int(crop_labels.size)
        valid = crop_labels >= 0
        source_ids = np.unique(crop_labels[valid])
        mapped = np.zeros(crop_labels.shape, dtype=np.int32)
        if source_ids.size:
            mapped[valid] = np.searchsorted(source_ids, crop_labels[valid]).astype(np.int32) + 1
        border_ids = set(
            int(value)
            for value in np.unique(
                np.concatenate((mapped[0, :], mapped[-1, :], mapped[:, 0], mapped[:, -1]))
            )
            if value > 0
        )

        if calibration:
            pixel_area = calibration.get("square_micrometers_per_pixel")
            if pixel_area is None:
                pixel_area = (
                    float(calibration["micrometers_per_pixel_x"])
                    * float(calibration["micrometers_per_pixel_y"])
                )
            diameter_unit = "μm"
        else:
            pixel_area = 1.0
            diameter_unit = "pixel"
        pixel_area = float(pixel_area)

        records = []
        excluded_small_count = 0
        excluded_small_area = 0
        included_edge_count = 0
        for region in measure.regionprops(mapped):
            area_pixels = int(region.area)
            if area_pixels < minimum_area:
                excluded_small_count += 1
                excluded_small_area += area_pixels
                continue
            if region.label in border_ids:
                included_edge_count += 1
            records.append(
                {
                    "diameter": 2.0 * math.sqrt(area_pixels * pixel_area / math.pi),
                    "area_pixels": area_pixels,
                    "touches_border": region.label in border_ids,
                }
            )

        bin_width = 1.0
        maximum = max((record["diameter"] for record in records), default=bin_width)
        upper_limit = max(bin_width, math.ceil(maximum / bin_width) * bin_width)
        bin_count = max(1, int(round(upper_limit / bin_width)))
        bins = [
            {
                "lower": index * bin_width,
                "upper": (index + 1) * bin_width,
                "label": f"[{index * bin_width:.4g}, {(index + 1) * bin_width:.4g}{']' if index == bin_count - 1 else ')'}",
                "region_count": 0,
                "area_pixels": 0,
                "area_fraction": 0.0,
            }
            for index in range(bin_count)
        ]
        for record in records:
            index = min(int(record["diameter"] // bin_width), bin_count - 1)
            bins[index]["region_count"] += 1
            bins[index]["area_pixels"] += int(record["area_pixels"])
        for row in bins:
            row["area_fraction"] = float(row["area_pixels"] / total_pixels) if total_pixels else 0.0

        included_area = sum(int(record["area_pixels"]) for record in records)
        return {
            "diameter_unit": diameter_unit,
            "bin_width": bin_width,
            "total_image_area_pixels": total_pixels,
            "total_image_area_physical": total_pixels * pixel_area if calibration else None,
            "included_region_count": len(records),
            "included_edge_region_count": included_edge_count,
            "excluded_small_region_count": excluded_small_count,
            "excluded_small_area_pixels": excluded_small_area,
            "included_grain_area_pixels": included_area,
            "included_grain_area_fraction": float(included_area / total_pixels) if total_pixels else 0.0,
            "unassigned_area_fraction": float(1.0 - included_area / total_pixels) if total_pixels else 1.0,
            "bins": bins,
        }

    @staticmethod
    def _cumulative_distribution_from_grains(
        grains: list[dict],
        calibration: dict | None,
        name: str,
    ) -> dict:
        """生成按晶粒数量和按晶粒面积加权的经验累积分布。"""
        if calibration:
            pixel_area = calibration.get("square_micrometers_per_pixel")
            if pixel_area is None:
                pixel_area = (
                    float(calibration["micrometers_per_pixel_x"])
                    * float(calibration["micrometers_per_pixel_y"])
                )
            diameter_unit = "μm"
        else:
            pixel_area = 1.0
            diameter_unit = "pixel"
        pixel_area = float(pixel_area)
        ordered = sorted(
            (
                2.0 * math.sqrt(float(grain["area_pixels"]) * pixel_area / math.pi),
                float(grain["area_pixels"]) * pixel_area,
            )
            for grain in grains
        )
        total_count = len(ordered)
        total_area = sum(area for _diameter, area in ordered)
        records = [
            {
                "diameter": 0.0,
                "rank": 0,
                "grain_area": 0.0,
                "cumulative_number_fraction": 0.0,
                "cumulative_area_fraction": 0.0,
            }
        ]
        cumulative_area = 0.0
        for index, (diameter, area) in enumerate(ordered, start=1):
            cumulative_area += area
            records.append(
                {
                    "diameter": float(diameter),
                    "rank": index,
                    "grain_area": float(area),
                    "cumulative_number_fraction": float(index / total_count) if total_count else 0.0,
                    "cumulative_area_fraction": float(cumulative_area / total_area) if total_area else 0.0,
                }
            )
        return {
            "name": name,
            "diameter_unit": diameter_unit,
            "grain_count": total_count,
            "total_grain_area": total_area,
            "records": records,
        }

    def _write_statistics_workbook(self, payload: dict, output_path: Path) -> None:
        self._run_excel_exporter(
            XLSX_EXPORTER,
            payload,
            output_path,
            timeout_seconds=240,
            temporary_prefix="grain_statistics_",
        )

    def _start_statistics_export(self, payload: dict, output_path: Path, context: dict) -> None:
        """Export in a worker so the Tk event loop remains responsive."""
        if self._statistics_export_thread is not None and self._statistics_export_thread.is_alive():
            messagebox.showinfo("正在导出", "已有一个 Excel 导出任务正在运行，请等待其完成。")
            return

        # Discard any stale result left by an interrupted UI session.
        while True:
            try:
                self._statistics_export_queue.get_nowait()
            except queue.Empty:
                break
        self._statistics_export_context = context
        self._statistics_export_started_at = time.monotonic()
        self.statistics_export_button.state(["disabled"])
        self.status_var.set("正在后台生成 Excel 工作簿（界面仍可操作）……")

        def worker() -> None:
            try:
                self._write_statistics_workbook(payload, output_path)
            except Exception as exc:
                self._statistics_export_queue.put({"ok": False, "error": str(exc)})
            else:
                self._statistics_export_queue.put({"ok": True})

        self._statistics_export_thread = threading.Thread(
            target=worker,
            name="grain-statistics-xlsx-export",
            daemon=True,
        )
        self._statistics_export_thread.start()
        self.root.after(150, self._poll_statistics_export)

    def _poll_statistics_export(self) -> None:
        try:
            result = self._statistics_export_queue.get_nowait()
        except queue.Empty:
            if self._statistics_export_thread is not None and self._statistics_export_thread.is_alive():
                elapsed = time.monotonic() - self._statistics_export_started_at
                self.status_var.set(f"正在后台生成 Excel 工作簿……已用时 {elapsed:.1f} 秒")
                self.root.after(250, self._poll_statistics_export)
                return
            result = {"ok": False, "error": "Excel 导出线程意外结束，未返回结果。"}

        context = self._statistics_export_context or {}
        self._statistics_export_thread = None
        self._statistics_export_context = None
        self.statistics_export_button.state(["!disabled"])
        if not result.get("ok"):
            error = str(result.get("error") or "未知错误")
            self.status_var.set("Excel 导出失败；请查看错误对话框。")
            messagebox.showerror("Excel导出失败", f"无法生成统计工作簿：\n{error}")
            return

        output_path = Path(context["output_path"])
        unit_note = "结果包含 μm 单位" if context["calibrated"] else "尚未标定，物理量留空并使用 pixel 单位分布"
        messagebox.showinfo(
            "统计完成",
            f"有效完整晶粒：{context['grain_count']} 个\n"
            f"排除边缘晶粒：{context['excluded_border']} 个\n"
            f"排除小区域：{context['excluded_small']} 个\n\n"
            f"直径分组步长：{context['bin_width']:.8g} {context['diameter_unit']}\n"
            f"分布图纵轴：Count\n\n"
            f"{unit_note}\n\nExcel：\n{output_path}",
        )
        self.status_var.set(f"统计完成：{context['grain_count']} 个完整晶粒，已保存 {output_path.name}")

    def calculate_and_export_statistics(self) -> None:
        if self._statistics_export_thread is not None and self._statistics_export_thread.is_alive():
            messagebox.showinfo("正在导出", "已有一个 Excel 导出任务正在运行，请等待其完成。")
            return
        if self.image is None or self.image_path is None:
            messagebox.showinfo("提示", "请先打开用于显示和裁剪的图像。")
            return
        if self.crop_box is None:
            messagebox.showinfo("提示", "请先框选统计区域。")
            return
        if self.statistics_labels is None or self.statistics_source_path is None:
            messagebox.showinfo("提示", "请加载实例标签 .npy 或纯黑白晶界图。")
            return
        try:
            minimum_area_value = self._positive_float(self.minimum_area_var.get())
            if minimum_area_value is None:
                raise ValueError("最小保留面积必须是正数。")
            minimum_area = max(1, int(round(minimum_area_value)))
            self.status_var.set("正在识别完整晶粒并计算尺寸……")
            self.root.update_idletasks()
            grains, excluded_border, excluded_small = self._measure_grains(minimum_area)
        except Exception as exc:
            messagebox.showerror("统计失败", str(exc))
            return
        if not grains:
            messagebox.showwarning("没有有效晶粒", "裁剪区域内没有满足条件且不接触裁剪边缘的完整晶粒。")
            return

        calibration = self._calibration()
        histogram = self._histogram_payload(grains, calibration)
        left, top, right, bottom = self._integer_crop_box()
        payload = {
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "display_image": str(self.image_path),
            "statistics_source": str(self.statistics_source_path),
            "label_method": self.statistics_label_method,
            "crop_box": {"left": left, "top": top, "right_exclusive": right, "bottom_exclusive": bottom},
            "crop_size": {"width": right - left, "height": bottom - top},
            "calibration": calibration,
            "minimum_area_pixels": minimum_area,
            "excluded_border_count": excluded_border,
            "excluded_small_count": excluded_small,
            "grains": grains,
            "histogram": histogram,
        }

        DEFAULT_STATISTICS_DIR.mkdir(parents=True, exist_ok=True)
        filename = filedialog.asksaveasfilename(
            title="保存晶粒统计 Excel",
            initialdir=DEFAULT_STATISTICS_DIR,
            initialfile=f"{self.image_path.stem}_grain_statistics.xlsx",
            defaultextension=".xlsx",
            filetypes=[("Excel 工作簿", "*.xlsx")],
        )
        if not filename:
            return
        output_path = self._non_overwriting_path(Path(filename).resolve())
        context = {
            "output_path": str(output_path),
            "grain_count": len(grains),
            "excluded_border": excluded_border,
            "excluded_small": excluded_small,
            "bin_width": float(histogram[0]["upper"] - histogram[0]["lower"]),
            "diameter_unit": "μm" if calibration is not None else "pixel",
            "calibrated": calibration is not None,
        }
        self._start_statistics_export(payload, output_path, context)

    def save_crop(self) -> None:
        source_image = self._source_image()
        if source_image is None or self.image_path is None:
            messagebox.showinfo("提示", "请先打开图像。")
            return
        if self.crop_box is None:
            messagebox.showinfo("提示", "请先框选裁剪区域，或点击“整幅图像”。")
            return

        left, top, right, bottom = self._integer_crop_box()
        if right - left < self.MIN_CROP_PIXELS or bottom - top < self.MIN_CROP_PIXELS:
            messagebox.showwarning("裁剪区域过小", "裁剪区域宽和高至少需要 4 pixel。")
            return

        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        initial_name = f"{self.image_path.stem}_crop.png"
        filename = filedialog.asksaveasfilename(
            title="保存裁剪图",
            initialdir=DEFAULT_OUTPUT_DIR,
            initialfile=initial_name,
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("TIFF", "*.tif"), ("JPEG", "*.jpg")],
        )
        if not filename:
            return

        requested_path = Path(filename).resolve()
        output_path = self._non_overwriting_path(requested_path)
        if output_path != requested_path:
            self.status_var.set(f"目标文件已存在，已自动改名为 {output_path.name}，未覆盖历史文件。")

        calibration = self._calibration()
        width_pixels = right - left
        height_pixels = bottom - top
        physical: dict[str, float] | None = None
        if calibration is not None:
            sx = calibration["micrometers_per_pixel_x"]
            sy = calibration["micrometers_per_pixel_y"]
            physical = {
                "crop_width_um": width_pixels * sx,
                "crop_height_um": height_pixels * sy,
                "crop_area_um2": width_pixels * height_pixels * sx * sy,
            }

        metadata = {
            "format_version": 1,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "source_image": str(self.image_path),
            "source_size_pixels": {"width": source_image.width, "height": source_image.height},
            "crop_box_source_pixels": {
                "left": left,
                "top": top,
                "right_exclusive": right,
                "bottom_exclusive": bottom,
            },
            "crop_size_pixels": {"width": width_pixels, "height": height_pixels},
            "calibration": calibration,
            "crop_physical_size": physical,
            "notes": (
                "right_exclusive 与 bottom_exclusive 遵循 Python/Pillow 裁剪坐标约定；"
                "等效圆直径可由 Deq = 2*sqrt(area_um2/pi) 计算。"
            ),
        }

        metadata_path = output_path.with_suffix(".crop_metadata.json")
        metadata_path = self._non_overwriting_path(metadata_path)
        try:
            crop = source_image.crop((left, top, right, bottom))
            save_image = crop
            if output_path.suffix.lower() in (".jpg", ".jpeg") and crop.mode not in ("RGB", "L"):
                save_image = crop.convert("RGB")
            save_image.save(output_path)
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            messagebox.showerror("保存失败", f"无法保存裁剪结果：\n{exc}")
            return

        calibration_text = "已保存尺度标定" if calibration is not None else "未标定，物理尺寸为空"
        messagebox.showinfo(
            "保存完成",
            f"裁剪图：\n{output_path}\n\n标定信息：\n{metadata_path}\n\n{calibration_text}",
        )
        self.status_var.set(f"保存完成：{output_path.name}（未覆盖任何已有文件）")


def main() -> None:
    root = tk.Tk()
    try:
        ttk.Style(root).theme_use("vista")
    except tk.TclError:
        pass
    GrainCropTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()
