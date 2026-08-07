#!/usr/bin/env python3
"""
几何图形矢量化识别系统 — 桌面 GUI 版
支持批量处理、单文件导出/合并导出、TikZ代码复制、原图/LaTeX预览
"""
import sys
import os
import math
import glob
import tempfile
import shutil
import subprocess
import gc
import pickle
import traceback
import faulthandler
import multiprocessing as mp
from datetime import datetime

# ── 确保 geometry_app 可导入 ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QListWidget, QListWidgetItem, QPushButton, QLabel,
    QTextEdit, QPlainTextEdit, QTabWidget, QFileDialog, QMessageBox,
    QStatusBar, QMenuBar, QMenu, QMenuBar, QToolBar, QComboBox,
    QDialog, QDialogButtonBox, QProgressBar, QFrame, QScrollArea,
    QGroupBox, QGridLayout, QButtonGroup, QRadioButton, QSizePolicy,
    QAbstractItemView, QStyle, QApplication as QA, QSlider
)
from PySide6.QtCore import (
    Qt, QThread, Signal, QSize, QTimer, QMutex, QMutexLocker,
    QRunnable, QThreadPool, QObject, QRect
)
from PySide6.QtGui import (
    QPixmap, QImage, QFont, QTextCursor, QAction, QActionGroup, QIcon,
    QPalette, QColor, QTextCharFormat, QSyntaxHighlighter,
    QFontDatabase, QClipboard, QPainter, QPen, QBrush, QTransform,
    QPageSize, QPageLayout, QTextBlockFormat
)

from geometry_app import GeometryRecognizer, RecognitionResult, safe_imread, create_recognizer, DETIKZIFY_AVAILABLE
from geometry_app.logger import logger, log_exception, get_log_file_path, LogSignal, read_recent_logs, write_crash_log, PROGRAM_DIR


# ═══════════════════════════════════════════════════════════════
# 全局异常捕获（防止闪退，自动保存 crash 日志）
# ═══════════════════════════════════════════════════════════════

def _global_excepthook(exc_type, exc_value, exc_tb):
    """全局未捕获异常处理——防止闪退，保存日志，弹窗提示"""
    tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    error_msg = f"{exc_type.__name__}: {exc_value}"

    # 紧急写入 crash_log.txt
    write_crash_log(error_msg, tb_str)

    # 日志记录
    try:
        logger.critical(f"未捕获异常: {error_msg}")
        for line in tb_str.split('\n'):
            if line.strip():
                logger.critical(f"  {line}")
    except Exception:
        pass

    # 弹窗提示（如果 QApplication 已存在）
    try:
        app = QApplication.instance()
        if app:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(
                None, "程序崩溃",
                f"程序遇到了未预期的错误，即将退出。\n\n"
                f"错误: {error_msg}\n\n"
                f"详细日志已保存到:\n{os.path.join(PROGRAM_DIR, 'crash_log.txt')}\n"
                f"程序目录下的 logs/ 文件夹"
            )
    except Exception:
        pass

    # 调用原始钩子
    sys.__excepthook__(exc_type, exc_value, exc_tb)


# 注册全局异常钩子
sys.excepthook = _global_excepthook


# ═══════════════════════════════════════════════════════════════
# TikZ 语法高亮
# ═══════════════════════════════════════════════════════════════

class TikZHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rules = []

        # 注释
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#6A9955"))
        self.rules.append((r"%.*$", fmt))

        # 关键字
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#569CD6"))
        fmt.setFontWeight(QFont.Weight.Bold)
        keywords = [
            r"\\documentclass", r"\\usepackage", r"\\begin", r"\\end",
            r"\\draw", r"\\node", r"\\coordinate", r"\\tikzset",
            r"\\path", r"\\fill", r"\\clip", r"\\scope", r"\\foreach"
        ]
        for kw in keywords:
            self.rules.append((kw, fmt))

        # 环境/命令
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#DCDCAA"))
        self.rules.append((r"\\(?:[a-zA-Z]+|\S)", fmt))

        # 坐标/数字
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#B5CEA8"))
        self.rules.append((r"-?\d+\.?\d*", fmt))

        # 标签文本
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#CE9178"))
        self.rules.append((r"\{[^}]*\}", fmt))

        # 方括号属性
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#D7BA7D"))
        self.rules.append((r"\[[^\]]*\]", fmt))

        import re
        self.rules = [(re.compile(p, re.MULTILINE), f) for p, f in self.rules]

    def highlightBlock(self, text):
        for pattern, fmt in self.rules:
            for match in pattern.finditer(text):
                start = match.start()
                length = match.end() - start
                self.setFormat(start, length, fmt)


# ═══════════════════════════════════════════════════════════════
# 子进程识别函数（防止 OpenCV segfault 导致 GUI 崩溃）
# ═══════════════════════════════════════════════════════════════

def _subprocess_recognize(image_path, backend, circle_tol, circle_hit, line_tol, line_hit, output_file):
    """
    在独立子进程中运行识别（用文件传递结果，避免 mp.Queue 在 PyInstaller 下的问题）
    即使 OpenCV 内部崩溃也不会影响 GUI 进程
    """
    try:
        # 子进程需要重新设置路径
        import sys
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, script_dir)

        # 启用 faulthandler 在子进程中也捕获 segfault
        faulthandler.enable()

        from geometry_app import create_recognizer, RecognitionResult
        from geometry_app.logger import logger, flush_log

        logger.info(f"子进程启动: 识别 {image_path}")
        flush_log()

        recognizer = create_recognizer(
            backend=backend,
            circle_pixel_tolerance=circle_tol,
            circle_hit_threshold=circle_hit,
            line_pixel_tolerance=line_tol,
            line_hit_threshold=line_hit,
        )
        result = recognizer.recognize(
            image_path,
            line_pixel_tolerance=line_tol,
            line_hit_threshold=line_hit,
        )
        recognizer.cleanup()

        logger.info(f"子进程识别完成: {image_path}")
        flush_log()

        # 将结果写入临时文件
        with open(output_file, 'wb') as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"子进程结果已写入: {output_file}")
        flush_log()
    except Exception as e:
        tb = traceback.format_exc()
        # 写入 crash 日志
        try:
            crash_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crash_log.txt")
            with open(crash_path, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"子进程识别异常: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"图片: {image_path}\n")
                f.write(f"错误: {type(e).__name__}: {str(e)}\n")
                f.write(f"堆栈:\n{tb}\n")
                f.write(f"{'='*60}\n")
        except Exception:
            pass
        # 异常也写入文件
        result = RecognitionResult()
        result.error = f"{type(e).__name__}: {str(e)}\n{tb}"
        try:
            with open(output_file, 'wb') as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════
# 识别工作线程
# ═══════════════════════════════════════════════════════════════

class RecognizeWorker(QThread):
    finished = Signal(str, object)  # image_path, RecognitionResult
    progress = Signal(str, int)     # message, percent

    def __init__(self, image_path, backend="cv", parent=None,
                 circle_pixel_tolerance=2, circle_hit_threshold=0.30,
                 line_pixel_tolerance=8, line_hit_threshold=0.60):
        super().__init__(parent)
        self.image_path = image_path
        self.backend = backend
        self.circle_pixel_tolerance = circle_pixel_tolerance
        self.circle_hit_threshold = circle_hit_threshold
        self.line_pixel_tolerance = line_pixel_tolerance
        self.line_hit_threshold = line_hit_threshold

    def run(self):
        try:
            self.progress.emit(f"识别中: {os.path.basename(self.image_path)}...", 10)

            # ── 使用临时文件传递结果（避免 mp.Queue 在 PyInstaller 下的问题） ──
            result_file = os.path.join(tempfile.gettempdir(),
                f"geo_recog_result_{os.getpid()}_{datetime.now().strftime('%H%M%S%f')}.pkl")

            # 使用 multiprocessing 在独立子进程中运行识别
            # 即使 OpenCV 内部 segfault，GUI 进程也不会崩溃
            ctx = mp.get_context('spawn')
            proc = ctx.Process(
                target=_subprocess_recognize,
                args=(self.image_path, self.backend,
                      self.circle_pixel_tolerance, self.circle_hit_threshold,
                      self.line_pixel_tolerance, self.line_hit_threshold,
                      result_file),
                daemon=True,
            )
            proc.start()
            self.progress.emit(f"识别中: {os.path.basename(self.image_path)}...", 30)

            # 等待子进程完成（最长 120 秒）
            proc.join(timeout=120)

            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
                raise Exception("识别超时（120秒），请检查图片是否过大或损坏")

            # 从临时文件读取结果
            result = None
            if os.path.exists(result_file) and os.path.getsize(result_file) > 0:
                try:
                    with open(result_file, 'rb') as f:
                        result = pickle.load(f)
                except Exception as e:
                    logger.error(f"读取子进程结果文件失败: {e}")

            # 清理临时文件
            try:
                if os.path.exists(result_file):
                    os.remove(result_file)
            except Exception:
                pass

            if result is None:
                if proc.exitcode != 0:
                    raise Exception(f"识别子进程异常退出 (exit code: {proc.exitcode})")
                else:
                    raise Exception("识别子进程未输出结果，可能发生了内部崩溃")

            # 确保 result 是 RecognitionResult 类型
            if not isinstance(result, RecognitionResult):
                # 从 dict 恢复
                if isinstance(result, dict):
                    r = RecognitionResult()
                    r.__dict__.update(result)
                    result = r

            self.progress.emit(f"编译中: {os.path.basename(self.image_path)}...", 80)
            self.finished.emit(self.image_path, result)

        except Exception as e:
            logger.error(f"识别工作线程异常: {type(e).__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            result = RecognitionResult()
            result.error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            self.finished.emit(self.image_path, result)


# ═══════════════════════════════════════════════════════════════
# 图片列表项（自定义 Widget）
# ═══════════════════════════════════════════════════════════════

class ImageListItem(QWidget):
    def __init__(self, path, index):
        super().__init__()
        self.path = path
        self.index = index
        self.status = "pending"  # pending, processing, done, failed
        self.result = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        self.status_label = QLabel("○")
        self.status_label.setFixedWidth(16)
        self.status_label.setStyleSheet("font-size: 14px;")
        layout.addWidget(self.status_label)

        self.name_label = QLabel(os.path.basename(path))
        self.name_label.setStyleSheet("font-size: 12px;")
        layout.addWidget(self.name_label, 1)

        self.setLayout(layout)

    def set_status(self, status):
        self.status = status
        icons = {
            "pending": ("○", "#888"),
            "processing": ("◌", "#2196F3"),
            "done": ("✓", "#4CAF50"),
            "failed": ("✗", "#F44336"),
        }
        icon, color = icons.get(status, ("○", "#888"))
        self.status_label.setText(icon)
        self.status_label.setStyleSheet(f"font-size: 14px; color: {color};")


# ═══════════════════════════════════════════════════════════════
# 主窗口
# ═══════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("几何图形矢量化识别系统")
        self.setMinimumSize(800, 600)   # 允许缩小到最小
        self.resize(1200, 800)

        # ── 数据 ──
        self.image_paths = []           # 所有图片路径
        self.results = {}               # path -> RecognitionResult
        self.current_index = -1
        self.recognizer_pool = {}       # path -> RecognizeWorker
        self.merge_tex_cache = ""       # 合并导出用

        # ── 构建 UI ──
        self._setup_menu()
        self._setup_central()
        self._setup_statusbar()

        # ── 样式 ──
        self._apply_style()

        # ── 定时器（非阻塞编译） ──
        self.compile_timer = QTimer(self)
        self.compile_timer.setSingleShot(True)
        self.compile_timer.timeout.connect(self._update_compile)

        # ── 连接日志系统到 GUI ──
        LogSignal.add_listener(self._on_log_message)

    # ────────────────────────────────────────────────────
    # 菜单栏
    # ────────────────────────────────────────────────────

    def _setup_menu(self):
        menubar = self.menuBar()

        # 文件
        file_menu = menubar.addMenu("文件(&F)")
        act_open_img = QAction("打开图片...", self)
        act_open_img.setShortcut("Ctrl+O")
        act_open_img.triggered.connect(self._on_open_images)
        file_menu.addAction(act_open_img)

        act_open_dir = QAction("打开文件夹...", self)
        act_open_dir.setShortcut("Ctrl+Shift+O")
        act_open_dir.triggered.connect(self._on_open_folder)
        file_menu.addAction(act_open_dir)

        file_menu.addSeparator()
        act_remove = QAction("移除选中", self)
        act_remove.setShortcut("Delete")
        act_remove.triggered.connect(self._on_remove_selected)
        file_menu.addAction(act_remove)

        act_clear = QAction("清空列表", self)
        act_clear.triggered.connect(self._on_clear_list)
        file_menu.addAction(act_clear)

        file_menu.addSeparator()
        act_exit = QAction("退出", self)
        act_exit.setShortcut("Ctrl+Q")
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # 识别
        rec_menu = menubar.addMenu("识别(&R)")

        # 后端选择子菜单
        backend_menu = rec_menu.addMenu("识别方法")
        self.backend_group = QActionGroup(self)
        self.backend_group.setExclusive(True)
        self.act_backend_cv = QAction("CV 几何算法", self, checkable=True)
        self.act_backend_cv.setChecked(True)
        self.act_backend_cv.setData("cv")
        backend_menu.addAction(self.act_backend_cv)
        self.backend_group.addAction(self.act_backend_cv)

        if DETIKZIFY_AVAILABLE:
            self.act_backend_ai = QAction("AI (DeTikZify)", self, checkable=True)
            self.act_backend_ai.setData("ai")
            backend_menu.addAction(self.act_backend_ai)
            self.backend_group.addAction(self.act_backend_ai)

        self.act_backend_auto = QAction("自动 (AI优先)", self, checkable=True)
        self.act_backend_auto.setData("auto")
        backend_menu.addAction(self.act_backend_auto)
        self.backend_group.addAction(self.act_backend_auto)

        self.backend_group.triggered.connect(self._on_backend_menu_changed)

        rec_menu.addSeparator()
        act_rec_cur = QAction("识别当前图片", self)
        act_rec_cur.setShortcut("F5")
        act_rec_cur.triggered.connect(self._on_recognize_current)
        rec_menu.addAction(act_rec_cur)

        act_rec_all = QAction("识别全部图片", self)
        act_rec_all.setShortcut("Ctrl+F5")
        act_rec_all.triggered.connect(self._on_recognize_all)
        rec_menu.addAction(act_rec_all)

        # 导出
        export_menu = menubar.addMenu("导出(&E)")
        act_exp_tex = QAction("导出 TEX...", self)
        act_exp_tex.setShortcut("Ctrl+S")
        act_exp_tex.triggered.connect(self._on_export_tex)
        export_menu.addAction(act_exp_tex)

        act_exp_png = QAction("导出 PNG...", self)
        act_exp_png.setShortcut("Ctrl+Shift+S")
        act_exp_png.triggered.connect(self._on_export_png)
        export_menu.addAction(act_exp_png)

        export_menu.addSeparator()
        act_merge = QAction("合并导出全部...", self)
        act_merge.setShortcut("Ctrl+M")
        act_merge.triggered.connect(self._on_merge_export)
        export_menu.addAction(act_merge)

        # 帮助
        help_menu = menubar.addMenu("帮助(&H)")
        act_about = QAction("关于", self)
        act_about.triggered.connect(self._on_about)
        help_menu.addAction(act_about)

    # ────────────────────────────────────────────────────
    # 中央区域
    # ────────────────────────────────────────────────────

    def _setup_central(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        splitter = QSplitter(Qt.Horizontal)

        # ── 左面板：图片列表 ──
        left_panel = self._create_left_panel()
        splitter.addWidget(left_panel)

        # ── 中面板：预览 ──
        center_panel = self._create_center_panel()
        splitter.addWidget(center_panel)

        # ── 右面板：代码 + 操作 ──
        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)

        splitter.setSizes([220, 620, 400])
        main_layout.addWidget(splitter)

    def _create_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        title = QLabel("📁 图片列表")
        title.setStyleSheet("font-size: 13px; font-weight: bold; padding: 4px;")
        layout.addWidget(title)

        # 列表
        self.list_widget = QListWidget()
        self.list_widget.setAlternatingRowColors(True)
        self.list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_widget.currentRowChanged.connect(self._on_selection_changed)
        layout.addWidget(self.list_widget, 1)

        # 操作按钮
        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton("+ 添加")
        self.btn_add.clicked.connect(self._on_open_images)
        self.btn_remove = QPushButton("— 移除")
        self.btn_remove.clicked.connect(self._on_remove_selected)
        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_remove)
        layout.addLayout(btn_layout)

        # 批量操作
        batch_group = QGroupBox("批量操作")
        batch_layout = QVBoxLayout(batch_group)
        self.btn_rec_all = QPushButton("▶ 全部识别")
        self.btn_rec_all.clicked.connect(self._on_recognize_all)
        batch_layout.addWidget(self.btn_rec_all)

        self.btn_export_all = QPushButton("📄 全部导出 TEX")
        self.btn_export_all.clicked.connect(self._on_export_all_tex)
        batch_layout.addWidget(self.btn_export_all)

        self.btn_merge = QPushButton("📑 合并导出")
        self.btn_merge.clicked.connect(self._on_merge_export)
        batch_layout.addWidget(self.btn_merge)
        layout.addWidget(batch_group)

        # 识别方法选择
        method_group = QGroupBox("识别方法")
        method_layout = QVBoxLayout(method_group)
        self.backend_combo = QComboBox()
        self.backend_combo.addItem("CV 几何算法", "cv")
        if DETIKZIFY_AVAILABLE:
            self.backend_combo.addItem("AI (DeTikZify)", "ai")
        self.backend_combo.addItem("自动 (AI优先)", "auto")
        self.backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        method_layout.addWidget(self.backend_combo)

        backend_status = QLabel("就绪")
        backend_status.setStyleSheet("font-size: 10px; color: #666;")
        if DETIKZIFY_AVAILABLE:
            backend_status.setText("✓ AI 后端可用")
            backend_status.setStyleSheet("font-size: 10px; color: #4CAF50;")
        else:
            backend_status.setText("AI 后端未安装 (可选)")
            backend_status.setStyleSheet("font-size: 10px; color: #999;")
        method_layout.addWidget(backend_status)
        layout.addWidget(method_group)

        # ── 圆形检测参数滑块 ──
        circle_group = QGroupBox("⚪ 圆形检测参数")
        circle_layout = QVBoxLayout(circle_group)
        circle_layout.setSpacing(2)

        # 像素搜索半径
        tol_layout = QHBoxLayout()
        tol_label = QLabel("像素搜索半径:")
        tol_label.setStyleSheet("font-size: 10px;")
        tol_layout.addWidget(tol_label)
        self.tol_slider = QSlider(Qt.Horizontal)
        self.tol_slider.setRange(1, 10)
        self.tol_slider.setValue(2)
        self.tol_slider.setTickPosition(QSlider.TicksBelow)
        self.tol_slider.setTickInterval(1)
        self.tol_slider.valueChanged.connect(self._on_tol_changed)
        tol_layout.addWidget(self.tol_slider, 1)
        self.tol_value_label = QLabel("2px")
        self.tol_value_label.setFixedWidth(40)
        self.tol_value_label.setStyleSheet("font-size: 10px; font-weight: bold; color: #1565C0;")
        tol_layout.addWidget(self.tol_value_label)
        circle_layout.addLayout(tol_layout)

        self.tol_hint_label = QLabel("建议: 标准打印图形 (2px)")
        self.tol_hint_label.setStyleSheet("font-size: 9px; color: #888; padding-left: 4px;")
        self.tol_hint_label.setWordWrap(True)
        circle_layout.addWidget(self.tol_hint_label)

        # 命中率阈值
        hit_layout = QHBoxLayout()
        hit_label = QLabel("命中率阈值:")
        hit_label.setStyleSheet("font-size: 10px;")
        hit_layout.addWidget(hit_label)
        self.hit_slider = QSlider(Qt.Horizontal)
        self.hit_slider.setRange(10, 100)
        self.hit_slider.setValue(30)
        self.hit_slider.setTickPosition(QSlider.TicksBelow)
        self.hit_slider.setTickInterval(10)
        self.hit_slider.valueChanged.connect(self._on_hit_changed)
        hit_layout.addWidget(self.hit_slider, 1)
        self.hit_value_label = QLabel("0.30")
        self.hit_value_label.setFixedWidth(40)
        self.hit_value_label.setStyleSheet("font-size: 10px; font-weight: bold; color: #E65100;")
        hit_layout.addWidget(self.hit_value_label)
        circle_layout.addLayout(hit_layout)

        self.hit_hint_label = QLabel("建议: 中等偏低（适合手绘圆） (0.30)")
        self.hit_hint_label.setStyleSheet("font-size: 9px; color: #888; padding-left: 4px;")
        self.hit_hint_label.setWordWrap(True)
        circle_layout.addWidget(self.hit_hint_label)

        circle_layout.addSpacing(2)
        reset_btn = QPushButton("恢复默认 (2px / 0.30)")
        reset_btn.setStyleSheet("font-size: 9px; padding: 2px 8px;")
        reset_btn.clicked.connect(self._on_reset_circle_params)
        circle_layout.addWidget(reset_btn)

        layout.addWidget(circle_group)

        # ── 直线检测参数滑块 ──
        line_group = QGroupBox("📏 直线检测参数")
        line_layout = QVBoxLayout(line_group)
        line_layout.setSpacing(2)

        # 像素匹配容差
        line_tol_layout = QHBoxLayout()
        line_tol_label = QLabel("像素匹配容差:")
        line_tol_label.setStyleSheet("font-size: 10px;")
        line_tol_layout.addWidget(line_tol_label)
        self.line_tol_slider = QSlider(Qt.Horizontal)
        self.line_tol_slider.setRange(1, 20)
        self.line_tol_slider.setValue(8)
        self.line_tol_slider.setTickPosition(QSlider.TicksBelow)
        self.line_tol_slider.setTickInterval(2)
        self.line_tol_slider.valueChanged.connect(self._on_line_tol_changed)
        line_tol_layout.addWidget(self.line_tol_slider, 1)
        self.line_tol_value_label = QLabel("8px")
        self.line_tol_value_label.setFixedWidth(40)
        self.line_tol_value_label.setStyleSheet("font-size: 10px; font-weight: bold; color: #1565C0;")
        line_tol_layout.addWidget(self.line_tol_value_label)
        line_layout.addLayout(line_tol_layout)

        self.line_tol_hint_label = QLabel("建议: 标准打印图形 (8px)")
        self.line_tol_hint_label.setStyleSheet("font-size: 9px; color: #888; padding-left: 4px;")
        self.line_tol_hint_label.setWordWrap(True)
        line_layout.addWidget(self.line_tol_hint_label)

        # 命中率阈值
        line_hit_layout = QHBoxLayout()
        line_hit_label = QLabel("命中率阈值:")
        line_hit_label.setStyleSheet("font-size: 10px;")
        line_hit_layout.addWidget(line_hit_label)
        self.line_hit_slider = QSlider(Qt.Horizontal)
        self.line_hit_slider.setRange(10, 100)
        self.line_hit_slider.setValue(60)
        self.line_hit_slider.setTickPosition(QSlider.TicksBelow)
        self.line_hit_slider.setTickInterval(10)
        self.line_hit_slider.valueChanged.connect(self._on_line_hit_changed)
        line_hit_layout.addWidget(self.line_hit_slider, 1)
        self.line_hit_value_label = QLabel("0.60")
        self.line_hit_value_label.setFixedWidth(40)
        self.line_hit_value_label.setStyleSheet("font-size: 10px; font-weight: bold; color: #E65100;")
        line_hit_layout.addWidget(self.line_hit_value_label)
        line_layout.addLayout(line_hit_layout)

        self.line_hit_hint_label = QLabel("建议: 中等（均衡） (0.60)")
        self.line_hit_hint_label.setStyleSheet("font-size: 9px; color: #888; padding-left: 4px;")
        self.line_hit_hint_label.setWordWrap(True)
        line_layout.addWidget(self.line_hit_hint_label)

        line_layout.addSpacing(2)
        line_reset_btn = QPushButton("恢复默认 (8px / 0.60)")
        line_reset_btn.setStyleSheet("font-size: 9px; padding: 2px 8px;")
        line_reset_btn.clicked.connect(self._on_reset_line_params)
        line_layout.addWidget(line_reset_btn)

        layout.addWidget(line_group)

        return panel

    def _create_center_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # 标题栏
        title_layout = QHBoxLayout()
        self.preview_title = QLabel("🖼 预览")
        self.preview_title.setStyleSheet("font-size: 13px; font-weight: bold; padding: 4px;")
        title_layout.addWidget(self.preview_title, 1)

        # 预览模式切换
        self.preview_mode = QButtonGroup(self)
        self.rb_original = QRadioButton("原图")
        self.rb_latex = QRadioButton("LaTeX渲染")
        self.rb_latex.setEnabled(False)
        self.rb_original.setChecked(True)
        self.preview_mode.addButton(self.rb_original, 0)
        self.preview_mode.addButton(self.rb_latex, 1)
        self.preview_mode.idClicked.connect(self._on_preview_mode_changed)
        title_layout.addWidget(self.rb_original)
        title_layout.addWidget(self.rb_latex)
        layout.addLayout(title_layout)

        # 预览区域（滚动）
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: 1px solid #ddd; background: #fafafa;")
        self.preview_label = QLabel("请添加图片并开始识别")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("font-size: 14px; color: #888;")
        scroll.setWidget(self.preview_label)
        layout.addWidget(scroll, 1)

        # 识别信息
        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet(
            "font-size: 11px; color: #333; padding: 4px; "
            "background: #f5f5f5; border: 1px solid #ddd; border-radius: 4px;"
        )
        self.info_label.setMaximumHeight(80)
        layout.addWidget(self.info_label)

        return panel

    def _create_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # 使用 TabWidget 切换代码/日志
        self.right_tabs = QTabWidget()
        self.right_tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #ccc; border-radius: 4px; }
            QTabBar::tab { padding: 4px 12px; font-size: 11px; }
            QTabBar::tab:selected { background: #fff; }
        """)

        # ── Tab 1: TikZ 代码 ──
        code_tab = QWidget()
        code_layout = QVBoxLayout(code_tab)
        code_layout.setContentsMargins(2, 2, 2, 2)
        code_layout.setSpacing(4)

        self.code_edit = QPlainTextEdit()
        self.code_edit.setReadOnly(True)
        self.code_edit.setFont(QFont("Consolas, 'Courier New', monospace", 10))
        self.code_edit.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.code_edit.setStyleSheet(
            "background: #1E1E1E; color: #D4D4D4; "
            "border: 1px solid #333; border-radius: 4px; padding: 4px;"
        )
        self.highlighter = TikZHighlighter(self.code_edit.document())
        code_layout.addWidget(self.code_edit, 1)

        # 操作按钮
        btn_layout = QHBoxLayout()
        self.btn_copy = QPushButton("📋 复制代码")
        self.btn_copy.clicked.connect(self._on_copy_code)
        btn_layout.addWidget(self.btn_copy)

        self.btn_export_tex = QPushButton("📄 导出 TEX")
        self.btn_export_tex.clicked.connect(self._on_export_tex)
        btn_layout.addWidget(self.btn_export_tex)

        self.btn_export_png = QPushButton("🖼 导出 PNG")
        self.btn_export_png.clicked.connect(self._on_export_png)
        btn_layout.addWidget(self.btn_export_png)

        code_layout.addLayout(btn_layout)

        self.code_status = QLabel("就绪")
        self.code_status.setStyleSheet("font-size: 11px; color: #666; padding: 2px;")
        code_layout.addWidget(self.code_status)

        self.right_tabs.addTab(code_tab, "📝 TikZ 代码")

        # ── Tab 2: 运行日志 ──
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        log_layout.setContentsMargins(2, 2, 2, 2)
        log_layout.setSpacing(4)

        # 日志路径提示
        log_path_label = QLabel(
            f"📁 日志文件: {os.path.join(get_log_file_path())}<br>"
            f"📁 崩溃日志: {os.path.join(PROGRAM_DIR, 'crash_log.txt')}"
        )
        log_path_label.setWordWrap(True)
        log_path_label.setStyleSheet(
            "font-size: 10px; color: #888; background: #2a2a2a; "
            "padding: 4px 8px; border-radius: 3px;"
        )
        log_layout.addWidget(log_path_label)

        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFont(QFont("Consolas, 'Courier New', monospace", 9))
        self.log_edit.setMaximumBlockCount(5000)
        self.log_edit.setStyleSheet(
            "background: #1E1E1E; color: #D4D4D4; "
            "border: 1px solid #333; border-radius: 4px; padding: 4px;"
        )
        log_layout.addWidget(self.log_edit, 1)

        log_btn_layout = QHBoxLayout()
        self.btn_clear_log = QPushButton("🧹 清空日志")
        self.btn_clear_log.clicked.connect(self._on_clear_log)
        log_btn_layout.addWidget(self.btn_clear_log)

        self.btn_open_log = QPushButton("📂 打开日志文件")
        self.btn_open_log.clicked.connect(self._on_open_log_file)
        log_btn_layout.addWidget(self.btn_open_log)

        self.btn_refresh_log = QPushButton("🔄 刷新日志")
        self.btn_refresh_log.clicked.connect(self._on_refresh_log)
        log_btn_layout.addWidget(self.btn_refresh_log)

        log_layout.addLayout(log_btn_layout)

        self.right_tabs.addTab(log_tab, "📋 运行日志")

        layout.addWidget(self.right_tabs)

        return panel

    # ────────────────────────────────────────────────────
    # 状态栏
    # ────────────────────────────────────────────────────

    def _setup_statusbar(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.status_label = QLabel("已就绪")
        self.statusbar.addWidget(self.status_label, 1)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setMaximumHeight(16)
        self.progress_bar.setVisible(False)
        self.statusbar.addPermanentWidget(self.progress_bar)

    # ────────────────────────────────────────────────────
    # 样式
    # ────────────────────────────────────────────────────

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow { background: #f0f0f0; }
            QGroupBox {
                font-size: 11px; font-weight: bold;
                border: 1px solid #ccc; border-radius: 4px;
                margin-top: 8px; padding-top: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 2px 6px;
            }
            QPushButton {
                padding: 5px 12px;
                border: 1px solid #ccc;
                border-radius: 4px;
                background: #fff;
                font-size: 11px;
            }
            QPushButton:hover {
                background: #e3f2fd;
                border-color: #2196F3;
            }
            QPushButton:pressed { background: #bbdefb; }
            QListWidget {
                border: 1px solid #ddd;
                border-radius: 4px;
                background: #fff;
                font-size: 11px;
            }
            QListWidget::item:selected {
                background: #e3f2fd;
                color: #000;
            }
            QRadioButton { font-size: 11px; }
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 3px;
                text-align: center;
                font-size: 10px;
            }
            QProgressBar::chunk {
                background: #4CAF50;
                border-radius: 2px;
            }
        """)

    # ────────────────────────────────────────────────────
    # 图片列表操作
    # ────────────────────────────────────────────────────

    def _add_images(self, paths):
        added = 0
        skipped = 0
        for path in paths:
            # 检查文件是否存在
            if not os.path.exists(path):
                logger.warning(f"跳过不存在的文件: {path}")
                continue

            ext = os.path.splitext(path)[1].lower()
            if ext not in ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'):
                logger.warning(f"跳过不支持的文件类型 [{ext}]: {path}")
                skipped += 1
                continue
            if path in self.image_paths:
                logger.debug(f"跳过重复文件: {path}")
                skipped += 1
                continue

            # 验证图片是否可读（用 safe_imread 预检）
            test_img = safe_imread(path)
            if test_img is None:
                logger.error(f"图片无法读取，跳过: {path}")
                skipped += 1
                continue

            self.image_paths.append(path)
            item = QListWidgetItem()
            widget = ImageListItem(path, len(self.image_paths) - 1)
            item.setSizeHint(widget.sizeHint())
            self.list_widget.addItem(item)
            self.list_widget.setItemWidget(item, widget)
            added += 1

        logger.info(f"添加了 {added} 张图片（跳过 {skipped} 张）")
        self._update_status()
        if self.image_paths and self.current_index < 0:
            self.list_widget.setCurrentRow(0)

    def _on_open_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择图片", "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.tiff *.webp);;所有文件 (*)"
        )
        if paths:
            logger.info(f"用户选择了 {len(paths)} 个文件")
            for p in paths[:5]:
                logger.debug(f"  - {p}")
            if len(paths) > 5:
                logger.debug(f"  ... 还有 {len(paths)-5} 个")
            self._add_images(paths)

    def _on_open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder:
            logger.info(f"打开文件夹: {folder}")
            paths = []
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp'):
                found = glob.glob(os.path.join(folder, ext))
                if found:
                    logger.debug(f"  找到 {len(found)} 个 {ext} 文件")
                    paths.extend(found)
                found = glob.glob(os.path.join(folder, ext.upper()))
                if found:
                    logger.debug(f"  找到 {len(found)} 个 {ext.upper()} 文件")
                    paths.extend(found)
            paths = sorted(set(paths))
            logger.info(f"文件夹中找到 {len(paths)} 个图片文件")
            if paths:
                self._add_images(paths)
            else:
                logger.warning(f"文件夹中没有图片文件: {folder}")
                QMessageBox.information(self, "提示", "文件夹中没有找到图片文件")

    def _on_remove_selected(self):
        rows = sorted(set(
            i.row() for i in self.list_widget.selectedIndexes()
        ), reverse=True)
        for row in rows:
            if 0 <= row < len(self.image_paths):
                path = self.image_paths[row]
                if path in self.results:
                    del self.results[path]
                del self.image_paths[row]
                self.list_widget.takeItem(row)
        self._update_status()
        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(min(self.list_widget.count() - 1, max(rows) if rows else 0))
        else:
            self._clear_preview()

    def _on_clear_list(self):
        self.image_paths.clear()
        self.results.clear()
        self.list_widget.clear()
        self._clear_preview()
        self._update_status()

    def _on_selection_changed(self, row):
        if 0 <= row < len(self.image_paths):
            self.current_index = row
            self._show_image(row)
        else:
            self.current_index = -1

    # ────────────────────────────────────────────────────
    # 预览
    # ────────────────────────────────────────────────────

    def _show_image(self, index):
        if not (0 <= index < len(self.image_paths)):
            return
        path = self.image_paths[index]
        self.preview_title.setText(f"🖼 预览: {os.path.basename(path)}")

        if self.rb_original.isChecked():
            self._show_original(path)
        else:
            self._show_latex_preview(path)

        self._update_info(index)
        self._update_code(index)

    def _show_original(self, path):
        logger.info(f"加载原图预览: {path}")
        logger.info(f"  文件是否存在: {os.path.exists(path)}")
        logger.info(f"  文件大小: {os.path.getsize(path) if os.path.exists(path) else 'N/A'} 字节")

        # 使用 QImage.fromData 加载（支持中文路径）
        try:
            with open(path, 'rb') as f:
                img_data = f.read()
            logger.info(f"  读取了 {len(img_data)} 字节")
            qimg = QImage.fromData(img_data)
            logger.info(f"  QImage.fromData: {'成功' if not qimg.isNull() else '失败'}")
            pixmap = QPixmap.fromImage(qimg)
        except Exception as e:
            logger.warning(f"QImage.fromData 加载失败: {e}")
            logger.info("  回退到 QPixmap(path) 直接加载")
            pixmap = QPixmap(path)

        if not pixmap.isNull():
            logger.info(f"  图片加载成功: {pixmap.width()}x{pixmap.height()}")
            scaled = pixmap.scaled(
                self.preview_label.size(), Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            # 在图片上叠加顶点标注
            if path in self.results and self.results[path].success:
                # 使用 Pixmap 绘制标注
                result = self.results[path]
                painter = QPainter(scaled)
                painter.setRenderHint(QPainter.Antialiasing)
                # 缩放因子
                sx = scaled.width() / pixmap.width()
                sy = scaled.height() / pixmap.height()
                for name, pt in result.key_points.items():
                    if name == 'O':
                        continue
                    x, y = int(pt[0] * sx), int(pt[1] * sy)
                    # 红色圆点
                    painter.setPen(QPen(QColor("#FF5722"), 3))
                    painter.setBrush(QBrush(QColor("#FF5722")))
                    painter.drawEllipse(x-4, y-4, 8, 8)
                    # 标签
                    painter.setPen(QPen(QColor("#FF5722"), 1))
                    font = QFont("sans-serif", 10, QFont.Weight.Bold)
                    painter.setFont(font)
                    painter.drawText(x+8, y-6, name)
                painter.end()

            self.preview_label.setPixmap(scaled)
        else:
            self.preview_label.setText("无法加载图片")
            logger.error(f"QPixmap 加载失败: {path}")
            logger.info("  尝试用 safe_imread 验证图片是否可读...")
            img = safe_imread(path)
            if img is not None:
                logger.info("  safe_imread 成功，是 Qt 格式兼容问题")
            else:
                logger.error("  safe_imread 也失败，图片文件可能已损坏")

    def _show_latex_preview(self, path):
        if path in self.results and self.results[path].success:
            preview_path = self.results[path].preview_image_path
            if preview_path and os.path.exists(preview_path):
                pixmap = QPixmap(preview_path)
                if not pixmap.isNull():
                    scaled = pixmap.scaled(
                        self.preview_label.size(), Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                    self.preview_label.setPixmap(scaled)
                    return
        self.preview_label.setText("尚无 LaTeX 渲染结果")

    def _on_preview_mode_changed(self, mode_id):
        if self.current_index >= 0:
            self._show_image(self.current_index)

    def _clear_preview(self):
        self.preview_label.setText("请添加图片并开始识别")
        self.preview_title.setText("🖼 预览")
        self.code_edit.clear()
        self.info_label.clear()
        self.code_status.setText("就绪")

    # ────────────────────────────────────────────────────
    # 信息显示
    # ────────────────────────────────────────────────────

    def _update_info(self, index):
        if not (0 <= index < len(self.image_paths)):
            self.info_label.clear()
            return
        path = self.image_paths[index]
        if path not in self.results:
            self.info_label.setText("⏳ 尚未识别")
            return
        result = self.results[path]
        if not result.success:
            self.info_label.setText(f"❌ 识别失败:\n{result.error[:200]}")
            return

        pts = result.key_points
        valid = result.valid_connections
        info = []
        info.append("✅ 识别成功")
        info.append(f"  顶点: {', '.join(pts.keys())}")
        info.append(f"  检测到线段: {len(valid)} 条")
        for p1, p2, conf in valid:
            info.append(f"    {p1}-{p2}: {conf:.0f}%")
        info.append(f"  构造层辅助线: DF, EG, DG, EF")
        self.info_label.setText("\n".join(info))

    def _update_code(self, index):
        if not (0 <= index < len(self.image_paths)):
            self.code_edit.clear()
            return
        path = self.image_paths[index]
        if path not in self.results or not self.results[path].success:
            self.code_edit.clear()
            self.code_status.setText("尚无代码")
            return

        tex = self.results[path].tex_code
        self.code_edit.setPlainText(tex)
        self.code_status.setText(f"代码行数: {tex.count(chr(10)) + 1}")

    def _update_status(self):
        self.status_label.setText(
            f"已加载 {len(self.image_paths)} 张图片 | "
            f"已识别 {sum(1 for r in self.results.values() if r.success)} 张"
        )

    # ────────────────────────────────────────────────────
    # 识别逻辑
    # ────────────────────────────────────────────────────

    def _on_recognize_current(self):
        if self.current_index < 0:
            QMessageBox.information(self, "提示", "请先选择一张图片")
            return
        path = self.image_paths[self.current_index]
        logger.info(f"用户请求识别当前图片: {path}")
        self._start_recognize(path)

    def _on_recognize_all(self):
        if not self.image_paths:
            QMessageBox.information(self, "提示", "请先添加图片")
            return
        pending = [p for p in self.image_paths if p not in self.results]
        logger.info(f"用户请求批量识别全部图片: {len(pending)} 张待识别")
        for path in pending:
            self._start_recognize(path)

    def _get_current_backend(self):
        """获取当前选中的识别后端"""
        return self.backend_combo.currentData()

    def _on_backend_changed(self, index):
        """下拉框后端选择变更"""
        backend = self.backend_combo.itemData(index)
        logger.info(f"识别方法切换为: {backend}")
        # 同步菜单选择
        for act in self.backend_group.actions():
            if act.data() == backend:
                act.setChecked(True)
                break

    def _on_backend_menu_changed(self, action):
        """菜单后端选择变更"""
        backend = action.data()
        logger.info(f"识别方法切换为: {backend}")
        # 同步下拉框选择
        for i in range(self.backend_combo.count()):
            if self.backend_combo.itemData(i) == backend:
                self.backend_combo.setCurrentIndex(i)
                break

    def _start_recognize(self, path):
        if path in self.recognizer_pool:
            logger.debug(f"图片已在识别中，跳过: {path}")
            return  # 已在识别中

        backend = self._get_current_backend()
        logger.info(f"启动识别线程: {path} (后端: {backend})")

        # 更新列表状态
        item = self._find_item_by_path(path)
        if item:
            item.set_status("processing")

        # 读取滑块参数
        tol = self.tol_slider.value()
        hit = self.hit_slider.value() / 100.0
        line_tol = self.line_tol_slider.value()
        line_hit = self.line_hit_slider.value() / 100.0
        logger.info(f"  圆形检测参数: 像素搜索半径={tol}px, 命中率阈值={hit:.2f}")
        logger.info(f"  直线检测参数: 像素匹配容差={line_tol}px, 命中率阈值={line_hit:.2f}")

        # 启动线程
        worker = RecognizeWorker(
            path, backend=backend,
            circle_pixel_tolerance=tol,
            circle_hit_threshold=hit,
            line_pixel_tolerance=line_tol,
            line_hit_threshold=line_hit,
        )
        worker.finished.connect(self._on_recognize_finished)
        worker.progress.connect(self._on_recognize_progress)
        self.recognizer_pool[path] = worker
        worker.start()

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"正在识别: {os.path.basename(path)}...")

    def _on_recognize_progress(self, message, percent):
        self.status_label.setText(message)
        self.progress_bar.setValue(percent)

    def _on_recognize_finished(self, image_path, result):
        if image_path in self.recognizer_pool:
            del self.recognizer_pool[image_path]

        self.results[image_path] = result

        if result.success:
            logger.info(f"识别成功: {image_path}")
        else:
            logger.error(f"识别失败: {image_path}")
            logger.error(f"  错误: {result.error[:200]}")

        # 更新列表状态
        item = self._find_item_by_path(image_path)
        if item:
            item.set_status("done" if result.success else "failed")
            item.result = result

        # 如果是当前选中的图片，更新显示
        if (self.current_index >= 0 and
            self.current_index < len(self.image_paths) and
            self.image_paths[self.current_index] == image_path):
            self._show_image(self.current_index)

        self._update_status()

        # 更新进度条
        if not self.recognizer_pool:
            self.progress_bar.setVisible(False)
            done = sum(1 for r in self.results.values() if r.success)
            failed = sum(1 for r in self.results.values() if not r.success)
            self.status_label.setText(f"识别完成: {done} 成功, {failed} 失败")
            logger.info(f"全部识别完成: {done} 成功, {failed} 失败")

    def _find_item_by_path(self, path):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            widget = self.list_widget.itemWidget(item)
            if widget and widget.path == path:
                return widget
        return None

    # ────────────────────────────────────────────────────
    # 导出
    # ────────────────────────────────────────────────────

    def _on_copy_code(self):
        tex = self.code_edit.toPlainText()
        if not tex:
            QMessageBox.information(self, "提示", "没有可复制的代码")
            return
        clipboard = QApplication.clipboard()
        clipboard.setText(tex)
        self.code_status.setText("✔ 已复制到剪贴板")
        QTimer.singleShot(2000, lambda: self.code_status.setText("代码已复制"))

    def _on_export_tex(self):
        path = self._get_current_image_path()
        if not path:
            QMessageBox.information(self, "提示", "请先选择一张图片")
            return
        if path not in self.results or not self.results[path].success:
            QMessageBox.information(self, "提示", "当前图片尚未识别成功")
            return

        default_name = os.path.splitext(os.path.basename(path))[0] + ".tex"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "导出 TEX", default_name,
            "LaTeX 文件 (*.tex);;所有文件 (*)"
        )
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(self.results[path].tex_code)
            self.status_label.setText(f"已导出: {save_path}")

    def _on_export_png(self):
        path = self._get_current_image_path()
        if not path:
            QMessageBox.information(self, "提示", "请先选择一张图片")
            return
        if path not in self.results or not self.results[path].success:
            QMessageBox.information(self, "提示", "当前图片尚未识别成功")
            return

        preview_path = self.results[path].preview_image_path
        if not preview_path or not os.path.exists(preview_path):
            # 重新编译
            recognizer = GeometryRecognizer()
            preview_path = recognizer.compile_tex(self.results[path].tex_code)
            self.results[path].preview_image_path = preview_path

        if not preview_path or not os.path.exists(preview_path):
            QMessageBox.warning(self, "错误", "编译失败，请检查 LaTeX 环境")
            return

        default_name = os.path.splitext(os.path.basename(path))[0] + ".png"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "导出 PNG", default_name,
            "PNG 图片 (*.png);;所有文件 (*)"
        )
        if save_path:
            shutil.copy2(preview_path, save_path)
            self.status_label.setText(f"已导出: {save_path}")

    def _on_export_all_tex(self):
        """批量导出所有识别结果的 TEX 文件"""
        if not self.results:
            QMessageBox.information(self, "提示", "没有已识别的结果")
            return

        folder = QFileDialog.getExistingDirectory(self, "选择导出文件夹")
        if not folder:
            return

        count = 0
        for path, result in self.results.items():
            if not result.success:
                continue
            name = os.path.splitext(os.path.basename(path))[0]
            tex_path = os.path.join(folder, f"{name}.tex")
            with open(tex_path, 'w', encoding='utf-8') as f:
                f.write(result.tex_code)
            count += 1

        QMessageBox.information(self, "导出完成", f"已导出 {count} 个 TEX 文件到:\n{folder}")

    def _on_merge_export(self):
        """合并导出所有识别结果为单个 PDF"""
        successful = [p for p, r in self.results.items() if r.success]
        if not successful:
            QMessageBox.information(self, "提示", "没有已识别成功的结果")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "合并导出", "merged_output.tex",
            "LaTeX 文件 (*.tex);;PDF 文件 (*.pdf);;所有文件 (*)"
        )
        if not save_path:
            return

        # 生成合并的 LaTeX 文档
        lines = []
        lines.append(r"\documentclass[a4paper]{article}")
        lines.append(r"\usepackage{tikz}")
        lines.append(r"\usetikzlibrary{arrows}")
        lines.append(r"\usepackage[margin=1cm]{geometry}")
        lines.append(r"\usepackage{graphicx}")
        lines.append("")
        lines.append(r"\begin{document}")
        lines.append(r"\title{几何图形识别结果汇总}")
        lines.append(r"\author{自动生成}")
        lines.append(r"\maketitle")
        lines.append(r"\tableofcontents")
        lines.append(r"\newpage")

        for i, path in enumerate(successful):
            result = self.results[path]
            name = os.path.basename(path)
            lines.append(f"")
            lines.append(r"\section{%s}" % name.replace('_', r'\_'))
            lines.append("")

            # 嵌入原图
            lines.append(r"\noindent\textbf{原图:}\\\\" )
            lines.append(r"\includegraphics[width=0.6\textwidth]{%s}" % path)
            lines.append("")

            # 嵌入 TikZ 代码
            lines.append(r"\noindent\textbf{识别结果:}\\\\" )
            lines.append(result.tex_code.replace(r"\begin{document}", "")
                                            .replace(r"\end{document}", "")
                                            .replace(r"\begin{tikzpicture}",
                                                     r"\begin{tikzpicture}[scale=0.8]"))
            lines.append("")
            lines.append(r"\noindent\textbf{TikZ 代码:}\\\\" )
            lines.append(r"\begin{verbatim}")
            lines.append(result.tex_code)
            lines.append(r"\end{verbatim}")
            lines.append(r"\newpage")

        lines.append(r"\end{document}")

        tex_content = "\n".join(lines)

        # 保存 TEX
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(tex_content)

        # 编译
        self.status_label.setText("正在编译合并文档...")
        out_dir = os.path.dirname(save_path)
        base_name = os.path.splitext(os.path.basename(save_path))[0]

        for i in range(2):
            subprocess.run(
                ['pdflatex', '-interaction=nonstopmode',
                 f'-output-directory={out_dir}', save_path],
                capture_output=True, text=True, timeout=60
            )

        pdf_path = os.path.join(out_dir, f"{base_name}.pdf")
        if os.path.exists(pdf_path):
            self.status_label.setText(f"合并导出完成: {pdf_path}")
            QMessageBox.information(self, "导出完成",
                f"合并 PDF 已生成:\n{pdf_path}")
        else:
            QMessageBox.warning(self, "编译失败",
                "LaTeX 编译失败，已保存 TEX 文件，请手动编译。")

    def _get_current_image_path(self):
        if self.current_index >= 0 and self.current_index < len(self.image_paths):
            return self.image_paths[self.current_index]
        return None

    # ────────────────────────────────────────────────────
    # 编译更新（定时器回调用）
    # ────────────────────────────────────────────────────

    def _update_compile(self):
        pass  # 预留

    # ────────────────────────────────────────────────────
    # 关于
    # ────────────────────────────────────────────────────

    def _on_about(self):
        ai_status = "✓ AI 后端已安装" if DETIKZIFY_AVAILABLE else "AI 后端未安装 (可选)"
        QMessageBox.about(self, "关于",
            "几何图形矢量化识别系统\n\n"
            "基于多特征融合的手绘几何图形矢量化识别算法\n\n"
            "功能：\n"
            "  • 手绘几何图形识别 (CV 几何算法)\n"
            "  • AI 智能识别 (DeTikZify 多模态大模型)\n"
            "  • 自动生成 TikZ 代码\n"
            "  • 批量处理 / 合并导出\n\n"
            f"识别后端: {ai_status}\n\n"
            "技术栈: Python + OpenCV + PySide6 + LaTeX + DeTikZify"
        )

    # ────────────────────────────────────────────────────
    # 圆形检测参数滑块回调
    # ────────────────────────────────────────────────────

    def _on_tol_changed(self, value):
        """像素搜索半径滑块变化"""
        self.tol_value_label.setText(f"{value}px")
        hint_text = self._get_tol_hint(value)
        self.tol_hint_label.setText(f"建议: {hint_text} ({value}px)")
        # 颜色渐变提示
        if value <= 2:
            self.tol_hint_label.setStyleSheet("font-size: 9px; color: #4CAF50; padding-left: 4px;")
        elif value <= 4:
            self.tol_hint_label.setStyleSheet("font-size: 9px; color: #FF9800; padding-left: 4px;")
        else:
            self.tol_hint_label.setStyleSheet("font-size: 9px; color: #F44336; padding-left: 4px;")
        logger.debug(f"像素搜索半径调整为: {value}px ({hint_text})")

    def _get_tol_hint(self, value):
        """根据像素搜索半径值返回建议文本"""
        if value <= 1:
            return "精确图形/扫描图"
        elif value <= 2:
            return "标准打印图形"
        elif value <= 3:
            return "手绘图形"
        elif value <= 5:
            return "粗糙手绘图"
        else:
            return "严重失真/模糊图"

    def _on_hit_changed(self, value):
        """命中率阈值滑块变化"""
        val = value / 100.0
        self.hit_value_label.setText(f"{val:.2f}")
        hint_text = self._get_hit_hint(val)
        self.hit_hint_label.setText(f"建议: {hint_text} ({val:.2f})")
        # 颜色渐变提示
        if val >= 0.50:
            self.hit_hint_label.setStyleSheet("font-size: 9px; color: #4CAF50; padding-left: 4px;")
        elif val >= 0.30:
            self.hit_hint_label.setStyleSheet("font-size: 9px; color: #FF9800; padding-left: 4px;")
        else:
            self.hit_hint_label.setStyleSheet("font-size: 9px; color: #F44336; padding-left: 4px;")
        logger.debug(f"命中率阈值调整为: {val:.2f} ({hint_text})")

    def _get_hit_hint(self, value):
        """根据命中率阈值返回建议文本"""
        if value < 0.20:
            return "宽松检测（易误检）"
        elif value < 0.30:
            return "低阈值（适合手绘粗略圆）"
        elif value < 0.40:
            return "中等偏低（适合手绘圆）"
        elif value < 0.50:
            return "中等（均衡）"
        elif value < 0.60:
            return "较高（适合精确圆）"
        elif value < 0.80:
            return "严格（仅精确图形）"
        else:
            return "极严格（仅完美圆形）"

    def _on_reset_circle_params(self):
        """恢复圆形检测参数为默认值"""
        self.tol_slider.setValue(2)
        self.hit_slider.setValue(30)
        self.tol_value_label.setText("2px")
        self.hit_value_label.setText("0.30")
        logger.info("圆形检测参数已恢复默认: 像素搜索半径=2px, 命中率阈值=0.30")

    # ────────────────────────────────────────────────────
    # 直线检测参数滑块回调
    # ────────────────────────────────────────────────────

    def _on_line_tol_changed(self, value):
        """直线像素匹配容差滑块变化"""
        self.line_tol_value_label.setText(f"{value}px")
        hint_text = self._get_line_tol_hint(value)
        self.line_tol_hint_label.setText(f"建议: {hint_text} ({value}px)")
        # 颜色渐变提示
        if value <= 5:
            self.line_tol_hint_label.setStyleSheet("font-size: 9px; color: #4CAF50; padding-left: 4px;")
        elif value <= 10:
            self.line_tol_hint_label.setStyleSheet("font-size: 9px; color: #FF9800; padding-left: 4px;")
        else:
            self.line_tol_hint_label.setStyleSheet("font-size: 9px; color: #F44336; padding-left: 4px;")
        logger.debug(f"直线像素匹配容差调整为: {value}px ({hint_text})")

    def _get_line_tol_hint(self, value):
        """根据直线像素匹配容差返回建议文本"""
        if value <= 3:
            return "精确图形/扫描图"
        elif value <= 5:
            return "标准打印图形"
        elif value <= 8:
            return "手绘图形"
        elif value <= 12:
            return "较粗糙手绘图"
        else:
            return "严重失真/模糊图"

    def _on_line_hit_changed(self, value):
        """直线命中率阈值滑块变化"""
        val = value / 100.0
        self.line_hit_value_label.setText(f"{val:.2f}")
        hint_text = self._get_line_hit_hint(val)
        self.line_hit_hint_label.setText(f"建议: {hint_text} ({val:.2f})")
        # 颜色渐变提示
        if val >= 0.70:
            self.line_hit_hint_label.setStyleSheet("font-size: 9px; color: #4CAF50; padding-left: 4px;")
        elif val >= 0.50:
            self.line_hit_hint_label.setStyleSheet("font-size: 9px; color: #FF9800; padding-left: 4px;")
        else:
            self.line_hit_hint_label.setStyleSheet("font-size: 9px; color: #F44336; padding-left: 4px;")
        logger.debug(f"直线命中率阈值调整为: {val:.2f} ({hint_text})")

    def _get_line_hit_hint(self, value):
        """根据直线命中率阈值返回建议文本"""
        if value < 0.20:
            return "宽松检测（易误检）"
        elif value < 0.35:
            return "低阈值（适合模糊手绘）"
        elif value < 0.50:
            return "中等偏低（适合手绘）"
        elif value < 0.60:
            return "中等（均衡）"
        elif value < 0.70:
            return "中等偏高（适合打印图）"
        elif value < 0.85:
            return "严格（仅清晰线段）"
        else:
            return "极严格（仅完美直线）"

    def _on_reset_line_params(self):
        """恢复直线检测参数为默认值"""
        self.line_tol_slider.setValue(8)
        self.line_hit_slider.setValue(60)
        self.line_tol_value_label.setText("8px")
        self.line_hit_value_label.setText("0.60")
        logger.info("直线检测参数已恢复默认: 像素匹配容差=8px, 命中率阈值=0.60")

    # ────────────────────────────────────────────────────
    # 析构
    # ────────────────────────────────────────────────────

    # ────────────────────────────────────────────────────
    # 日志面板
    # ────────────────────────────────────────────────────

    def _on_log_message(self, level, message):
        """接收 LogSignal 的日志消息，显示到 GUI 日志面板"""
        try:
            color_map = {
                "DEBUG": "#888888",
                "INFO": "#D4D4D4",
                "WARNING": "#FFD700",
                "ERROR": "#FF6B6B",
                "CRITICAL": "#FF4444",
            }
            color = color_map.get(level, "#D4D4D4")
            timestamp = datetime.now().strftime("%H:%M:%S")
            html = f'<span style="color:{color}">[{timestamp}] [{level}] {message}</span><br>'
            self.log_edit.appendHtml(html)
            # 自动滚动到底部
            cursor = self.log_edit.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.log_edit.setTextCursor(cursor)
        except Exception:
            pass

    def _on_clear_log(self):
        """清空日志面板"""
        self.log_edit.clear()
        logger.info("日志已清空")

    def _on_open_log_file(self):
        """打开日志文件所在目录"""
        log_path = get_log_file_path()
        log_dir = os.path.dirname(log_path)
        try:
            if sys.platform == 'win32':
                os.startfile(log_dir)
            elif sys.platform == 'darwin':
                subprocess.run(['open', log_dir], check=False)
            else:
                subprocess.run(['xdg-open', log_dir], check=False)
            logger.info(f"打开日志目录: {log_dir}")
        except Exception as e:
            logger.warning(f"打开日志目录失败: {e}")

    def _on_refresh_log(self):
        """从日志文件重新加载最新日志到面板"""
        lines = read_recent_logs(200)
        self.log_edit.clear()
        for line in lines:
            # 解析日志级别进行颜色标记
            if "[ERROR]" in line:
                level = "ERROR"
            elif "[WARNING]" in line:
                level = "WARNING"
            elif "[INFO]" in line:
                level = "INFO"
            else:
                level = "DEBUG"
            self._on_log_message(level, line)
        logger.info(f"已刷新日志面板: {len(lines)} 行")

    def closeEvent(self, event):
        # 等待所有识别线程结束
        for worker in self.recognizer_pool.values():
            worker.wait(5000)
        super().closeEvent(event)


# ═══════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════

def main():
    try:
        # ── PyInstaller 兼容：让 multiprocessing 子进程能正确启动 ──
        mp.freeze_support()

        # 启用 faulthandler 捕获 C 层 segfault，输出到程序目录
        crash_fd_path = os.path.join(PROGRAM_DIR, "crash_faulthandler.log")
        try:
            crash_fd = open(crash_fd_path, "a")
            faulthandler.enable(file=crash_fd)
            logger.info(f"faulthandler 已启用: {crash_fd_path}")
        except Exception as e:
            logger.warning(f"faulthandler 启用失败: {e}")

        # 高DPI支持
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
        app = QApplication(sys.argv)
        app.setApplicationName("几何图形矢量化识别系统")

        # 设置全局字体
        font = QFont("Microsoft YaHei, 'Noto Sans CJK SC', 'PingFang SC', sans-serif", 10)
        app.setFont(font)

        # 日志记录启动信息
        logger.info("程序启动")
        logger.info(f"Python 版本: {sys.version}")
        logger.info(f"程序目录: {PROGRAM_DIR}")
        logger.info(f"日志目录: {os.path.join(PROGRAM_DIR, 'logs')}")

        window = MainWindow()
        window.show()

        # 包裹事件循环，捕获 Qt 事件中的异常
        exit_code = app.exec()
        logger.info(f"程序正常退出，退出码: {exit_code}")
        sys.exit(exit_code)
    except Exception as e:
        tb_str = traceback.format_exc()
        write_crash_log(f"main() 异常: {type(e).__name__}: {e}", tb_str)
        logger.critical(f"程序启动/运行异常: {e}\n{tb_str}")
        raise


if __name__ == "__main__":
    main()