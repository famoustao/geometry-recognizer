"""
全局配置与自适应阈值模块

基于第4章的自适应阈值优化方案，实现所有关键阈值的动态计算。
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import math


@dataclass
class AdaptiveThresholds:
    """
    自适应阈值集合

    所有阈值根据图像分辨率、线条粗细动态缩放。
    """

    # ====== 图像预处理阈值 ======
    # 高斯滤波核大小
    gaussian_kernel_size: int = 5
    # 自适应二值化块大小 (必须为奇数)
    adaptive_block_size: int = 31
    # 自适应二值化C值
    adaptive_c: float = 2.0

    # ====== 形态学阈值 ======
    # 形态学核大小基准 (将根据线条粗细自适应)
    morph_kernel_base: int = 3
    # 断线修复最大膨胀迭代次数
    max_dilate_iterations: int = 3
    # 毛刺去除腐蚀迭代次数
    erode_iterations: int = 1

    # ====== 骨架化阈值 ======
    # 骨架剪枝长度阈值 (像素)
    prune_length_base: int = 10

    # ====== DBSCAN聚类阈值 ======
    # 邻域距离基准 (像素) - 随分辨率线性缩放
    dbscan_eps_base: float = 5.0
    # 最小样本数
    dbscan_min_samples: int = 2

    # ====== 直曲分类阈值 ======
    # 伸长率基准 r0 = 1.12 (第4章)
    elongation_ratio_base: float = 1.12
    # 短线拉伸上浮系数
    short_line_elongation_boost: float = 0.08
    # 长线下调系数
    long_line_elongation_penalty: float = 0.05

    # ====== 直线校验阈值 ======
    # 方向一致性角度基准 (度)
    angle_consistency_base: float = 15.0
    # 虚线/射线放宽角度
    dashed_angle_tolerance: float = 20.0
    # 像素重合度基准
    pixel_overlap_base: float = 0.6
    # 淡线条重合度降低值
    faint_line_overlap_penalty: float = 0.15

    # ====== RANSAC拟合阈值 ======
    # RANSAC距离阈值 (像素)
    ransac_distance_base: float = 2.0
    # 圆弧拟合残差阈值 (像素)
    arc_residual_threshold: float = 3.0

    # ====== 圆弧识别阈值 ======
    # 向量夹角均值基准 (度) - 第4章
    arc_angle_mean_base: float = 8.0
    # 短弧放宽角度
    short_arc_angle_tolerance: float = 12.0

    # ====== 顶点聚类阈值 ======
    # 顶点融合距离 (像素)
    vertex_fusion_distance_base: float = 5.0

    # ====== 几何规整阈值 ======
    # 垂直判定角度容差 (度)
    perpendicular_tolerance: float = 5.0
    # 平行判定角度容差 (度)
    parallel_tolerance: float = 5.0
    # 等长判定比例容差
    equal_length_tolerance: float = 0.05
    # 共线判定角度容差 (度)
    collinear_tolerance: float = 3.0

    # ====== 虚拟顶点阈值 ======
    # 垂足投影距离阈值 (像素)
    foot_distance_threshold: float = 5.0
    # 延长线交点外推比例
    extension_ratio: float = 1.5

    # ====== 图像分辨率 ======
    image_width: int = 0
    image_height: int = 0
    # 参考分辨率 (用于缩放)
    reference_resolution: float = 1000.0

    # ====== LaTeX输出阈值 ======
    # 坐标归一化范围
    tikz_coord_range: float = 5.0
    # 坐标精度 (小数位数)
    coord_precision: int = 3

    # ====== 性能优化 ======
    # 骨架下采样步长
    skeleton_downsample_step: int = 2
    # 分块处理的行数
    tile_height: int = 500

    def get_resolution_scale(self) -> float:
        """根据图像分辨率获取缩放因子"""
        if self.image_width == 0 or self.image_height == 0:
            return 1.0
        diagonal = math.sqrt(self.image_width**2 + self.image_height**2)
        ref_diagonal = math.sqrt(2) * self.reference_resolution
        return max(0.5, min(2.0, diagonal / ref_diagonal))

    @property
    def dbscan_eps(self) -> float:
        """自适应DBSCAN邻域距离"""
        return self.dbscan_eps_base * self.get_resolution_scale()

    @property
    def vertex_fusion_distance(self) -> float:
        """自适应顶点融合距离"""
        return self.vertex_fusion_distance_base * self.get_resolution_scale()

    @property
    def prune_length(self) -> int:
        """自适应骨架剪枝长度"""
        return max(5, int(self.prune_length_base * self.get_resolution_scale()))

    @property
    def ransac_distance(self) -> float:
        """自适应RANSAC距离阈值"""
        return self.ransac_distance_base * self.get_resolution_scale()

    @property
    def morph_kernel_size(self) -> int:
        """自适应形态学核大小"""
        base = self.morph_kernel_base
        scale = self.get_resolution_scale()
        size = max(3, int(base * scale))
        return size if size % 2 == 1 else size + 1

    def get_elongation_threshold(self, segment_length: float) -> float:
        """
        自适应伸长率阈值 (第4章第1条)
        短线自动上浮，长线自动下调
        """
        threshold = self.elongation_ratio_base
        if segment_length < 50:
            threshold += self.short_line_elongation_boost
        elif segment_length > 200:
            threshold -= self.long_line_elongation_penalty
        return max(1.01, min(1.30, threshold))

    def get_angle_consistency_threshold(self, line_type: str = "solid") -> float:
        """
        自适应方向一致性角度阈值 (第4章第2条)
        虚线/射线放宽至20度
        """
        base = self.angle_consistency_base
        if line_type in ("dashed", "ray", "dash_dot"):
            return self.dashed_angle_tolerance
        return base

    def get_pixel_overlap_threshold(self, is_faint: bool = False) -> float:
        """
        自适应像素重合度阈值 (第4章第3条)
        淡手绘线条降低至0.45
        """
        if is_faint:
            return self.pixel_overlap_base - self.faint_line_overlap_penalty
        return self.pixel_overlap_base

    def get_arc_angle_threshold(self, arc_length: float, total_length: float) -> float:
        """
        自适应圆弧角度阈值 (第4章第6条)
        短弧放宽至12度
        """
        ratio = arc_length / max(total_length, 1)
        if ratio < 0.3:
            return self.short_arc_angle_tolerance
        return self.arc_angle_mean_base


@dataclass
class GeometryConfig:
    """
    全局几何识别配置

    包含所有可调参数，提供默认值。
    """
    # 自适应阈值
    thresholds: AdaptiveThresholds = field(default_factory=AdaptiveThresholds)

    # ====== 预处理控制 ======
    enable_perspective_correction: bool = True
    enable_illumination_equalization: bool = True
    enable_text_masking: bool = True
    enable_morphology_repair: bool = True
    enable_skeleton_pruning: bool = True

    # ====== 顶点检测控制 ======
    enable_dbscan_clustering: bool = True
    enable_virtual_vertices: bool = True
    enable_line_type_detection: bool = True

    # ====== 图元识别控制 ======
    enable_adaptive_classification: bool = True
    enable_elliptic_arc_detection: bool = True
    enable_triple_line_verification: bool = True

    # ====== 拓扑后处理控制 ======
    enable_full_intersection: bool = True
    enable_geometric_regularization: bool = True
    enable_hierarchical_storage: bool = True

    # ====== LaTeX输出控制 ======
    enable_latex_output: bool = True
    enable_coordinate_normalization: bool = True
    enable_auto_naming: bool = True
    latex_template: str = "standalone"

    # ====== 性能控制 ======
    enable_downsampling: bool = True
    enable_tiled_processing: bool = False
    max_image_size: int = 2000

    # ====== 交互控制 ======
    enable_manual_edit: bool = False
    interactive_mode: bool = False

    # ====== 场景拓展 ======
    enable_3d_recognition: bool = False
    enable_batch_processing: bool = False

    def update_image_size(self, width: int, height: int):
        """根据图像尺寸更新自适应阈值"""
        self.thresholds.image_width = width
        self.thresholds.image_height = height