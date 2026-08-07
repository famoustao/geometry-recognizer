"""
前置图像预处理模块 (第3.1节)

实现:
1. 透视畸变校正
2. 自适应光照均衡化
3. 文字实例分割（轻量方法）
4. 自适应形态学断线修复
5. 骨架细化+分叉剪枝
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from .config import GeometryConfig
from .utils import estimate_line_width


class ImagePreprocessor:
    """
    图像预处理器

    处理流程:
    输入图像 → 透视校正 → 光照均衡 → 灰度化 → 二值化
    → 文字屏蔽 → 形态学修复 → 骨架化 → 剪枝
    """

    def __init__(self, config: GeometryConfig):
        self.config = config
        self.th = config.thresholds
        self.original_image: Optional[np.ndarray] = None
        self.processed_image: Optional[np.ndarray] = None
        self.binary_image: Optional[np.ndarray] = None
        self.skeleton: Optional[np.ndarray] = None
        self.text_mask: Optional[np.ndarray] = None
        self.line_width: int = 3

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        执行完整预处理流水线

        Args:
            image: 输入图像 (H, W, C) BGR或灰度

        Returns:
            (processed_image, binary_image, skeleton, text_mask)
        """
        self.original_image = image.copy()

        # 灰度化
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 更新配置中的图像尺寸
        self.config.update_image_size(gray.shape[1], gray.shape[0])

        # 1. 透视畸变校正
        if self.config.enable_perspective_correction:
            gray = self._perspective_correction(gray)

        # 2. 光照均衡化
        if self.config.enable_illumination_equalization:
            gray = self._illumination_equalization(gray)

        # 3. 自适应二值化
        binary = self._adaptive_binarization(gray)

        # 4. 文字屏蔽
        text_mask = np.zeros_like(binary)
        if self.config.enable_text_masking:
            text_mask = self._text_instance_segmentation(binary, gray)
            # 只屏蔽文字内部，保留轮廓
            binary_cleaned = self._apply_text_mask(binary, text_mask)
        else:
            binary_cleaned = binary.copy()

        # 估计线条宽度
        self.line_width = estimate_line_width(binary_cleaned)

        # 5. 形态学断线修复
        if self.config.enable_morphology_repair:
            binary_cleaned = self._morphological_repair(binary_cleaned)

        # 6. 骨架化
        skeleton = self._skeletonize(binary_cleaned)

        # 7. 骨架剪枝
        if self.config.enable_skeleton_pruning:
            skeleton = self._skeleton_pruning(skeleton)

        self.processed_image = gray
        self.binary_image = binary_cleaned
        self.skeleton = skeleton
        self.text_mask = text_mask

        return gray, binary_cleaned, skeleton, text_mask

    # ============================================================
    # 1. 透视畸变校正 (第3.1.1节)
    # ============================================================

    def _perspective_correction(self, gray: np.ndarray) -> np.ndarray:
        """
        图纸边框霍夫检测 + 四点透视变换矫正

        检测图像中的最大四边形轮廓，进行透视变换。
        """
        h, w = gray.shape

        # 边缘检测
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # 霍夫线检测
        lines = cv2.HoughLines(edges, 1, np.pi / 180, max(100, int(min(h, w) * 0.15)))

        if lines is None or len(lines) < 4:
            return gray

        # 找到四条边界线
        # 按角度分类: 水平 (0±30°) 和 垂直 (90±30°)
        h_lines = []  # 水平线: rho, theta
        v_lines = []  # 垂直线

        for line in lines:
            rho, theta = line[0]
            deg = np.degrees(theta)
            if deg < 30 or deg > 150:
                v_lines.append((rho, theta))
            elif 60 < deg < 120:
                h_lines.append((rho, theta))

        if len(h_lines) < 2 or len(v_lines) < 2:
            return gray

        # 取最外边的线
        h_lines.sort(key=lambda x: x[0])
        v_lines.sort(key=lambda x: x[0])

        top_line = h_lines[0]
        bottom_line = h_lines[-1]
        left_line = v_lines[0]
        right_line = v_lines[-1]

        # 计算四个交点
        corners = self._line_intersection_perspective(
            top_line, left_line,
            top_line, right_line,
            bottom_line, left_line,
            bottom_line, right_line
        )

        if corners is None or len(corners) != 4:
            return gray

        # 排序: 左上, 右上, 右下, 左下
        corners = self._order_corners(corners)

        # 目标矩形
        dst_width = max(int(np.linalg.norm(corners[1] - corners[0])),
                        int(np.linalg.norm(corners[2] - corners[3])))
        dst_height = max(int(np.linalg.norm(corners[3] - corners[0])),
                         int(np.linalg.norm(corners[2] - corners[1])))

        if dst_width < 10 or dst_height < 10:
            return gray

        dst_pts = np.array([
            [0, 0],
            [dst_width - 1, 0],
            [dst_width - 1, dst_height - 1],
            [0, dst_height - 1]
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(
            corners.astype(np.float32), dst_pts
        )
        return cv2.warpPerspective(gray, matrix, (dst_width, dst_height))

    def _line_intersection_perspective(self, *lines) -> Optional[np.ndarray]:
        """计算四条直线两两相交的四个交点"""
        corners = []
        line_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
        line_list = list(lines)

        for i, j in line_pairs:
            rho1, theta1 = line_list[i]
            rho2, theta2 = line_list[j]

            A = np.array([
                [np.cos(theta1), np.sin(theta1)],
                [np.cos(theta2), np.sin(theta2)]
            ])
            b = np.array([rho1, rho2])

            try:
                pt = np.linalg.solve(A, b)
                corners.append(pt)
            except np.linalg.LinAlgError:
                return None

        return np.array(corners)

    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """将四个角点排序为 左上, 右上, 右下, 左下"""
        # 按x排序
        sorted_x = corners[np.argsort(corners[:, 0])]
        left = sorted_x[:2]
        right = sorted_x[2:]

        # 左列按y排序
        left = left[np.argsort(left[:, 1])]
        # 右列按y排序
        right = right[np.argsort(right[:, 1])]

        return np.array([left[0], right[0], right[1], left[1]])

    # ============================================================
    # 2. 光照均衡化 (第3.1.1节)
    # ============================================================

    def _illumination_equalization(self, gray: np.ndarray) -> np.ndarray:
        """
        自适应光照均衡化

        使用CLAHE + 背景估计，消除纸张阴影、明暗不均。
        """
        # CLAHE 限制对比度自适应直方图均衡
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)

        # 使用高斯模糊估计背景光照
        large_kernel = max(31, min(gray.shape) // 8)
        if large_kernel % 2 == 0:
            large_kernel += 1
        background = cv2.GaussianBlur(gray, (large_kernel, large_kernel), 0)

        # 背景归一化
        bg_mean = np.mean(background)
        if bg_mean > 0:
            normalized = np.clip(equalized * (bg_mean / (background + 1)), 0, 255).astype(np.uint8)
        else:
            normalized = equalized

        return normalized

    # ============================================================
    # 3. 自适应二值化
    # ============================================================

    def _adaptive_binarization(self, gray: np.ndarray) -> np.ndarray:
        """
        自适应阈值二值化

        使用Otsu + 自适应阈值组合。
        """
        # 高斯滤波去噪
        kernel_size = self.th.gaussian_kernel_size
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

        # Otsu全局阈值
        _, otsu_binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # 自适应局部阈值
        block_size = self.th.adaptive_block_size
        if block_size % 2 == 0:
            block_size += 1
        adaptive_binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, self.th.adaptive_c
        )

        # 融合: 取两者交集（保留更可靠的线条）
        binary = cv2.bitwise_and(otsu_binary, adaptive_binary)

        # 形态学开运算去噪点
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        return binary

    # ============================================================
    # 4. 文字实例分割 (第3.1.2节)
    # ============================================================

    def _text_instance_segmentation(self, binary: np.ndarray, gray: np.ndarray) -> np.ndarray:
        """
        轻量文字实例分割

        基于连通域分析的文字笔画检测，区分几何线条与手写文字。
        不使用深度模型，采用形态学+几何特征区分。

        Returns:
            text_mask: 文字区域掩码 (255=文字)
        """
        h, w = binary.shape

        # 连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        text_mask = np.zeros_like(binary)

        # 文字区域判定特征:
        # 1. 面积较小 (不是主要几何线条)
        # 2. 宽高比接近字符比例
        # 3. 紧致度较高
        # 4. 笔画密度模式
        total_area = np.sum(binary > 0)
        if total_area == 0:
            return text_mask

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            x, y, bw, bh = (stats[i, cv2.CC_STAT_LEFT],
                            stats[i, cv2.CC_STAT_TOP],
                            stats[i, cv2.CC_STAT_WIDTH],
                            stats[i, cv2.CC_STAT_HEIGHT])

            if area < 10 or area > total_area * 0.3:
                continue

            # 宽高比
            aspect_ratio = max(bw, bh) / max(min(bw, bh), 1)
            # 填充率
            fill_ratio = area / max(bw * bh, 1)
            # 相对面积
            relative_area = area / max(total_area, 1)

            # 文字区域特征:
            # - 宽高比在1.0~3.0之间（字符形状）
            # - 填充率适中（笔画结构）
            # - 面积较小（相对于整图）
            is_text = (
                1.0 < aspect_ratio < 3.0 and
                0.15 < fill_ratio < 0.65 and
                relative_area < 0.08
            )

            if is_text:
                # 获取该连通域的轮廓
                component_mask = (labels == i).astype(np.uint8) * 255
                # 略膨胀以覆盖文字区域内部
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                component_mask = cv2.dilate(component_mask, kernel, iterations=1)
                text_mask = cv2.bitwise_or(text_mask, component_mask)

        return text_mask

    def _apply_text_mask(self, binary: np.ndarray, text_mask: np.ndarray) -> np.ndarray:
        """
        应用文字掩膜

        仅屏蔽文字内部填充区域，保留紧贴顶点的字母标注。
        """
        if np.sum(text_mask) == 0:
            return binary

        # 对文字区域进行腐蚀，保留边缘（字母轮廓可能也是几何的一部分）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        text_eroded = cv2.erode(text_mask, kernel, iterations=1)

        # 从二值图中移除文字内部区域
        cleaned = binary.copy()
        cleaned[text_eroded > 0] = 0

        return cleaned

    # ============================================================
    # 5. 自适应形态学修复 (第3.1.3节)
    # ============================================================

    def _morphological_repair(self, binary: np.ndarray) -> np.ndarray:
        """
        自适应分层形态学断线修复

        1. 微小毛刺: 小尺度腐蚀去除
        2. 线段缺口: 分段膨胀连通
        """
        # 根据线条宽度自适应核大小
        base_size = self.th.morph_kernel_size
        line_width = self.line_width

        # 腐蚀去毛刺
        erode_size = max(3, int(base_size * 0.5))
        if erode_size % 2 == 0:
            erode_size += 1
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))

        # 膨胀连通
        dilate_size = max(3, int(base_size * 1.2))
        if dilate_size % 2 == 0:
            dilate_size += 1
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))

        # 先腐蚀去毛刺
        repaired = cv2.erode(binary, erode_kernel, iterations=self.th.erode_iterations)

        # 再膨胀连通断线
        for i in range(self.th.max_dilate_iterations):
            repaired = cv2.dilate(repaired, dilate_kernel, iterations=1)
            # 检查膨胀效果
            # 如果膨胀过度导致大面积粘连，提前终止
            if np.sum(repaired > 0) > np.sum(binary > 0) * 1.5:
                break

        # 恢复原大小（可选）
        # 使用闭运算保持线条宽度
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        repaired = cv2.morphologyEx(repaired, cv2.MORPH_CLOSE, close_kernel)

        return repaired

    # ============================================================
    # 6. 骨架化 (第3.1.4节)
    # ============================================================

    def _skeletonize(self, binary: np.ndarray) -> np.ndarray:
        """
        形态学骨架化

        生成单像素中心线。
        """
        from skimage.morphology import skeletonize

        binary_bool = binary > 0
        if np.sum(binary_bool) == 0:
            return binary

        skeleton_bool = skeletonize(binary_bool)
        skeleton = (skeleton_bool * 255).astype(np.uint8)

        return skeleton

    # ============================================================
    # 7. 骨架剪枝 (第3.1.4节)
    # ============================================================

    def _skeleton_pruning(self, skeleton: np.ndarray) -> np.ndarray:
        """
        骨架剪枝算法

        剔除长度低于阈值的转角、交叉处伪分叉骨架。
        """
        min_length = self.th.prune_length

        skeleton = skeleton.copy()
        h, w = skeleton.shape

        # 找到所有端点 (8邻域中只有1个邻域点)
        kernel = np.ones((3, 3), np.uint8)
        neighbor_count = cv2.filter2D(
            (skeleton > 0).astype(np.uint8), -1, kernel
        )
        neighbor_count[skeleton == 0] = 0
        # 端点: 8邻域中恰好有2个前景点（自身+1个邻居）
        endpoints = (neighbor_count == 2)

        # 找到所有分支点 (8邻域中有>=4个前景点，即>=3个邻居)
        branch_points = (neighbor_count >= 4)

        # 从端点出发，跟踪分支直到遇到分支点或长度达标
        endpoints_indices = np.argwhere(endpoints)

        for ep_y, ep_x in endpoints_indices:
            # BFS跟踪分支
            branch_pixels = []
            visited = set()
            queue = [(ep_y, ep_x)]
            visited.add((ep_y, ep_x))

            while queue and len(branch_pixels) < min_length:
                cy, cx = queue.pop(0)
                branch_pixels.append((cy, cx))

                # 检查是否到达分支点
                if branch_points[cy, cx]:
                    break

                # 检查3x3邻域
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if skeleton[ny, nx] > 0 and (ny, nx) not in visited:
                                visited.add((ny, nx))
                                queue.append((ny, nx))

            # 如果分支长度小于阈值，剪枝
            if len(branch_pixels) < min_length:
                for py, px in branch_pixels:
                    # 保留分支点
                    if not branch_points[py, px]:
                        skeleton[py, px] = 0

        return skeleton