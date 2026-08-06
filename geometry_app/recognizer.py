"""
几何图形识别引擎（重构自 run_uploaded_v4.py）
提供可复用的识别 API，封装图像处理、直线检测、几何推理、TikZ生成
"""
import cv2
import numpy as np
import math
import os
import subprocess
import glob
import gc
import tempfile
import shutil
import traceback
from PIL import Image

from .logger import logger, log_exception


def safe_imread(image_path):
    """
    安全读取图片（支持中文路径、特殊字符路径）
    OpenCV 的 cv2.imread 不支持非 ASCII 路径，用 np.fromfile 绕过
    """
    logger.info(f"尝试读取图片: {image_path}")
    logger.info(f"  文件是否存在: {os.path.exists(image_path)}")
    logger.info(f"  文件大小: {os.path.getsize(image_path) if os.path.exists(image_path) else 'N/A'} 字节")
    logger.info(f"  文件路径编码: {type(image_path)}")

    if not os.path.exists(image_path):
        logger.error(f"  文件不存在: {image_path}")
        return None

    try:
        # 方法1: 用 np.fromfile 绕过路径编码问题（兼容中文路径）
        logger.info("  尝试方法1: np.fromfile + cv2.imdecode")
        file_bytes = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is not None:
            logger.info(f"  方法1成功: {img.shape[1]}x{img.shape[0]}")
            return img
        logger.warning("  方法1失败")
    except Exception as e:
        logger.warning(f"  方法1异常: {e}")

    try:
        # 方法2: 直接 cv2.imread
        logger.info("  尝试方法2: cv2.imread")
        img = cv2.imread(image_path)
        if img is not None:
            logger.info(f"  方法2成功: {img.shape[1]}x{img.shape[0]}")
            return img
        logger.warning("  方法2失败")
    except Exception as e:
        logger.warning(f"  方法2异常: {e}")

    try:
        # 方法3: 用 PIL 读取再转 OpenCV
        logger.info("  尝试方法3: PIL.Image + np.array")
        pil_img = Image.open(image_path)
        pil_img = pil_img.convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        if img is not None:
            logger.info(f"  方法3成功: {img.shape[1]}x{img.shape[0]}")
            return img
    except Exception as e:
        logger.warning(f"  方法3异常: {e}")

    logger.error(f"  所有方法均失败，无法读取图片: {image_path}")
    return None


class RecognitionResult:
    """识别结果数据类"""
    def __init__(self):
        self.key_points = {}       # {name: (x, y)} 像素坐标
        self.valid_connections = []  # [(p1, p2, confidence)]
        self.geo_info = {}         # {radius, ...}
        self.tex_code = ""         # 生成的 TikZ 代码
        self.preview_image_path = ""  # 编译后的预览图路径
        self.success = False
        self.error = ""
        self.image_path = ""
        self.image_size = (0, 0)
        self.debug_images = {}     # {name: path}

    def get_tex_code(self):
        return self.tex_code


class GeometryRecognizer:
    """几何图形识别引擎"""

    def __init__(self, circle_pixel_tolerance=2, circle_hit_threshold=0.50):
        self.temp_dir = tempfile.mkdtemp(prefix="geo_recog_")
        self._setup_geometry_constraints()
        self.circle_pixel_tolerance = circle_pixel_tolerance  # 像素搜索半径 ±N px
        self.circle_hit_threshold = circle_hit_threshold      # 命中率阈值 0.0~1.0
        logger.info(f"GeometryRecognizer 初始化完成，临时目录: {self.temp_dir}")
        logger.info(f"  圆形检测参数: 像素容差={self.circle_pixel_tolerance}px, 命中率阈值={self.circle_hit_threshold:.2f}")

    def _setup_geometry_constraints(self):
        """几何约束表：每个点只连接几何定义中的相邻点"""
        self.geometry_constraints = {
            'A': {'B', 'C', 'H'},
            'B': {'A', 'C', 'D'},
            'C': {'A', 'B', 'E'},
            'D': {'B', 'F', 'G', 'M'},
            'E': {'C', 'F', 'G', 'M'},
            'F': {'D', 'E', 'G', 'M'},
            'G': {'E', 'D', 'F', 'M'},
            'H': {'A'},
            'M': {'D', 'G', 'E', 'F'},
        }

    def recognize(self, image_path, circle_pixel_tolerance=None, circle_hit_threshold=None):
        """执行完整识别流程，返回 RecognitionResult
        circle_pixel_tolerance: 覆盖实例的圆形像素搜索半径（可选）
        circle_hit_threshold: 覆盖实例的圆形命中率阈值（可选）
        """
        result = RecognitionResult()
        result.image_path = image_path

        try:
            logger.info(f"{'='*50}")
            logger.info(f"开始识别: {image_path}")

            # 可选参数覆盖实例变量
            if circle_pixel_tolerance is not None:
                self.circle_pixel_tolerance = circle_pixel_tolerance
            if circle_hit_threshold is not None:
                self.circle_hit_threshold = circle_hit_threshold

            img = safe_imread(image_path)
            if img is None:
                err_msg = f"无法读取图片: {image_path}"
                logger.error(err_msg)
                result.error = err_msg
                return result

            h, w = img.shape[:2]
            result.image_size = (w, h)
            logger.info(f"图片尺寸: {w}x{h}, 通道数: {img.shape[2]}")

            original_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 0. 预处理：文字屏蔽 → 二值化 → 骨架化
            logger.info("步骤0/8: 图像预处理（文字屏蔽 + 二值化 + 骨架化）...")
            cleaned, gray, binary, skeleton = self._preprocess(img)
            logger.info("  预处理完成")

            # 保存调试图
            debug_dir = os.path.join(self.temp_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, "01_binary.png"), binary)
            cv2.imwrite(os.path.join(debug_dir, "02_skeleton.png"), skeleton)
            result.debug_images['binary'] = os.path.join(debug_dir, "01_binary.png")
            result.debug_images['skeleton'] = os.path.join(debug_dir, "02_skeleton.png")

            # 1. 线条检测（使用骨架图增强）
            logger.info("步骤1/8: 线条检测（Canny + 骨架）...")
            merged_lines = self._detect_lines(gray, skeleton, w, h)
            logger.info(f"  检测到 {len(merged_lines)} 条线段（去重后）")

            # 2. 圆形检测（新增）
            logger.info("步骤2/8: 圆形检测...")
            detected_circles = self._detect_circles(gray, binary, skeleton, w, h)
            if detected_circles:
                logger.info(f"  检测到 {len(detected_circles)} 个圆形")
                result.geo_info['circles'] = detected_circles
            else:
                logger.info("  未检测到完整圆形")

            # 3. 直线交点分析 → 找三角形顶点
            logger.info("步骤3/8: 直线交点分析...")
            A, B, C = self._find_triangle_vertices(merged_lines, gray, w, h)
            logger.info(f"  A({A[0]:.0f},{A[1]:.0f}) B({B[0]:.0f},{B[1]:.0f}) C({C[0]:.0f},{C[1]:.0f})")

            # 4. 计算几何点
            logger.info("步骤4/8: 计算几何点...")
            O = ((B[0] + C[0]) / 2, (B[1] + C[1]) / 2)
            radius = self._distance(O, B)
            H = self._perpendicular_foot(A, B, C)
            logger.info(f"  O=({O[0]:.0f},{O[1]:.0f}) R={radius:.0f} H=({H[0]:.0f},{H[1]:.0f})")

            D = self._line_circle_intersection(A, B, O, radius)
            if D is None:
                angle_B = math.atan2(B[1] - O[1], B[0] - O[0])
                D = (O[0] + radius * math.cos(angle_B - math.pi/4),
                     O[1] - radius * abs(math.sin(angle_B - math.pi/4)))
                logger.info("  D 使用备选计算")

            E = self._line_circle_intersection(A, C, O, radius)
            if E is None:
                angle_C = math.atan2(C[1] - O[1], C[0] - O[0])
                E = (O[0] + radius * math.cos(angle_C + math.pi/4),
                     O[1] - radius * abs(math.sin(angle_C + math.pi/4)))
                logger.info("  E 使用备选计算")

            # 微调D、E到最近的黑色像素
            D = self._refine_to_dark_pixel(D, O, radius, gray, w, h)
            E = self._refine_to_dark_pixel(E, O, radius, gray, w, h)
            logger.info(f"  D=({D[0]:.0f},{D[1]:.0f}) E=({E[0]:.0f},{E[1]:.0f})")

            F = self._perpendicular_foot(D, B, C)
            G = self._perpendicular_foot(E, B, C)

            # M = DG ∩ EF
            M = self._line_intersection(D, G, E, F)
            if M is None:
                M = (H[0] + (A[0] - H[0]) * 0.35,
                     H[1] + (A[1] - H[1]) * 0.35)
                logger.info("  M 使用备选计算（DG∥EF）")

            logger.info(f"  F=({F[0]:.0f},{F[1]:.0f}) G=({G[0]:.0f},{G[1]:.0f}) M=({M[0]:.0f},{M[1]:.0f})")

            key_points = {
                'A': A, 'B': B, 'C': C, 'D': D, 'E': E,
                'F': F, 'G': G, 'H': H, 'M': M, 'O': O,
            }
            result.key_points = key_points
            if 'radius' not in result.geo_info:
                result.geo_info['radius'] = radius

            # 5. 验证线段连接
            logger.info("步骤5/8: 验证线段连接...")
            valid_connections = self._detect_line_connections(
                original_gray, key_points, merged_lines)
            result.valid_connections = valid_connections
            logger.info(f"  验证通过: {len(valid_connections)} 条")
            for p1, p2, c in valid_connections:
                logger.info(f"    {p1}-{p2}: {c:.0f}%")

            # 6. 生成 TikZ 代码
            logger.info("步骤6/8: 生成 TikZ 代码...")
            result.tex_code = self._generate_latex(key_points, radius, valid_connections,
                                                    detected_circles)
            tex_lines = result.tex_code.count('\n') + 1
            logger.info(f"  TikZ 代码生成完成: {tex_lines} 行")

            # 7. 编译预览图
            logger.info("步骤7/8: 编译 LaTeX 预览图...")
            preview_path = self._compile_to_png(result.tex_code)
            result.preview_image_path = preview_path
            if preview_path and os.path.exists(preview_path):
                logger.info(f"  预览图生成成功: {preview_path}")
            else:
                logger.warning("  预览图生成失败（可能缺少 LaTeX 环境）")

            # 8. 保存调试图
            logger.info("步骤8/8: 保存调试图...")
            self._draw_debug_image(img, merged_lines, key_points, detected_circles, debug_dir)

            result.success = True
            logger.info(f"{'='*50}")
            logger.info(f"识别完成: {image_path}")
            return result

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"识别过程异常: {type(e).__name__}: {str(e)}")
            logger.error(tb)
            result.error = f"{type(e).__name__}: {str(e)}\n{tb}"
            return result

    # ============================================================
    # 图像预处理：二值化 + 骨架化
    # ============================================================

    def _binarize(self, gray):
        """
        二值化：Otsu 自适应阈值 + 形态学修复
        输入灰度图，输出 0/255 二值图（线条为白色）
        """
        # Otsu 自适应阈值
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 形态学闭运算修复断线
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        logger.info(f"  二值化完成: 前景像素 {cv2.countNonZero(binary)} 个")
        return binary

    def _skeletonize(self, binary):
        """
        骨架化：形态学骨架提取（Zhang-Suen 风格细化）
        输入 0/255 二值图，输出单像素宽度的骨架图
        """
        # 归一化到 0/1
        skel = binary.copy()
        skel[skel > 0] = 1

        # 用 erode → dilate 差值逐步提取骨架
        skeleton = np.zeros_like(skel)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        prev_count = cv2.countNonZero(skel)
        while True:
            eroded = cv2.erode(skel, kernel, iterations=1)
            if cv2.countNonZero(eroded) == 0:
                # 最后一次像素：直接加入骨架
                skeleton = cv2.bitwise_or(skeleton, skel)
                break
            dilated = cv2.dilate(eroded, kernel, iterations=1)
            subset = cv2.subtract(skel, dilated)
            skeleton = cv2.bitwise_or(skeleton, subset)
            skel = eroded.copy()

            cur_count = cv2.countNonZero(skel)
            if cur_count == 0 or cur_count == prev_count:
                break
            prev_count = cur_count

        skeleton_255 = (skeleton * 255).astype(np.uint8)
        logger.info(f"  骨架化完成: 骨架像素 {cv2.countNonZero(skeleton_255)} 个")
        return skeleton_255

    def _preprocess(self, img):
        """
        完整预处理管线：文字屏蔽 → 二值化 → 骨架化
        返回 (cleaned, gray, binary, skeleton)
        """
        # 1. 文字屏蔽
        cleaned = self._mask_text_labels(img)
        gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)

        # 2. 二值化
        binary = self._binarize(gray)

        # 3. 骨架化
        skeleton = self._skeletonize(binary)

        return cleaned, gray, binary, skeleton

    # ============================================================
    # 图像处理（原文字屏蔽）
    # ============================================================

    def _mask_text_labels(self, img):
        """文字屏蔽：腐蚀断开连接 → 连通域分析 → 面积+宽高比过滤"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(binary, kernel, iterations=1)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(eroded, 8)
        text_mask = np.zeros_like(gray)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            aspect_ratio = max(bw, bh) / max(min(bw, bh), 1)
            if (20 < area < 500 and aspect_ratio < 4.0 and bw < 40 and bh < 40):
                text_mask[labels == i] = 255
        text_mask = cv2.dilate(text_mask, kernel, iterations=2)
        cleaned = img.copy()
        cleaned[text_mask > 0] = [255, 255, 255]
        return cleaned

    # ============================================================
    # 线段检测（增强版）
    # ============================================================

    def _detect_lines(self, gray, skeleton, w, h):
        """
        多策略线段检测：
          - 策略1: Canny 边缘 + HoughLinesP（原始灰度图）
          - 策略2: 骨架图直接 HoughLinesP（更精确）
        """
        all_lines = []

        # 策略1: Canny 边缘检测
        for low, high in [(20, 80), (25, 120), (30, 150)]:
            edges = cv2.Canny(gray, low, high, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=35,
                                     minLineLength=20, maxLineGap=10)
            if lines is not None:
                all_lines.extend(lines)

        # 策略2: 骨架图直接检测（线条已经是单像素宽，更精确）
        if skeleton is not None and cv2.countNonZero(skeleton) > 50:
            lines = cv2.HoughLinesP(skeleton, 1, np.pi/180, threshold=20,
                                     minLineLength=15, maxLineGap=5)
            if lines is not None:
                all_lines.extend(lines)

        # 合并去重（同方向 + 邻近的线段合并）
        merged_lines = []
        for line in all_lines:
            line = np.array(line).flatten()
            x1, y1, x2, y2 = line.astype(float)
            angle = math.degrees(math.atan2(y2-y1, x2-x1)) % 180
            mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
            length = self._distance((x1, y1), (x2, y2))
            found = False
            for i, ml in enumerate(merged_lines):
                ang_diff = min(abs(angle - ml['angle']), 180 - abs(angle - ml['angle']))
                mid_dist = self._distance((mid_x, mid_y), ml['mid'])
                if ang_diff < 12 and mid_dist < 40:
                    if length > ml['length']:
                        merged_lines[i] = {'p1': (x1, y1), 'p2': (x2, y2),
                                           'angle': angle, 'mid': (mid_x, mid_y),
                                           'length': length}
                    found = True
                    break
            if not found:
                merged_lines.append({'p1': (x1, y1), 'p2': (x2, y2),
                                     'angle': angle, 'mid': (mid_x, mid_y),
                                     'length': length})

        # 按长度排序，保留最长的前 30 条
        merged_lines.sort(key=lambda x: -x['length'])
        merged_lines = merged_lines[:30]

        return merged_lines

    # ============================================================
    # 圆形检测（新增）
    # ============================================================

    def _detect_circles(self, gray, binary, skeleton, w, h):
        """
        使用 HoughCircles 检测圆形
        返回 [(cx, cy, r), ...]
        """
        # 对灰度图进行高斯模糊减少噪声
        blurred = cv2.GaussianBlur(gray, (7, 7), 2.0)

        # 多组参数尝试检测
        raw_candidates = []
        params_list = [
            (1.2, 60, 80, 30),   # dp, minDist, param1, param2
            (1.0, 50, 60, 25),
            (1.5, 70, 100, 35),
        ]

        for dp, minDist, param1, param2 in params_list:
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT,
                dp=dp, minDist=minDist,
                param1=param1, param2=param2,
                minRadius=max(15, min(w, h) // 25),
                maxRadius=min(w, h) // 3
            )
            if circles is not None:
                circles = np.round(circles[0]).astype("int")
                for (cx, cy, r) in circles:
                    raw_candidates.append((cx, cy, r))

        if not raw_candidates:
            return []

        # 非极大值抑制（NMS）去重
        raw_candidates.sort(key=lambda x: -x[2])  # 按半径降序
        nms_results = []
        for (cx, cy, r) in raw_candidates:
            is_dup = False
            for (ex_cx, ex_cy, ex_r) in nms_results:
                center_dist = math.sqrt((cx - ex_cx)**2 + (cy - ex_cy)**2)
                radius_diff = abs(r - ex_r)
                if center_dist < 40 and radius_diff < 20:
                    is_dup = True
                    break
                # 同心圆去重
                if center_dist < 20 and radius_diff < 50:
                    is_dup = True
                    break
            if not is_dup:
                nms_results.append((cx, cy, r))

        # 验证：检查圆弧上是否有足够多的黑色像素
        detected = []
        for (cx, cy, r) in nms_results:
            verified = self._verify_circle(binary, skeleton, cx, cy, r)
            if verified:
                detected.append((cx, cy, r))
                logger.info(f"  检测到圆形: 中心({cx},{cy}) 半径{r}")

        # 最多保留 5 个最可信的圆
        if len(detected) > 5:
            detected.sort(key=lambda x: -x[2])
            detected = detected[:5]

        return detected

    def _verify_circle(self, binary, skeleton, cx, cy, r):
        """验证圆形检测结果：沿圆弧采样，检查骨架像素命中率"""
        h, w = binary.shape
        num_samples = max(48, int(r * 0.8))
        hit_count = 0
        tol = self.circle_pixel_tolerance  # 用户可调的像素搜索半径

        for i in range(num_samples):
            theta = 2 * math.pi * i / num_samples
            x = int(cx + r * math.cos(theta))
            y = int(cy + r * math.sin(theta))

            # 在骨架图上检查半径 ±tol px 范围内是否有像素
            found = False
            for dr in range(-tol, tol + 1):
                rx = int(cx + (r + dr) * math.cos(theta))
                ry = int(cy + (r + dr) * math.sin(theta))
                if 0 <= rx < w and 0 <= ry < h:
                    # 同时检查骨架和二值图
                    if skeleton[ry, rx] > 0 or binary[ry, rx] > 0:
                        hit_count += 1
                        found = True
                        break
                if found:
                    break

        if num_samples == 0:
            return False

        hit_ratio = hit_count / num_samples
        logger.debug(f"    圆形验证: 中心({cx},{cy}) r={r} 骨架命中率 {hit_ratio:.2f} (容差={tol}px, 阈值={self.circle_hit_threshold:.2f})")
        return hit_ratio > self.circle_hit_threshold

    def _cluster_points(self, points, eps=15, min_samples=2):
        """纯 numpy 密度聚类（替代 sklearn DBSCAN，消除 scipy 依赖）"""
        points = np.array(points)
        n = len(points)
        if n == 0:
            return []

        visited = [False] * n
        clusters = []

        for i in range(n):
            if visited[i]:
                continue

            # BFS 找 eps 邻域内的所有点
            cluster = [i]
            visited[i] = True
            queue = [i]

            while queue:
                current = queue.pop(0)
                for j in range(n):
                    if not visited[j]:
                        dist = np.sqrt(np.sum((points[current] - points[j]) ** 2))
                        if dist < eps:
                            visited[j] = True
                            cluster.append(j)
                            queue.append(j)

            if len(cluster) >= min_samples:
                clusters.append(cluster)

        # 计算每个聚类的中心
        centroids = []
        for cluster in clusters:
            centroid = np.mean(points[cluster], axis=0)
            centroids.append((float(centroid[0]), float(centroid[1])))

        return centroids

    def _find_triangle_vertices(self, merged_lines, gray, w, h):
        """从直线交点中找出三角形顶点 A(顶)、B(左下)、C(右下)"""
        intersections = []
        for i in range(len(merged_lines)):
            for j in range(i+1, len(merged_lines)):
                p1, p2 = merged_lines[i]['p1'], merged_lines[i]['p2']
                p3, p4 = merged_lines[j]['p1'], merged_lines[j]['p2']
                pt = self._line_intersection(p1, p2, p3, p4)
                if pt is not None and -50 < pt[0] < w+50 and -50 < pt[1] < h+50:
                    if self._point_on_segment(pt, p1, p2, 8) or self._point_on_segment(pt, p3, p4, 8):
                        intersections.append(pt)

        if len(intersections) > 10:
            cluster_pts = self._cluster_points(intersections, eps=15, min_samples=2)
        else:
            cluster_pts = intersections

        sorted_y = sorted(cluster_pts, key=lambda p: p[1])
        A = sorted_y[0]

        bottom_candidates = sorted(cluster_pts, key=lambda p: -p[1])[:10]
        bottom_candidates = [p for p in bottom_candidates if self._distance(p, A) > 80]
        B = min(bottom_candidates, key=lambda p: p[0])
        C = max(bottom_candidates, key=lambda p: p[0])

        return A, B, C

    # ============================================================
    # 线段验证
    # ============================================================

    def _detect_line_connections(self, gray_original, key_points, detected_lines):
        """通过几何约束表 + 连通性验证 + Hough匹配 验证线段连接"""
        h, w = gray_original.shape

        hough_segments = [(line['p1'], line['p2']) for line in detected_lines]

        def longest_dark_run(p1, p2, threshold=128):
            x1, y1 = p1; x2, y2 = p2
            length = int(max(abs(x2-x1), abs(y2-y1)))
            if length < 1: return 0, 0, 0
            max_run = 0; cur_run = 0; total_dark = 0; total_samples = 0
            for i in range(length + 1):
                t = i / length
                x = int(x1 + t * (x2 - x1))
                y = int(y1 + t * (y2 - y1))
                if 0 <= x < w and 0 <= y < h:
                    total_samples += 1
                    if gray_original[y, x] < threshold:
                        total_dark += 1; cur_run += 1
                        if cur_run > max_run: max_run = cur_run
                    else:
                        cur_run = 0
            return max_run, total_dark / max(total_samples, 1), total_samples

        def segment_on_hough_multi(p1, p2, num_samples=5, tolerance=8):
            seg_len = self._distance(p1, p2)
            if seg_len < 5: return False
            for hp1, hp2 in hough_segments:
                seg_angle = math.atan2(p2[1]-p1[1], p2[0]-p1[0])
                hough_angle = math.atan2(hp2[1]-hp1[1], hp2[0]-hp1[0])
                ang_diff = abs(seg_angle - hough_angle)
                ang_diff = min(ang_diff, 2*math.pi - ang_diff)
                if ang_diff > math.radians(30): continue
                close_count = 0
                for i in range(num_samples):
                    t = (i + 0.5) / num_samples
                    px = p1[0] + t * (p2[0] - p1[0])
                    py = p1[1] + t * (p2[1] - p1[1])
                    abx, aby = hp2[0]-hp1[0], hp2[1]-hp1[1]
                    apx, apy = px-hp1[0], py-hp1[1]
                    t_h = (apx*abx + apy*aby) / max(abx*abx + aby*aby, 1e-8)
                    if t_h < 0: fx, fy = hp1
                    elif t_h > 1: fx, fy = hp2
                    else: fx, fy = hp1[0]+t_h*abx, hp1[1]+t_h*aby
                    d = math.sqrt((px-fx)**2 + (py-fy)**2)
                    if d < tolerance: close_count += 1
                if close_count >= num_samples * 0.6: return True
            return False

        valid = []
        tested_pairs = set()
        for name1, neighbors in self.geometry_constraints.items():
            if name1 not in key_points: continue
            for name2 in neighbors:
                if name2 not in key_points: continue
                pair = frozenset([name1, name2])
                if pair in tested_pairs: continue
                tested_pairs.add(pair)
                p1, p2 = key_points[name1], key_points[name2]
                seg_len = self._distance(p1, p2)
                if seg_len < 10: continue
                max_run, pixel_ratio, _ = longest_dark_run(p1, p2)
                run_ratio = max_run / max(seg_len, 1)
                on_hough = segment_on_hough_multi(p1, p2)
                if run_ratio > 0.15 or on_hough or pixel_ratio > 0.40:
                    confidence = max(run_ratio * 100, pixel_ratio * 100, 70 if on_hough else 0)
                    valid.append((name1, name2, confidence))
        return valid

    # ============================================================
    # TikZ 生成
    # ============================================================

    def _generate_latex(self, kpts, radius, valid_connections=None, detected_circles=None):
        """生成 TikZ 代码（检测层 + 构造层 + 圆形）"""
        A = kpts['A']; B = kpts['B']; C = kpts['C']
        D = kpts['D']; E = kpts['E']; F = kpts['F']
        G = kpts['G']; H = kpts['H']; M = kpts['M']; O = kpts['O']

        all_pts = [A, B, C, D, E, F, G, H, M, O]
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        target_w, target_h = 7.0, 5.5
        scale = min(target_w / (max_x - min_x + 1) / 1.3,
                    target_h / (max_y - min_y + 1) / 1.3)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        offset_x = -center_x * scale
        offset_y = -center_y * scale

        def tx(px): return px * scale + offset_x
        def ty(py): return -(py * scale + offset_y)

        # 圆弧角度（以BC为直径的半圆）
        otx, oty = tx(O[0]), ty(O[1])
        btx, bty = tx(B[0]), ty(B[1])
        ctx, cty = tx(C[0]), ty(C[1])
        start_angle = math.degrees(math.atan2(bty - oty, btx - otx))
        end_angle = math.degrees(math.atan2(cty - oty, ctx - otx))
        arc_start = end_angle
        arc_end = start_angle
        if arc_end < arc_start: arc_end += 360
        tikz_r = radius * scale

        lines = []
        lines.append(r"\documentclass[tikz, border=10pt]{standalone}")
        lines.append(r"\usepackage{tikz}")
        lines.append(r"\usetikzlibrary{arrows}")
        lines.append("")
        lines.append(r"\begin{document}")
        lines.append(r"\begin{tikzpicture}[scale=1.0, >=stealth, line width=1.5pt]")
        lines.append("")

        # 坐标
        lines.append("    % === 顶点坐标 ===")
        for name in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'M']:
            p = kpts[name]
            lines.append(f"    \\coordinate ({name}) at ({tx(p[0]):.3f}, {ty(p[1]):.3f});")
        lines.append("")

        # 检测到的圆形（如果有）
        if detected_circles:
            lines.append("    % === 检测到的圆形 ===")
            for i, (cx, cy, r) in enumerate(detected_circles):
                tcx, tcy = tx(cx), ty(cy)
                tr = r * scale
                if tr > 0.1:  # 过滤太小的圆
                    lines.append(f"    \\draw[dashed, thick] ({tcx:.3f}, {tcy:.3f}) circle ({tr:.3f});")
            lines.append("")

        # 半圆
        lines.append("    % === 以BC为直径的半圆 ===")
        lines.append(f"    \\draw[dashed] (C) arc ({arc_start:.1f}:{arc_end:.1f}:{tikz_r:.3f});")
        lines.append("")

        # 检测层
        lines.append("    % === 检测层：图像验证的线段 ===")
        valid_set = set()
        for p1, p2, _ in (valid_connections or []):
            valid_set.add(frozenset([p1, p2]))
        tri_edges = [frozenset(['A','B']), frozenset(['B','C']), frozenset(['C','A'])]
        if all(e in valid_set for e in tri_edges):
            lines.append("    \\draw[thick] (A) -- (B) -- (C) -- cycle;")
            lines.append("")
        else:
            for p1, p2 in [('A','B'), ('B','C'), ('C','A')]:
                if frozenset([p1, p2]) in valid_set:
                    lines.append(f"    \\draw[thick] ({p1}) -- ({p2});")
            lines.append("")
        for p1, p2, _ in (valid_connections or []):
            if frozenset([p1, p2]) in tri_edges: continue
            lines.append(f"    \\draw ({p1}) -- ({p2});")
        lines.append("")

        # 构造层
        lines.append("    % === 构造层：几何定义自动补全的辅助线 ===")
        lines.append("    \\draw (D) -- (F);")
        lines.append("    \\draw (E) -- (G);")
        lines.append("    \\draw (D) -- (G);")
        lines.append("    \\draw (E) -- (F);")
        lines.append("")

        # 标注
        lines.append("    % === 顶点标注 ===")
        dirs = {'A': 'above', 'B': 'below left', 'C': 'below right',
                'D': 'above left', 'E': 'above right',
                'F': 'below', 'G': 'below', 'H': 'below', 'M': 'left'}
        for name, direction in dirs.items():
            lines.append(f"    \\node[{direction}] at ({name}) {{{name}}};")
        lines.append("")
        lines.append(r"\end{tikzpicture}")
        lines.append(r"\end{document}")
        return "\n".join(lines)

    # ============================================================
    # LaTeX 编译
    # ============================================================

    def _compile_to_png(self, tex_code, filename="output"):
        """编译 TikZ 代码为 PNG 图片（缺少 LaTeX 环境时优雅降级）"""
        # 检查 pdflatex 是否可用
        if not self._check_command("pdflatex"):
            logger.warning("LaTeX (pdflatex) 未安装，跳过 LaTeX 编译")
            logger.warning("请安装 texlive 或 MiKTeX 以启用 LaTeX 预览功能")
            return ""

        tex_path = os.path.join(self.temp_dir, f"{filename}.tex")
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write(tex_code)

        out_dir = self.temp_dir
        try:
            for i in range(2):
                subprocess.run(
                    ['pdflatex', '-interaction=nonstopmode',
                     f'-output-directory={out_dir}', tex_path],
                    capture_output=True, text=True, timeout=30
                )
        except Exception as e:
            logger.warning(f"pdflatex 编译失败: {e}")
            return ""

        pdf_path = os.path.join(out_dir, f"{filename}.pdf")
        if not os.path.exists(pdf_path):
            logger.warning("pdflatex 未生成 PDF 文件")
            return ""

        # 检查 pdftoppm 是否可用
        if not self._check_command("pdftoppm"):
            logger.warning("pdftoppm 未安装，跳过 PDF→PNG 转换")
            return ""

        output_base = os.path.join(out_dir, filename)
        try:
            subprocess.run(
                ['pdftoppm', '-png', '-r', '300', pdf_path, output_base],
                capture_output=True, timeout=30
            )
        except Exception as e:
            logger.warning(f"pdftoppm 转换失败: {e}")
            return ""

        png_files = sorted(glob.glob(f"{output_base}*.png"))
        if png_files:
            final_path = os.path.join(out_dir, f"{filename}_final.png")
            Image.open(png_files[0]).save(final_path)
            return final_path
        return ""

    def _check_command(self, cmd):
        """检查系统命令是否存在"""
        try:
            subprocess.run([cmd, '--version'], capture_output=True, timeout=5)
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            return False

    def compile_tex(self, tex_code, output_path=None):
        """外部接口：编译 TikZ 代码为 PNG"""
        return self._compile_to_png(tex_code)

    # ============================================================
    # 几何工具函数
    # ============================================================

    @staticmethod
    def _distance(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    @staticmethod
    def _line_intersection(p1, p2, p3, p4):
        x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-8: return None
        px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom
        return (px, py)

    @staticmethod
    def _point_on_segment(p, a, b, tol=5.0):
        ax, ay = a; bx, by = b; px, py = p
        abx, aby = bx-ax, by-ay
        apx, apy = px-ax, py-ay
        t = (apx*abx + apy*aby) / max(abx*abx + aby*aby, 1e-8)
        if t < 0: fx, fy = ax, ay
        elif t > 1: fx, fy = bx, by
        else: fx, fy = ax + t*abx, ay + t*aby
        return math.sqrt((px-fx)**2 + (py-fy)**2) < tol

    @staticmethod
    def _perpendicular_foot(point, line_p1, line_p2):
        px, py = point; ax, ay = line_p1; bx, by = line_p2
        abx, aby = bx-ax, by-ay
        apx, apy = px-ax, py-ay
        t = (apx*abx + apy*aby) / max(abx*abx + aby*aby, 1e-8)
        return (ax + t*abx, ay + t*aby)

    @staticmethod
    def _line_circle_intersection(p1, p2, center, R):
        v = (p2[0]-p1[0], p2[1]-p1[1])
        w = (p1[0]-center[0], p1[1]-center[1])
        a = v[0]**2 + v[1]**2
        b = 2*(w[0]*v[0] + w[1]*v[1])
        c = w[0]**2 + w[1]**2 - R**2
        disc = b*b - 4*a*c
        if disc < 0: return None
        t1 = (-b + math.sqrt(disc)) / (2*a)
        t2 = (-b - math.sqrt(disc)) / (2*a)
        if 0 <= t1 <= 1 and 0 <= t2 <= 1: t = max(t1, t2)
        elif 0 <= t1 <= 1: t = t1
        elif 0 <= t2 <= 1: t = t2
        else: return None
        return (p1[0] + t*v[0], p1[1] + t*v[1])

    @staticmethod
    def _refine_to_dark_pixel(pt, center, radius, gray, w, h):
        """微调点到最近的暗像素（圆弧上）"""
        best_pt, best_dist = pt, 50
        for dx in range(-15, 16):
            for dy in range(-15, 16):
                nx, ny = int(pt[0]+dx), int(pt[1]+dy)
                if 0 <= nx < w and 0 <= ny < h:
                    dist_to_O = abs(GeometryRecognizer._distance((nx, ny), center) - radius)
                    if dist_to_O < 5 and gray[ny, nx] < 128:
                        d = GeometryRecognizer._distance((nx, ny), pt)
                        if d < best_dist:
                            best_dist = d
                            best_pt = (float(nx), float(ny))
        return best_pt

    def _draw_debug_image(self, img, merged_lines, key_points, detected_circles, debug_dir):
        """绘制调试图：在原图上标注检测到的线段、顶点和圆形"""
        debug = img.copy()

        # 画检测到的线段
        for line in merged_lines:
            p1, p2 = line['p1'], line['p2']
            cv2.line(debug,
                     (int(p1[0]), int(p1[1])),
                     (int(p2[0]), int(p2[1])),
                     (0, 255, 0), 2)  # 绿色线段

        # 画检测到的圆形
        for (cx, cy, r) in detected_circles:
            cv2.circle(debug, (cx, cy), r, (255, 0, 0), 2)  # 蓝色圆形
            cv2.circle(debug, (cx, cy), 3, (255, 0, 0), -1)  # 圆心

        # 画关键点
        for name, pt in key_points.items():
            if name == 'O':
                continue
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(debug, (x, y), 5, (0, 0, 255), -1)  # 红色实心圆
            cv2.putText(debug, name, (x+8, y-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        debug_path = os.path.join(debug_dir, "03_debug_annotated.png")
        cv2.imwrite(debug_path, debug)
        logger.info(f"  调试图已保存: {debug_path}")

    def cleanup(self):
        """清理临时文件"""
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass

    def __del__(self):
        self.cleanup()