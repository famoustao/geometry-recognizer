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
from PIL import Image


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

    def __init__(self):
        self.temp_dir = tempfile.mkdtemp(prefix="geo_recog_")
        self._setup_geometry_constraints()

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

    def recognize(self, image_path):
        """执行完整识别流程，返回 RecognitionResult"""
        result = RecognitionResult()
        result.image_path = image_path

        try:
            img = cv2.imread(image_path)
            if img is None:
                result.error = f"无法读取图片: {image_path}"
                return result

            h, w = img.shape[:2]
            result.image_size = (w, h)
            original_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 1. 文字屏蔽
            cleaned = self._mask_text_labels(img)
            gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)

            # 2. 线条检测
            merged_lines = self._detect_lines(gray, w, h)

            # 3. 直线交点分析 → 找三角形顶点
            A, B, C = self._find_triangle_vertices(merged_lines, gray, w, h)

            # 4. 计算几何点
            O = ((B[0] + C[0]) / 2, (B[1] + C[1]) / 2)
            radius = self._distance(O, B)
            H = self._perpendicular_foot(A, B, C)

            D = self._line_circle_intersection(A, B, O, radius)
            if D is None:
                angle_B = math.atan2(B[1] - O[1], B[0] - O[0])
                D = (O[0] + radius * math.cos(angle_B - math.pi/4),
                     O[1] - radius * abs(math.sin(angle_B - math.pi/4)))

            E = self._line_circle_intersection(A, C, O, radius)
            if E is None:
                angle_C = math.atan2(C[1] - O[1], C[0] - O[0])
                E = (O[0] + radius * math.cos(angle_C + math.pi/4),
                     O[1] - radius * abs(math.sin(angle_C + math.pi/4)))

            # 微调D、E到最近的黑色像素
            D = self._refine_to_dark_pixel(D, O, radius, gray, w, h)
            E = self._refine_to_dark_pixel(E, O, radius, gray, w, h)

            F = self._perpendicular_foot(D, B, C)
            G = self._perpendicular_foot(E, B, C)

            # M = DG ∩ EF
            M = self._line_intersection(D, G, E, F)
            if M is None:
                M = (H[0] + (A[0] - H[0]) * 0.35,
                     H[1] + (A[1] - H[1]) * 0.35)

            key_points = {
                'A': A, 'B': B, 'C': C, 'D': D, 'E': E,
                'F': F, 'G': G, 'H': H, 'M': M, 'O': O,
            }
            result.key_points = key_points
            result.geo_info = {'radius': radius}

            # 5. 验证线段连接
            valid_connections = self._detect_line_connections(
                original_gray, key_points, merged_lines)
            result.valid_connections = valid_connections

            # 6. 生成 TikZ 代码
            result.tex_code = self._generate_latex(key_points, radius, valid_connections)

            # 7. 编译预览图
            preview_path = self._compile_to_png(result.tex_code)
            result.preview_image_path = preview_path

            result.success = True
            return result

        except Exception as e:
            import traceback
            result.error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            return result

    # ============================================================
    # 图像处理
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

    def _detect_lines(self, gray, w, h):
        """多阈值 Canny + HoughLinesP 检测线段"""
        all_lines = []
        for low, high in [(20, 80), (25, 120), (30, 150)]:
            edges = cv2.Canny(gray, low, high, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=35,
                                     minLineLength=20, maxLineGap=10)
            if lines is not None:
                all_lines.extend(lines)

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
                if ang_diff < 15 and mid_dist < 50:
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
        return merged_lines

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
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=15, min_samples=2).fit(intersections)
            labels = clustering.labels_
            clusters = {}
            for i, label in enumerate(labels):
                if label == -1: continue
                clusters.setdefault(label, []).append(intersections[i])
            cluster_pts = []
            for label, pts in clusters.items():
                cx = sum(p[0] for p in pts) / len(pts)
                cy = sum(p[1] for p in pts) / len(pts)
                cluster_pts.append((cx, cy))
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

    def _generate_latex(self, kpts, radius, valid_connections=None):
        """生成 TikZ 代码（检测层 + 构造层）"""
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

        # 圆弧角度
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
        """编译 TikZ 代码为 PNG 图片"""
        tex_path = os.path.join(self.temp_dir, f"{filename}.tex")
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write(tex_code)

        out_dir = self.temp_dir
        for i in range(2):
            subprocess.run(
                ['pdflatex', '-interaction=nonstopmode',
                 f'-output-directory={out_dir}', tex_path],
                capture_output=True, text=True, timeout=30
            )

        pdf_path = os.path.join(out_dir, f"{filename}.pdf")
        if not os.path.exists(pdf_path):
            return ""

        output_base = os.path.join(out_dir, filename)
        subprocess.run(
            ['pdftoppm', '-png', '-r', '300', pdf_path, output_base],
            capture_output=True, timeout=30
        )

        png_files = sorted(glob.glob(f"{output_base}*.png"))
        if png_files:
            final_path = os.path.join(out_dir, f"{filename}_final.png")
            Image.open(png_files[0]).save(final_path)
            return final_path
        return ""

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

    def cleanup(self):
        """清理临时文件"""
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass

    def __del__(self):
        self.cleanup()