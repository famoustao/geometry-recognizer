"""
直线、圆弧图元识别模块 (第3.3节)

核心优化:
1. 分段自适应伸长率直曲粗分类
2. 圆弧多层识别升级 (长短自适应 + 椭圆拟合)
3. 直线三重校验机制
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from sklearn.cluster import DBSCAN
from .config import GeometryConfig
from .data_structures import (
    Vertex, Point, Primitive, PrimitiveType, LineType,
    GeometryLayer, ConfidenceSource
)
from .utils import (
    elongate_ratio, fit_line_ransac, fit_line_orthogonal,
    fit_circle_least_squares, fit_ellipse_least_squares,
    resample_contour, compute_curvature, line_angle_degrees,
    angle_between_degrees, point_line_distance,
    detect_line_pattern
)


class PrimitiveRecognizer:
    """
    图元识别器

    识别流程:
    骨架分段 → 伸长率直曲分类 → 曲线细分 → 直线拟合 → 直线校验
    → 圆弧识别 → 椭圆弧识别
    """

    def __init__(self, config: GeometryConfig):
        self.config = config
        self.th = config.thresholds
        self._primitive_counter: int = 0

    def _next_primitive_id(self, prefix: str = "P") -> str:
        """生成下一个图元ID"""
        self._primitive_counter += 1
        return f"{prefix}{self._primitive_counter}"

    def recognize(
        self,
        skeleton: np.ndarray,
        binary_image: np.ndarray,
        vertices: Dict[str, Vertex]
    ) -> List[Primitive]:
        """
        完整图元识别流水线

        Args:
            skeleton: 骨架图像
            binary_image: 二值化图像
            vertices: 顶点字典

        Returns:
            图元列表
        """
        # 1. 提取骨架路径
        paths = self._extract_skeleton_paths(skeleton)

        # 2. 路径分段
        segments = self._segment_paths(paths, vertices)

        # 3. 直曲粗分类
        primitives = []
        for seg in segments:
            primitives.extend(self._classify_segment(seg, vertices))

        # 4. 直线校验
        if self.config.enable_triple_line_verification:
            primitives = self._triple_line_verification(primitives)

        # 5. 圆弧合并与校验
        primitives = self._merge_arcs(primitives)
        primitives = self._arc_topology_validation(primitives, vertices)

        return primitives

    # ============================================================
    # 1. 骨架路径提取
    # ============================================================

    def _extract_skeleton_paths(self, skeleton: np.ndarray) -> List[np.ndarray]:
        """从骨架中提取所有连续路径"""
        skeleton_bool = skeleton > 0

        if np.sum(skeleton_bool) == 0:
            return []

        num_labels, labels = cv2.connectedComponents(
            skeleton_bool.astype(np.uint8), connectivity=8
        )

        paths = []
        for i in range(1, num_labels):
            ys, xs = np.where(labels == i)
            if len(ys) < 5:
                continue

            path = np.column_stack([xs, ys]).astype(np.float64)
            if self.config.enable_downsampling:
                step = self.th.skeleton_downsample_step
                path = path[::step]
            paths.append(path)

        return paths

    # ============================================================
    # 2. 路径分段 (第3.3.1节)
    # ============================================================

    def _segment_paths(
        self,
        paths: List[np.ndarray],
        vertices: Dict[str, Vertex]
    ) -> List[np.ndarray]:
        """
        路径分段

        在顶点位置将路径切分为子段，避免交叉重叠区域误判。
        """
        segments = []

        for path in paths:
            if len(path) < 5:
                continue

            # 找到路径上靠近顶点的位置
            split_indices = self._find_split_points(path, vertices)

            if not split_indices:
                segments.append(path)
                continue

            # 分段
            split_indices = sorted(set([0] + split_indices + [len(path) - 1]))
            for i in range(len(split_indices) - 1):
                start = split_indices[i]
                end = split_indices[i + 1] + 1
                if end - start >= 5:
                    segments.append(path[start:end])

        return segments

    def _find_split_points(
        self,
        path: np.ndarray,
        vertices: Dict[str, Vertex]
    ) -> List[int]:
        """找到路径上靠近顶点的分裂点"""
        if not vertices:
            return []

        split_indices = []
        vertex_positions = np.array([
            (v.position.x, v.position.y) for v in vertices.values()
        ])

        for i, pt in enumerate(path):
            if i % 5 != 0:  # 降低采样频率
                continue
            distances = np.linalg.norm(vertex_positions - pt, axis=1)
            if np.min(distances) < 3.0:
                split_indices.append(i)

        return split_indices

    # ============================================================
    # 3. 直曲粗分类 (第3.3.1节)
    # ============================================================

    def _classify_segment(
        self,
        segment: np.ndarray,
        vertices: Dict[str, Vertex]
    ) -> List[Primitive]:
        """
        分段自适应伸长率直曲粗分类

        将长骨架路径均等分段，逐段计算伸长率。
        多数分段达标即判定为直线。
        """
        if len(segment) < 5:
            return []

        # 将路径分成若干段
        seg_length = len(segment)
        num_sub_segs = max(2, seg_length // 20)
        sub_seg_size = seg_length // num_sub_segs

        straight_votes = 0
        curve_votes = 0

        for i in range(num_sub_segs):
            start = i * sub_seg_size
            end = min((i + 1) * sub_seg_size, seg_length)
            if end - start < 5:
                continue

            sub_seg = segment[start:end]
            er = elongate_ratio(sub_seg)
            threshold = self.th.get_elongation_threshold(
                np.linalg.norm(sub_seg[0] - sub_seg[-1])
            )

            if er >= threshold:
                straight_votes += 1
            else:
                curve_votes += 1

        # 多数分段达标即判定为直线
        if straight_votes >= curve_votes:
            return self._fit_line(segment, vertices)
        else:
            return self._classify_curve(segment, vertices)

    # ============================================================
    # 4. 直线拟合
    # ============================================================

    def _fit_line(
        self,
        segment: np.ndarray,
        vertices: Dict[str, Vertex]
    ) -> List[Primitive]:
        """RANSAC + 正交最小二乘稳健直线拟合"""
        if len(segment) < 5:
            return []

        # RANSAC拟合
        ransac_dist = self.th.ransac_distance
        line_params, inlier_mask = fit_line_ransac(segment, ransac_dist)

        if line_params is None:
            return []

        # 找到端点
        inlier_pts = segment[inlier_mask]
        if len(inlier_pts) < 5:
            return []

        # 计算端点 (投影到直线上)
        line_start = inlier_pts[0]
        line_end = inlier_pts[-1]

        # 检测线型
        line_type = detect_line_pattern(segment)

        # 创建图元
        prim = Primitive(
            id=self._next_primitive_id("L"),
            type=PrimitiveType.LINE_SEGMENT,
            layer=GeometryLayer.CONTOUR,
            line_type=line_type,
            params={
                'slope': float(line_params[0]),
                'intercept': float(line_params[1]),
                'length': float(np.linalg.norm(line_end - line_start)),
                'angle': float(np.degrees(np.arctan2(
                    line_end[1] - line_start[1],
                    line_end[0] - line_start[0]
                ))),
                'inlier_count': int(np.sum(inlier_mask)),
                'inlier_ratio': float(np.sum(inlier_mask) / max(len(segment), 1)),
            },
            confidence=0.8,
            pixel_points=segment
        )

        return [prim]

    # ============================================================
    # 5. 曲线分类 (第3.3.2节)
    # ============================================================

    def _classify_curve(
        self,
        segment: np.ndarray,
        vertices: Dict[str, Vertex]
    ) -> List[Primitive]:
        """
        曲线片段三层细分

        折线 / 标准圆弧 / 椭圆弧 / 无效曲线
        """
        if len(segment) < 5:
            return []

        # 尝试圆弧拟合
        arc_result = self._fit_arc(segment)
        if arc_result:
            return [arc_result]

        # 尝试椭圆弧拟合
        if self.config.enable_elliptic_arc_detection:
            ellipse_result = self._fit_elliptic_arc(segment)
            if ellipse_result:
                return [ellipse_result]

        # 检查是否为折线 (多个直线段)
        polyline_result = self._fit_polyline(segment)
        if polyline_result:
            return polyline_result

        # 无效曲线
        return []

    # ============================================================
    # 6. 圆弧拟合 (第3.3.2节)
    # ============================================================

    def _fit_arc(self, segment: np.ndarray) -> Optional[Primitive]:
        """
        圆弧拟合

        弧长自适应角度阈值，支持标准圆弧识别。
        """
        if len(segment) < 5:
            return None

        # 最小二乘圆拟合
        center, radius = fit_circle_least_squares(segment)
        if center is None or radius < 5:
            return None

        # 计算拟合误差
        residuals = np.abs(
            np.linalg.norm(segment - np.array([[center.x, center.y]]), axis=1) - radius
        )
        rmse = float(np.sqrt(np.mean(residuals**2)))

        # 自适应残差阈值
        residual_threshold = self.th.arc_residual_threshold * \
            self.th.get_resolution_scale()

        if rmse > residual_threshold:
            return None

        # 计算起始/终止角度
        angles = np.arctan2(
            segment[:, 1] - center.y,
            segment[:, 0] - center.x
        )
        start_angle = float(angles[0])
        end_angle = float(angles[-1])

        # 弧长
        arc_length = float(np.sum(np.linalg.norm(
            np.diff(segment, axis=0), axis=1
        )))

        # 自适应角度阈值
        angle_threshold = self.th.get_arc_angle_threshold(
            arc_length, np.linalg.norm(segment[0] - segment[-1])
        )

        # 向量夹角均值检验
        if len(segment) >= 5:
            vec_angles = []
            for i in range(1, len(segment) - 1):
                v1 = segment[i] - segment[i-1]
                v2 = segment[i+1] - segment[i]
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    angle = angle_between_degrees(v1, v2)
                    vec_angles.append(angle)

            if vec_angles:
                mean_angle = np.mean(vec_angles)
                if mean_angle < angle_threshold:
                    # 太直，可能是直线误判
                    return None

        # 判断是否为完整圆
        span_angle = abs(end_angle - start_angle)
        if span_angle > 2 * np.pi - 0.1:
            prim_type = PrimitiveType.CIRCLE
        else:
            prim_type = PrimitiveType.ARC

        # 检测线型
        line_type = detect_line_pattern(segment)

        prim = Primitive(
            id=self._next_primitive_id("A"),
            type=prim_type,
            layer=GeometryLayer.CONTOUR,
            line_type=line_type,
            params={
                'center_x': float(center.x),
                'center_y': float(center.y),
                'radius': float(radius),
                'start_angle': start_angle,
                'end_angle': end_angle,
                'arc_length': arc_length,
                'rmse': rmse,
                'span_angle': float(span_angle),
            },
            confidence=float(max(0.5, 1.0 - rmse / residual_threshold)),
            pixel_points=segment
        )

        return prim

    # ============================================================
    # 7. 椭圆弧拟合 (第3.3.2节)
    # ============================================================

    def _fit_elliptic_arc(self, segment: np.ndarray) -> Optional[Primitive]:
        """
        椭圆弧拟合

        区分标准圆弧与椭圆弧。
        """
        if len(segment) < 10:
            return None

        ellipse = fit_ellipse_least_squares(segment)
        if ellipse is None:
            return None

        center = ellipse['center']
        a_axis = ellipse['a']
        b_axis = ellipse['b']

        if a_axis < 5 or b_axis < 5:
            return None

        # 计算拟合误差
        angles = np.arctan2(
            segment[:, 1] - center.y,
            segment[:, 0] - center.x
        )
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        ellipse_pts = np.column_stack([
            center.x + a_axis * cos_a,
            center.y + b_axis * sin_a
        ])
        # 应用旋转
        rot_angle = ellipse['angle']
        rot_mat = np.array([
            [np.cos(rot_angle), -np.sin(rot_angle)],
            [np.sin(rot_angle), np.cos(rot_angle)]
        ])
        ellipse_pts = np.dot(ellipse_pts - np.array([[center.x, center.y]]), rot_mat.T) + \
                      np.array([[center.x, center.y]])

        residuals = np.linalg.norm(segment - ellipse_pts, axis=1)
        rmse = float(np.sqrt(np.mean(residuals**2)))

        if rmse > self.th.arc_residual_threshold * 1.5:
            return None

        # 计算起始/终止角度
        start_angle = float(angles[0])
        end_angle = float(angles[-1])

        prim = Primitive(
            id=self._next_primitive_id("E"),
            type=PrimitiveType.ELLIPTIC_ARC,
            layer=GeometryLayer.CONTOUR,
            params={
                'center_x': float(center.x),
                'center_y': float(center.y),
                'a': a_axis,
                'b': b_axis,
                'angle': ellipse['angle'],
                'start_angle': start_angle,
                'end_angle': end_angle,
                'rmse': rmse,
            },
            confidence=float(max(0.4, 1.0 - rmse / (self.th.arc_residual_threshold * 1.5))),
            pixel_points=segment
        )

        return prim

    # ============================================================
    # 8. 折线拟合
    # ============================================================

    def _fit_polyline(self, segment: np.ndarray) -> Optional[List[Primitive]]:
        """
        折线拟合

        将曲线分段拟合为多个直线段。
        """
        if len(segment) < 10:
            return None

        # 使用Ramer-Douglas-Peucker算法简化
        epsilon = self.th.ransac_distance * 2
        try:
            simplified = cv2.approxPolyDP(
                segment.astype(np.float32), epsilon, False
            )
            if simplified is None or len(simplified) < 3:
                return None
        except cv2.error:
            return None

        simplified = simplified.squeeze()
        if len(simplified.shape) < 2:
            return None

        # 创建直线段
        primitives = []
        for i in range(len(simplified) - 1):
            p1 = simplified[i]
            p2 = simplified[i + 1]
            length = np.linalg.norm(p2 - p1)
            if length < 5:
                continue

            angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))

            # 找到对应的像素点
            mask = np.zeros(len(segment), dtype=bool)
            for j, pt in enumerate(segment):
                d = point_line_distance(
                    Point(pt[0], pt[1]),
                    Point(p1[0], p1[1]),
                    Point(p2[0], p2[1])
                )
                if d < self.th.ransac_distance:
                    mask[j] = True

            prim = Primitive(
                id=self._next_primitive_id("L"),
                type=PrimitiveType.LINE_SEGMENT,
                layer=GeometryLayer.CONTOUR,
                params={
                    'slope': float(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])),
                    'length': float(length),
                    'angle': float(angle),
                    'inlier_count': int(np.sum(mask)),
                },
                confidence=0.7,
                pixel_points=segment[mask] if np.sum(mask) > 0 else None
            )
            primitives.append(prim)

        return primitives if primitives else None

    # ============================================================
    # 9. 直线三重校验 (第3.3.3节)
    # ============================================================

    def _triple_line_verification(self, primitives: List[Primitive]) -> List[Primitive]:
        """
        直线三重校验机制

        1. 全局向量方向一致性校验
        2. 平行分组联合校验
        3. 像素重合度自适应阈值校验
        """
        verified = []

        # 分离直线和其他图元
        lines = [p for p in primitives if p.type == PrimitiveType.LINE_SEGMENT]
        others = [p for p in primitives if p.type != PrimitiveType.LINE_SEGMENT]

        # 方向一致性校验
        valid_lines = []
        for line in lines:
            if self._check_direction_consistency(line):
                valid_lines.append(line)
            else:
                # 不通过的直线降级为曲线
                line.type = PrimitiveType.CURVE
                line.confidence *= 0.5
                others.append(line)

        # 平行分组联合校验
        groups = self._group_parallel_lines(valid_lines)
        for group in groups:
            if self._check_parallel_group(group):
                verified.extend(group)
            else:
                for line in group:
                    line.confidence *= 0.8
                    verified.append(line)

        # 像素重合度校验
        final_lines = []
        for line in verified:
            if self._check_pixel_overlap(line):
                final_lines.append(line)
            else:
                others.append(line)

        return final_lines + others

    def _check_direction_consistency(self, prim: Primitive) -> bool:
        """全局向量方向一致性校验"""
        if prim.pixel_points is None or len(prim.pixel_points) < 10:
            return True

        pts = prim.pixel_points

        # 计算整体方向
        overall_angle = prim.params.get('angle', 0)

        # 计算局部切向量方向
        local_angles = []
        for i in range(1, len(pts)):
            dx = pts[i, 0] - pts[i-1, 0]
            dy = pts[i, 1] - pts[i-1, 1]
            if abs(dx) > 0.5 or abs(dy) > 0.5:
                angle = np.degrees(np.arctan2(dy, dx))
                local_angles.append(angle)

        if not local_angles:
            return True

        # 计算中位数偏差
        deviations = [abs(a - overall_angle) % 180 for a in local_angles]
        deviations = [min(d, 180 - d) for d in deviations]
        median_dev = np.median(deviations)

        # 自适应角度阈值
        angle_threshold = self.th.get_angle_consistency_threshold(
            prim.line_type.value
        )

        return median_dev < angle_threshold

    def _group_parallel_lines(self, lines: List[Primitive]) -> List[List[Primitive]]:
        """平行分组"""
        if len(lines) < 2:
            return [[l] for l in lines]

        groups = []
        used = set()

        for i, l1 in enumerate(lines):
            if i in used:
                continue
            group = [l1]
            used.add(i)
            angle1 = l1.params.get('angle', 0) % 180

            for j, l2 in enumerate(lines):
                if j in used:
                    continue
                angle2 = l2.params.get('angle', 0) % 180
                diff = abs(angle1 - angle2) % 180
                diff = min(diff, 180 - diff)

                if diff < self.th.parallel_tolerance:
                    group.append(l2)
                    used.add(j)

            groups.append(group)

        return groups

    def _check_parallel_group(self, group: List[Primitive]) -> bool:
        """平行分组联合校验"""
        if len(group) < 2:
            return True

        # 检查组内线段的一致性
        angles = [l.params.get('angle', 0) % 180 for l in group]
        mean_angle = np.mean(angles)
        deviations = [abs(a - mean_angle) % 180 for a in angles]
        deviations = [min(d, 180 - d) for d in deviations]

        return np.std(deviations) < self.th.parallel_tolerance * 0.8

    def _check_pixel_overlap(self, prim: Primitive) -> bool:
        """像素重合度自适应阈值校验"""
        if prim.pixel_points is None or len(prim.pixel_points) < 10:
            return True

        pts = prim.pixel_points

        # 将点投影到直线上，检查覆盖范围
        angle = prim.params.get('angle', 0)
        rad = np.radians(angle)
        proj = pts[:, 0] * np.cos(rad) + pts[:, 1] * np.sin(rad)

        # 检查投影覆盖的范围
        coverage = (np.max(proj) - np.min(proj)) / max(len(pts), 1)
        expected = np.linalg.norm(pts[0] - pts[-1]) / max(len(pts), 1)

        ratio = coverage / max(expected, 1e-10)
        threshold = self.th.get_pixel_overlap_threshold()

        return ratio > threshold

    # ============================================================
    # 10. 圆弧合并与校验 (第3.3.2节)
    # ============================================================

    def _merge_arcs(self, primitives: List[Primitive]) -> List[Primitive]:
        """
        同圆心、同半径分段圆弧自动合并
        """
        arcs = [p for p in primitives if p.type == PrimitiveType.ARC]
        others = [p for p in primitives if p.type != PrimitiveType.ARC]

        if len(arcs) < 2:
            return primitives

        merged = []
        used = set()

        for i, a1 in enumerate(arcs):
            if i in used:
                continue
            group = [a1]
            used.add(i)

            cx1 = a1.params.get('center_x')
            cy1 = a1.params.get('center_y')
            r1 = a1.params.get('radius')

            for j, a2 in enumerate(arcs):
                if j in used:
                    continue
                cx2 = a2.params.get('center_x')
                cy2 = a2.params.get('center_y')
                r2 = a2.params.get('radius')

                if cx1 is None or cx2 is None:
                    continue

                dist = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
                if dist < 5 and abs(r1 - r2) < 3:
                    group.append(a2)
                    used.add(j)

            if len(group) > 1:
                # 合并圆弧
                merged_arc = self._merge_arc_group(group)
                if merged_arc:
                    merged.append(merged_arc)
            else:
                merged.append(a1)

        return merged + others

    def _merge_arc_group(self, arcs: List[Primitive]) -> Optional[Primitive]:
        """合并一组同圆心同半径的圆弧"""
        if not arcs:
            return None

        base = arcs[0]
        all_angles = []
        for arc in arcs:
            all_angles.append(arc.params.get('start_angle', 0))
            all_angles.append(arc.params.get('end_angle', 0))

        if not all_angles:
            return None

        start_angle = min(all_angles)
        end_angle = max(all_angles)

        merged = Primitive(
            id=self._next_primitive_id("AM"),
            type=PrimitiveType.ARC,
            layer=GeometryLayer.CONTOUR,
            params={
                'center_x': base.params.get('center_x'),
                'center_y': base.params.get('center_y'),
                'radius': base.params.get('radius'),
                'start_angle': start_angle,
                'end_angle': end_angle,
                'merged_from': [a.id for a in arcs],
            },
            confidence=max(a.confidence for a in arcs)
        )

        return merged

    def _arc_topology_validation(
        self,
        primitives: List[Primitive],
        vertices: Dict[str, Vertex]
    ) -> List[Primitive]:
        """
        圆弧拓扑绑定校验

        圆弧首尾顶点必须与全局有效顶点匹配。
        """
        validated = []
        vertex_positions = np.array([
            (v.position.x, v.position.y) for v in vertices.values()
        ])

        for prim in primitives:
            if prim.type not in (PrimitiveType.ARC, PrimitiveType.CIRCLE):
                validated.append(prim)
                continue

            if prim.type == PrimitiveType.CIRCLE:
                validated.append(prim)
                continue

            # 检查圆弧端点
            if prim.pixel_points is not None and len(prim.pixel_points) >= 2:
                start_pt = prim.pixel_points[0]
                end_pt = prim.pixel_points[-1]

                # 检查是否靠近某个顶点
                start_near = np.min(np.linalg.norm(
                    vertex_positions - start_pt, axis=1
                )) if len(vertex_positions) > 0 else float('inf')

                end_near = np.min(np.linalg.norm(
                    vertex_positions - end_pt, axis=1
                )) if len(vertex_positions) > 0 else float('inf')

                if start_near > 10 and end_near > 10:
                    # 两端都不靠近任何顶点，可能是游离弧
                    prim.confidence *= 0.5

            validated.append(prim)

        return validated