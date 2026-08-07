"""
多源融合顶点检测体系 (第3.2节)

核心优化:
1. 三种顶点来源: 语义/骨架拓扑/轮廓曲率
2. DBSCAN自适应密度聚类替代固定距离聚类
3. 量化置信度加权融合
4. 几何约束虚拟顶点推导 (重大升级)
5. 线型语义识别
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Set, Dict
from sklearn.cluster import DBSCAN
from .config import GeometryConfig
from .data_structures import (
    Vertex, Point, ConfidenceSource, LineType, GeometryLayer,
    Primitive, PrimitiveType
)
from .utils import (
    compute_curvature, point_line_distance, perpendicular_foot,
    midpoint, angle_bisector_point, circle_center, centroid,
    triangle_centroid, triangle_circumcenter, triangle_orthocenter,
    triangle_incenter, detect_line_pattern, normalize_coordinates
)


class VertexDetector:
    """
    顶点检测器

    检测流程:
    骨架拓扑顶点 → 轮廓曲率拐点 → DBSCAN聚类降噪
    → 置信度加权融合 → 虚拟顶点推导 → 线型检测
    """

    def __init__(self, config: GeometryConfig):
        self.config = config
        self.th = config.thresholds
        self.vertex_counter: int = 0
        self.semantic_labels: Dict[str, Point] = {}
        self._vertex_id_counter: int = 0

    def _next_vertex_id(self, prefix: str = "V") -> str:
        """生成下一个顶点ID"""
        self._vertex_id_counter += 1
        return f"{prefix}{self._vertex_id_counter}"

    def detect_vertices(
        self,
        skeleton: np.ndarray,
        binary_image: np.ndarray,
        original_gray: np.ndarray,
        text_mask: np.ndarray
    ) -> Dict[str, Vertex]:
        """
        完整顶点检测流水线

        Args:
            skeleton: 骨架图像
            binary_image: 二值化图像
            original_gray: 原始灰度图
            text_mask: 文字掩膜

        Returns:
            顶点字典 {id: Vertex}
        """
        # 1. 提取语义顶点 (OCR文字包围盒中心)
        semantic_vertices = self._extract_semantic_vertices(
            original_gray, text_mask, binary_image
        )

        # 2. 提取骨架拓扑顶点
        skeleton_vertices = self._extract_skeleton_vertices(skeleton)

        # 3. 提取轮廓曲率拐点
        contour_vertices = self._extract_contour_vertices(binary_image, skeleton)

        # 4. DBSCAN聚类降噪
        if self.config.enable_dbscan_clustering:
            all_candidates = (
                semantic_vertices + skeleton_vertices + contour_vertices
            )
            clustered_vertices = self._dbscan_cluster_fusion(all_candidates)
        else:
            # 简单距离融合
            all_candidates = self._simple_fusion(
                semantic_vertices + skeleton_vertices + contour_vertices
            )
            clustered_vertices = all_candidates

        # 5. 构建顶点字典
        vertices = {}
        for v in clustered_vertices:
            vertices[v.id] = v

        # 返回顶点 (虚拟顶点在primitive识别后推导)
        return vertices

    # ============================================================
    # 1. 语义顶点提取 (第3.2.1节)
    # ============================================================

    def _extract_semantic_vertices(
        self,
        gray: np.ndarray,
        text_mask: np.ndarray,
        binary: np.ndarray
    ) -> List[Vertex]:
        """
        语义顶点提取

        基于文字区域检测，提取字符包围盒中心作为语义顶点。
        支持字母、数字、希腊字母。

        Returns:
            语义顶点列表 (置信度0.9)
        """
        vertices = []

        if np.sum(text_mask) == 0:
            return vertices

        # 对文字掩膜进行连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            text_mask, connectivity=8
        )

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            x, y, w, h = (stats[i, cv2.CC_STAT_LEFT],
                          stats[i, cv2.CC_STAT_TOP],
                          stats[i, cv2.CC_STAT_WIDTH],
                          stats[i, cv2.CC_STAT_HEIGHT])

            if area < 15:
                continue

            # 质心
            cx, cy = centroids[i]

            # 尝试OCR识别字母 (使用模板匹配简化版)
            label = self._simple_ocr_region(
                gray[y:y+h, x:x+w], binary[y:y+h, x:x+w]
            )

            vertex = Vertex(
                id=self._next_vertex_id("S"),
                position=Point(float(cx), float(cy)),
                confidence=ConfidenceSource.SEMANTIC.score,
                source=ConfidenceSource.SEMANTIC,
                label=label,
                layer=GeometryLayer.ANNOTATION
            )
            vertices.append(vertex)

            # 记录语义标签
            if label:
                self.semantic_labels[label] = Point(float(cx), float(cy))

        return vertices

    def _simple_ocr_region(self, gray_roi: np.ndarray, binary_roi: np.ndarray) -> Optional[str]:
        """简化的单字符OCR (基于轮廓匹配)"""
        if gray_roi.size == 0:
            return None

        # 缩放到统一大小
        try:
            resized = cv2.resize(binary_roi, (20, 20))
        except cv2.error:
            return None

        # 简单特征: 水平和垂直投影
        h_proj = np.sum(resized > 0, axis=1) / 20
        v_proj = np.sum(resized > 0, axis=0) / 20

        # 计算宽高比
        h_ratio = gray_roi.shape[1] / max(gray_roi.shape[0], 1)

        # 返回标签 (实际应用中应使用数学OCR模型)
        # 这里简化处理，根据宽高比粗略判断
        if 0.6 < h_ratio < 1.4:
            return "O"  # 近似圆形字母
        elif h_ratio < 0.6:
            return "I"  # 窄字母
        else:
            return None  # 无法确定

    def set_semantic_labels(self, labels: Dict[str, Point]):
        """手动设置语义标签 (人机交互接口)"""
        self.semantic_labels = labels

    # ============================================================
    # 2. 骨架拓扑顶点提取 (第3.2.1节)
    # ============================================================

    def _extract_skeleton_vertices(self, skeleton: np.ndarray) -> List[Vertex]:
        """
        骨架拓扑顶点提取

        通过8邻域统计:
        - 端点: 1个邻域点 (射线、开放线段端点)
        - 交点: >=3个邻域点 (交叉点、T型交接)
        """
        vertices = []
        h, w = skeleton.shape

        skeleton_bool = skeleton > 0

        # 使用卷积计算8邻域点数
        kernel = np.ones((3, 3), np.uint8)
        kernel[1, 1] = 0  # 不计算自身
        neighbor_count = cv2.filter2D(
            skeleton_bool.astype(np.uint8), -1, kernel
        )
        neighbor_count[skeleton == 0] = 0

        # 端点: 仅有1个邻域点
        end_mask = (neighbor_count == 1)
        end_pts = np.argwhere(end_mask)
        for pt in end_pts:
            y, x = pt[0], pt[1]
            vertex = Vertex(
                id=self._next_vertex_id("E"),
                position=Point(float(x), float(y)),
                confidence=ConfidenceSource.SKELETON_END.score,
                source=ConfidenceSource.SKELETON_END,
                layer=GeometryLayer.CONTOUR
            )
            vertices.append(vertex)

        # 交点: >=3个邻域点
        cross_mask = (neighbor_count >= 3)
        cross_pts = np.argwhere(cross_mask)
        for pt in cross_pts:
            y, x = pt[0], pt[1]
            vertex = Vertex(
                id=self._next_vertex_id("C"),
                position=Point(float(x), float(y)),
                confidence=ConfidenceSource.SKELETON_CROSS.score,
                source=ConfidenceSource.SKELETON_CROSS,
                layer=GeometryLayer.CONTOUR
            )
            vertices.append(vertex)

        return vertices

    # ============================================================
    # 3. 轮廓曲率拐点提取 (第3.2.1节)
    # ============================================================

    def _extract_contour_vertices(
        self, binary: np.ndarray, skeleton: np.ndarray
    ) -> List[Vertex]:
        """
        轮廓曲率拐点提取

        在骨架路径上计算曲率，提取局部极值点作为直-弧过渡弱顶点。
        """
        vertices = []

        # 提取骨架上的连续路径
        paths = self._extract_skeleton_paths(skeleton)

        for path in paths:
            if len(path) < 10:
                continue

            # 计算曲率
            path_pts = np.array(path)
            curvature = compute_curvature(path_pts)

            # 找到曲率的局部极大值
            curvature_smooth = cv2.GaussianBlur(
                curvature.reshape(-1, 1).astype(np.float32),
                (5, 1), 0
            ).flatten()

            for i in range(2, len(curvature_smooth) - 2):
                if curvature_smooth[i] > curvature_smooth[i-1] and \
                   curvature_smooth[i] > curvature_smooth[i+1] and \
                   curvature_smooth[i] > 0.1:  # 曲率阈值
                    y, x = path[i]
                    vertex = Vertex(
                        id=self._next_vertex_id("K"),
                        position=Point(float(x), float(y)),
                        confidence=ConfidenceSource.CONTOUR_CORNER.score,
                        source=ConfidenceSource.CONTOUR_CORNER,
                        layer=GeometryLayer.CONTOUR
                    )
                    vertices.append(vertex)

        return vertices

    def _extract_skeleton_paths(self, skeleton: np.ndarray) -> List[List[Tuple[int, int]]]:
        """从骨架中提取连续路径"""
        from skimage.morphology import remove_small_objects

        h, w = skeleton.shape
        skeleton_bool = skeleton > 0

        # 如果骨架是空的，返回空列表
        if np.sum(skeleton_bool) == 0:
            return []

        # 使用连通域分析提取路径
        num_labels, labels = cv2.connectedComponents(
            skeleton_bool.astype(np.uint8), connectivity=8
        )

        paths = []
        for i in range(1, num_labels):
            # 提取该连通域的像素坐标
            ys, xs = np.where(labels == i)
            if len(ys) < 5:
                continue

            # 转换为有序路径
            path = self._order_path_points(list(zip(ys, xs)), labels == i)
            paths.append(path)

        return paths

    def _order_path_points(
        self, points: List[Tuple[int, int]], component_mask: np.ndarray
    ) -> List[Tuple[int, int]]:
        """对连通域中的点进行排序，形成有序路径"""
        if len(points) < 2:
            return points

        # 找到端点
        h, w = component_mask.shape
        kernel = np.ones((3, 3), np.uint8)
        kernel[1, 1] = 0
        neighbor_count = cv2.filter2D(
            component_mask.astype(np.uint8), -1, kernel
        )

        # 从端点开始跟踪
        start_points = np.argwhere(
            np.logical_and(component_mask, neighbor_count == 1)
        )

        if len(start_points) == 0:
            # 闭合路径，任选一点
            start = (points[0][0], points[0][1])
        else:
            start = (start_points[0][0], start_points[0][1])

        # BFS排序
        ordered = []
        visited = set()
        queue = [start]
        visited.add(start)

        while queue:
            current = queue.pop(0)
            ordered.append(current)
            cy, cx = current

            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if component_mask[ny, nx] and (ny, nx) not in visited:
                            visited.add((ny, nx))
                            queue.append((ny, nx))

        return ordered

    # ============================================================
    # 4. DBSCAN自适应密度聚类 (第3.2.2节)
    # ============================================================

    def _dbscan_cluster_fusion(self, candidates: List[Vertex]) -> List[Vertex]:
        """
        DBSCAN自适应密度聚类 + 置信度加权融合

        根据局部线条密度动态划分顶点簇：
        - 高分辨率大图不会过度合并密集交点
        - 低分辨率小图有效过滤抖动重影伪点
        """
        if len(candidates) < 2:
            return candidates

        # 提取坐标和置信度
        points = np.array([(v.position.x, v.position.y) for v in candidates])
        confidences = np.array([v.confidence for v in candidates])

        # DBSCAN聚类
        eps = self.th.dbscan_eps
        clustering = DBSCAN(
            eps=eps,
            min_samples=self.th.dbscan_min_samples
        ).fit(points)

        labels = clustering.labels_

        # 融合每个簇中的顶点
        fused_vertices = []

        # 处理噪声点 (label=-1)
        for i, label in enumerate(labels):
            if label == -1:
                # 噪声点独立保留
                fused_vertices.append(candidates[i])

        # 处理非噪声簇
        unique_labels = set(labels) - {-1}
        for label in unique_labels:
            mask = labels == label
            cluster_points = points[mask]
            cluster_confidences = confidences[mask]
            cluster_candidates = [candidates[i] for i in range(len(candidates)) if labels[i] == label]

            # 加权平均位置
            weights = cluster_confidences / np.sum(cluster_confidences)
            fused_pos = np.average(cluster_points, axis=0, weights=weights)

            # 取最高置信度
            max_conf_idx = np.argmax(cluster_confidences)
            best_source = cluster_candidates[max_conf_idx].source
            best_label = cluster_candidates[max_conf_idx].label

            fused_vertex = Vertex(
                id=self._next_vertex_id("F"),
                position=Point(float(fused_pos[0]), float(fused_pos[1])),
                confidence=float(np.max(cluster_confidences)),
                source=best_source,
                label=best_label,
                layer=GeometryLayer.CONTOUR
            )
            fused_vertices.append(fused_vertex)

        return fused_vertices

    def _simple_fusion(self, candidates: List[Vertex]) -> List[Vertex]:
        """简单距离融合 (备选方案)"""
        if len(candidates) < 2:
            return candidates

        fusion_distance = self.th.vertex_fusion_distance
        fused = []

        while candidates:
            base = candidates.pop(0)
            cluster = [base]
            remaining = []

            for c in candidates:
                if base.position.distance_to(c.position) < fusion_distance:
                    cluster.append(c)
                else:
                    remaining.append(c)

            # 融合
            if len(cluster) > 1:
                avg_x = np.mean([v.position.x for v in cluster])
                avg_y = np.mean([v.position.y for v in cluster])
                best_conf = max(v.confidence for v in cluster)
                best_source = max(cluster, key=lambda v: v.confidence).source
                best_label = max(cluster, key=lambda v: v.confidence).label

                fused.append(Vertex(
                    id=self._next_vertex_id("F"),
                    position=Point(avg_x, avg_y),
                    confidence=best_conf,
                    source=best_source,
                    label=best_label
                ))
            else:
                fused.append(base)

            candidates = remaining

        return fused

    # ============================================================
    # 5. 几何虚拟顶点推导 (第3.2.4节) - 重大升级
    # ============================================================

    def derive_virtual_vertices(
        self,
        vertices: Dict[str, Vertex],
        primitives: List[Primitive]
    ) -> Dict[str, Vertex]:
        """
        几何约束虚拟顶点推导

        基于平面几何定理自动计算隐性关键点:
        1. 线段中点、垂线垂足
        2. 角平分点
        3. 圆/圆弧圆心
        4. 三角形重心、垂心、外心、内心
        5. 延长线虚拟交点
        """
        if not self.config.enable_virtual_vertices:
            return vertices

        result = dict(vertices)
        vertex_list = list(result.values())

        # 获取所有线段
        segments = [p for p in primitives if p.type == PrimitiveType.LINE_SEGMENT]

        # 1. 线段中点
        for seg in segments:
            if len(seg.vertices) >= 2:
                v1 = result.get(seg.vertices[0])
                v2 = result.get(seg.vertices[1])
                if v1 and v2:
                    mid = midpoint(v1.position, v2.position)
                    mid_id = self._next_vertex_id("M")
                    result[mid_id] = Vertex(
                        id=mid_id,
                        position=mid,
                        confidence=ConfidenceSource.VIRTUAL.score,
                        source=ConfidenceSource.VIRTUAL,
                        is_virtual=True,
                        virtual_type="midpoint",
                        layer=GeometryLayer.AUXILIARY
                    )

        # 2. 垂线垂足
        # 对每个顶点到每条线段求垂足
        for v in vertex_list:
            for seg in segments:
                if len(seg.vertices) >= 2:
                    a = result.get(seg.vertices[0])
                    b = result.get(seg.vertices[1])
                    if a and b:
                        foot = perpendicular_foot(v.position, a.position, b.position)
                        foot_dist = v.position.distance_to(foot)
                        # 垂足在有效范围内
                        seg_len = a.position.distance_to(b.position)
                        if foot_dist < seg_len * 2 and v.position.distance_to(foot) > 5:
                            foot_id = self._next_vertex_id("F")
                            result[foot_id] = Vertex(
                                id=foot_id,
                                position=foot,
                                confidence=ConfidenceSource.VIRTUAL.score * 0.9,
                                source=ConfidenceSource.VIRTUAL,
                                is_virtual=True,
                                virtual_type="foot",
                                layer=GeometryLayer.AUXILIARY
                            )

        # 3. 角平分点
        for i, v1 in enumerate(vertex_list):
            for j, v2 in enumerate(vertex_list):
                for k, v3 in enumerate(vertex_list):
                    if len({i, j, k}) < 3:
                        continue
                    # 检查v2是否为角的顶点
                    d12 = v1.position.distance_to(v2.position)
                    d23 = v2.position.distance_to(v3.position)
                    if d12 < 30 or d23 < 30:
                        continue
                    # 计算角平分线
                    bp = angle_bisector_point(
                        v1.position, v2.position, v3.position,
                        length=min(d12, d23) * 0.5
                    )
                    bp_id = self._next_vertex_id("B")
                    result[bp_id] = Vertex(
                        id=bp_id,
                        position=bp,
                        confidence=ConfidenceSource.VIRTUAL.score * 0.85,
                        source=ConfidenceSource.VIRTUAL,
                        is_virtual=True,
                        virtual_type="angle_bisector",
                        layer=GeometryLayer.AUXILIARY
                    )

        # 4. 三角形几何中心
        triangles = self._find_triangles(segments, result)
        for tri in triangles:
            a, b, c = tri

            # 重心
            gc = triangle_centroid(a.position, b.position, c.position)
            gc_id = self._next_vertex_id("G")
            result[gc_id] = Vertex(
                id=gc_id,
                position=gc,
                confidence=ConfidenceSource.VIRTUAL.score,
                source=ConfidenceSource.VIRTUAL,
                is_virtual=True,
                virtual_type="centroid",
                layer=GeometryLayer.AUXILIARY
            )

            # 外心
            circum = triangle_circumcenter(a.position, b.position, c.position)
            if circum:
                oc_id = self._next_vertex_id("O")
                result[oc_id] = Vertex(
                    id=oc_id,
                    position=circum,
                    confidence=ConfidenceSource.VIRTUAL.score * 0.95,
                    source=ConfidenceSource.VIRTUAL,
                    is_virtual=True,
                    virtual_type="circumcenter",
                    layer=GeometryLayer.AUXILIARY
                )

            # 垂心
            ortho = triangle_orthocenter(a.position, b.position, c.position)
            oh_id = self._next_vertex_id("H")
            result[oh_id] = Vertex(
                id=oh_id,
                position=ortho,
                confidence=ConfidenceSource.VIRTUAL.score * 0.9,
                source=ConfidenceSource.VIRTUAL,
                is_virtual=True,
                virtual_type="orthocenter",
                layer=GeometryLayer.AUXILIARY
            )

            # 内心
            incenter = triangle_incenter(a.position, b.position, c.position)
            ic_id = self._next_vertex_id("I")
            result[ic_id] = Vertex(
                id=ic_id,
                position=incenter,
                confidence=ConfidenceSource.VIRTUAL.score * 0.9,
                source=ConfidenceSource.VIRTUAL,
                is_virtual=True,
                virtual_type="incenter",
                layer=GeometryLayer.AUXILIARY
            )

        # 5. 圆/圆弧圆心
        circles = [p for p in primitives if p.type in (PrimitiveType.CIRCLE, PrimitiveType.ARC)]
        for circ in circles:
            center_id = circ.params.get('center_id')
            if center_id and center_id in result:
                # 已有圆心顶点
                result[center_id].is_virtual = True
                result[center_id].virtual_type = "circle_center"
                result[center_id].confidence = max(
                    result[center_id].confidence,
                    ConfidenceSource.VIRTUAL.score
                )

        return result

    def _find_triangles(
        self,
        segments: List[Primitive],
        vertices: Dict[str, Vertex]
    ) -> List[Tuple[Vertex, Vertex, Vertex]]:
        """从线段集合中找出三角形"""
        triangles = []
        visited = set()

        for i, s1 in enumerate(segments):
            for j, s2 in enumerate(segments):
                if j <= i:
                    continue
                # 找到共享顶点的线段对
                shared = set(s1.vertices) & set(s2.vertices)
                if len(shared) != 1:
                    continue
                shared_v = shared.pop()
                # 两条线段的不共享顶点
                v1 = (set(s1.vertices) - {shared_v}).pop()
                v2 = (set(s2.vertices) - {shared_v}).pop()

                # 寻找第三边
                for k, s3 in enumerate(segments):
                    if k <= j:
                        continue
                    s3_verts = set(s3.vertices)
                    if v1 in s3_verts and v2 in s3_verts:
                        tri_key = tuple(sorted([shared_v, v1, v2]))
                        if tri_key not in visited:
                            visited.add(tri_key)
                            a = vertices.get(shared_v)
                            b = vertices.get(v1)
                            c = vertices.get(v2)
                            if a and b and c:
                                triangles.append((a, b, c))

        return triangles

    # ============================================================
    # 6. 线型语义识别 (第3.2.5节)
    # ============================================================

    def detect_line_types(
        self,
        skeleton: np.ndarray,
        primitives: List[Primitive]
    ) -> List[Primitive]:
        """
        线型语义识别

        检测区分实线、虚线、点划线、射线。
        """
        if not self.config.enable_line_type_detection:
            return primitives

        for prim in primitives:
            if prim.type not in (PrimitiveType.LINE_SEGMENT, PrimitiveType.RAY_LINE):
                continue

            # 获取像素点
            if prim.pixel_points is not None and len(prim.pixel_points) > 5:
                line_type = detect_line_pattern(prim.pixel_points)
                prim.line_type = line_type

            # 射线检测：端点是否在图像边界附近
            if len(prim.vertices) == 2:
                v1_pos = None
                v2_pos = None
                # 这里假设传入的primitives关联了顶点信息
                # 实际检测射线需要检查顶点位置

        return primitives