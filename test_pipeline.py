"""
几何识别流水线综合测验脚本

测试内容:
1. 核心数据结构创建与操作
2. 几何工具函数正确性
3. 合成测试图像生成与识别
4. 完整流水线端到端测试
5. LaTeX代码生成验证
6. 人机交互接口测试
"""

import sys
import os
import numpy as np
import math

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry_recognizer import (
    Vertex, Point, ConfidenceSource, LineType, PrimitiveType,
    Primitive, GeometryLayer, GeometryResult, GeometryConfig,
    GeometryRecognizerPipeline
)
from geometry_recognizer.utils import (
    angle_between, line_angle, point_line_distance,
    point_line_segment_distance, line_intersection,
    midpoint, perpendicular_foot, circle_center,
    triangle_centroid, triangle_circumcenter, triangle_orthocenter,
    triangle_incenter, elongate_ratio, fit_circle_least_squares,
    fit_line_ransac, normalize_coordinates, compute_curvature,
    line_arc_intersection
)


class TestGeometryRecognizer:
    """几何识别器测验类"""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def assert_almost_equal(self, a, b, tol=1e-4, msg=""):
        """断言两个浮点数近似相等"""
        if abs(a - b) > tol:
            self.failed += 1
            err = f"ASSERT FAIL: {msg}: expected {b}, got {a} (tol={tol})"
            self.errors.append(err)
            print(f"  ✗ {err}")
        else:
            self.passed += 1
            print(f"  ✓ {msg}")

    def assert_true(self, cond, msg=""):
        if not cond:
            self.failed += 1
            err = f"ASSERT FAIL: {msg}"
            self.errors.append(err)
            print(f"  ✗ {err}")
        else:
            self.passed += 1
            print(f"  ✓ {msg}")

    # ============================================================
    # 测试1: 核心数据结构
    # ============================================================

    def test_data_structures(self):
        print("\n" + "=" * 60)
        print("测试1: 核心数据结构")
        print("=" * 60)

        # 1.1 Point
        p1 = Point(1.0, 2.0)
        p2 = Point(4.0, 6.0)
        self.assert_almost_equal(p1.distance_to(p2), 5.0, msg="Point distance")
        self.assert_almost_equal(p1.x, 1.0, msg="Point.x")
        self.assert_almost_equal(p1.y, 2.0, msg="Point.y")

        # 1.2 Vertex
        v = Vertex(id="V1", position=Point(3, 4),
                   confidence=0.9, source=ConfidenceSource.SEMANTIC,
                   label="A")
        self.assert_true(v.id == "V1", msg="Vertex id")
        self.assert_almost_equal(v.position.x, 3.0, msg="Vertex position.x")
        self.assert_true(v.source == ConfidenceSource.SEMANTIC, msg="Vertex source")
        self.assert_true(v.label == "A", msg="Vertex label")
        self.assert_true(v.is_virtual == False, msg="Vertex is_virtual default")

        # 1.3 ConfidenceSource scores
        self.assert_almost_equal(ConfidenceSource.SEMANTIC.score, 0.9,
                                 msg="SEMANTIC score")
        self.assert_almost_equal(ConfidenceSource.SKELETON_CROSS.score, 0.7,
                                 msg="CROSS score")
        self.assert_almost_equal(ConfidenceSource.SKELETON_END.score, 0.5,
                                 msg="END score")
        self.assert_almost_equal(ConfidenceSource.CONTOUR_CORNER.score, 0.3,
                                 msg="CORNER score")
        self.assert_almost_equal(ConfidenceSource.VIRTUAL.score, 0.8,
                                 msg="VIRTUAL score")

        # 1.4 Primitive
        prim = Primitive(
            id="L1", type=PrimitiveType.LINE_SEGMENT,
            vertices=["V1", "V2"],
            params={"length": 5.0, "angle": 45.0}
        )
        self.assert_true(prim.id == "L1", msg="Primitive id")
        self.assert_true(prim.type == PrimitiveType.LINE_SEGMENT, msg="Primitive type")
        self.assert_true(len(prim.vertices) == 2, msg="Primitive vertices count")

        # 1.5 GeometryResult
        result = GeometryResult()
        result.add_vertex(v)
        result.add_primitive(prim)
        self.assert_true("V1" in result.vertices, msg="Result add_vertex")
        self.assert_true(len(result.primitives) == 1, msg="Result add_primitive")
        self.assert_true(len(result.contours) == 1, msg="Result contour layer")

        # 1.6 to_dict serialization
        d = result.to_dict()
        self.assert_true("vertices" in d, msg="Result to_dict vertices")
        self.assert_true("primitives" in d, msg="Result to_dict primitives")
        self.assert_true(d["primitives"][0]["type"] == "line_segment",
                         msg="Result to_dict primitive type")

    # ============================================================
    # 测试2: 几何工具函数
    # ============================================================

    def test_geometry_utils(self):
        print("\n" + "=" * 60)
        print("测试2: 几何工具函数")
        print("=" * 60)

        # 2.1 点到直线距离
        p = Point(0, 0)
        a = Point(1, 0)
        b = Point(1, 1)
        dist = point_line_distance(p, a, b)
        self.assert_almost_equal(dist, 1.0, msg="Point-line distance (vertical)")

        # 2.2 点到线段距离
        dist_seg = point_line_segment_distance(Point(2, 0.5), Point(0, 0), Point(1, 1))
        self.assert_true(dist_seg > 0, msg="Point-segment distance")

        # 2.3 直线交点
        p1, p2 = Point(0, 0), Point(1, 1)
        p3, p4 = Point(0, 1), Point(1, 0)
        ip = line_intersection(p1, p2, p3, p4)
        self.assert_true(ip is not None, msg="Line intersection exists")
        self.assert_almost_equal(ip.x, 0.5, msg="Line intersection x")
        self.assert_almost_equal(ip.y, 0.5, msg="Line intersection y")

        # 2.4 平行线交点
        p1, p2 = Point(0, 0), Point(1, 1)
        p3, p4 = Point(1, 0), Point(2, 1)
        ip = line_intersection(p1, p2, p3, p4)
        self.assert_true(ip is None, msg="Parallel lines no intersection")

        # 2.5 中点
        mid = midpoint(Point(0, 0), Point(2, 4))
        self.assert_almost_equal(mid.x, 1.0, msg="Midpoint x")
        self.assert_almost_equal(mid.y, 2.0, msg="Midpoint y")

        # 2.6 垂足
        foot = perpendicular_foot(Point(2, 0), Point(0, 0), Point(1, 1))
        self.assert_almost_equal(foot.x, 1.0, msg="Foot x")
        self.assert_almost_equal(foot.y, 1.0, msg="Foot y")

        # 2.7 三点定圆
        center = circle_center(Point(0, 0), Point(1, 0), Point(0, 1))
        self.assert_true(center is not None, msg="Circle center exists")
        self.assert_almost_equal(center.x, 0.5, msg="Circle center x")
        self.assert_almost_equal(center.y, 0.5, msg="Circle center y")

        # 2.8 三角形重心
        c = triangle_centroid(Point(0, 0), Point(3, 0), Point(0, 3))
        self.assert_almost_equal(c.x, 1.0, msg="Centroid x")
        self.assert_almost_equal(c.y, 1.0, msg="Centroid y")

        # 2.9 三角形外心
        circ = triangle_circumcenter(Point(0, 0), Point(2, 0), Point(0, 2))
        self.assert_true(circ is not None, msg="Circumcenter exists")
        self.assert_almost_equal(circ.x, 1.0, msg="Circumcenter x")
        self.assert_almost_equal(circ.y, 1.0, msg="Circumcenter y")

        # 2.10 三角形垂心
        ortho = triangle_orthocenter(Point(0, 0), Point(2, 0), Point(0, 2))
        self.assert_almost_equal(ortho.x, 0.0, msg="Orthocenter x")
        self.assert_almost_equal(ortho.y, 0.0, msg="Orthocenter y")

        # 2.11 三角形内心
        incent = triangle_incenter(Point(0, 0), Point(3, 0), Point(0, 4))
        self.assert_true(incent is not None, msg="Incenter exists")
        self.assert_almost_equal(incent.x, 1.0, msg="Incenter x")

        # 2.12 伸长率 (直线)
        straight_pts = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        er = elongate_ratio(straight_pts)
        self.assert_almost_equal(er, 1.0, msg="Elongation ratio (straight)")

        # 2.13 伸长率 (曲线)
        curve_pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
        er = elongate_ratio(curve_pts)
        self.assert_true(er < 1.0, msg="Elongation ratio (curve) < 1")

        # 2.14 圆拟合
        circle_pts = np.array([
            [1 + np.cos(t), 1 + np.sin(t)]
            for t in np.linspace(0, 2*np.pi, 20)
        ])
        center_fit, radius_fit = fit_circle_least_squares(circle_pts)
        self.assert_true(center_fit is not None, msg="Circle fit exists")
        self.assert_almost_equal(center_fit.x, 1.0, msg="Circle fit center x")
        self.assert_almost_equal(center_fit.y, 1.0, msg="Circle fit center y")
        self.assert_almost_equal(radius_fit, 1.0, msg="Circle fit radius")

        # 2.15 直线拟合
        line_pts = np.array([
            [i, 2*i + 3 + np.random.normal(0, 0.1)]
            for i in range(10)
        ])
        params, inliers = fit_line_ransac(line_pts, distance_threshold=0.5)
        self.assert_true(params is not None, msg="Line RANSAC fit exists")
        self.assert_true(np.sum(inliers) >= 8, msg="Line RANSAC inliers > 8")

        # 2.16 坐标归一化
        vertices = {
            "V1": Vertex(id="V1", position=Point(0, 0)),
            "V2": Vertex(id="V2", position=Point(100, 100)),
        }
        norm_vertices, scale, offset = normalize_coordinates(vertices, target_range=5.0)
        self.assert_true(scale > 0, msg="Normalization scale > 0")
        if "V1" in norm_vertices and norm_vertices["V1"].normalized_pos:
            self.assert_almost_equal(
                norm_vertices["V1"].normalized_pos.x, -2.5, msg="Normalized V1 x"
            )

        # 2.17 曲率计算
        curv = compute_curvature(straight_pts)
        self.assert_true(all(c < 0.1 for c in curv), msg="Curvature (straight) < 0.1")

        # 2.18 直线-圆弧交点
        p1, p2 = Point(0, 0), Point(2, 0)
        cx, r = Point(0, 0), 1.0
        intersections = line_arc_intersection(p1, p2, cx, r, 0, math.pi)
        self.assert_true(len(intersections) > 0, msg="Line-arc intersection exists")

    # ============================================================
    # 测试3: 合成图像生成与识别
    # ============================================================

    def test_synthetic_image_pipeline(self):
        print("\n" + "=" * 60)
        print("测试3: 合成图像端到端识别")
        print("=" * 60)

        # 3.1 生成合成测试图像
        image = self._create_synthetic_geometry()
        self.assert_true(image is not None, msg="Synthetic image created")
        self.assert_true(image.shape[0] > 0, msg="Image has height")

        # 3.2 配置
        config = GeometryConfig()
        config.enable_perspective_correction = False
        config.enable_dbscan_clustering = True
        config.enable_virtual_vertices = True
        config.enable_triple_line_verification = True
        config.enable_full_intersection = True
        config.enable_geometric_regularization = True
        config.enable_latex_output = True
        config.thresholds.image_width = image.shape[1]
        config.thresholds.image_height = image.shape[0]

        # 3.3 运行流水线
        pipeline = GeometryRecognizerPipeline(config)
        result = pipeline.process(image)

        # 3.4 验证结果
        self.assert_true(result.success, msg=f"Pipeline success: {result.error_message}")

        # 3.5 验证顶点
        vertex_count = len(result.vertices)
        print(f"  检测到顶点数: {vertex_count}")
        self.assert_true(vertex_count >= 3,
                         msg=f"Vertex count >= 3 (got {vertex_count})")

        # 3.6 验证图元
        prim_count = len(result.primitives)
        print(f"  检测到图元数: {prim_count}")
        self.assert_true(prim_count >= 1,
                         msg=f"Primitive count >= 1 (got {prim_count})")

        # 3.7 验证分层存储
        contour_count = len(result.contours)
        aux_count = len(result.auxiliaries)
        print(f"  轮廓层: {contour_count}, 辅助层: {aux_count}")
        self.assert_true(contour_count + aux_count > 0,
                         msg="Total primitives > 0")

        # 3.8 验证LaTeX代码
        if result.latex_code:
            print(f"  LaTeX代码长度: {len(result.latex_code)} 字符")
            self.assert_true("\\documentclass" in result.latex_code,
                             msg="LaTeX has documentclass")
            self.assert_true("\\begin{tikzpicture}" in result.latex_code,
                             msg="LaTeX has tikzpicture")
            self.assert_true("\\end{tikzpicture}" in result.latex_code,
                             msg="LaTeX has end tikzpicture")
            self.assert_true("\\coordinate" in result.latex_code,
                             msg="LaTeX has coordinate definitions")
            self.assert_true("\\draw" in result.latex_code,
                             msg="LaTeX has draw commands")

            # 输出LaTeX代码片段
            print(f"\n  ---- LaTeX代码片段 (前600字符) ----")
            print(result.latex_code[:600])
            print("  ---- 结束 ----")
        else:
            self.assert_true(False, msg="LaTeX code generated")

        # 3.9 导出结果字典
        result_dict = result.to_dict()
        self.assert_true("vertices" in result_dict, msg="Result dict has vertices")
        self.assert_true("primitives" in result_dict, msg="Result dict has primitives")

    def _create_synthetic_geometry(self) -> np.ndarray:
        """
        创建合成几何测试图像

        包含: 三角形 + 圆 + 直线
        """
        h, w = 400, 600
        image = np.ones((h, w), dtype=np.uint8) * 255

        # 三角形 A(100, 300), B(300, 300), C(200, 100)
        triangle = np.array([[100, 300], [300, 300], [200, 100]], dtype=np.int32)
        cv2 = __import__('cv2')
        cv2.polylines(image, [triangle], isClosed=True, color=0, thickness=2)

        # 圆 圆心(450, 200), 半径80
        cv2.circle(image, (450, 200), 80, 0, thickness=2)

        # 标注文字
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, 'A', (85, 315), font, 0.6, 0, 1)
        cv2.putText(image, 'B', (305, 315), font, 0.6, 0, 1)
        cv2.putText(image, 'C', (195, 85), font, 0.6, 0, 1)

        # 添加少量噪声
        noise = np.random.randint(0, 30, image.shape, dtype=np.uint8)
        image = cv2.add(image, noise)

        # 转为3通道BGR
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        return image_bgr

    # ============================================================
    # 测试4: 人机交互接口
    # ============================================================

    def test_manual_interaction(self):
        print("\n" + "=" * 60)
        print("测试4: 人机交互接口")
        print("=" * 60)

        config = GeometryConfig()
        config.enable_latex_output = True
        pipeline = GeometryRecognizerPipeline(config)

        # 创建一个空结果
        result = GeometryResult()
        result.image_shape = (100, 100)
        result.success = True

        # 4.1 手动添加顶点
        result = pipeline.manual_add_vertex(result, Point(10, 20), label="A")
        self.assert_true("V1" in result.vertices, msg="Manual add vertex exists")
        self.assert_almost_equal(result.vertices["V1"].position.x, 10.0,
                                 msg="Manual add vertex x")
        self.assert_true(result.vertices["V1"].label == "A", msg="Manual add label")

        # 4.2 手动编辑顶点
        result = pipeline.manual_edit_vertex(result, "V1", Point(30, 40))
        self.assert_almost_equal(result.vertices["V1"].position.x, 30.0,
                                 msg="Manual edit vertex x")
        self.assert_true(result.vertices["V1"].source == ConfidenceSource.MANUAL,
                         msg="Manual edit source")

        # 4.3 手动添加图元
        result = pipeline.manual_add_vertex(result, Point(50, 60), label="B")
        result = pipeline.manual_add_primitive(
            result, PrimitiveType.LINE_SEGMENT, ["V1", "V2"]
        )
        self.assert_true(len(result.primitives) == 1, msg="Manual add primitive")

        # 4.4 手动删除顶点
        result = pipeline.manual_delete_vertex(result, "V2")
        self.assert_true("V2" not in result.vertices, msg="Manual delete vertex")

        # 4.5 LaTeX生成
        self.assert_true(result.latex_code is not None, msg="Manual edit LaTeX")

    # ============================================================
    # 测试5: 自适应阈值
    # ============================================================

    def test_adaptive_thresholds(self):
        print("\n" + "=" * 60)
        print("测试5: 自适应阈值")
        print("=" * 60)

        config = GeometryConfig()
        config.update_image_size(2000, 1500)

        # 5.1 分辨率缩放
        scale = config.thresholds.get_resolution_scale()
        self.assert_true(scale > 0, msg=f"Resolution scale = {scale:.2f}")

        # 5.2 DBSCAN eps
        eps = config.thresholds.dbscan_eps
        self.assert_true(eps > 0, msg=f"DBSCAN eps = {eps:.2f}")

        # 5.3 伸长率阈值
        short_th = config.thresholds.get_elongation_threshold(30)
        long_th = config.thresholds.get_elongation_threshold(300)
        self.assert_true(short_th >= long_th,
                         msg="Short line threshold >= long line threshold")

        # 5.4 角度一致性阈值
        solid_th = config.thresholds.get_angle_consistency_threshold("solid")
        dashed_th = config.thresholds.get_angle_consistency_threshold("dashed")
        self.assert_true(dashed_th >= solid_th,
                         msg="Dashed line threshold >= solid")

        # 5.5 圆弧角度阈值
        short_arc_th = config.thresholds.get_arc_angle_threshold(20, 100)
        long_arc_th = config.thresholds.get_arc_angle_threshold(80, 100)
        self.assert_true(short_arc_th >= long_arc_th,
                         msg="Short arc threshold >= long arc threshold")

    # ============================================================
    # 运行所有测试
    # ============================================================

    def run_all(self):
        """运行所有测试"""
        print("=" * 60)
        print("几何识别算法综合测验")
        print("=" * 60)

        self.test_data_structures()
        self.test_geometry_utils()
        self.test_adaptive_thresholds()
        self.test_synthetic_image_pipeline()
        self.test_manual_interaction()

        # 汇总
        print("\n" + "=" * 60)
        print("测验结果汇总")
        print("=" * 60)
        print(f"  通过: {self.passed}")
        print(f"  失败: {self.failed}")
        print(f"  总计: {self.passed + self.failed}")

        if self.failed > 0:
            print("\n失败详情:")
            for err in self.errors:
                print(f"  - {err}")

        success_rate = self.passed / max(self.passed + self.failed, 1) * 100
        print(f"\n  通过率: {success_rate:.1f}%")

        return self.failed == 0


if __name__ == "__main__":
    tester = TestGeometryRecognizer()
    success = tester.run_all()
    sys.exit(0 if success else 1)