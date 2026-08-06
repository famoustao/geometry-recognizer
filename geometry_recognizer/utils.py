"""
工具函数模块

提供各类数学计算、几何运算、图像处理等辅助函数。
"""

import numpy as np
import math
from typing import List, Tuple, Optional, Union
from .data_structures import Point, Vertex, LineType


# ============================================================
# 数学工具函数
# ============================================================

def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """计算两个向量之间的夹角 (弧度)"""
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm < 1e-10:
        return 0.0
    cos_a = np.clip(dot / norm, -1.0, 1.0)
    return math.acos(cos_a)


def angle_between_degrees(v1: np.ndarray, v2: np.ndarray) -> float:
    """计算两个向量之间的夹角 (度)"""
    return math.degrees(angle_between(v1, v2))


def line_angle(p1: Point, p2: Point) -> float:
    """计算线段的方向角 (弧度, [-π, π])"""
    return math.atan2(p2.y - p1.y, p2.x - p1.x)


def line_angle_degrees(p1: Point, p2: Point) -> float:
    """计算线段的方向角 (度)"""
    return math.degrees(line_angle(p1, p2))


def point_line_distance(p: Point, a: Point, b: Point) -> float:
    """点到直线的距离"""
    return abs(np.cross(b.to_array() - a.to_array(),
                        a.to_array() - p.to_array())) / \
           max(np.linalg.norm(b.to_array() - a.to_array()), 1e-10)


def point_line_segment_distance(p: Point, a: Point, b: Point) -> float:
    """点到线段的最短距离"""
    ab = b.to_array() - a.to_array()
    ap = p.to_array() - a.to_array()
    t = np.dot(ap, ab) / max(np.dot(ab, ab), 1e-10)
    t = np.clip(t, 0, 1)
    proj = a.to_array() + t * ab
    return float(np.linalg.norm(p.to_array() - proj))


def line_intersection(p1: Point, p2: Point, p3: Point, p4: Point) -> Optional[Point]:
    """计算两条直线 (p1-p2, p3-p4) 的交点"""
    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y
    x3, y3 = p3.x, p3.y
    x4, y4 = p4.x, p4.y

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None  # 平行或重合

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    return Point(x, y)


def line_arc_intersection(
    p1: Point, p2: Point,
    center: Point, radius: float,
    start_angle: float, end_angle: float
) -> List[Point]:
    """
    计算直线与圆弧的交点

    Args:
        p1, p2: 直线上的两点
        center: 圆弧圆心
        radius: 圆弧半径
        start_angle, end_angle: 圆弧起始/终止角度 (弧度)

    Returns:
        交点列表 (在圆弧上的点)
    """
    # 直线参数方程: P = p1 + t * (p2 - p1)
    # 代入圆方程: |P - center|^2 = radius^2
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    fx = p1.x - center.x
    fy = p1.y - center.y

    a = dx*dx + dy*dy
    b = 2 * (fx*dx + fy*dy)
    c = fx*fx + fy*fy - radius*radius

    if abs(a) < 1e-10:
        return []

    disc = b*b - 4*a*c
    if disc < 0:
        return []

    sqrt_disc = math.sqrt(disc)
    t1 = (-b + sqrt_disc) / (2*a)
    t2 = (-b - sqrt_disc) / (2*a)

    intersections = []
    for t in (t1, t2):
        x = p1.x + t * dx
        y = p1.y + t * dy
        pt = Point(x, y)

        # 检查是否在圆弧上
        angle = math.atan2(y - center.y, x - center.x)
        if is_angle_in_arc(angle, start_angle, end_angle):
            intersections.append(pt)

    return intersections


def arc_arc_intersection(
    c1: Point, r1: float, a1_start: float, a1_end: float,
    c2: Point, r2: float, a2_start: float, a2_end: float
) -> List[Point]:
    """
    计算两个圆弧的交点

    Returns:
        交点列表
    """
    d = c1.distance_to(c2)

    # 无交点或重合
    if d > r1 + r2 + 1e-6 or d < abs(r1 - r2) - 1e-6 or d < 1e-10:
        return []

    # 计算交点
    a = (r1*r1 - r2*r2 + d*d) / (2*d)
    h = math.sqrt(max(0, r1*r1 - a*a))

    x0 = c1.x + a * (c2.x - c1.x) / d
    y0 = c1.y + a * (c2.y - c1.y) / d

    rx = -h * (c2.y - c1.y) / d
    ry = h * (c2.x - c1.x) / d

    pts = []
    for sign in (1, -1):
        pt = Point(x0 + sign * rx, y0 + sign * ry)
        angle1 = math.atan2(pt.y - c1.y, pt.x - c1.x)
        angle2 = math.atan2(pt.y - c2.y, pt.x - c2.x)
        if is_angle_in_arc(angle1, a1_start, a1_end) and \
           is_angle_in_arc(angle2, a2_start, a2_end):
            pts.append(pt)

    return pts


def is_angle_in_arc(angle: float, start: float, end: float) -> bool:
    """判断角度是否在圆弧区间内 (处理跨0度的情况)"""
    if start <= end:
        return start - 0.01 <= angle <= end + 0.01
    else:
        return angle >= start - 0.01 or angle <= end + 0.01


def perpendicular_foot(p: Point, a: Point, b: Point) -> Point:
    """计算点P到直线AB的垂足"""
    ab = b.to_array() - a.to_array()
    ap = p.to_array() - a.to_array()
    t = np.dot(ap, ab) / max(np.dot(ab, ab), 1e-10)
    foot = a.to_array() + t * ab
    return Point(float(foot[0]), float(foot[1]))


def midpoint(p1: Point, p2: Point) -> Point:
    """计算线段中点"""
    return Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)


def angle_bisector_point(p1: Point, vertex: Point, p2: Point, length: float = 1.0) -> Point:
    """
    计算角平分线上一点

    Args:
        p1, vertex, p2: 角的两边和顶点
        length: 角平分线长度

    Returns:
        角平分线上距顶点length的点
    """
    v1 = (p1.to_array() - vertex.to_array())
    v2 = (p2.to_array() - vertex.to_array())
    v1_norm = v1 / max(np.linalg.norm(v1), 1e-10)
    v2_norm = v2 / max(np.linalg.norm(v2), 1e-10)
    bisector = v1_norm + v2_norm
    if np.linalg.norm(bisector) < 1e-10:
        # 180度角，取垂直方向
        bisector = np.array([-v1_norm[1], v1_norm[0]])
    bisector = bisector / max(np.linalg.norm(bisector), 1e-10) * length
    return Point(vertex.x + bisector[0], vertex.y + bisector[1])


def circle_center(p1: Point, p2: Point, p3: Point) -> Optional[Point]:
    """三点确定圆心"""
    d = 2 * (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))
    if abs(d) < 1e-10:
        return None
    ux = ((p1.x**2 + p1.y**2) * (p2.y - p3.y) +
          (p2.x**2 + p2.y**2) * (p3.y - p1.y) +
          (p3.x**2 + p3.y**2) * (p1.y - p2.y)) / d
    uy = ((p1.x**2 + p1.y**2) * (p3.x - p2.x) +
          (p2.x**2 + p2.y**2) * (p1.x - p3.x) +
          (p3.x**2 + p3.y**2) * (p2.x - p1.x)) / d
    return Point(ux, uy)


def centroid(points: List[Point]) -> Point:
    """计算点集的质心"""
    if not points:
        return Point(0, 0)
    x = sum(p.x for p in points) / len(points)
    y = sum(p.y for p in points) / len(points)
    return Point(x, y)


def triangle_centroid(a: Point, b: Point, c: Point) -> Point:
    """三角形重心"""
    return Point((a.x + b.x + c.x) / 3, (a.y + b.y + c.y) / 3)


def triangle_circumcenter(a: Point, b: Point, c: Point) -> Optional[Point]:
    """三角形外心"""
    return circle_center(a, b, c)


def triangle_orthocenter(a: Point, b: Point, c: Point) -> Point:
    """三角形垂心"""
    # 垂心 = 三顶点坐标和 - 2 * 外心坐标
    circum = triangle_circumcenter(a, b, c)
    if circum is None:
        return centroid([a, b, c])
    return Point(a.x + b.x + c.x - 2 * circum.x,
                 a.y + b.y + c.y - 2 * circum.y)


def triangle_incenter(a: Point, b: Point, c: Point) -> Point:
    """三角形内心"""
    la = a.distance_to(b)
    lb = b.distance_to(c)
    lc = c.distance_to(a)
    perimeter = la + lb + lc
    if perimeter < 1e-10:
        return centroid([a, b, c])
    return Point((lb * a.x + lc * b.x + la * c.x) / perimeter,
                 (lb * a.y + lc * b.y + la * c.y) / perimeter)


def fit_line_ransac(points: np.ndarray, distance_threshold: float = 2.0,
                    max_iterations: int = 100) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    RANSAC直线拟合

    Args:
        points: Nx2 点集
        distance_threshold: 内点距离阈值
        max_iterations: 最大迭代次数

    Returns:
        (line_params, inlier_mask)
        line_params: [斜率, 截距] 或 None (失败)
        inlier_mask: 内点布尔掩码
    """
    n = len(points)
    if n < 2:
        return None, np.zeros(n, dtype=bool)

    best_inliers = np.zeros(n, dtype=bool)
    best_params = None
    best_count = 0

    # 使用确定的采样策略
    num_trials = min(max_iterations, n * (n - 1) // 2)

    for _ in range(num_trials):
        # 随机选择两点
        idx = np.random.choice(n, 2, replace=False)
        p1, p2 = points[idx[0]], points[idx[1]]

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        if abs(dx) < 1e-10:
            # 垂直线
            slope = float('inf')
            intercept = p1[0]
            distances = np.abs(points[:, 0] - intercept)
        else:
            slope = dy / dx
            intercept = p1[1] - slope * p1[0]
            distances = np.abs(points[:, 1] - (slope * points[:, 0] + intercept))

        inliers = distances < distance_threshold
        count = np.sum(inliers)

        if count > best_count:
            best_count = count
            best_inliers = inliers
            if abs(dx) < 1e-10:
                best_params = np.array([float('inf'), intercept])
            else:
                best_params = np.array([slope, intercept])

    # 用所有内点重新拟合 (正交最小二乘)
    if best_count >= 2:
        inlier_pts = points[best_inliers]
        best_params = fit_line_orthogonal(inlier_pts)

    return best_params, best_inliers


def fit_line_orthogonal(points: np.ndarray) -> np.ndarray:
    """
    正交最小二乘直线拟合

    Returns:
        [斜率, 截距] 或 [inf, x_intercept] (垂直线)
    """
    if len(points) < 2:
        return np.array([0.0, 0.0])

    # 使用PCA
    mean = np.mean(points, axis=0)
    centered = points - mean
    cov = np.cov(centered.T)

    if np.linalg.matrix_rank(cov) < 2:
        # 所有点共线，使用普通最小二乘
        return fit_line_least_squares(points)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # 主成分方向
    main_direction = eigenvectors[:, 1]  # 最大特征值对应的特征向量

    if abs(main_direction[0]) < 1e-10:
        # 垂直线
        return np.array([float('inf'), mean[0]])

    slope = main_direction[1] / main_direction[0]
    intercept = mean[1] - slope * mean[0]
    return np.array([slope, intercept])


def fit_line_least_squares(points: np.ndarray) -> np.ndarray:
    """普通最小二乘直线拟合"""
    x, y = points[:, 0], points[:, 1]
    A = np.vstack([x, np.ones_like(x)]).T
    try:
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        return np.array([slope, intercept])
    except np.linalg.LinAlgError:
        return np.array([0.0, 0.0])


def fit_circle_least_squares(points: np.ndarray) -> Tuple[Optional[Point], float]:
    """
    最小二乘圆拟合

    Returns:
        (center, radius) 或 (None, 0)
    """
    if len(points) < 3:
        return None, 0

    x, y = points[:, 0], points[:, 1]

    # 代数拟合方法
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x**2 + y**2)

    try:
        result = np.linalg.lstsq(A, b, rcond=None)[0]
        cx = -result[0] / 2
        cy = -result[1] / 2
        r = math.sqrt(max(0, cx**2 + cy**2 - result[2]))

        if r > 0 and not (math.isnan(cx) or math.isnan(cy) or math.isnan(r)):
            return Point(cx, cy), r
    except np.linalg.LinAlgError:
        pass

    return None, 0


def fit_ellipse_least_squares(points: np.ndarray) -> Optional[dict]:
    """
    最小二乘椭圆拟合

    Returns:
        {center, a, b, angle} 或 None
    """
    if len(points) < 5:
        return None

    try:
        x, y = points[:, 0].astype(np.float64), points[:, 1].astype(np.float64)
        # 构建设计矩阵
        D = np.column_stack([x*x, x*y, y*y, x, y, np.ones_like(x)])

        # 约束矩阵
        S = np.zeros((6, 6))
        S[0, 2] = 2
        S[2, 0] = 2
        S[1, 1] = -1

        # 广义特征值求解
        try:
            DT_D = D.T @ D
            S_inv = np.linalg.pinv(S)
            DT_D_S_inv = DT_D @ S_inv
            eigvals, eigvecs = np.linalg.eig(DT_D_S_inv)

            # 找到最小正特征值对应的特征向量
            min_pos_idx = np.argmin(np.abs(eigvals))
            params = np.real(eigvecs[:, min_pos_idx])
        except np.linalg.LinAlgError:
            return None

        a_val, b_val, c_val, d_val, e_val, f_val = params

        # 验证椭圆参数
        disc = b_val**2 - 4*a_val*c_val
        if disc >= 0 or abs(a_val) < 1e-10 or abs(c_val) < 1e-10:
            return None

        # 计算中心
        cx = (2*c_val*d_val - b_val*e_val) / (b_val**2 - 4*a_val*c_val)
        cy = (2*a_val*e_val - b_val*d_val) / (b_val**2 - 4*a_val*c_val)

        # 计算轴长和角度
        M = np.array([[a_val, b_val/2],
                      [b_val/2, c_val]])
        eigvals_M, eigvecs_M = np.linalg.eig(M)

        angle = math.atan2(eigvecs_M[1, 0], eigvecs_M[0, 0])

        # 计算半轴长
        F = f_val + (d_val**2 * c_val - b_val*d_val*e_val + a_val*e_val**2) / \
            (b_val**2 - 4*a_val*c_val)
        F = np.real(F)

        if F >= 0 or abs(eigvals_M[0]) < 1e-10 or abs(eigvals_M[1]) < 1e-10:
            return None

        a_axis = math.sqrt(abs(F / eigvals_M[0]))
        b_axis = math.sqrt(abs(F / eigvals_M[1]))

        if a_axis < 1 or b_axis < 1:
            return None

        center = Point(float(cx), float(cy))
        return {
            'center': center,
            'a': float(max(a_axis, b_axis)),
            'b': float(min(a_axis, b_axis)),
            'angle': float(angle % math.pi),
        }

    except Exception:
        return None


def resample_contour(contour: np.ndarray, num_points: int = 100) -> np.ndarray:
    """等距重采样轮廓点"""
    if len(contour) < 2:
        return contour

    # 计算累积弧长
    diffs = np.diff(contour, axis=0)
    distances = np.sqrt(np.sum(diffs**2, axis=1))
    cum_dist = np.concatenate([[0], np.cumsum(distances)])
    total_length = cum_dist[-1]

    if total_length < 1e-10:
        return contour

    # 等距采样
    uniform_dist = np.linspace(0, total_length, num_points)
    return np.array([
        contour[np.searchsorted(cum_dist, d) - 1] if d > 0 else contour[0]
        for d in uniform_dist
    ])


def compute_curvature(points: np.ndarray) -> np.ndarray:
    """
    计算轮廓曲率

    Args:
        points: Nx2 点集

    Returns:
        曲率数组 (长度 N)
    """
    if len(points) < 3:
        return np.zeros(len(points))

    n = len(points)
    curvature = np.zeros(n)

    for i in range(1, n - 1):
        p_prev = points[i - 1]
        p_curr = points[i]
        p_next = points[i + 1]

        v1 = p_curr - p_prev
        v2 = p_next - p_curr

        angle = abs(math.atan2(np.cross(v1, v2), np.dot(v1, v2)))
        dist = np.linalg.norm(p_prev - p_next)

        if dist > 1e-6:
            curvature[i] = 2 * math.sin(angle) / dist

    return curvature


def elongate_ratio(points: np.ndarray) -> float:
    """
    计算伸长率 (端点距离 / 路径长度)

    Returns:
        伸长率 (<=1, 越接近1越直)
    """
    if len(points) < 2:
        return 1.0

    end_dist = np.linalg.norm(points[0] - points[-1])
    path_length = 0
    for i in range(1, len(points)):
        path_length += np.linalg.norm(points[i] - points[i-1])

    if path_length < 1e-10:
        return 1.0

    return end_dist / path_length


def detect_line_pattern(points: np.ndarray) -> LineType:
    """
    检测线条纹理类型 (实线/虚线/点划线/射线)

    基于像素间隔模式分析

    Args:
        points: Nx2 骨架点集

    Returns:
        LineType 枚举值
    """
    if len(points) < 2:
        return LineType.SOLID

    # 计算连续点之间的间隔
    diffs = np.diff(points, axis=0)
    gaps = np.sqrt(np.sum(diffs**2, axis=1))

    if len(gaps) < 5:
        return LineType.SOLID

    # 检测间隔模式
    mean_gap = np.mean(gaps)
    std_gap = np.std(gaps)

    # 如果有较大的间隙变化，可能是虚线
    if std_gap > mean_gap * 0.5:
        # 检测是否有重复的"有-无-有"模式
        gap_ratio = gaps / max(mean_gap, 1e-10)
        large_gaps = np.sum(gap_ratio > 2.0)
        if large_gaps > len(gaps) * 0.2:
            return LineType.DASHED

    return LineType.SOLID


def estimate_line_width(binary_image: np.ndarray) -> int:
    """估计图像中线条的平均宽度 (像素)"""
    from scipy.ndimage import distance_transform_edt

    # 距离变换
    dist = distance_transform_edt(binary_image)
    # 提取骨架上的距离值
    skeleton = np.zeros_like(binary_image)
    from skimage.morphology import skeletonize
    try:
        skeleton = skeletonize(binary_image > 0)
    except Exception:
        pass

    if np.sum(skeleton) > 0:
        widths = dist[skeleton > 0] * 2
        if len(widths) > 0:
            return max(1, int(np.median(widths)))

    return 3


def normalize_coordinates(
    vertices: dict,
    target_range: float = 5.0,
    margin: float = 0.5
) -> Tuple[dict, float, Point]:
    """
    坐标归一化

    Args:
        vertices: 顶点字典 {id: Vertex}
        target_range: 目标坐标范围 [-target_range, target_range]
        margin: 边距比例

    Returns:
        (更新后的顶点字典, 缩放因子, 偏移量)
    """
    if not vertices:
        return vertices, 1.0, Point(0, 0)

    positions = np.array([(v.position.x, v.position.y) for v in vertices.values()])
    min_x, min_y = positions.min(axis=0)
    max_x, max_y = positions.max(axis=0)

    # 计算中心和范围
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    range_x = max_x - min_x
    range_y = max_y - min_y

    if range_x < 1e-10 and range_y < 1e-10:
        return vertices, 1.0, Point(0, 0)

    # 缩放因子
    scale = 2 * target_range * (1 - margin) / max(range_x, range_y, 1e-10)

    # 偏移量
    offset = Point(-center_x, -center_y)

    # 归一化
    for v in vertices.values():
        nx = (v.position.x + offset.x) * scale
        ny = (v.position.y + offset.y) * scale
        v.normalized_pos = Point(nx, ny)

    return vertices, scale, offset


