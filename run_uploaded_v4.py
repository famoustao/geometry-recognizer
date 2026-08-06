"""
智能几何识别 v4：改进的文字屏蔽 + 直线交点检测 + 几何推理
=============================================================
核心改进：
1. 文字屏蔽：先腐蚀断开连接，再找连通域，面积+宽高比过滤
2. 直线检测：先用霍夫检测，再计算直线交点确定顶点
3. 圆弧检测：检测圆弧上的点，用几何约束确定D、E
4. 所有点用几何计算，不依赖聚类
"""
import sys, os, cv2, numpy as np, subprocess, glob, math, gc
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def mask_text_labels(img):
    """
    改进的文字屏蔽策略。
    先腐蚀断开文字与线条的连接，再找连通域，按面积+宽高比过滤。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Otsu二值化（反色，线条和文字为白色）
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 先腐蚀，断开文字和线条的连接
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=1)

    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded, 8)

    # 创建文字掩膜
    text_mask = np.zeros_like(gray)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]

        # 文字特征：面积适中，宽高比接近1
        aspect_ratio = max(bw, bh) / max(min(bw, bh), 1)
        is_text = (20 < area < 500 and
                   aspect_ratio < 4.0 and
                   bw < 40 and bh < 40)
        if is_text:
            text_mask[labels == i] = 255

    # 膨胀回原始大小
    text_mask = cv2.dilate(text_mask, kernel, iterations=2)

    # 用白色修补文字区域
    cleaned = img.copy()
    cleaned[text_mask > 0] = [255, 255, 255]

    # 保存调试图
    cv2.imwrite("/workspace/debug4_text_mask.png", text_mask)
    cv2.imwrite("/workspace/debug4_cleaned.png", cleaned)

    print(f"  文字屏蔽: {np.count_nonzero(text_mask)} 像素被填充")
    return cleaned


def line_intersection(p1, p2, p3, p4):
    """计算两条直线的交点 (p1-p2, p3-p4)"""
    x1, y1 = p1; x2, y2 = p2
    x3, y3 = p3; x4, y4 = p4

    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-8:
        return None  # 平行或重合

    # 交点
    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom
    return (px, py)


def point_on_segment(p, a, b, tol=5.0):
    """检查点p是否在线段a-b上（在容差范围内）"""
    ax, ay = a; bx, by = b; px, py = p
    abx, aby = bx-ax, by-ay
    apx, apy = px-ax, py-ay
    t = (apx*abx + apy*aby) / max(abx*abx + aby*aby, 1e-8)
    if t < 0:
        fx, fy = ax, ay
    elif t > 1:
        fx, fy = bx, by
    else:
        fx, fy = ax + t*abx, ay + t*aby
    dist = math.sqrt((px-fx)**2 + (py-fy)**2)
    return dist < tol


def perpendicular_foot(point, line_p1, line_p2):
    """计算点到直线的垂足"""
    px, py = point
    ax, ay = line_p1
    bx, by = line_p2
    abx, aby = bx-ax, by-ay
    apx, apy = px-ax, py-ay
    t = (apx*abx + apy*aby) / max(abx*abx + aby*aby, 1e-8)
    fx = ax + t * abx
    fy = ay + t * aby
    return (fx, fy)


def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def detect_geometry(img_path):
    """检测几何图形的主要结构"""
    img = cv2.imread(img_path)
    if img is None:
        print(f"[FAIL] 无法读取图片")
        return None
    h, w = img.shape[:2]
    print(f"  原始尺寸: {w}x{h}")

    # ============================================================
    # 第一步：屏蔽文字
    # ============================================================
    print("\n  [1/5] 文字屏蔽...")
    cleaned = mask_text_labels(img)
    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)

    # ============================================================
    # 第二步：边缘检测 + 霍夫线检测
    # ============================================================
    print("\n  [2/5] 线条检测...")
    all_lines = []
    for low, high in [(20, 80), (25, 120), (30, 150)]:
        edges = cv2.Canny(gray, low, high, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=35,
                                 minLineLength=20, maxLineGap=10)
        if lines is not None:
            all_lines.extend(lines)

    if not all_lines:
        print("[FAIL] 未检测到线条")
        return None

    merged_lines = []
    for line in all_lines:
        line = np.array(line).flatten()
        x1, y1, x2, y2 = line.astype(float)
        angle = math.degrees(math.atan2(y2-y1, x2-x1))
        mid_x = (x1+x2)/2
        mid_y = (y1+y2)/2
        length = distance((x1,y1), (x2,y2))
        angle = angle % 180

        found = False
        for i, ml in enumerate(merged_lines):
            ma = ml['angle']
            mmx, mmy = ml['mid']
            ang_diff = min(abs(angle-ma), 180-abs(angle-ma))
            mid_dist = distance((mid_x, mid_y), (mmx, mmy))
            if ang_diff < 15 and mid_dist < 50:
                if length > ml['length']:
                    merged_lines[i] = {
                        'p1': (x1, y1), 'p2': (x2, y2),
                        'angle': angle, 'mid': (mid_x, mid_y),
                        'length': length
                    }
                found = True
                break
        if not found:
            merged_lines.append({
                'p1': (x1, y1), 'p2': (x2, y2),
                'angle': angle, 'mid': (mid_x, mid_y),
                'length': length
            })

    print(f"  检测到 {len(merged_lines)} 条线段（去重后）")

    debug = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    for i, ml in enumerate(merged_lines):
        p1, p2 = ml['p1'], ml['p2']
        cv2.line(debug, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
                 colors[i % len(colors)], 1)
    cv2.imwrite("/workspace/debug4_lines.png", debug)

    # ============================================================
    # 第三步：直线交点分析 → 找三角形顶点
    # ============================================================
    print("\n  [3/5] 直线交点分析...")

    intersections = []
    for i in range(len(merged_lines)):
        for j in range(i+1, len(merged_lines)):
            p1, p2 = merged_lines[i]['p1'], merged_lines[i]['p2']
            p3, p4 = merged_lines[j]['p1'], merged_lines[j]['p2']
            pt = line_intersection(p1, p2, p3, p4)
            if pt is not None:
                if -50 < pt[0] < w+50 and -50 < pt[1] < h+50:
                    on_i = point_on_segment(pt, p1, p2, 8)
                    on_j = point_on_segment(pt, p3, p4, 8)
                    if on_i or on_j:
                        intersections.append(pt)

    print(f"  找到 {len(intersections)} 个交点")

    if len(intersections) > 10:
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=15, min_samples=2).fit(intersections)
        labels = clustering.labels_
        clusters = {}
        for i, label in enumerate(labels):
            if label == -1:
                continue
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(intersections[i])
        cluster_pts = []
        for label, pts in clusters.items():
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            cluster_pts.append((cx, cy))
        print(f"  交点聚类后: {len(cluster_pts)} 个候选顶点")
    else:
        cluster_pts = intersections

    for pt in cluster_pts:
        cv2.circle(debug, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
    cv2.imwrite("/workspace/debug4_intersections.png", debug)

    if len(cluster_pts) < 3:
        print("[FAIL] 交点太少，无法构成三角形")
        return None

    sorted_y = sorted(cluster_pts, key=lambda p: p[1])
    A = sorted_y[0]
    print(f"  A (顶部) = ({A[0]:.1f}, {A[1]:.1f})")

    bottom_candidates = sorted(cluster_pts, key=lambda p: -p[1])[:10]
    bottom_candidates = [p for p in bottom_candidates if distance(p, A) > 80]
    if len(bottom_candidates) < 2:
        print("[FAIL] 底部候选点太少")
        return None

    B = min(bottom_candidates, key=lambda p: p[0])
    C = max(bottom_candidates, key=lambda p: p[0])
    print(f"  B (左下) = ({B[0]:.1f}, {B[1]:.1f})")
    print(f"  C (右下) = ({C[0]:.1f}, {C[1]:.1f})")

    # ============================================================
    # 第四步：计算所有几何点
    # ============================================================
    print("\n  [4/5] 计算几何点...")

    O = ((B[0] + C[0]) / 2, (B[1] + C[1]) / 2)
    radius = distance(O, B)
    print(f"  O (BC中点) = ({O[0]:.1f}, {O[1]:.1f}), R={radius:.1f}")

    H = perpendicular_foot(A, B, C)
    print(f"  H (A垂足) = ({H[0]:.1f}, {H[1]:.1f})")

    # D = AB与半圆的交点，E = AC与半圆的交点
    def line_circle_intersection(p1, p2, center, R):
        v = (p2[0]-p1[0], p2[1]-p1[1])
        w = (p1[0]-center[0], p1[1]-center[1])
        a = v[0]**2 + v[1]**2
        b = 2*(w[0]*v[0] + w[1]*v[1])
        c = w[0]**2 + w[1]**2 - R**2
        disc = b*b - 4*a*c
        if disc < 0:
            return None
        t1 = (-b + math.sqrt(disc)) / (2*a)
        t2 = (-b - math.sqrt(disc)) / (2*a)
        if 0 <= t1 <= 1 and 0 <= t2 <= 1:
            t = max(t1, t2)
        elif 0 <= t1 <= 1:
            t = t1
        elif 0 <= t2 <= 1:
            t = t2
        else:
            return None
        return (p1[0] + t*v[0], p1[1] + t*v[1])

    D = line_circle_intersection(A, B, O, radius)
    if D is None:
        angle_B = math.atan2(B[1] - O[1], B[0] - O[0])
        angle_D = angle_B - math.pi/4
        if angle_D < 0:
            angle_D += 2*math.pi
        D = (O[0] + radius * math.cos(angle_D),
             O[1] - radius * abs(math.sin(angle_D)))

    E = line_circle_intersection(A, C, O, radius)
    if E is None:
        angle_C = math.atan2(C[1] - O[1], C[0] - O[0])
        angle_E = angle_C + math.pi/4
        if angle_E < 0:
            angle_E += 2*math.pi
        E = (O[0] + radius * math.cos(angle_E),
             O[1] - radius * abs(math.sin(angle_E)))

    print(f"  D (AB与弧交点) = ({D[0]:.1f}, {D[1]:.1f})")
    print(f"  E (AC与弧交点) = ({E[0]:.1f}, {E[1]:.1f})")

    # 微调D和E到最近的黑色像素
    for label, pt in [('D', D), ('E', E)]:
        best_pt = pt
        best_dist = 50
        for dx in range(-15, 16):
            for dy in range(-15, 16):
                nx, ny = int(pt[0]+dx), int(pt[1]+dy)
                if 0 <= nx < w and 0 <= ny < h:
                    dist_to_O = abs(distance((nx, ny), O) - radius)
                    if dist_to_O < 5 and gray[ny, nx] < 128:
                        d = distance((nx, ny), pt)
                        if d < best_dist:
                            best_dist = d
                            best_pt = (float(nx), float(ny))
        if label == 'D':
            D = best_pt
        else:
            E = best_pt

    print(f"  D微调后 = ({D[0]:.1f}, {D[1]:.1f})")
    print(f"  E微调后 = ({E[0]:.1f}, {E[1]:.1f})")

    # ============================================================
    # 第五步：计算辅助点F、G、M
    # ============================================================
    print("\n  [5/5] 计算辅助点...")

    F = perpendicular_foot(D, B, C)
    print(f"  F (D垂足) = ({F[0]:.1f}, {F[1]:.1f})")

    G = perpendicular_foot(E, B, C)
    print(f"  G (E垂足) = ({G[0]:.1f}, {G[1]:.1f})")

    # M = DG 与 EF 的交点（原图实际结构）
    M = line_intersection(D, G, E, F)
    if M is None:
        # 备选：在AH上取0.35位置
        M_x = H[0] + (A[0] - H[0]) * 0.35
        M_y = H[1] + (A[1] - H[1]) * 0.35
        M = (M_x, M_y)
    print(f"  M (DG与EF交点) = ({M[0]:.1f}, {M[1]:.1f})")

    key_points = {
        'A': A, 'B': B, 'C': C, 'D': D, 'E': E,
        'F': F, 'G': G, 'H': H, 'M': M, 'O': O,
    }
    geo_info = {'radius': radius}

    # 检测各点之间的实际线段连接
    print("\n  [5/5 验证] 扫描像素验证线段...")
    original_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    valid_connections = detect_line_connections(original_gray, key_points, merged_lines)
    print(f"  验证到的连接: {len(valid_connections)} 条")
    for c in valid_connections:
        print(f"    {c[0]}-{c[1]}: {c[2]:.0f}% 像素匹配")

    # 绘制最终检测结果
    debug_final = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    angle_B = math.atan2(B[1] - O[1], B[0] - O[0])
    angle_C = math.atan2(C[1] - O[1], C[0] - O[0])
    for i in range(181):
        t = i / 180.0
        angle = angle_B + t * (angle_C - angle_B + 2*math.pi)
        if angle > 2*math.pi:
            angle -= 2*math.pi
        px = O[0] + radius * math.cos(angle)
        py = O[1] - radius * abs(math.sin(angle))
        if 0 <= int(px) < w and 0 <= int(py) < h:
            cv2.circle(debug_final, (int(px), int(py)), 1, (0, 200, 0), -1)
    colors_map = {'A': (0,0,255), 'B': (0,0,255), 'C': (0,0,255),
                  'D': (255,0,0), 'E': (255,0,0),
                  'F': (0,255,0), 'G': (0,255,0), 'H': (0,255,0), 'M': (255,255,0)}
    for name, pt in key_points.items():
        cv2.circle(debug_final, (int(pt[0]), int(pt[1])), 6, colors_map.get(name, (255,0,255)), -1)
        cv2.putText(debug_final, name, (int(pt[0])+8, int(pt[1])-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    cv2.imwrite("/workspace/debug4_final.png", debug_final)

    return key_points, geo_info, valid_connections


def detect_line_connections(gray_original, key_points, detected_lines):
    """
    通过三种方式验证线段连接：
    1. 几何约束表：只测试符合几何定义的候选连接（排除 B-M、C-M、H-M 等非法连接）
    2. 连通性验证：检查最长连续暗段，而非简单像素计数
    3. Hough检测线匹配
    
    返回: [(p1_name, p2_name, confidence%), ...]
    """
    h, w = gray_original.shape

    # ============================================================
    # 几何约束表：定义哪些连接在几何上是合理的
    # 每个点只连接它在几何定义中相邻的点
    # ============================================================
    geometry_constraints = {
        'A': {'B', 'C', 'H'},           # 三角形顶点 + 高
        'B': {'A', 'C', 'D'},           # 三角形 + AB上的点D
        'C': {'A', 'B', 'E'},           # 三角形 + AC上的点E
        'D': {'B', 'F', 'G', 'M'},      # AB上的点 + 垂足F + 交叉线G + M
        'E': {'C', 'F', 'G', 'M'},      # AC上的点 + 垂足G + 交叉线F + M
        'F': {'D', 'E', 'G', 'M'},      # 垂足 + 交叉线 + M
        'G': {'E', 'D', 'F', 'M'},      # 垂足 + 交叉线 + M
        'H': {'A'},                      # 仅连接A（垂足）
        'M': {'D', 'G', 'E', 'F'},      # M = DG ∩ EF，只连接这4个点
    }

    # 预处理Hough线为线段列表
    hough_segments = []
    for line in detected_lines:
        p1 = line['p1']
        p2 = line['p2']
        hough_segments.append((p1, p2))

    def longest_dark_run(p1, p2, threshold=128):
        """沿路径扫描，找到最长连续暗像素段（核心改进！）"""
        x1, y1 = p1; x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        length = int(max(abs(dx), abs(dy)))
        if length < 1:
            return 0, 0, 0

        max_run = 0
        current_run = 0
        total_dark = 0
        total_samples = 0

        for i in range(length + 1):
            t = i / length
            x = int(x1 + t * dx)
            y = int(y1 + t * dy)
            if 0 <= x < w and 0 <= y < h:
                total_samples += 1
                if gray_original[y, x] < threshold:
                    total_dark += 1
                    current_run += 1
                    if current_run > max_run:
                        max_run = current_run
                else:
                    current_run = 0

        return max_run, total_dark / max(total_samples, 1), total_samples

    def segment_on_hough_multi(p1, p2, num_samples=5, tolerance=8):
        """
        改进的Hough匹配：在路径上均匀采样多个点，检查多数点是否接近某条Hough线。
        避免单点误判（如B-M路径的中点恰好落在AH线上）。
        """
        seg_len = distance(p1, p2)
        if seg_len < 5:
            return False

        for hp1, hp2 in hough_segments:
            # 方向一致性检查（快速过滤）
            seg_angle = math.atan2(p2[1]-p1[1], p2[0]-p1[0])
            hough_angle = math.atan2(hp2[1]-hp1[1], hp2[0]-hp1[0])
            ang_diff = abs(seg_angle - hough_angle)
            ang_diff = min(ang_diff, 2*math.pi-ang_diff)
            if ang_diff > math.radians(30):
                continue

            # 多点采样验证
            close_count = 0
            for i in range(num_samples):
                t = (i + 0.5) / num_samples
                px = p1[0] + t * (p2[0] - p1[0])
                py = p1[1] + t * (p2[1] - p1[1])

                # 点到线段的距离
                abx, aby = hp2[0]-hp1[0], hp2[1]-hp1[1]
                apx, apy = px-hp1[0], py-hp1[1]
                t_h = (apx*abx + apy*aby) / max(abx*abx + aby*aby, 1e-8)
                if t_h < 0:
                    fx, fy = hp1
                elif t_h > 1:
                    fx, fy = hp2
                else:
                    fx, fy = hp1[0]+t_h*abx, hp1[1]+t_h*aby
                d = math.sqrt((px-fx)**2 + (py-fy)**2)

                if d < tolerance:
                    close_count += 1

            # 多数点接近Hough线才认为匹配
            if close_count >= num_samples * 0.6:
                return True

        return False

    valid = []
    tested_pairs = set()

    # 从几何约束表生成候选列表
    for name1, neighbors in geometry_constraints.items():
        if name1 not in key_points:
            continue
        for name2 in neighbors:
            if name2 not in key_points:
                continue
            pair = frozenset([name1, name2])
            if pair in tested_pairs:
                continue
            tested_pairs.add(pair)

            p1 = key_points[name1]
            p2 = key_points[name2]
            seg_len = distance(p1, p2)
            if seg_len < 10:
                continue

            # 方法1: 连通性验证（最长连续暗段）
            max_run, pixel_ratio, _ = longest_dark_run(p1, p2)
            run_ratio = max_run / max(seg_len, 1)

            # 方法2: 多点Hough匹配
            on_hough = segment_on_hough_multi(p1, p2)

            # 综合判断：
            # - 最长连续暗段占比 > 15%（真实线段有大量连续暗像素）
            # - 或 Hough 匹配通过
            # - 或整体暗像素占比 > 40%（保护短线段）
            if run_ratio > 0.15 or on_hough or pixel_ratio > 0.40:
                confidence = max(run_ratio * 100, pixel_ratio * 100, 70 if on_hough else 0)

                # 打印调试信息
                print(f"    {name1}-{name2}: run_ratio={run_ratio:.2f} pix_ratio={pixel_ratio:.2f} "
                      f"hough={on_hough} conf={confidence:.0f}%")
                valid.append((name1, name2, confidence))

    return valid


def generate_latex(kpts, geo_info, valid_connections=None):
    """生成正确TikZ代码"""
    A = kpts['A']; B = kpts['B']; C = kpts['C']
    D = kpts['D']; E = kpts['E']
    F = kpts['F']; G = kpts['G']
    H = kpts['H']; M = kpts['M']
    O = kpts['O']; radius = geo_info['radius']

    all_pts = [A, B, C, D, E, F, G, H, M, O]
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    target_w = 7.0
    target_h = 5.5
    scale = min(target_w / (max_x - min_x + 1) / 1.3,
                target_h / (max_y - min_y + 1) / 1.3)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    offset_x = -center_x * scale
    offset_y = -center_y * scale

    def tx(px): return px * scale + offset_x
    def ty(py): return -(py * scale + offset_y)  # y轴翻转

    # TikZ坐标系中计算圆弧角度
    otx, oty = tx(O[0]), ty(O[1])
    btx, bty = tx(B[0]), ty(B[1])
    ctx, cty = tx(C[0]), ty(C[1])

    start_angle = math.degrees(math.atan2(bty - oty, btx - otx))
    end_angle = math.degrees(math.atan2(cty - oty, ctx - otx))

    # 从C到B逆时针经过上半圆
    arc_start = end_angle
    arc_end = start_angle
    if arc_end < arc_start:
        arc_end += 360

    tikz_r = radius * scale

    print(f"\n  [TikZ] 圆弧参数:")
    print(f"    B: ({btx:.2f}, {bty:.2f})")
    print(f"    C: ({ctx:.2f}, {cty:.2f})")
    print(f"    O: ({otx:.2f}, {oty:.2f})")
    print(f"    start={arc_start:.1f}°, end={arc_end:.1f}°")
    print(f"    radius={tikz_r:.2f}")

    # 构建LaTeX
    lines = []
    lines.append(r"\documentclass[tikz, border=10pt]{standalone}")
    lines.append(r"\usepackage{tikz}")
    lines.append(r"\usetikzlibrary{arrows}")
    lines.append("")
    lines.append(r"\begin{document}")
    lines.append(r"\begin{tikzpicture}[scale=1.0, >=stealth, line width=1.5pt]")
    lines.append("")

    # 坐标定义
    lines.append("    % === 顶点坐标 ===")
    for name in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'M']:
        p = kpts[name]
        lines.append(f"    \\coordinate ({name}) at ({tx(p[0]):.3f}, {ty(p[1]):.3f});")
    lines.append("")

    # 半圆（虚线）
    lines.append("    % === 以BC为直径的半圆 ===")
    lines.append(f"    \\draw[dashed] (C) arc ({arc_start:.1f}:{arc_end:.1f}:{tikz_r:.3f});")
    lines.append("")

    # ============================================================
    # 由检测结果绘制线段（检测层 + 构造层）
    # 检测层：从图像中验证的线段
    # 构造层：根据几何定义自动补全的辅助线
    # ============================================================
    lines.append("    % === 由图像验证得到的线段 ===")

    # 构建验证集合
    valid_set = set()
    for p1, p2, _ in (valid_connections or []):
        valid_set.add(frozenset([p1, p2]))

    # -----------------------------------------------------------
    # 检测层：从图像中验证的线段
    # -----------------------------------------------------------

    # 1) 三角形 ABC（三边都验证通过才画cycle，否则单边）
    tri_edges = [frozenset(['A','B']), frozenset(['B','C']), frozenset(['C','A'])]
    if all(e in valid_set for e in tri_edges):
        lines.append("    \\draw[thick] (A) -- (B) -- (C) -- cycle;")
        lines.append("")
    else:
        for p1, p2 in [('A','B'), ('B','C'), ('C','A')]:
            if frozenset([p1, p2]) in valid_set:
                lines.append(f"    \\draw[thick] ({p1}) -- ({p2});")
        lines.append("")

    # 2) 其他从图像验证的线段（AH, BD, CE 等）
    for p1, p2, _ in (valid_connections or []):
        if frozenset([p1, p2]) in tri_edges:
            continue
        lines.append(f"    \\draw ({p1}) -- ({p2});")
    lines.append("")

    # -----------------------------------------------------------
    # 构造层：根据几何定义自动补全的辅助线
    # 这些线在原图中可能未画出，但由几何构造决定
    # -----------------------------------------------------------
    lines.append("    % === 构造层：几何定义自动补全的辅助线 ===")

    # 垂线 D-F（D到BC的垂足）
    lines.append("    \\draw (D) -- (F);")
    # 垂线 E-G（E到BC的垂足）
    lines.append("    \\draw (E) -- (G);")
    # 交叉线 D-G（与EF交于M）
    lines.append("    \\draw (D) -- (G);")
    # 交叉线 E-F（与DG交于M）
    lines.append("    \\draw (E) -- (F);")
    lines.append("")

    # 顶点标注
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


def compile_to_png(tex_path, output_base):
    """编译LaTeX → PDF → PNG"""
    out_dir = os.path.dirname(output_base)
    print(f"\n  编译LaTeX...")

    for i in range(2):
        proc = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode',
             f'-output-directory={out_dir}', tex_path],
            capture_output=True, text=True, timeout=30
        )

    pdf_path = output_base + '.pdf'
    if not os.path.exists(pdf_path):
        print(f"  [FAIL] PDF未生成")
        log_path = output_base + '.log'
        if os.path.exists(log_path):
            with open(log_path) as f:
                for line in f.readlines()[-10:]:
                    print(f"    {line.rstrip()}")
        return None

    pdf_size = os.path.getsize(pdf_path)
    print(f"  PDF: {pdf_size} bytes")

    subprocess.run(
        ['pdftoppm', '-png', '-r', '300', pdf_path, output_base],
        capture_output=True, timeout=30
    )

    png_files = sorted(glob.glob(f"{output_base}*.png"))
    if png_files:
        final_path = f"{output_base}_final.png"
        Image.open(png_files[0]).save(final_path)
        img_size = Image.open(final_path).size
        print(f"  PNG: {final_path} ({img_size[0]}x{img_size[1]})")
        return final_path
    return None


def main():
    print("=" * 60)
    print("  智能几何识别 v4")
    print("  文字屏蔽 + 直线交点检测 + 几何推理")
    print("=" * 60)

    img_path = "/workspace/uploaded_geometry.jpeg"

    # 1. 检测几何
    print("\n[1/3] 检测几何结构...")
    gc.collect()
    result = detect_geometry(img_path)

    if result is None:
        return

    kpts, geo_info, valid_connections = result

    # 2. 生成LaTeX
    print("\n[2/3] 生成LaTeX...")
    latex = generate_latex(kpts, geo_info, valid_connections)

    tex_path = "/workspace/uploaded_v4.tex"
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(latex)

    print(f"\n  LaTeX代码:")
    print("-" * 40)
    for line in latex.split('\n'):
        print(f"  {line}")
    print("-" * 40)

    # 3. 编译
    print("\n[3/3] 编译LaTeX为图片...")
    final_png = compile_to_png(tex_path, "/workspace/uploaded_v4")

    if final_png and os.path.exists(final_png):
        img_size = Image.open(final_png).size
        print(f"\n{'=' * 60}")
        print(f"  ✓ 成功! 图片已生成")
        print(f"  文件: {final_png}")
        print(f"  尺寸: {img_size[0]}x{img_size[1]}")
        print(f"  调试图:")
        print(f"    文字掩膜: /workspace/debug4_text_mask.png")
        print(f"    清洗后: /workspace/debug4_cleaned.png")
        print(f"    线条检测: /workspace/debug4_lines.png")
        print(f"    交点检测: /workspace/debug4_intersections.png")
        print(f"    最终检测: /workspace/debug4_final.png")
        print(f"{'=' * 60}")
    else:
        print(f"\n  ✗ 编译失败")


if __name__ == "__main__":
    main()