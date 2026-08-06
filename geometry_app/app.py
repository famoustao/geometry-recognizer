"""几何识别 Web 应用 - 批量处理 + 导出"""
import os, sys, uuid, threading, json, shutil, glob, math, gc, subprocess, time
import cv2, numpy as np
from flask import Flask, render_template, request, jsonify, send_file, url_for
from io import StringIO
from PIL import Image

# ---------- PyInstaller 兼容路径 ----------
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
    APP_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    APP_DIR = BASE_DIR

sys.path.insert(0, BASE_DIR)

# 替换模块导入为直接内联，避免依赖外部文件
# 下面直接引入需要的函数（复制自 run_uploaded_v4.py）
import math

def line_intersection(p1, p2, p3, p4):
    x1, y1 = p1; x2, y2 = p2
    x3, y3 = p3; x4, y4 = p4
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-8:
        return None
    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom
    return (px, py)

def point_on_segment(p, a, b, tol=5.0):
    ax, ay = a; bx, by = b; px, py = p
    abx, aby = bx-ax, by-ay
    apx, apy = px-ax, py-ay
    t = (apx*abx + apy*aby) / max(abx*abx + aby*aby, 1e-8)
    if t < 0: fx, fy = ax, ay
    elif t > 1: fx, fy = bx, by
    else: fx, fy = ax + t*abx, ay + t*aby
    dist = math.sqrt((px-fx)**2 + (py-fy)**2)
    return dist < tol

def perpendicular_foot(point, line_p1, line_p2):
    px, py = point; ax, ay = line_p1; bx, by = line_p2
    abx, aby = bx-ax, by-ay
    apx, apy = px-ax, py-ay
    t = (apx*abx + apy*aby) / max(abx*abx + aby*aby, 1e-8)
    return (ax + t * abx, ay + t * aby)

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def detect_line_connections(gray_original, key_points, detected_lines):
    h, w = gray_original.shape
    geometry_constraints = {
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
    hough_segments = [(line['p1'], line['p2']) for line in detected_lines]

    def longest_dark_run(p1, p2, threshold=128):
        x1, y1 = p1; x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        length = int(max(abs(dx), abs(dy)))
        if length < 1: return 0, 0, 0
        max_run = 0; current_run = 0; total_dark = 0; total_samples = 0
        for i in range(length + 1):
            t = i / length
            x = int(x1 + t * dx); y = int(y1 + t * dy)
            if 0 <= x < w and 0 <= y < h:
                total_samples += 1
                if gray_original[y, x] < threshold:
                    total_dark += 1; current_run += 1
                    if current_run > max_run: max_run = current_run
                else: current_run = 0
        return max_run, total_dark / max(total_samples, 1), total_samples

    def segment_on_hough_multi(p1, p2, num_samples=5, tolerance=8):
        seg_len = distance(p1, p2)
        if seg_len < 5: return False
        for hp1, hp2 in hough_segments:
            seg_angle = math.atan2(p2[1]-p1[1], p2[0]-p1[0])
            hough_angle = math.atan2(hp2[1]-hp1[1], hp2[0]-hp1[0])
            ang_diff = min(abs(seg_angle - hough_angle), 2*math.pi-abs(seg_angle - hough_angle))
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
                if math.sqrt((px-fx)**2 + (py-fy)**2) < tolerance: close_count += 1
            if close_count >= num_samples * 0.6: return True
        return False

    valid = []; tested_pairs = set()
    for name1, neighbors in geometry_constraints.items():
        if name1 not in key_points: continue
        for name2 in neighbors:
            if name2 not in key_points: continue
            pair = frozenset([name1, name2])
            if pair in tested_pairs: continue
            tested_pairs.add(pair)
            p1, p2 = key_points[name1], key_points[name2]
            seg_len = distance(p1, p2)
            if seg_len < 10: continue
            max_run, pixel_ratio, _ = longest_dark_run(p1, p2)
            run_ratio = max_run / max(seg_len, 1)
            on_hough = segment_on_hough_multi(p1, p2)
            if run_ratio > 0.15 or on_hough or pixel_ratio > 0.40:
                confidence = max(run_ratio * 100, pixel_ratio * 100, 70 if on_hough else 0)
                valid.append((name1, name2, confidence))
    return valid

def mask_text_labels(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=1)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded, 8)
    text_mask = np.zeros_like(gray)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        aspect_ratio = max(bw, bh) / max(min(bw, bh), 1)
        if 20 < area < 500 and aspect_ratio < 4.0 and bw < 40 and bh < 40:
            text_mask[labels == i] = 255
    text_mask = cv2.dilate(text_mask, kernel, iterations=2)
    cleaned = img.copy()
    cleaned[text_mask > 0] = [255, 255, 255]
    return cleaned

def generate_latex(kpts, geo_info, valid_connections=None):
    A = kpts['A']; B = kpts['B']; C = kpts['C']
    D = kpts['D']; E = kpts['E']; F = kpts['F']
    G = kpts['G']; H = kpts['H']; M = kpts['M']; O = kpts['O']
    radius = geo_info['radius']
    all_pts = [A, B, C, D, E, F, G, H, M, O]
    xs = [p[0] for p in all_pts]; ys = [p[1] for p in all_pts]
    min_x, max_x = min(xs), max(xs); min_y, max_y = min(ys), max(ys)
    target_w = 7.0; target_h = 5.5
    scale = min(target_w / (max_x - min_x + 1) / 1.3, target_h / (max_y - min_y + 1) / 1.3)
    center_x = (min_x + max_x) / 2; center_y = (min_y + max_y) / 2
    offset_x = -center_x * scale; offset_y = -center_y * scale
    def tx(px): return px * scale + offset_x
    def ty(py): return -(py * scale + offset_y)
    otx, oty = tx(O[0]), ty(O[1])
    btx, bty = tx(B[0]), ty(B[1])
    ctx, cty = tx(C[0]), ty(C[1])
    start_angle = math.degrees(math.atan2(bty - oty, btx - otx))
    end_angle = math.degrees(math.atan2(cty - oty, ctx - otx))
    arc_start = end_angle; arc_end = start_angle
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
    lines.append("    % === 顶点坐标 ===")
    for name in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'M']:
        p = kpts[name]
        lines.append(f"    \\coordinate ({name}) at ({tx(p[0]):.3f}, {ty(p[1]):.3f});")
    lines.append("")
    lines.append("    % === 以BC为直径的半圆 ===")
    lines.append(f"    \\draw[dashed] (C) arc ({arc_start:.1f}:{arc_end:.1f}:{tikz_r:.3f});")
    lines.append("")
    lines.append("    % === 由图像验证得到的线段 ===")
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
    lines.append("    % === 构造层：几何定义自动补全的辅助线 ===")
    lines.append("    \\draw (D) -- (F);")
    lines.append("    \\draw (E) -- (G);")
    lines.append("    \\draw (D) -- (G);")
    lines.append("    \\draw (E) -- (F);")
    lines.append("")
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

# ---------- Flask 应用 ----------
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
app.config['UPLOAD_FOLDER'] = os.path.join(APP_DIR, 'uploads')
app.config['RESULT_FOLDER'] = os.path.join(APP_DIR, 'results')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

tasks = {}

def process_single_image(img_path, task_id, img_name):
    try:
        img = cv2.imread(img_path)
        if img is None: return {'error': f'无法读取图片: {img_name}'}
        h, w = img.shape[:2]
        cleaned = mask_text_labels(img)
        gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
        all_lines = []
        for low, high in [(20, 80), (25, 120), (30, 150)]:
            edges = cv2.Canny(gray, low, high, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=35, minLineLength=20, maxLineGap=10)
            if lines is not None: all_lines.extend(lines)
        if not all_lines: return {'error': f'未检测到线条: {img_name}'}
        merged_lines = []
        for line in all_lines:
            line = np.array(line).flatten()
            x1, y1, x2, y2 = line.astype(float)
            angle = math.degrees(math.atan2(y2-y1, x2-x1)) % 180
            mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
            length = distance((x1, y1), (x2, y2))
            found = False
            for i, ml in enumerate(merged_lines):
                ma = ml['angle']; mmx, mmy = ml['mid']
                if min(abs(angle-ma), 180-abs(angle-ma)) < 15 and distance((mid_x, mid_y), (mmx, mmy)) < 50:
                    if length > ml['length']:
                        merged_lines[i] = {'p1': (x1, y1), 'p2': (x2, y2), 'angle': angle, 'mid': (mid_x, mid_y), 'length': length}
                    found = True; break
            if not found:
                merged_lines.append({'p1': (x1, y1), 'p2': (x2, y2), 'angle': angle, 'mid': (mid_x, mid_y), 'length': length})
        intersections = []
        for i in range(len(merged_lines)):
            for j in range(i+1, len(merged_lines)):
                p1, p2 = merged_lines[i]['p1'], merged_lines[i]['p2']
                p3, p4 = merged_lines[j]['p1'], merged_lines[j]['p2']
                pt = line_intersection(p1, p2, p3, p4)
                if pt is not None and -50 < pt[0] < w+50 and -50 < pt[1] < h+50:
                    if point_on_segment(pt, p1, p2, 8) or point_on_segment(pt, p3, p4, 8):
                        intersections.append(pt)
        if len(intersections) > 10:
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=15, min_samples=2).fit(intersections)
            labels = clustering.labels_
            clusters = {}
            for i, label in enumerate(labels):
                if label == -1: continue
                clusters.setdefault(label, []).append(intersections[i])
            cluster_pts = [(sum(p[0] for p in pts)/len(pts), sum(p[1] for p in pts)/len(pts)) for pts in clusters.values()]
        else:
            cluster_pts = intersections
        if len(cluster_pts) < 3: return {'error': f'交点太少: {img_name}'}
        sorted_y = sorted(cluster_pts, key=lambda p: p[1])
        A = sorted_y[0]
        bottom_candidates = [p for p in sorted(cluster_pts, key=lambda p: -p[1])[:10] if distance(p, A) > 80]
        if len(bottom_candidates) < 2: return {'error': f'底部候选点太少: {img_name}'}
        B = min(bottom_candidates, key=lambda p: p[0])
        C = max(bottom_candidates, key=lambda p: p[0])
        O = ((B[0] + C[0]) / 2, (B[1] + C[1]) / 2)
        radius = distance(O, B)
        H = perpendicular_foot(A, B, C)
        def line_circle_intersection(p1, p2, center, R):
            v = (p2[0]-p1[0], p2[1]-p1[1])
            w = (p1[0]-center[0], p1[1]-center[1])
            a = v[0]**2 + v[1]**2; b = 2*(w[0]*v[0] + w[1]*v[1]); c = w[0]**2 + w[1]**2 - R**2
            disc = b*b - 4*a*c
            if disc < 0: return None
            t1 = (-b + math.sqrt(disc))/(2*a); t2 = (-b - math.sqrt(disc))/(2*a)
            if 0 <= t1 <= 1 and 0 <= t2 <= 1: t = max(t1, t2)
            elif 0 <= t1 <= 1: t = t1
            elif 0 <= t2 <= 1: t = t2
            else: return None
            return (p1[0] + t*v[0], p1[1] + t*v[1])
        D = line_circle_intersection(A, B, O, radius)
        if D is None:
            angle_B = math.atan2(B[1]-O[1], B[0]-O[0])
            angle_D = angle_B - math.pi/4
            if angle_D < 0: angle_D += 2*math.pi
            D = (O[0] + radius*math.cos(angle_D), O[1] - radius*abs(math.sin(angle_D)))
        E = line_circle_intersection(A, C, O, radius)
        if E is None:
            angle_C = math.atan2(C[1]-O[1], C[0]-O[0])
            angle_E = angle_C + math.pi/4
            if angle_E < 0: angle_E += 2*math.pi
            E = (O[0] + radius*math.cos(angle_E), O[1] - radius*abs(math.sin(angle_E)))
        for label, pt in [('D', D), ('E', E)]:
            best_pt = pt; best_dist = 50
            for dx in range(-15, 16):
                for dy in range(-15, 16):
                    nx, ny = int(pt[0]+dx), int(pt[1]+dy)
                    if 0 <= nx < w and 0 <= ny < h and abs(distance((nx, ny), O)-radius) < 5 and gray[ny, nx] < 128:
                        d = distance((nx, ny), pt)
                        if d < best_dist: best_dist = d; best_pt = (float(nx), float(ny))
            if label == 'D': D = best_pt
            else: E = best_pt
        F = perpendicular_foot(D, B, C)
        G = perpendicular_foot(E, B, C)
        M = line_intersection(D, G, E, F)
        if M is None: M = (H[0] + (A[0]-H[0])*0.35, H[1] + (A[1]-H[1])*0.35)
        key_points = {'A': A, 'B': B, 'C': C, 'D': D, 'E': E, 'F': F, 'G': G, 'H': H, 'M': M, 'O': O}
        geo_info = {'radius': radius}
        original_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        valid_connections = detect_line_connections(original_gray, key_points, merged_lines)
        latex = generate_latex(key_points, geo_info, valid_connections)
        result_id = f"{task_id}_{img_name.replace('.', '_')}"
        preview_path = os.path.join(app.config['RESULT_FOLDER'], f"{result_id}_preview.png")
        angle_B = math.atan2(B[1]-O[1], B[0]-O[0])
        angle_C = math.atan2(C[1]-O[1], C[0]-O[0])
        debug = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for i in range(181):
            t = i / 180.0
            angle = angle_B + t * (angle_C - angle_B + 2*math.pi)
            if angle > 2*math.pi: angle -= 2*math.pi
            px = O[0] + radius*math.cos(angle); py = O[1] - radius*abs(math.sin(angle))
            if 0 <= int(px) < w and 0 <= int(py) < h: cv2.circle(debug, (int(px), int(py)), 1, (0,200,0), -1)
        colors_map = {'A': (0,0,255), 'B': (0,0,255), 'C': (0,0,255), 'D': (255,0,0), 'E': (255,0,0),
                      'F': (0,255,0), 'G': (0,255,0), 'H': (0,255,0), 'M': (255,255,0)}
        for name, pt in key_points.items():
            cv2.circle(debug, (int(pt[0]), int(pt[1])), 6, colors_map.get(name, (255,0,255)), -1)
            cv2.putText(debug, name, (int(pt[0])+8, int(pt[1])-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        cv2.imwrite(preview_path, debug)
        tex_path = os.path.join(app.config['RESULT_FOLDER'], f"{result_id}.tex")
        with open(tex_path, 'w', encoding='utf-8') as f: f.write(latex)
        return {
            'success': True, 'name': img_name, 'result_id': result_id,
            'preview': f'/results/{result_id}_preview.png',
            'tex_path': f'/results/{result_id}.tex', 'tex_content': latex,
            'detected_lines': len(valid_connections),
            'points': {k: f'({v[0]:.1f}, {v[1]:.1f})' for k, v in key_points.items()},
        }
    except Exception as e:
        return {'error': f'处理 {img_name} 失败: {str(e)}'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'files' not in request.files: return jsonify({'error': '请选择文件'})
    files = request.files.getlist('files')
    if not files or files[0].filename == '': return jsonify({'error': '请选择文件'})
    task_id = str(uuid.uuid4())
    file_data_list = []
    for f in files:
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in ('.jpg', '.jpeg', '.png', '.bmp', '.tiff'):
            file_data_list.append({'name': f.filename, 'data': None, 'error': '不支持的文件格式'})
        else:
            file_data_list.append({'name': f.filename, 'data': f.read(), 'error': None})
    tasks[task_id] = {'status': 'processing', 'progress': 0, 'results': [], 'total': len(file_data_list)}
    def process(file_data_list, task_id):
        results = []
        for i, fd in enumerate(file_data_list):
            if fd['error']:
                results.append({'error': fd['error'], 'name': fd['name']})
                tasks[task_id]['progress'] = int((i + 1) / len(file_data_list) * 100)
                continue
            ext = os.path.splitext(fd['name'])[1].lower()
            safe_name = f"{uuid.uuid4().hex}{ext}"
            path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
            with open(path, 'wb') as f: f.write(fd['data'])
            result = process_single_image(path, task_id, fd['name'])
            results.append(result)
            tasks[task_id]['progress'] = int((i + 1) / len(file_data_list) * 100)
        tasks[task_id]['status'] = 'completed'
        tasks[task_id]['results'] = results
    thread = threading.Thread(target=process, args=(file_data_list, task_id))
    thread.start()
    return jsonify({'task_id': task_id})

@app.route('/status/<task_id>')
def status(task_id):
    if task_id not in tasks: return jsonify({'status': 'not_found'})
    t = tasks[task_id]
    return jsonify({'status': t['status'], 'progress': t.get('progress', 0), 'total': t.get('total', 0), 'results': t.get('results', []) if t['status'] == 'completed' else []})

@app.route('/results/<path:filename>')
def results(filename):
    return send_file(os.path.join(app.config['RESULT_FOLDER'], filename))

@app.route('/export/<task_id>')
def export(task_id):
    result_id = request.args.get('result_id')
    if not result_id: return jsonify({'error': '缺少 result_id'})
    tex_path = os.path.join(app.config['RESULT_FOLDER'], f"{result_id}.tex")
    if os.path.exists(tex_path):
        return send_file(tex_path, as_attachment=True, download_name=f"{result_id}.tex")
    return jsonify({'error': '文件不存在'})

@app.route('/export_merge/<task_id>')
def export_merge(task_id):
    if task_id not in tasks or tasks[task_id]['status'] != 'completed': return jsonify({'error': '任务未完成'})
    results = tasks[task_id]['results']
    valid = [r for r in results if r.get('success')]
    merge_tex = r"""\documentclass[tikz, border=10pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{arrows}
\begin{document}
"""
    for i, r in enumerate(valid):
        tex = r['tex_content']
        tex = tex.replace(r'\begin{document}', '').replace(r'\end{document}', '')
        tex = tex.replace(r'\begin{tikzpicture}', '\\begin{tikzpicture}[scale=1.0, >=stealth, line width=1.5pt]')
        merge_tex += f"\\section*{{{r['name']}}}\n{tex}\n\n"
    merge_tex += r"\end{document}"
    merge_path = os.path.join(app.config['RESULT_FOLDER'], f"{task_id}_merged.tex")
    with open(merge_path, 'w', encoding='utf-8') as f: f.write(merge_tex)
    return send_file(merge_path, as_attachment=True, download_name=f"merged_results.tex")

@app.route('/export_png/<task_id>')
def export_png(task_id):
    result_id = request.args.get('result_id')
    if not result_id: return jsonify({'error': '缺少 result_id'})
    png_path = os.path.join(app.config['RESULT_FOLDER'], f"{result_id}_preview.png")
    if os.path.exists(png_path):
        return send_file(png_path, as_attachment=True, download_name=f"{result_id}.png")
    return jsonify({'error': '文件不存在'})