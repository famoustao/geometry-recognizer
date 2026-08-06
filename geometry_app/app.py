"""几何识别 Web 应用 - 批量处理 + 导出"""
import os, sys, uuid, threading, json, shutil, glob, math, gc, subprocess, time
import cv2, numpy as np
from flask import Flask, render_template, request, jsonify, send_file, url_for
from io import StringIO
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_uploaded_v4 import (
    mask_text_labels, line_intersection, point_on_segment,
    perpendicular_foot, distance, detect_line_connections,
    generate_latex
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
app.config['UPLOAD_FOLDER'] = '/workspace/geometry_app/uploads'
app.config['RESULT_FOLDER'] = '/workspace/geometry_app/results'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# 任务状态跟踪
tasks = {}


def process_single_image(img_path, task_id, img_name):
    """处理单张图片"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return {'error': f'无法读取图片: {img_name}'}

        h, w = img.shape[:2]

        # 文字屏蔽
        cleaned = mask_text_labels(img)
        gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)

        # 线条检测
        all_lines = []
        for low, high in [(20, 80), (25, 120), (30, 150)]:
            edges = cv2.Canny(gray, low, high, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=35,
                                     minLineLength=20, maxLineGap=10)
            if lines is not None:
                all_lines.extend(lines)

        if not all_lines:
            return {'error': f'未检测到线条: {img_name}'}

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
                ang_diff = min(abs(angle-ma), 180-abs(angle-ma))
                mid_dist = distance((mid_x, mid_y), (mmx, mmy))
                if ang_diff < 15 and mid_dist < 50:
                    if length > ml['length']:
                        merged_lines[i] = {'p1': (x1, y1), 'p2': (x2, y2), 'angle': angle,
                                           'mid': (mid_x, mid_y), 'length': length}
                    found = True
                    break
            if not found:
                merged_lines.append({'p1': (x1, y1), 'p2': (x2, y2), 'angle': angle,
                                     'mid': (mid_x, mid_y), 'length': length})

        # 交点分析
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
            cluster_pts = []
            for label, pts in clusters.items():
                cx = sum(p[0] for p in pts) / len(pts)
                cy = sum(p[1] for p in pts) / len(pts)
                cluster_pts.append((cx, cy))
        else:
            cluster_pts = intersections

        if len(cluster_pts) < 3:
            return {'error': f'交点太少，无法构成三角形: {img_name}'}

        # 定位三角形顶点
        sorted_y = sorted(cluster_pts, key=lambda p: p[1])
        A = sorted_y[0]
        bottom_candidates = sorted(cluster_pts, key=lambda p: -p[1])[:10]
        bottom_candidates = [p for p in bottom_candidates if distance(p, A) > 80]
        if len(bottom_candidates) < 2:
            return {'error': f'底部候选点太少: {img_name}'}
        B = min(bottom_candidates, key=lambda p: p[0])
        C = max(bottom_candidates, key=lambda p: p[0])

        # 几何计算
        O = ((B[0] + C[0]) / 2, (B[1] + C[1]) / 2)
        radius = distance(O, B)
        H = perpendicular_foot(A, B, C)

        def line_circle_intersection(p1, p2, center, R):
            v = (p2[0]-p1[0], p2[1]-p1[1])
            w = (p1[0]-center[0], p1[1]-center[1])
            a = v[0]**2 + v[1]**2
            b = 2*(w[0]*v[0] + w[1]*v[1])
            c = w[0]**2 + w[1]**2 - R**2
            disc = b*b - 4*a*c
            if disc < 0: return None
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
            if angle_D < 0: angle_D += 2*math.pi
            D = (O[0] + radius * math.cos(angle_D), O[1] - radius * abs(math.sin(angle_D)))

        E = line_circle_intersection(A, C, O, radius)
        if E is None:
            angle_C = math.atan2(C[1] - O[1], C[0] - O[0])
            angle_E = angle_C + math.pi/4
            if angle_E < 0: angle_E += 2*math.pi
            E = (O[0] + radius * math.cos(angle_E), O[1] - radius * abs(math.sin(angle_E)))

        # 微调D、E
        for label, pt in [('D', D), ('E', E)]:
            best_pt = pt; best_dist = 50
            for dx in range(-15, 16):
                for dy in range(-15, 16):
                    nx, ny = int(pt[0]+dx), int(pt[1]+dy)
                    if 0 <= nx < w and 0 <= ny < h:
                        if abs(distance((nx, ny), O) - radius) < 5 and gray[ny, nx] < 128:
                            d = distance((nx, ny), pt)
                            if d < best_dist:
                                best_dist = d; best_pt = (float(nx), float(ny))
            if label == 'D': D = best_pt
            else: E = best_pt

        # 辅助点
        F = perpendicular_foot(D, B, C)
        G = perpendicular_foot(E, B, C)
        M = line_intersection(D, G, E, F)
        if M is None:
            M_x = H[0] + (A[0] - H[0]) * 0.35
            M_y = H[1] + (A[1] - H[1]) * 0.35
            M = (M_x, M_y)

        key_points = {
            'A': A, 'B': B, 'C': C, 'D': D, 'E': E,
            'F': F, 'G': G, 'H': H, 'M': M, 'O': O,
        }
        geo_info = {'radius': radius}

        original_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        valid_connections = detect_line_connections(original_gray, key_points, merged_lines)

        # 生成 TikZ
        latex = generate_latex(key_points, geo_info, valid_connections)

        # 保存结果图片（预览）
        result_id = f"{task_id}_{img_name.replace('.', '_')}"
        preview_path = os.path.join(app.config['RESULT_FOLDER'], f"{result_id}_preview.png")

        # 绘制检测结果图
        angle_B = math.atan2(B[1] - O[1], B[0] - O[0])
        angle_C = math.atan2(C[1] - O[1], C[0] - O[0])
        debug = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for i in range(181):
            t = i / 180.0
            angle = angle_B + t * (angle_C - angle_B + 2*math.pi)
            if angle > 2*math.pi: angle -= 2*math.pi
            px = O[0] + radius * math.cos(angle)
            py = O[1] - radius * abs(math.sin(angle))
            if 0 <= int(px) < w and 0 <= int(py) < h:
                cv2.circle(debug, (int(px), int(py)), 1, (0, 200, 0), -1)
        colors_map = {'A': (0,0,255), 'B': (0,0,255), 'C': (0,0,255),
                      'D': (255,0,0), 'E': (255,0,0),
                      'F': (0,255,0), 'G': (0,255,0), 'H': (0,255,0), 'M': (255,255,0)}
        for name, pt in key_points.items():
            cv2.circle(debug, (int(pt[0]), int(pt[1])), 6, colors_map.get(name, (255,0,255)), -1)
            cv2.putText(debug, name, (int(pt[0])+8, int(pt[1])-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        cv2.imwrite(preview_path, debug)

        # 保存 TikZ 文件
        tex_path = os.path.join(app.config['RESULT_FOLDER'], f"{result_id}.tex")
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write(latex)

        return {
            'success': True,
            'name': img_name,
            'result_id': result_id,
            'preview': f'/results/{result_id}_preview.png',
            'tex_path': f'/results/{result_id}.tex',
            'tex_content': latex,
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
    if 'files' not in request.files:
        return jsonify({'error': '请选择文件'})

    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({'error': '请选择文件'})

    task_id = str(uuid.uuid4())

    # 在线程启动前读取所有文件数据到内存
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
            with open(path, 'wb') as f:
                f.write(fd['data'])

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
    if task_id not in tasks:
        return jsonify({'status': 'not_found'})
    t = tasks[task_id]
    return jsonify({
        'status': t['status'],
        'progress': t.get('progress', 0),
        'total': t.get('total', 0),
        'results': t.get('results', []) if t['status'] == 'completed' else [],
    })


@app.route('/results/<path:filename>')
def results(filename):
    return send_file(os.path.join(app.config['RESULT_FOLDER'], filename))


@app.route('/export/<task_id>')
def export(task_id):
    """导出单个 TeX 文件"""
    result_id = request.args.get('result_id')
    if not result_id:
        return jsonify({'error': '缺少 result_id'})
    tex_path = os.path.join(app.config['RESULT_FOLDER'], f"{result_id}.tex")
    if os.path.exists(tex_path):
        return send_file(tex_path, as_attachment=True, download_name=f"{result_id}.tex")
    return jsonify({'error': '文件不存在'})


@app.route('/export_merge/<task_id>')
def export_merge(task_id):
    """合并导出所有结果为一份文档"""
    if task_id not in tasks or tasks[task_id]['status'] != 'completed':
        return jsonify({'error': '任务未完成'})

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
        tex = tex.replace(r'\begin{tikzpicture}', f'\\begin{{tikzpicture}}[scale=1.0, >=stealth, line width=1.5pt]')
        merge_tex += f"\\section*{{{r['name']}}}\n{tex}\n\n"
    merge_tex += r"\end{document}"

    merge_path = os.path.join(app.config['RESULT_FOLDER'], f"{task_id}_merged.tex")
    with open(merge_path, 'w', encoding='utf-8') as f:
        f.write(merge_tex)

    return send_file(merge_path, as_attachment=True, download_name=f"merged_results.tex")


@app.route('/export_png/<task_id>')
def export_png(task_id):
    """导出单个 PNG 图片"""
    result_id = request.args.get('result_id')
    if not result_id:
        return jsonify({'error': '缺少 result_id'})
    png_path = os.path.join(app.config['RESULT_FOLDER'], f"{result_id}_preview.png")
    if os.path.exists(png_path):
        return send_file(png_path, as_attachment=True, download_name=f"{result_id}.png")
    return jsonify({'error': '文件不存在'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)