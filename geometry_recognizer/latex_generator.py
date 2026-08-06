"""
LaTeX TikZ代码自动生成模块 (第3.5节)

核心优化:
1. 坐标归一化处理
2. 顶点自动命名规则
3. 图元语法精准匹配
4. 代码轻量化封装
"""

import numpy as np
from typing import List, Dict, Optional, Set, Tuple
from .config import GeometryConfig
from .data_structures import (
    Vertex, Point, Primitive, PrimitiveType, LineType,
    GeometryLayer, GeometryResult, ConfidenceSource
)
from .utils import normalize_coordinates


class LatexGenerator:
    """
    LaTeX TikZ代码生成器

    支持输出:
    - 实线线段: \draw (A)--(B);
    - 虚线辅助线: \draw[dashed] (M)--(C);
    - 射线: \draw (A)--++(3,0);
    - 圆形: \draw (O) circle (r);
    - 圆弧: arc 标准语法
    - 顶点标注: \node[方位] at (点) {$字母$};
    """

    def __init__(self, config: GeometryConfig):
        self.config = config
        self.th = config.thresholds
        # 顶点命名映射: vertex_id -> latex_name
        self.name_map: Dict[str, str] = {}
        # 已使用的命名
        self.used_names: Set[str] = set()
        # 占位符计数器
        self.placeholder_counter: int = 0

    def generate(self, result: GeometryResult) -> str:
        """
        生成完整LaTeX TikZ代码

        Args:
            result: 几何识别结果

        Returns:
            完整LaTeX代码字符串
        """
        if not result.success:
            return self._generate_error_template(result.error_message or "Unknown error")

        # 1. 坐标归一化
        if self.config.enable_coordinate_normalization:
            vertices, scale, offset = normalize_coordinates(
                result.vertices,
                target_range=self.th.tikz_coord_range
            )
            result.vertices = vertices
            result.scale = scale
            result.offset = offset

        # 2. 顶点自动命名
        if self.config.enable_auto_naming:
            self._auto_name_vertices(result)

        # 3. 生成TikZ代码
        tikz_code = self._generate_tikz_code(result)

        # 4. 封装完整LaTeX模板
        latex_code = self._wrap_latex_template(tikz_code, result)

        result.latex_code = latex_code
        return latex_code

    # ============================================================
    # 顶点命名 (第3.5.2节)
    # ============================================================

    def _auto_name_vertices(self, result: GeometryResult):
        """
        顶点自动命名规则

        1. OCR语义顶点沿用原图字母 (A, B, C, O)
        2. 无标注拓扑顶点自动生成序列名 (P1, P2, O1, M1)
        3. 虚拟几何关键点命名 (M: 中点, F: 垂足, O: 圆心, G: 重心)
        """
        self.name_map = {}
        self.used_names = set()

        # 第一轮: 语义顶点
        for vid, v in result.vertices.items():
            if v.label and v.label.isalpha() and len(v.label) == 1:
                name = v.label.upper()
                if name not in self.used_names:
                    self.name_map[vid] = name
                    self.used_names.add(name)

        # 第二轮: 虚拟顶点按类型命名
        name_counters = {
            'M': 0,  # 中点
            'F': 0,  # 垂足
            'O': 0,  # 圆心/外心
            'G': 0,  # 重心
            'H': 0,  # 垂心
            'I': 0,  # 内心
            'B': 0,  # 角平分点
            'P': 0,  # 通用顶点
            'C': 0,  # 交点
            'E': 0,  # 端点
        }

        for vid, v in result.vertices.items():
            if vid in self.name_map:
                continue

            if v.is_virtual and v.virtual_type:
                prefix = self._get_virtual_prefix(v.virtual_type)
                name_counters[prefix] = name_counters.get(prefix, 0) + 1
                name = f"{prefix}{name_counters[prefix]}"
                # 处理冲突
                while name in self.used_names:
                    name_counters[prefix] = name_counters.get(prefix, 0) + 1
                    name = f"{prefix}{name_counters[prefix]}"
                self.name_map[vid] = name
                self.used_names.add(name)

        # 第三轮: 其余顶点使用P序列
        for vid, v in result.vertices.items():
            if vid in self.name_map:
                continue
            name_counters['P'] = name_counters.get('P', 0) + 1
            name = f"P{name_counters['P']}"
            while name in self.used_names:
                name_counters['P'] = name_counters.get('P', 0) + 1
                name = f"P{name_counters['P']}"
            self.name_map[vid] = name
            self.used_names.add(name)

    def _get_virtual_prefix(self, virtual_type: str) -> str:
        """根据虚拟顶点类型获取命名前缀"""
        prefix_map = {
            'midpoint': 'M',
            'foot': 'F',
            'circle_center': 'O',
            'centroid': 'G',
            'circumcenter': 'O',
            'orthocenter': 'H',
            'incenter': 'I',
            'angle_bisector': 'B',
            'intersection_line_line': 'C',
            'intersection_line_arc': 'C',
            'intersection_arc_arc': 'C',
        }
        return prefix_map.get(virtual_type, 'P')

    def _get_name(self, vid: str) -> str:
        """获取顶点的LaTeX命名"""
        return self.name_map.get(vid, f"P{vid}")

    # ============================================================
    # TikZ代码生成 (第3.5.3节)
    # ============================================================

    def _generate_tikz_code(self, result: GeometryResult) -> str:
        """
        生成TikZ绘图代码

        按图层顺序输出: 辅助层 → 轮廓层 → 标注层
        """
        lines = []

        # 1. 坐标定义
        lines.append("    % === 坐标定义 ===")
        for vid, v in result.vertices.items():
            name = self._get_name(vid)
            pos = v.normalized_pos or v.position
            lines.append(
                f"    \\coordinate ({name}) at ({pos.x:.{self.th.coord_precision}f}, "
                f"{pos.y:.{self.th.coord_precision}f});"
            )

        lines.append("")

        # 2. 辅助层 (先绘制，作为底层)
        aux_lines = []
        for prim in result.auxiliaries:
            code = self._generate_primitive_tikz(prim, result)
            if code:
                aux_lines.append(code)

        if aux_lines:
            lines.append("    % === 辅助元素 ===")
            lines.extend(aux_lines)
            lines.append("")

        # 3. 轮廓层
        contour_lines = []
        for prim in result.contours:
            code = self._generate_primitive_tikz(prim, result)
            if code:
                contour_lines.append(code)

        if contour_lines:
            lines.append("    % === 基础轮廓 ===")
            lines.extend(contour_lines)
            lines.append("")

        # 4. 标注层
        annotation_lines = []
        for prim in result.annotations:
            code = self._generate_primitive_tikz(prim, result)
            if code:
                annotation_lines.append(code)

        # 顶点标注
        vertex_labels = self._generate_vertex_labels(result)
        if vertex_labels:
            lines.append("    % === 顶点标注 ===")
            lines.extend(vertex_labels)
            lines.append("")

        if annotation_lines:
            lines.append("    % === 其他标注 ===")
            lines.extend(annotation_lines)

        return "\n".join(lines)

    def _generate_primitive_tikz(self, prim: Primitive, result: GeometryResult) -> Optional[str]:
        """生成单个图元的TikZ代码"""
        if prim.tikz_code:
            return prim.tikz_code

        if prim.type == PrimitiveType.LINE_SEGMENT:
            return self._gen_line_segment(prim, result)
        elif prim.type == PrimitiveType.CIRCLE:
            return self._gen_circle(prim, result)
        elif prim.type == PrimitiveType.ARC:
            return self._gen_arc(prim, result)
        elif prim.type == PrimitiveType.ELLIPTIC_ARC:
            return self._gen_elliptic_arc(prim, result)
        elif prim.type == PrimitiveType.RAY_LINE:
            return self._gen_ray(prim, result)
        elif prim.type == PrimitiveType.POINT:
            return self._gen_point(prim, result)

        return None

    def _get_draw_options(self, prim: Primitive) -> str:
        """获取TikZ绘图选项"""
        options = []

        if prim.line_type == LineType.DASHED:
            options.append("dashed")
        elif prim.line_type == LineType.DASH_DOT:
            options.append("dash dot")
        elif prim.line_type == LineType.HIDDEN:
            options.append("dashed, gray")
        elif prim.line_type == LineType.RAY:
            options.append("-latex")

        if prim.layer == GeometryLayer.AUXILIARY:
            if "dashed" not in str(options):
                options.append("dashed")
            options.append("gray")

        if prim.confidence < 0.5:
            options.append("gray!50")

        return f"[{','.join(options)}]" if options else ""

    def _gen_line_segment(self, prim: Primitive, result: GeometryResult) -> str:
        """生成线段代码"""
        if len(prim.vertices) < 2:
            return None

        v1 = self._get_name(prim.vertices[0])
        v2 = self._get_name(prim.vertices[-1])
        opts = self._get_draw_options(prim)

        return f"    \\draw{opts} ({v1}) -- ({v2});"

    def _gen_circle(self, prim: Primitive, result: GeometryResult) -> str:
        """生成圆形代码"""
        center_id = prim.params.get('center_id')
        radius = prim.params.get('radius', 0)

        if center_id and center_id in self.name_map:
            center_name = self._get_name(center_id)
            # 归一化后的半径
            norm_radius = radius * result.scale
            opts = self._get_draw_options(prim)
            return f"    \\draw{opts} ({center_name}) circle ({norm_radius:.{self.th.coord_precision}f});"
        else:
            cx = prim.params.get('center_x', 0)
            cy = prim.params.get('center_y', 0)
            radius = prim.params.get('radius', 0)
            # 使用匿名坐标
            opts = self._get_draw_options(prim)
            return f"    \\draw{opts} ({cx:.{self.th.coord_precision}f}, {cy:.{self.th.coord_precision}f}) circle ({radius:.{self.th.coord_precision}f});"

    def _gen_arc(self, prim: Primitive, result: GeometryResult) -> str:
        """生成圆弧代码"""
        center_id = prim.params.get('center_id')
        cx = prim.params.get('center_x', 0)
        cy = prim.params.get('center_y', 0)
        radius = prim.params.get('radius', 0)
        start_angle = prim.params.get('start_angle', 0)
        end_angle = prim.params.get('end_angle', 0)

        # 转换为度
        start_deg = np.degrees(start_angle)
        end_deg = np.degrees(end_angle)

        # 计算起始点
        start_x = cx + radius * np.cos(start_angle)
        start_y = cy + radius * np.sin(start_angle)

        opts = self._get_draw_options(prim)

        return (
            f"    \\draw{opts} ({start_x:.{self.th.coord_precision}f}, "
            f"{start_y:.{self.th.coord_precision}f}) "
            f"arc ({start_deg:.1f}:{end_deg:.1f}:{radius:.{self.th.coord_precision}f});"
        )

    def _gen_elliptic_arc(self, prim: Primitive, result: GeometryResult) -> str:
        """生成椭圆弧代码"""
        cx = prim.params.get('center_x', 0)
        cy = prim.params.get('center_y', 0)
        a = prim.params.get('a', 0)
        b = prim.params.get('b', 0)
        angle = prim.params.get('angle', 0)
        start_angle = prim.params.get('start_angle', 0)
        end_angle = prim.params.get('end_angle', 0)

        start_deg = np.degrees(start_angle)
        end_deg = np.degrees(end_angle)
        rot_deg = np.degrees(angle)
        opts = self._get_draw_options(prim)

        if rot_deg != 0:
            return (
                f"    \\draw{opts} [rotate around={{{rot_deg:.1f}:({cx:.{self.th.coord_precision}f}, "
                f"{cy:.{self.th.coord_precision}f})}}] "
                f"({cx + a * np.cos(start_angle):.{self.th.coord_precision}f}, "
                f"{cy + b * np.sin(start_angle):.{self.th.coord_precision}f}) "
                f"arc ({start_deg:.1f}:{end_deg:.1f}:{a:.{self.th.coord_precision}f} "
                f"and {b:.{self.th.coord_precision}f});"
            )
        else:
            return (
                f"    \\draw{opts} ({cx + a * np.cos(start_angle):.{self.th.coord_precision}f}, "
                f"{cy + b * np.sin(start_angle):.{self.th.coord_precision}f}) "
                f"arc ({start_deg:.1f}:{end_deg:.1f}:{a:.{self.th.coord_precision}f} "
                f"and {b:.{self.th.coord_precision}f});"
            )

    def _gen_ray(self, prim: Primitive, result: GeometryResult) -> str:
        """生成射线代码"""
        if len(prim.vertices) < 2:
            return None

        start = self._get_name(prim.vertices[0])
        end = self._get_name(prim.vertices[-1])
        opts = self._get_draw_options(prim)

        # 射线使用带箭头的线段
        return f"    \\draw[{opts}, -latex] ({start}) -- ({end});"

    def _gen_point(self, prim: Primitive, result: GeometryResult) -> str:
        """生成孤点代码"""
        if not prim.vertices:
            return None
        name = self._get_name(prim.vertices[0])
        return f"    \\fill ({name}) circle (2pt);"

    def _generate_vertex_labels(self, result: GeometryResult) -> List[str]:
        """生成顶点标注代码"""
        labels = []
        direction_map = {
            'A': 'left', 'B': 'right', 'C': 'above', 'D': 'below',
            'O': 'above', 'P': 'right', 'M': 'below',
        }

        for vid, v in result.vertices.items():
            name = self._get_name(vid)
            if not name or len(name) > 3:
                continue

            # 确定标注方向
            direction = direction_map.get(name, 'right')

            # 语义顶点使用数学模式
            label_text = f"${name}$"

            # 虚拟顶点用小字体
            if v.is_virtual:
                labels.append(
                    f"    \\node[{direction}, font=\\tiny] at ({name}) {{{label_text}}};"
                )
            else:
                labels.append(
                    f"    \\node[{direction}] at ({name}) {{{label_text}}};"
                )

        return labels

    # ============================================================
    # LaTeX模板封装 (第3.5.4节)
    # ============================================================

    def _wrap_latex_template(self, tikz_code: str, result: GeometryResult) -> str:
        """
        封装完整LaTeX standalone模板

        包含必要的tikz依赖包，无需手动补充环境代码。
        """
        template = r"""\documentclass[tikz, border=5pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{arrows, decorations.pathmorphing, backgrounds, positioning, fit, petri}

\begin{document}
\begin{tikzpicture}[scale=1.0, >=stealth]

%s

\end{tikzpicture}
\end{document}
"""
        return template % tikz_code

    def _generate_error_template(self, error_message: str) -> str:
        """生成错误模板"""
        template = r"""\documentclass[tikz, border=5pt]{standalone}
\usepackage{tikz}

\begin{document}
\begin{tikzpicture}
    \node[align=center, text=red] at (0,0) {
        Recognition Failed\\[4pt]
        \footnotesize %s
    };
\end{tikzpicture}
\end{document}
"""
        return template % error_message

    # ============================================================
    # 批量导出 (第3.6.3节)
    # ============================================================

    def batch_export(self, results: List[GeometryResult], filenames: List[str]) -> List[Tuple[str, str]]:
        """
        批量导出 .tex 源文件

        Args:
            results: 几何识别结果列表
            filenames: 输出文件名列表

        Returns:
            [(filename, latex_code), ...]
        """
        outputs = []
        for result, filename in zip(results, filenames):
            latex_code = self.generate(result)
            outputs.append((filename, latex_code))
        return outputs