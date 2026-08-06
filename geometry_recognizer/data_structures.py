"""
数据结构定义模块

定义整个系统中使用的核心数据结构，包括：
- 顶点 (Vertex) 及其置信度来源、线型、图元类型
- 图元 (Primitive) 及其分层存储
- 几何结果 (GeometryResult) 完整输出
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum, auto
import numpy as np


class ConfidenceSource(Enum):
    """顶点置信度来源枚举"""
    SEMANTIC = 0.9       # 语义顶点（OCR识别）
    SKELETON_CROSS = 0.7 # 骨架交叉交点
    SKELETON_END = 0.5   # 骨架开放端点
    CONTOUR_CORNER = 0.3 # 轮廓曲率拐点
    VIRTUAL = 0.8        # 几何虚拟顶点（推导）
    MANUAL = 1.0         # 人工修正顶点

    def __init__(self, score: float):
        self.score = score


class LineType(Enum):
    """线型枚举"""
    SOLID = "solid"          # 实线
    DASHED = "dashed"        # 虚线
    DASH_DOT = "dash_dot"    # 点划线
    RAY = "ray"              # 射线
    HIDDEN = "hidden"        # 隐藏线（辅助线）


class PrimitiveType(Enum):
    """图元类型枚举"""
    LINE_SEGMENT = "line_segment"           # 直线段
    CIRCLE = "circle"                       # 圆
    ARC = "arc"                             # 圆弧
    ELLIPTIC_ARC = "elliptic_arc"           # 椭圆弧
    RAY_LINE = "ray_line"                   # 射线
    POINT = "point"                         # 孤点
    POLYLINE = "polyline"                   # 折线
    CURVE = "curve"                         # 一般曲线（未分类）
    ANGLE_MARK = "angle_mark"              # 角度标注弧
    DIMENSION = "dimension"                 # 尺寸标注


class GeometryLayer(Enum):
    """几何图层枚举"""
    CONTOUR = "contour"         # 基础轮廓层
    AUXILIARY = "auxiliary"     # 辅助元素层
    ANNOTATION = "annotation"   # 标注层


@dataclass
class Point:
    """二维点"""
    x: float
    y: float

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def distance_to(self, other: 'Point') -> float:
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def __add__(self, other: 'Point') -> 'Point':
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Point') -> 'Point':
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> 'Point':
        return Point(self.x * scalar, self.y * scalar)

    def __repr__(self) -> str:
        return f"P({self.x:.2f}, {self.y:.2f})"


@dataclass
class Vertex:
    """
    顶点数据结构

    Attributes:
        id: 顶点唯一标识
        position: 二维坐标 (Point)
        confidence: 置信度 [0, 1]
        source: 置信度来源
        label: OCR识别的标签 (如 'A', 'B', 'O')
        layer: 所属图层
        normalized_pos: 归一化后的坐标 (用于LaTeX输出)
        is_virtual: 是否为虚拟推导顶点
        virtual_type: 虚拟顶点类型描述
    """
    id: str
    position: Point
    confidence: float = 0.5
    source: ConfidenceSource = ConfidenceSource.SKELETON_END
    label: Optional[str] = None
    layer: GeometryLayer = GeometryLayer.CONTOUR
    normalized_pos: Optional[Point] = None
    is_virtual: bool = False
    virtual_type: Optional[str] = None

    def __repr__(self) -> str:
        return f"V({self.id}: {self.position}, conf={self.confidence:.2f})"


@dataclass
class Primitive:
    """
    图元数据结构

    Attributes:
        id: 图元唯一标识
        type: 图元类型
        layer: 所属图层
        line_type: 线型
        vertices: 关联顶点ID列表 (有序)
        params: 几何参数字典
            - line: {'slope', 'intercept', 'angle', 'length'}
            - circle: {'center_id', 'radius'}
            - arc: {'center_id', 'radius', 'start_angle', 'end_angle', 'direction'}
            - elliptic_arc: {'center_id', 'a', 'b', 'angle', 'start_angle', 'end_angle'}
        confidence: 识别置信度
        pixel_points: 原始像素点集 (用于拟合)
        tikz_code: 预生成的TikZ代码片段（可选）
    """
    id: str
    type: PrimitiveType
    layer: GeometryLayer = GeometryLayer.CONTOUR
    line_type: LineType = LineType.SOLID
    vertices: List[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    confidence: float = 0.5
    pixel_points: Optional[np.ndarray] = None
    tikz_code: Optional[str] = None

    def __repr__(self) -> str:
        return f"Prim({self.id}: {self.type.value}, line={self.line_type.value})"


@dataclass
class GeometryResult:
    """
    完整几何识别结果

    Attributes:
        vertices: 顶点字典 {id: Vertex}
        primitives: 图元列表
        contours: 轮廓层图元列表
        auxiliaries: 辅助层图元列表
        annotations: 标注层图元列表
        image_shape: 原始图像尺寸 (h, w)
        scale: 像素坐标到归一化坐标的缩放因子
        offset: 归一化坐标偏移量
        latex_code: 完整LaTeX代码
        success: 识别是否成功
        error_message: 错误信息（如有）
    """
    vertices: dict = field(default_factory=dict)
    primitives: List[Primitive] = field(default_factory=list)
    contours: List[Primitive] = field(default_factory=list)
    auxiliaries: List[Primitive] = field(default_factory=list)
    annotations: List[Primitive] = field(default_factory=list)
    image_shape: Tuple[int, int] = (0, 0)
    scale: float = 1.0
    offset: Point = field(default_factory=lambda: Point(0, 0))
    latex_code: Optional[str] = None
    success: bool = False
    error_message: Optional[str] = None

    def get_vertex_by_id(self, vid: str) -> Optional[Vertex]:
        return self.vertices.get(vid)

    def get_vertex_position(self, vid: str) -> Optional[Point]:
        v = self.vertices.get(vid)
        return v.position if v else None

    def add_vertex(self, vertex: Vertex):
        self.vertices[vertex.id] = vertex

    def add_primitive(self, prim: Primitive):
        self.primitives.append(prim)
        if prim.layer == GeometryLayer.CONTOUR:
            self.contours.append(prim)
        elif prim.layer == GeometryLayer.AUXILIARY:
            self.auxiliaries.append(prim)
        elif prim.layer == GeometryLayer.ANNOTATION:
            self.annotations.append(prim)

    def get_layer_primitives(self, layer: GeometryLayer) -> List[Primitive]:
        if layer == GeometryLayer.CONTOUR:
            return self.contours
        elif layer == GeometryLayer.AUXILIARY:
            return self.auxiliaries
        elif layer == GeometryLayer.ANNOTATION:
            return self.annotations
        return []

    def to_dict(self) -> dict:
        """将结果转为可序列化字典"""
        return {
            'vertices': {
                vid: {
                    'id': v.id,
                    'position': (v.position.x, v.position.y),
                    'confidence': v.confidence,
                    'source': v.source.name,
                    'label': v.label,
                    'layer': v.layer.value,
                    'normalized_pos': (v.normalized_pos.x, v.normalized_pos.y) if v.normalized_pos else None,
                    'is_virtual': v.is_virtual,
                    'virtual_type': v.virtual_type,
                }
                for vid, v in self.vertices.items()
            },
            'primitives': [
                {
                    'id': p.id,
                    'type': p.type.value,
                    'layer': p.layer.value,
                    'line_type': p.line_type.value,
                    'vertices': p.vertices,
                    'params': {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                               for k, v in p.params.items()},
                    'confidence': p.confidence,
                }
                for p in self.primitives
            ],
            'image_shape': self.image_shape,
            'success': self.success,
            'error_message': self.error_message,
        }