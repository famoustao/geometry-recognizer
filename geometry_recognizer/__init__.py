"""
Geometry Recognizer: 基于多特征融合的手绘几何图形矢量化识别系统
=============================================================
优化升级完整版 - 适配中学平面几何手绘图像识别

核心流程:
  透视畸变校正 → 光照均衡 → 文字屏蔽 → 形态学修复 → 骨架化
  → 多源顶点检测 → DBSCAN聚类 → 置信度融合 → 虚拟顶点推导
  → 直曲分类 → 圆弧识别 → 直线校验 → 全局求交修正
  → 几何弱规整 → 分层存储 → LaTeX/TikZ代码生成
"""

from .data_structures import (
    Vertex, ConfidenceSource, LineType, PrimitiveType,
    Primitive, GeometryLayer, GeometryResult, Point
)
from .config import GeometryConfig, AdaptiveThresholds
from .preprocessing import ImagePreprocessor
from .vertex_detection import VertexDetector
from .primitive_recognition import PrimitiveRecognizer
from .topology import TopologyProcessor
from .latex_generator import LatexGenerator
from .main import GeometryRecognizerPipeline

__version__ = "2.0.0"
__all__ = [
    'Vertex', 'ConfidenceSource', 'LineType', 'PrimitiveType',
    'Primitive', 'GeometryLayer', 'GeometryResult', 'Point',
    'GeometryConfig', 'AdaptiveThresholds',
    'ImagePreprocessor', 'VertexDetector', 'PrimitiveRecognizer',
    'TopologyProcessor', 'LatexGenerator', 'GeometryRecognizerPipeline',
]