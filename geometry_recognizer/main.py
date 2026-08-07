"""
主流水线封装模块

整合所有子模块，提供完整的端到端识别流水线。
"""

import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple
from .config import GeometryConfig
from .data_structures import (
    Vertex, Point, Primitive, GeometryResult, GeometryLayer,
    PrimitiveType, LineType, ConfidenceSource
)
from .preprocessing import ImagePreprocessor
from .vertex_detection import VertexDetector
from .primitive_recognition import PrimitiveRecognizer
from .topology import TopologyProcessor
from .latex_generator import LatexGenerator


class GeometryRecognizerPipeline:
    """
    几何识别主流水线

    完整流程:
    图像输入 → 预处理 → 顶点检测 → 图元识别 → 拓扑后处理 → LaTeX输出

    Usage:
        config = GeometryConfig()
        pipeline = GeometryRecognizerPipeline(config)
        result = pipeline.process(image)
        print(result.latex_code)
    """

    def __init__(self, config: Optional[GeometryConfig] = None):
        """
        初始化流水线

        Args:
            config: 几何识别配置，None则使用默认配置
        """
        self.config = config or GeometryConfig()

        # 初始化各子模块
        self.preprocessor = ImagePreprocessor(self.config)
        self.vertex_detector = VertexDetector(self.config)
        self.primitive_recognizer = PrimitiveRecognizer(self.config)
        self.topology_processor = TopologyProcessor(self.config)
        self.latex_generator = LatexGenerator(self.config)

        # 调试/中间结果存储
        self.debug_info: dict = {}

    def process(self, image: np.ndarray) -> GeometryResult:
        """
        执行完整识别流水线

        Args:
            image: 输入图像 (H, W, C) BGR 或 (H, W) 灰度

        Returns:
            GeometryResult 完整几何识别结果
        """
        # 初始化结果
        result = GeometryResult()
        result.image_shape = image.shape[:2]

        try:
            # ====== 阶段1: 图像预处理 ======
            gray, binary, skeleton, text_mask = self.preprocessor.process(image)
            self.debug_info['binary'] = binary
            self.debug_info['skeleton'] = skeleton
            self.debug_info['text_mask'] = text_mask

            # ====== 阶段2: 顶点检测 ======
            vertices = self.vertex_detector.detect_vertices(
                skeleton, binary, gray, text_mask
            )

            # ====== 阶段3: 图元识别 ======
            primitives = self.primitive_recognizer.recognize(
                skeleton, binary, vertices
            )

            # ====== 阶段4: 虚拟顶点推导 ======
            vertices = self.vertex_detector.derive_virtual_vertices(
                vertices, primitives
            )

            # ====== 阶段5: 线型检测 ======
            primitives = self.vertex_detector.detect_line_types(
                skeleton, primitives
            )

            # ====== 阶段6: 拓扑后处理 ======
            result = self.topology_processor.process(vertices, primitives)
            result.image_shape = image.shape[:2]

            # ====== 阶段7: LaTeX代码生成 ======
            if self.config.enable_latex_output:
                latex_code = self.latex_generator.generate(result)
                result.latex_code = latex_code

            result.success = True

        except Exception as e:
            result.success = False
            result.error_message = str(e)

        return result

    def process_with_visualization(
        self,
        image: np.ndarray
    ) -> Tuple[GeometryResult, Dict[str, np.ndarray]]:
        """
        处理并返回可视化中间结果

        Returns:
            (result, visualization_dict)
        """
        result = self.process(image)
        vis = self._create_visualization(result)
        return result, vis

    def _create_visualization(self, result: GeometryResult) -> Dict[str, np.ndarray]:
        """创建可视化中间结果图"""
        vis = {}

        if 'binary' in self.debug_info:
            vis['binary'] = self.debug_info['binary']

        if 'skeleton' in self.debug_info:
            vis['skeleton'] = self.debug_info['skeleton']

        # 顶点可视化
        if result.vertices:
            h, w = result.image_shape
            vertex_vis = np.ones((h, w, 3), dtype=np.uint8) * 255
            for vid, v in result.vertices.items():
                x, y = int(v.position.x), int(v.position.y)
                color = (0, 255, 0) if v.is_virtual else (0, 0, 255)
                cv2.circle(vertex_vis, (x, y), 4, color, -1)
                if v.label:
                    cv2.putText(vertex_vis, v.label, (x+5, y-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            vis['vertices'] = vertex_vis

        return vis

    # ============================================================
    # 人机交互接口 (第3.6.2节)
    # ============================================================

    def manual_edit_vertex(
        self,
        result: GeometryResult,
        vertex_id: str,
        new_position: Point
    ) -> GeometryResult:
        """
        手动编辑顶点坐标

        Args:
            result: 当前几何结果
            vertex_id: 顶点ID
            new_position: 新坐标

        Returns:
            更新后的几何结果
        """
        if vertex_id in result.vertices:
            result.vertices[vertex_id].position = new_position
            result.vertices[vertex_id].source = ConfidenceSource.MANUAL
            result.vertices[vertex_id].confidence = ConfidenceSource.MANUAL.score

            # 重新生成LaTeX
            if self.config.enable_latex_output:
                result.latex_code = self.latex_generator.generate(result)

        return result

    def manual_add_vertex(
        self,
        result: GeometryResult,
        position: Point,
        label: Optional[str] = None
    ) -> GeometryResult:
        """
        手动添加顶点

        Args:
            result: 当前几何结果
            position: 顶点坐标
            label: 顶点标签

        Returns:
            更新后的几何结果
        """
        # 生成顶点ID
        existing_ids = [int(vid[1:]) for vid in result.vertices.keys()
                        if vid[0] == 'V' and vid[1:].isdigit()]
        next_id = max(existing_ids) + 1 if existing_ids else 1
        vid = f"V{next_id}"

        vertex = Vertex(
            id=vid,
            position=position,
            confidence=ConfidenceSource.MANUAL.score,
            source=ConfidenceSource.MANUAL,
            label=label,
            layer=GeometryLayer.CONTOUR
        )
        result.add_vertex(vertex)

        # 重新生成LaTeX
        if self.config.enable_latex_output:
            result.latex_code = self.latex_generator.generate(result)

        return result

    def manual_delete_vertex(
        self,
        result: GeometryResult,
        vertex_id: str
    ) -> GeometryResult:
        """
        手动删除顶点

        Args:
            result: 当前几何结果
            vertex_id: 要删除的顶点ID

        Returns:
            更新后的几何结果
        """
        if vertex_id in result.vertices:
            del result.vertices[vertex_id]

            # 移除关联图元中的顶点引用
            for prim in result.primitives:
                if vertex_id in prim.vertices:
                    prim.vertices.remove(vertex_id)

            # 重新生成LaTeX
            if self.config.enable_latex_output:
                result.latex_code = self.latex_generator.generate(result)

        return result

    def manual_add_primitive(
        self,
        result: GeometryResult,
        prim_type: PrimitiveType,
        vertex_ids: List[str],
        line_type: LineType = LineType.SOLID,
        layer: GeometryLayer = GeometryLayer.CONTOUR
    ) -> GeometryResult:
        """
        手动添加图元

        Args:
            result: 当前几何结果
            prim_type: 图元类型
            vertex_ids: 关联顶点ID列表
            line_type: 线型
            layer: 图层

        Returns:
            更新后的几何结果
        """
        existing_ids = [int(pid[1:]) for pid in
                        [p.id for p in result.primitives]
                        if pid[0] == 'P' and pid[1:].isdigit()]
        next_id = max(existing_ids) + 1 if existing_ids else 1

        prim = Primitive(
            id=f"P{next_id}",
            type=prim_type,
            layer=layer,
            line_type=line_type,
            vertices=vertex_ids,
            confidence=1.0
        )
        result.add_primitive(prim)

        # 重新生成LaTeX
        if self.config.enable_latex_output:
            result.latex_code = self.latex_generator.generate(result)

        return result

    # ============================================================
    # 批量处理 (第3.6.3节)
    # ============================================================

    def batch_process(
        self,
        images: List[np.ndarray]
    ) -> List[GeometryResult]:
        """
        批量处理多张图片

        Args:
            images: 图像列表

        Returns:
            几何识别结果列表
        """
        return [self.process(img) for img in images]

    def batch_process_files(
        self,
        image_paths: List[str]
    ) -> List[GeometryResult]:
        """
        批量处理多个图片文件

        Args:
            image_paths: 图片文件路径列表

        Returns:
            几何识别结果列表
        """
        results = []
        for path in image_paths:
            image = cv2.imread(path)
            if image is not None:
                result = self.process(image)
                results.append(result)
        return results

    def batch_export_tex(
        self,
        results: List[GeometryResult],
        output_dir: str
    ) -> List[str]:
        """
        批量导出 .tex 源文件

        Args:
            results: 几何识别结果列表
            output_dir: 输出目录

        Returns:
            导出的文件路径列表
        """
        import os
        exported = []
        for i, result in enumerate(results):
            if result.latex_code:
                filename = f"geometry_output_{i:04d}.tex"
                filepath = os.path.join(output_dir, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(result.latex_code)
                exported.append(filepath)
        return exported