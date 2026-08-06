"""
DeTikZify AI 识别后端
基于 DeTikZify 多模态大模型，将图片通过 AI 生成 TikZ 代码

安装依赖（可选）：
  pip install 'detikzify @ git+https://github.com/potamides/DeTikZify'

需要安装系统依赖：
  - TeX Live 2023+ (pdflatex, pdftoppm)
  - ghostscript
  - poppler
"""
import os
import sys
import gc
import time
import subprocess
import traceback
from datetime import datetime

from .recognizer import GeometryRecognizer, RecognitionResult, safe_imread
from .logger import logger

# ──────────────────────────────────────────────────────────────
# 检查 DeTikZify 是否可用
# ──────────────────────────────────────────────────────────────

DETIKZIFY_AVAILABLE = False
DETIKZIFY_ERROR = ""

try:
    import torch
    # 只做轻量检查，不实际加载模型
    import transformers
    DETIKZIFY_AVAILABLE = True
except ImportError as e:
    DETIKZIFY_ERROR = f"缺少依赖: {e}"
    logger.warning(f"DeTikZify 后端不可用: {DETIKZIFY_ERROR}")
    logger.warning("如需使用 AI 识别，请安装: pip install 'detikzify @ git+https://github.com/potamides/DeTikZify'")


def is_detikzify_available():
    """检查 DeTikZify 是否可导入"""
    return DETIKZIFY_AVAILABLE


def get_detikzify_error():
    """获取 DeTikZify 不可用的原因"""
    return DETIKZIFY_ERROR


# ──────────────────────────────────────────────────────────────
# 检查 LaTeX 环境
# ──────────────────────────────────────────────────────────────

def _check_command(cmd):
    """检查系统命令是否存在"""
    try:
        subprocess.run([cmd, '--version'], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return False


LATEX_AVAILABLE = _check_command("pdflatex")
PDFTOPPM_AVAILABLE = _check_command("pdftoppm")


# ──────────────────────────────────────────────────────────────
# DeTikZify 识别后端
# ──────────────────────────────────────────────────────────────

class DeTikZifyRecognizer(GeometryRecognizer):
    """
    基于 DeTikZify AI 模型的几何图形识别引擎
    继承自 GeometryRecognizer 接口，保持 API 兼容

    用法同 GeometryRecognizer:
      recognizer = DeTikZifyRecognizer(model_name="nllg/detikzify-v2.5-8b")
      result = recognizer.recognize("image.jpg")
    """

    def __init__(self, model_name="nllg/detikzify-v2.5-8b", device_map="auto",
                 torch_dtype="bfloat16", use_mcts=False, mcts_timeout=300):
        super().__init__()

        if not DETIKZIFY_AVAILABLE:
            raise ImportError(
                f"DeTikZify 不可用: {DETIKZIFY_ERROR}\n"
                "请安装: pip install 'detikzify @ git+https://github.com/potamides/DeTikZify'"
            )

        self.model_name = model_name
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.use_mcts = use_mcts
        self.mcts_timeout = mcts_timeout
        self._pipeline = None
        self._model_loaded = False

        logger.info(f"DeTikZifyRecognizer 初始化:")
        logger.info(f"  模型: {model_name}")
        logger.info(f"  MCTS: {'启用' if use_mcts else '禁用'}")
        logger.info(f"  LaTeX: {'可用' if LATEX_AVAILABLE else '不可用'}")

    def _load_model(self):
        """延迟加载模型（只在首次识别时加载）"""
        if self._model_loaded:
            return

        logger.info(f"正在加载 DeTikZify 模型: {self.model_name}...")
        logger.info("  首次加载可能需要下载模型文件（约 16GB）")
        start_time = time.time()

        try:
            from detikzify.model import load
            from detikzify.infer import DetikzifyPipeline

            model, processor = load(
                model_name_or_path=self.model_name,
                device_map=self.device_map,
                torch_dtype=self.torch_dtype,
            )
            self._pipeline = DetikzifyPipeline(model, processor)
            self._model_loaded = True

            elapsed = time.time() - start_time
            logger.info(f"模型加载完成，耗时 {elapsed:.1f} 秒")

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            logger.error(traceback.format_exc())
            raise

    def recognize(self, image_path):
        """使用 DeTikZify AI 模型识别图片，返回 RecognitionResult"""
        result = RecognitionResult()
        result.image_path = image_path

        try:
            logger.info(f"{'='*50}")
            logger.info(f"DeTikZify AI 识别: {image_path}")

            # 1. 加载图片
            img = safe_imread(image_path)
            if img is None:
                err_msg = f"无法读取图片: {image_path}"
                logger.error(err_msg)
                result.error = err_msg
                return result

            h, w = img.shape[:2]
            result.image_size = (w, h)
            logger.info(f"图片尺寸: {w}x{h}")

            # 2. 加载模型（延迟加载）
            self._load_model()

            # 3. AI 生成 TikZ 代码
            logger.info("AI 生成 TikZ 代码中...")
            start_time = time.time()

            if self.use_mcts:
                logger.info(f"使用 MCTS 推理（超时 {self.mcts_timeout}s）...")
                figs = set()
                for score, fig in self._pipeline.simulate(
                    image=image_path, timeout=self.mcts_timeout
                ):
                    figs.add((score, fig))
                # 取最优结果
                from operator import itemgetter
                best = sorted(figs, key=itemgetter(0))[-1][1]
                fig = best
            else:
                fig = self._pipeline.sample(image=image_path)

            elapsed = time.time() - start_time
            logger.info(f"AI 生成完成，耗时 {elapsed:.1f} 秒")

            # 4. 提取 TikZ 代码
            if hasattr(fig, 'tex_code') and fig.tex_code:
                result.tex_code = fig.tex_code
                tex_lines = result.tex_code.count('\n') + 1
                logger.info(f"TikZ 代码: {tex_lines} 行")
            elif hasattr(fig, 'code') and fig.code:
                result.tex_code = fig.code
            else:
                result.tex_code = str(fig)
                logger.info(f"TikZ 代码: {len(result.tex_code)} 字符")

            # 5. 编译预览图
            logger.info("编译 LaTeX 预览图...")
            preview_path = self._compile_to_png(result.tex_code)
            result.preview_image_path = preview_path
            if preview_path and os.path.exists(preview_path):
                logger.info(f"预览图: {preview_path}")
            else:
                logger.warning("预览图生成失败（可能缺少 LaTeX 环境）")

            # 6. 尝试提取关键点（DeTikZify 不直接提供，留空）
            result.key_points = {}
            result.valid_connections = []
            result.geo_info = {}

            # 尝试保存 TikZ 代码到文件
            tex_path = os.path.join(self.temp_dir, "detikzify_output.tex")
            with open(tex_path, 'w', encoding='utf-8') as f:
                f.write(result.tex_code)
            logger.info(f"TikZ 代码已保存: {tex_path}")

            result.success = True
            logger.info(f"{'='*50}")
            logger.info(f"DeTikZify 识别完成")
            return result

        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"DeTikZify 识别异常: {type(e).__name__}: {str(e)}")
            logger.error(tb)
            result.error = f"{type(e).__name__}: {str(e)}\n{tb}"
            return result

    def cleanup(self):
        """清理资源"""
        if self._pipeline is not None:
            try:
                del self._pipeline
            except Exception:
                pass
            self._pipeline = None
            self._model_loaded = False
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        super().cleanup()


# ──────────────────────────────────────────────────────────────
# 工厂函数
# ──────────────────────────────────────────────────────────────

def create_recognizer(backend="auto", **kwargs):
    """
    创建识别器实例

    参数:
      backend: "auto" | "cv" | "ai"
        - "auto" : 优先使用 AI（DeTikZify），不可用时回退到 CV
        - "cv"   : 强制使用 CV 几何算法
        - "ai"   : 强制使用 AI（DeTikZify），不可用时抛异常
      **kwargs: 传递给具体识别器的参数
        circle_pixel_tolerance: 圆形像素搜索半径（默认 2）
        circle_hit_threshold: 圆形命中率阈值（默认 0.50）

    返回:
      GeometryRecognizer 或 DeTikZifyRecognizer 实例
    """
    if backend == "ai" or (backend == "auto" and DETIKZIFY_AVAILABLE):
        if not DETIKZIFY_AVAILABLE:
            if backend == "ai":
                raise ImportError(f"DeTikZify 不可用: {DETIKZIFY_ERROR}")
            logger.info("DeTikZify 不可用，回退到 CV 几何算法")
            return GeometryRecognizer(**kwargs)
        logger.info("使用 DeTikZify AI 识别后端")
        # 提取 DeTikZifyRecognizer 的专属参数
        detikzify_kwargs = {}
        for key in ['model_name', 'device_map', 'torch_dtype', 'use_mcts', 'mcts_timeout']:
            if key in kwargs:
                detikzify_kwargs[key] = kwargs.pop(key)
        return DeTikZifyRecognizer(**detikzify_kwargs, **kwargs)
    else:
        logger.info("使用 CV 几何算法识别后端")
        return GeometryRecognizer(**kwargs)