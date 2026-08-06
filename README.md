# 基于多特征融合的手绘几何图形矢量化识别算法

手绘几何图形 → 矢量化识别 → LaTeX/TikZ代码 → 编译为图片

## 算法流程

1. **文字屏蔽**：连通域分析，腐蚀断开文字与线条的连接
2. **线条检测**：多阈值Canny边缘检测 + HoughLinesP，合并相近线段
3. **直线交点分析**：计算所有线段交点，DBSCAN聚类得到候选顶点
4. **几何推理**：识别三角形顶点，计算半圆、垂足、交点
5. **线段验证**：
   - 几何约束表：只测试符合几何定义的连接
   - 连通性验证：最长连续暗段检测
   - 多点Hough匹配
6. **TikZ生成**：检测层（图像验证）+ 构造层（几何定义自动补全）

## 使用

```bash
# 安装依赖
pip install -r requirements.txt --break-system-packages

# 运行
python run_uploaded_v4.py
```

## 输出
- `uploaded_v4.tex` - 生成的TikZ代码
- `uploaded_v4_final.png` - 编译后的图片
- `debug4_*.png` - 各步骤调试图