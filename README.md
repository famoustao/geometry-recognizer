# 基于多特征融合的手绘几何图形矢量化识别系统

手绘几何图形 → 矢量化识别 → LaTeX/TikZ代码 → 编译为图片

## 功能

- **命令行模式**：`python run_uploaded_v4.py` — 单张图片识别
- **桌面 GUI 模式**：`python gui_app.py` — 批量处理、预览、导出

## 桌面 GUI 功能

| 功能 | 说明 |
|------|------|
| 图片管理 | 添加单张/整个文件夹，移除/清空列表 |
| 批量识别 | 一键识别全部图片，后台线程不卡界面 |
| 原图预览 | 显示原图 + 自动标注识别出的顶点(A,B,C,...) |
| LaTeX预览 | 显示编译后的 TikZ 渲染结果 |
| 代码显示 | 深色主题 + 语法高亮（关键字/数字/注释/标签分色） |
| 复制代码 | 一键复制到剪贴板 |
| 导出 TEX | 单文件导出 .tex |
| 导出 PNG | 单文件导出编译后的 .png |
| 批量导出 | 全部导出为独立 .tex 文件 |
| 合并导出 | 所有识别结果合并为一个 PDF（含原图+代码+渲染） |

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

## 安装与运行

### 方式一：源码运行

```bash
# 安装依赖
pip install -r requirements.txt --break-system-packages

# 系统依赖（LaTeX编译用）
apt-get install -y texlive-latex-base texlive-latex-extra texlive-pictures poppler-utils libegl1-mesa

# 命令行模式
python run_uploaded_v4.py

# 桌面 GUI 模式
python gui_app.py
```

### 方式二：下载编译好的 EXE（Windows）

从 [GitHub Releases](https://github.com/famoustao/geometry-recognizer/releases) 或 Actions 产物下载 `GeometryRecognizer-Windows-x64.zip`，解压后双击运行即可。

> **注意**：EXE 版本不包含 LaTeX 编译环境，LaTeX 预览功能不可用，但 TikZ 代码导出功能正常。

### 方式三：自行编译

```bash
pip install pyinstaller
pyinstaller --onefile --windowed --name GeometryRecognizer \
  --add-data "geometry_app;geometry_app" \
  --add-data "geometry_recognizer;geometry_recognizer" \
  gui_app.py
```

## 项目结构

```
├── gui_app.py                    # 桌面 GUI 主程序（入口）
├── run_uploaded_v4.py            # 命令行版本
├── geometry_app/
│   ├── __init__.py
│   └── recognizer.py             # 识别引擎（可复用 API）
├── geometry_recognizer/          # 原算法库（模块化）
│   ├── config.py
│   ├── data_structures.py
│   ├── latex_generator.py
│   ├── main.py
│   ├── preprocessing.py
│   ├── primitive_recognition.py
│   ├── topology.py
│   ├── utils.py
│   └── vertex_detection.py
├── uploaded_geometry.jpeg        # 测试图片
└── .github/workflows/
    ├── build.yml                 # 自动编译 EXE
    └── compile.yml               # 自动 LaTeX 编译
```

## GitHub Actions 自动编译

每次推送代码到 `main` 分支或打 `v*` 标签时，自动执行：

1. **build.yml** → Windows/macOS 双平台编译为独立 EXE/APP
2. **compile.yml** → Linux 环境运行识别算法并编译 LaTeX 结果

编译产物可在 Actions 页面下载，或发布为 Release。