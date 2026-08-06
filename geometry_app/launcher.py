#!/usr/bin/env python3
"""几何识别工具 - 桌面启动器"""
import os, sys, threading, webbrowser, time, socket

# PyInstaller 打包后，模板文件在 sys._MEIPASS 中
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
    # 确保能找到 app 模块
    sys.path.insert(0, BASE_DIR)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from app import app

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def open_browser(port, delay=1.5):
    time.sleep(delay)
    webbrowser.open(f'http://localhost:{port}')

if __name__ == '__main__':
    port = 5000
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('', port))
        s.close()
    except OSError:
        port = find_free_port()

    print(f"  ╔════════════════════════════════════════════╗")
    print(f"  ║       📐 几何识别工具 v4.0                 ║")
    print(f"  ║  基于多特征融合的手绘几何图形矢量化识别     ║")
    print(f"  ╠════════════════════════════════════════════╣")
    print(f"  ║  正在启动服务...                           ║")
    print(f"  ║  浏览器已自动打开                          ║")
    print(f"  ║  访问地址: http://localhost:{port}            ║")
    print(f"  ║  关闭此窗口即停止服务                      ║")
    print(f"  ╚════════════════════════════════════════════╝")

    threading.Thread(target=open_browser, args=(port,), daemon=True).start()
    app.run(host='127.0.0.1', port=port, debug=False, threaded=True)