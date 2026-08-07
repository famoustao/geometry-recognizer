"""
日志系统模块
- 自动保存到程序目录下的 logs/run_YYYYMMDD.log
- 支持文件日志 + 控制台日志 + GUI 日志信号
- 自动识别 EXE 目录（PyInstaller 打包后也能正确保存）
- 紧急 crash 日志写入程序目录下的 crash_log.txt
"""
import os
import sys
import logging
import traceback
from datetime import datetime
from logging.handlers import RotatingFileHandler


def get_program_dir():
    """获取程序所在目录（兼容 PyInstaller EXE 和源码运行）"""
    if getattr(sys, 'frozen', False):
        # PyInstaller 打包后的 EXE
        return os.path.dirname(sys.executable)
    else:
        # 源码运行
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# 程序目录（用户可访问的目录，不是临时目录）
PROGRAM_DIR = get_program_dir()
LOG_DIR = os.path.join(PROGRAM_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# 日志级别映射
LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class LogSignal:
    """日志信号（用于 GUI 接收日志）"""
    _listeners = []

    @classmethod
    def add_listener(cls, callback):
        cls._listeners.append(callback)

    @classmethod
    def remove_listener(cls, callback):
        if callback in cls._listeners:
            cls._listeners.remove(callback)

    @classmethod
    def emit(cls, level, message):
        for cb in cls._listeners:
            try:
                cb(level, message)
            except Exception:
                pass


class GuiLogHandler(logging.Handler):
    """将日志发送到 GUI 信号"""
    def emit(self, record):
        try:
            level = record.levelname
            msg = self.format(record)
            LogSignal.emit(level, msg)
        except Exception:
            pass


def get_logger(name="geometry_recog"):
    """获取或创建日志器"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    # ── 文件日志（带轮转，最大 5MB，保留 3 个备份）──
    log_file = os.path.join(LOG_DIR, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    # ── 控制台日志 ──
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_fmt = logging.Formatter(
        '[%(levelname)s] %(message)s'
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # ── GUI 日志 ──
    gui_handler = GuiLogHandler()
    gui_handler.setLevel(logging.INFO)
    gui_fmt = logging.Formatter('%(message)s')
    gui_handler.setFormatter(gui_fmt)
    logger.addHandler(gui_handler)

    logger.info(f"日志文件: {log_file}")
    logger.info(f"程序目录: {PROGRAM_DIR}")
    # 直接刷新 handler，避免引用尚未定义好的全局 logger
    for handler in logger.handlers:
        try:
            handler.flush()
        except Exception:
            pass
    return logger


def flush_log():
    """强制刷新所有日志处理器，确保日志立即写入磁盘"""
    global logger
    try:
        for handler in logger.handlers:
            try:
                handler.flush()
            except Exception:
                pass
    except Exception:
        pass


# 全局默认日志器
logger = get_logger()


def log_exception(logger_instance=None, message="异常"):
    """记录异常堆栈"""
    lg = logger_instance or logger
    lg.error(f"{message}:\n{traceback.format_exc()}")


def get_log_file_path():
    """获取当前日志文件路径"""
    log_file = os.path.join(LOG_DIR, f"geometry_recog_{datetime.now().strftime('%Y%m%d')}.log")
    return log_file


def read_recent_logs(lines=100):
    """读取最近 N 行日志"""
    log_file = get_log_file_path()
    if not os.path.exists(log_file):
        return [f"[INFO] 日志文件不存在: {log_file}"]

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
        return [line.rstrip() for line in all_lines[-lines:]]
    except Exception as e:
        return [f"[ERROR] 读取日志失败: {e}"]


def write_crash_log(error_message, traceback_str):
    """紧急写入 crash 日志到程序目录（确保即使崩溃也能找到）"""
    crash_path = os.path.join(PROGRAM_DIR, "crash_log.txt")
    try:
        with open(crash_path, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"崩溃时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"程序目录: {PROGRAM_DIR}\n")
            f.write(f"错误: {error_message}\n")
            f.write(f"堆栈:\n{traceback_str}\n")
            f.write(f"{'='*60}\n")
        print(f"[紧急] crash 日志已保存: {crash_path}")
    except Exception as e:
        print(f"[紧急] 无法写入 crash 日志: {e}")