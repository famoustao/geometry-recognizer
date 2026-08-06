"""
日志系统模块
支持文件日志 + 控制台日志 + 日志轮转
"""
import os
import sys
import logging
import traceback
from datetime import datetime
from logging.handlers import RotatingFileHandler


# 全局日志目录
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
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

    return logger


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